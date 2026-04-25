use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::ffi::CStr;

/* ── C FFI declarations ──────────────────────────────────── */

#[repr(C)]
struct CpuInfo {
    vendor: [u8; 64],
    brand: [u8; 128],
    arch: [u8; 32],
    physical_cores: i32,
    logical_cores: i32,
    numa_nodes: i32,
    has_sse2: i32,
    has_sse3: i32,
    has_ssse3: i32,
    has_sse41: i32,
    has_sse42: i32,
    has_avx: i32,
    has_avx2: i32,
    has_fma3: i32,
    has_avx512f: i32,
    has_avx512bw: i32,
    has_avx512vl: i32,
    has_neon: i32,
    has_sve: i32,
    has_sve2: i32,
    has_fp16: i32,
    l1d_cache_bytes: usize,
    l1i_cache_bytes: usize,
    l2_cache_bytes: usize,
    l3_cache_bytes: usize,
    cache_line_bytes: usize,
    base_freq_mhz: i32,
}

#[repr(C)]
struct MemInfo {
    total_ram_bytes: usize,
    available_ram_bytes: usize,
    page_size_bytes: usize,
    huge_pages_supported: i32,
    huge_page_size_bytes: usize,
}

#[repr(C)]
struct GpuInfo {
    cuda_device_count: i32,
    cuda_device_name: [u8; 128],
    cuda_total_mem_bytes: usize,
    cuda_compute_major: i32,
    cuda_compute_minor: i32,
    opencl_available: i32,
}

#[repr(C)]
struct OsInfo {
    os_name: [u8; 64],
    os_release: [u8; 128],
    hostname: [u8; 128],
    pid: i32,
}

#[repr(C)]
struct Suitability {
    overall_status: i32,
    cpu_ok: i32,
    ram_ok: i32,
    simd_ok: i32,
    messages: [[u8; 256]; 8],
    message_count: i32,
}

extern "C" {
    fn dtflowcv_normalize_dispatch(
        input: *const u8,
        output: *mut f32,
        height: usize,
        width: usize,
        channels: usize,
        mean: *const f32,
        std: *const f32,
    );
    fn dtflowcv_cpu_has_avx512f() -> i32;
    fn dtflowcv_cpu_has_sse2() -> i32;
    fn dtflowcv_box_iou_matrix(
        boxes_a: *const f32,
        n_a: usize,
        boxes_b: *const f32,
        n_b: usize,
        out: *mut f32,
    );

    fn dtflowcv_detect_cpu(out: *mut CpuInfo);
    fn dtflowcv_detect_mem(out: *mut MemInfo);
    fn dtflowcv_detect_gpu(out: *mut GpuInfo);
    fn dtflowcv_detect_os(out: *mut OsInfo);
    fn dtflowcv_check_suitability(cpu: *const CpuInfo, mem: *const MemInfo, out: *mut Suitability);
    fn dtflowcv_hw_recommendations(
        cpu: *const CpuInfo,
        mem: *const MemInfo,
        gpu: *const GpuInfo,
        output: *mut u8,
        output_size: usize,
    );
}

/* ── Helpers ─────────────────────────────────────────────── */

fn c_buf_to_string(buf: &[u8]) -> String {
    let nul_pos = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..nul_pos]).trim().to_string()
}

/* ── Python-exposed functions ────────────────────────────── */

#[pyfunction]
fn capabilities() -> BTreeMap<&'static str, bool> {
    let mut out = BTreeMap::new();
    out.insert("c_preprocess_kernel", true);
    out.insert("sse2_kernel", true);
    out.insert("dispatch_kernel", true);
    out.insert("box_iou_matrix", true);
    out.insert("hwinfo", true);
    out.insert("cpu_avx512f", unsafe { dtflowcv_cpu_has_avx512f() == 1 });
    out.insert("cpu_sse2", unsafe { dtflowcv_cpu_has_sse2() == 1 });
    out
}

#[pyfunction]
fn normalize_hwc_u8_to_chw_f32<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, u8>,
    mean: [f32; 3],
    std: [f32; 3],
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let shape = image.shape();
    let height = shape[0];
    let width = shape[1];
    let channels = shape[2];
    if channels != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "image must have HWC shape with 3 channels",
        ));
    }
    if !image.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "image must be C-contiguous",
        ));
    }
    let input = image
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("image must be contiguous"))?;

    let output = PyArray3::<f32>::zeros(py, [3, height, width], false);
    {
        let mut output_rw = output.readwrite();
        let output_slice = output_rw
            .as_slice_mut()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("output must be contiguous"))?;

        let input_ptr = input.as_ptr();
        let output_ptr = output_slice.as_mut_ptr();
        let mean_arr = mean;
        let std_arr = std;
        let h = height;
        let w = width;

        py.allow_threads(|| {
            unsafe {
                dtflowcv_normalize_dispatch(
                    input_ptr, output_ptr, h, w, 3,
                    mean_arr.as_ptr(), std_arr.as_ptr(),
                );
            }
        });
    }
    Ok(output)
}

#[pyfunction]
fn box_iou_matrix<'py>(
    py: Python<'py>,
    boxes_a: PyReadonlyArray2<'py, f32>,
    boxes_b: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape_a = boxes_a.shape();
    let shape_b = boxes_b.shape();
    if shape_a[1] != 4 || shape_b[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("boxes must have shape (N, 4)"));
    }
    if !boxes_a.is_c_contiguous() || !boxes_b.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err("boxes must be C-contiguous"));
    }
    let n_a = shape_a[0];
    let n_b = shape_b[0];
    let a_slice = boxes_a.as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("boxes_a must be contiguous"))?;
    let b_slice = boxes_b.as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("boxes_b must be contiguous"))?;

    let output = PyArray2::<f32>::zeros(py, [n_a, n_b], false);
    {
        let mut output_rw = output.readwrite();
        let output_slice = output_rw.as_slice_mut()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("output must be contiguous"))?;
        let a_ptr = a_slice.as_ptr();
        let b_ptr = b_slice.as_ptr();
        let out_ptr = output_slice.as_mut_ptr();
        py.allow_threads(|| {
            unsafe { dtflowcv_box_iou_matrix(a_ptr, n_a, b_ptr, n_b, out_ptr); }
        });
    }
    Ok(output)
}

#[pyfunction]
fn nms_boxes<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'py, f32>,
    scores: PyReadonlyArray1<'py, f32>,
    iou_threshold: f32,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let n = boxes.shape()[0];
    if boxes.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("boxes must be (N,4)"));
    }
    if scores.shape()[0] != n {
        return Err(pyo3::exceptions::PyValueError::new_err("scores length must match boxes"));
    }
    let b = boxes.as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("boxes must be contiguous"))?;
    let s = scores.as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("scores must be contiguous"))?;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b_idx| s[b_idx].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut suppressed = vec![false; n];
    let mut keep: Vec<i64> = Vec::with_capacity(n);

    for &idx in &order {
        if suppressed[idx] { continue; }
        keep.push(idx as i64);
        let ax1 = b[idx * 4];
        let ay1 = b[idx * 4 + 1];
        let ax2 = b[idx * 4 + 2];
        let ay2 = b[idx * 4 + 3];
        let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);

        for &other in &order {
            if suppressed[other] || other == idx { continue; }
            let bx1 = b[other * 4];
            let by1 = b[other * 4 + 1];
            let bx2 = b[other * 4 + 2];
            let by2 = b[other * 4 + 3];
            let ix1 = ax1.max(bx1);
            let iy1 = ay1.max(by1);
            let ix2 = ax2.min(bx2);
            let iy2 = ay2.min(by2);
            let iw = (ix2 - ix1).max(0.0);
            let ih = (iy2 - iy1).max(0.0);
            let inter = iw * ih;
            let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
            let union = area_a + area_b - inter;
            if union > 0.0 && inter / union >= iou_threshold {
                suppressed[other] = true;
            }
        }
    }
    Ok(PyArray1::from_vec(py, keep))
}

/// Full hardware detection report as a Python dict.
#[pyfunction]
fn hardware_info() -> BTreeMap<String, PyObject> {
    Python::with_gil(|py| {
        let mut result = BTreeMap::new();

        unsafe {
            // CPU
            let mut cpu = std::mem::zeroed::<CpuInfo>();
            dtflowcv_detect_cpu(&mut cpu);

            let mut cpu_map = BTreeMap::<String, PyObject>::new();
            cpu_map.insert("arch".into(), c_buf_to_string(&cpu.arch).into_pyobject(py).unwrap().into_any().unbind());
            cpu_map.insert("vendor".into(), c_buf_to_string(&cpu.vendor).into_pyobject(py).unwrap().into_any().unbind());
            cpu_map.insert("brand".into(), c_buf_to_string(&cpu.brand).into_pyobject(py).unwrap().into_any().unbind());
            cpu_map.insert("physical_cores".into(), cpu.physical_cores.into_pyobject(py).unwrap().into_any().unbind());
            cpu_map.insert("logical_cores".into(), cpu.logical_cores.into_pyobject(py).unwrap().into_any().unbind());
            cpu_map.insert("numa_nodes".into(), cpu.numa_nodes.into_pyobject(py).unwrap().into_any().unbind());
            cpu_map.insert("base_freq_mhz".into(), cpu.base_freq_mhz.into_pyobject(py).unwrap().into_any().unbind());

            // SIMD flags
            let mut simd = BTreeMap::<String, PyObject>::new();
            simd.insert("sse2".into(), (cpu.has_sse2 != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("sse3".into(), (cpu.has_sse3 != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("ssse3".into(), (cpu.has_ssse3 != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("sse41".into(), (cpu.has_sse41 != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("sse42".into(), (cpu.has_sse42 != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("avx".into(), (cpu.has_avx != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("avx2".into(), (cpu.has_avx2 != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("fma3".into(), (cpu.has_fma3 != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("avx512f".into(), (cpu.has_avx512f != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("avx512bw".into(), (cpu.has_avx512bw != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("avx512vl".into(), (cpu.has_avx512vl != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("neon".into(), (cpu.has_neon != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("sve".into(), (cpu.has_sve != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("sve2".into(), (cpu.has_sve2 != 0).into_pyobject(py).unwrap().into_any().unbind());
            simd.insert("fp16".into(), (cpu.has_fp16 != 0).into_pyobject(py).unwrap().into_any().unbind());
            cpu_map.insert("simd".into(), simd.into_pyobject(py).unwrap().into_any().unbind());

            // Cache
            let mut cache = BTreeMap::<String, PyObject>::new();
            cache.insert("l1d_bytes".into(), cpu.l1d_cache_bytes.into_pyobject(py).unwrap().into_any().unbind());
            cache.insert("l1i_bytes".into(), cpu.l1i_cache_bytes.into_pyobject(py).unwrap().into_any().unbind());
            cache.insert("l2_bytes".into(), cpu.l2_cache_bytes.into_pyobject(py).unwrap().into_any().unbind());
            cache.insert("l3_bytes".into(), cpu.l3_cache_bytes.into_pyobject(py).unwrap().into_any().unbind());
            cache.insert("line_bytes".into(), cpu.cache_line_bytes.into_pyobject(py).unwrap().into_any().unbind());
            cpu_map.insert("cache".into(), cache.into_pyobject(py).unwrap().into_any().unbind());

            result.insert("cpu".into(), cpu_map.into_pyobject(py).unwrap().into_any().unbind());

            // Memory
            let mut mem = std::mem::zeroed::<MemInfo>();
            dtflowcv_detect_mem(&mut mem);
            let mut mem_map = BTreeMap::<String, PyObject>::new();
            mem_map.insert("total_bytes".into(), mem.total_ram_bytes.into_pyobject(py).unwrap().into_any().unbind());
            mem_map.insert("available_bytes".into(), mem.available_ram_bytes.into_pyobject(py).unwrap().into_any().unbind());
            mem_map.insert("page_size".into(), mem.page_size_bytes.into_pyobject(py).unwrap().into_any().unbind());
            mem_map.insert("huge_pages".into(), (mem.huge_pages_supported != 0).into_pyobject(py).unwrap().into_any().unbind());
            mem_map.insert("huge_page_size".into(), mem.huge_page_size_bytes.into_pyobject(py).unwrap().into_any().unbind());
            result.insert("memory".into(), mem_map.into_pyobject(py).unwrap().into_any().unbind());

            // GPU
            let mut gpu = std::mem::zeroed::<GpuInfo>();
            dtflowcv_detect_gpu(&mut gpu);
            let mut gpu_map = BTreeMap::<String, PyObject>::new();
            gpu_map.insert("cuda_device_count".into(), gpu.cuda_device_count.into_pyobject(py).unwrap().into_any().unbind());
            gpu_map.insert("cuda_device_name".into(), c_buf_to_string(&gpu.cuda_device_name).into_pyobject(py).unwrap().into_any().unbind());
            gpu_map.insert("cuda_total_mem_bytes".into(), gpu.cuda_total_mem_bytes.into_pyobject(py).unwrap().into_any().unbind());
            gpu_map.insert("cuda_compute".into(), format!("{}.{}", gpu.cuda_compute_major, gpu.cuda_compute_minor).into_pyobject(py).unwrap().into_any().unbind());
            result.insert("gpu".into(), gpu_map.into_pyobject(py).unwrap().into_any().unbind());

            // OS
            let mut os = std::mem::zeroed::<OsInfo>();
            dtflowcv_detect_os(&os);
            let mut os_map = BTreeMap::<String, PyObject>::new();
            os_map.insert("name".into(), c_buf_to_string(&os.os_name).into_pyobject(py).unwrap().into_any().unbind());
            os_map.insert("release".into(), c_buf_to_string(&os.os_release).into_pyobject(py).unwrap().into_any().unbind());
            os_map.insert("hostname".into(), c_buf_to_string(&os.hostname).into_pyobject(py).unwrap().into_any().unbind());
            result.insert("os".into(), os_map.into_pyobject(py).unwrap().into_any().unbind());

            // Suitability
            let mut suit = std::mem::zeroed::<Suitability>();
            dtflowcv_check_suitability(&cpu, &mem, &mut suit);
            let mut suit_map = BTreeMap::<String, PyObject>::new();
            let status_str = match suit.overall_status {
                0 => "OK",
                1 => "WARNING",
                _ => "INSUFFICIENT",
            };
            suit_map.insert("status".into(), status_str.into_pyobject(py).unwrap().into_any().unbind());
            suit_map.insert("cpu_ok".into(), (suit.cpu_ok != 0).into_pyobject(py).unwrap().into_any().unbind());
            suit_map.insert("ram_ok".into(), (suit.ram_ok != 0).into_pyobject(py).unwrap().into_any().unbind());
            suit_map.insert("simd_ok".into(), (suit.simd_ok != 0).into_pyobject(py).unwrap().into_any().unbind());
            let mut msgs = Vec::new();
            for i in 0..suit.message_count as usize {
                msgs.push(c_buf_to_string(&suit.messages[i]));
            }
            suit_map.insert("messages".into(), msgs.into_pyobject(py).unwrap().into_any().unbind());
            result.insert("suitability".into(), suit_map.into_pyobject(py).unwrap().into_any().unbind());

            // Recommendations text
            let mut rec_buf = vec![0u8; 4096];
            dtflowcv_hw_recommendations(&cpu, &mem, &gpu, rec_buf.as_mut_ptr(), 4096);
            let rec_str = c_buf_to_string(&rec_buf);
            result.insert("recommendations".into(), rec_str.into_pyobject(py).unwrap().into_any().unbind());
        }

        result
    })
}

#[pymodule]
fn dtflowcv_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_hwc_u8_to_chw_f32, module)?)?;
    module.add_function(wrap_pyfunction!(box_iou_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(nms_boxes, module)?)?;
    module.add_function(wrap_pyfunction!(hardware_info, module)?)?;
    Ok(())
}
