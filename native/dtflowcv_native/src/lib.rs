use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::collections::BTreeMap;

extern "C" {
    fn dtflowcv_normalize_hwc_u8_to_chw_f32(
        input: *const u8,
        output: *mut f32,
        height: usize,
        width: usize,
        channels: usize,
        mean: *const f32,
        std: *const f32,
    );
    fn dtflowcv_cpu_has_avx512f() -> i32;
}

#[pyfunction]
fn capabilities() -> BTreeMap<&'static str, bool> {
    let mut out = BTreeMap::new();
    out.insert("c_preprocess_kernel", true);
    out.insert("cpu_avx512f", unsafe { dtflowcv_cpu_has_avx512f() == 1 });
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
        unsafe {
            dtflowcv_normalize_hwc_u8_to_chw_f32(
                input.as_ptr(),
                output_slice.as_mut_ptr(),
                height,
                width,
                channels,
                mean.as_ptr(),
                std.as_ptr(),
            );
        }
    }
    Ok(output)
}

#[pymodule]
fn dtflowcv_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_hwc_u8_to_chw_f32, module)?)?;
    Ok(())
}
