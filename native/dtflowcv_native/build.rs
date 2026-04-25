fn main() {
    cc::Build::new()
        .file("c/preprocess.c")
        .file("c/hwinfo.c")
        .include("c")
        .flag_if_supported("-O3")
        .flag_if_supported("-msse2")
        .flag_if_supported("-ffast-math")
        .flag_if_supported("-funroll-loops")
        .flag_if_supported("-march=native")
        .compile("dtflowcv_native_c");
}
