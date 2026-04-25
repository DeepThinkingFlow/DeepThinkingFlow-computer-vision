fn main() {
    cc::Build::new()
        .file("c/preprocess.c")
        .include("c")
        .flag_if_supported("-O3")
        .compile("dtflowcv_preprocess");
}
