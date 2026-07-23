fn main() {
    println!("cargo:rerun-if-changed=src/sleep_diagnostics_macos.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("src/sleep_diagnostics_macos.c")
            .flag_if_supported("-std=c11")
            .compile("neuracore_sleep_diagnostics");
    }
}
