fn main() {
    // Embed icon resource into the Windows .exe (for both CLI and GUI binaries)
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.compile().expect("Failed to compile Windows resource");
    }
}
