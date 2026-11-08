// The window and Explorer icon, built into Sable.exe (resource 1).
fn main() {
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon_with_id("app.ico", "1");
        res.set("ProductName", "Sable");
        res.set("FileDescription", "Sable");
        let _ = res.compile();
    }
    println!("cargo:rerun-if-changed=app.ico");
}
