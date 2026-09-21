//! Real runtime test of `spoof::install`/`remove` end to end: patches this binary's
//! own `GetModuleFileNameW` IAT slot, confirms a call asking about this process's own
//! module now reports the spoofed `"nvngx.dll"`, confirms a call about something else
//! is unaffected, then removes the spoof and confirms the real name comes back.
//!
//! Run under Wine: `cargo run --example spoof_install_test --target x86_64-pc-windows-gnu -p neural-forge-helper`

use std::ffi::c_void;

use neural_forge_helper::spoof;

#[link(name = "kernel32")]
extern "system" {
    fn GetModuleHandleW(module_name: *const u16) -> *mut c_void;
    fn GetModuleFileNameW(module: *mut c_void, filename: *mut u16, size: u32) -> u32;
}

fn get_module_file_name(module: *mut c_void) -> String {
    let mut buf = [0u16; 512];
    // SAFETY: `buf` is a valid, writable buffer of the length passed.
    let len = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) };
    String::from_utf16_lossy(&buf[..len as usize])
}

fn main() {
    // SAFETY: NULL means "this process's own main module".
    let this_module: *mut c_void = unsafe { GetModuleHandleW(std::ptr::null()) };
    let kernel32_name_w: Vec<u16> = "kernel32".encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: `kernel32_name_w` is a valid NUL-terminated UTF-16 string; `kernel32.dll`
    // is always loaded.
    let kernel32 = unsafe { GetModuleHandleW(kernel32_name_w.as_ptr()) };

    let real_name = get_module_file_name(this_module);
    println!("before install, this module's real name: {real_name}");
    assert!(!real_name.is_empty());

    // SAFETY: `this_module` is a valid, currently-loaded module handle.
    let Some(installed) = (unsafe { spoof::install(this_module) }) else {
        panic!("FAIL: spoof::install returned None");
    };

    let spoofed_name = get_module_file_name(this_module);
    println!("after install, this module's reported name: {spoofed_name}");
    assert_eq!(spoofed_name, "nvngx.dll", "FAIL: expected the spoofed name");

    // A query about a *different* module must be completely unaffected -- the spoof
    // only ever intercepts a query about this process's own module.
    let kernel32_name = get_module_file_name(kernel32);
    println!("kernel32's own reported name (should be real, unaffected): {kernel32_name}");
    assert!(
        kernel32_name.to_lowercase().contains("kernel32"),
        "FAIL: spoofing this module's identity affected an unrelated module's query"
    );

    // SAFETY: `this_module` is still loaded (it's this process's own main module).
    unsafe { spoof::remove(installed) };

    let restored_name = get_module_file_name(this_module);
    println!("after remove, this module's reported name: {restored_name}");
    assert_eq!(restored_name, real_name, "FAIL: the real name did not come back after remove");

    println!("PASS: install/remove both work, and only this process's own identity was ever affected");
}
