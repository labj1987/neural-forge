//! Real runtime test of `spoof::find_imported_function_slots`, needing no NVIDIA DLL at
//! all: a compiled `x86_64-pc-windows-gnu` binary imports `GetModuleFileNameW` from
//! `KERNEL32.dll` itself (indirectly, via the Rust standard library), so this binary's
//! own loaded image is a real, self-contained PE import table to test the parser
//! against.
//!
//! Run under Wine: `cargo run --example spoof_test --target x86_64-pc-windows-gnu -p neural-forge-helper`

use std::ffi::c_void;

use neural_forge_helper::spoof;

#[link(name = "kernel32")]
extern "system" {
    fn GetModuleHandleW(module_name: *const u16) -> *mut c_void;
    fn GetModuleFileNameW(module: *mut c_void, filename: *mut u16, size: u32) -> u32;
}

fn main() {
    // SAFETY: NULL means "this process's own main module", which is always valid.
    let this_module: *mut c_void = unsafe { GetModuleHandleW(std::ptr::null()) };
    println!("this_module = {this_module:?}");
    assert!(!this_module.is_null());

    // SAFETY: `this_module` is a valid, currently-loaded module handle (this process's
    // own, per `GetModuleHandleW` above).
    let slots = unsafe { spoof::find_imported_function_slots(this_module, "GetModuleFileNameW") };
    println!("find_imported_function_slots(GetModuleFileNameW) = {slots:?}");
    let Some(&slot) = slots.first() else {
        panic!("FAIL: did not find GetModuleFileNameW in this binary's own import table");
    };

    // The slot must actually hold a real, callable function pointer -- i.e. the parser
    // found the *real* IAT entry (already resolved by the loader before main() ran),
    // not some other, coincidentally-zero or garbage location.
    // SAFETY: `slot` was just returned by `find_imported_function_slots` as pointing at
    // a valid, in-image, pointer-sized IAT entry.
    let resolved = unsafe { *slot };
    println!("resolved function pointer at slot: {resolved:#x}");
    assert_ne!(resolved, 0, "FAIL: the slot found holds a null/unresolved pointer");

    // Informational only, not asserted: `GetModuleFileNameW` referenced this way is
    // this binary's own call stub/thunk address, not necessarily bit-identical to the
    // real IAT slot's resolved value -- `resolved != 0` above is the load-bearing check.
    let directly_linked = GetModuleFileNameW as *const () as usize;
    println!("GetModuleFileNameW (linked directly, informational) = {directly_linked:#x}");

    println!("PASS: found a real, resolved IAT slot for GetModuleFileNameW");
}
