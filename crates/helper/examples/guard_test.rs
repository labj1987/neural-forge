//! Real runtime test of `guard::guarded`: deliberately faults (dereferences an
//! unmapped address) inside a guarded closure and confirms the process survives and
//! reports the SEH exception code, instead of crashing outright. Also confirms a
//! normal, faultless call still returns the real result through the same path.
//!
//! Deliberately *not* a literal null (`0 as *const _`) dereference: Rust's debug-mode
//! "unsafe precondition checks" detect that specific case and turn it into a clean
//! panic (which this workspace's `panic = "abort"` then turns into a fail-fast,
//! `0xC0000409`) before the CPU ever actually faults -- confirmed the hard way, this
//! test's first version used exactly that and never touched the VEH handler at all. A
//! high, clearly-unmapped-but-non-null address has no such special-cased check and
//! reaches a real hardware page fault, which is what this test needs to prove
//! anything about `guard::guarded` at all.
//!
//! Run under Wine: `cargo run --example guard_test --target x86_64-pc-windows-gnu -p neural-forge-helper`
use neural_forge_helper::guard;

fn main() {
    guard::install();

    let (ok_result, ok_seh) = guard::guarded(|| 40 + 2, -1);
    println!("no-fault call: result={ok_result} seh={ok_seh:#x} (expect 42, 0x0)");
    assert_eq!(ok_result, 42);
    assert_eq!(ok_seh, 0);

    println!("about to deliberately fault inside a guarded call...");
    let (fault_result, fault_seh) = guard::guarded(
        || {
            let p = 0xdead_beef_0000usize as *const i32;
            // SAFETY: none -- this is the deliberate fault this test exists to trigger.
            unsafe { std::ptr::read_volatile(p) }
        },
        -1,
    );
    println!("faulted call: result={fault_result} seh={fault_seh:#x}");
    // EXCEPTION_ACCESS_VIOLATION = 0xC0000005.
    assert_eq!(fault_result, -1, "the guard should have returned the fail value, not a real read");
    assert_eq!(fault_seh, 0xC000_0005, "expected EXCEPTION_ACCESS_VIOLATION");

    println!("PASS: the process survived a real access violation inside guard::guarded");
}
