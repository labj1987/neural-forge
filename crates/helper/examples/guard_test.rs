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

#[link(name = "kernel32")]
extern "system" {
    fn OutputDebugStringA(s: *const std::ffi::c_char);
}

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

    // The fault latches process-wide: no later guarded call may run at all, because the
    // faulting DLL may still hold its locks.
    assert_eq!(guard::faulted(), Some(0xC000_0005));
    let mut ran = false;
    let (latched_result, latched_seh) = guard::guarded(
        || {
            ran = true;
            5
        },
        -1,
    );
    println!("call after the fault: ran={ran} result={latched_result} seh={latched_seh:#x} (expect false, -1, 0xc0000005)");
    assert!(!ran, "a guarded call ran after a caught fault");
    assert_eq!((latched_result, latched_seh), (-1, 0xC000_0005));
    guard::clear_fault_latch_for_test();

    // Wine raises DBG_PRINTEXCEPTION_C for every OutputDebugString. It must pass straight
    // through the guard: no jump, the closure runs to completion, and no fault is reported.
    let (dbg_result, dbg_seh) = guard::guarded(
        || {
            // SAFETY: a NUL-terminated string.
            unsafe { OutputDebugStringA(c"guard_test: OutputDebugString inside a guarded call".as_ptr()) };
            7
        },
        -1,
    );
    println!("OutputDebugString call: result={dbg_result} seh={dbg_seh:#x} (expect 7, 0x0)");
    assert_eq!((dbg_result, dbg_seh), (7, 0), "a debug-print exception must not be treated as a fault");

    // A result bigger than a register must come back intact through the C shim.
    let (wide, wide_seh) = guard::guarded(|| [7u64; 8], [0u64; 8]);
    assert_eq!((wide, wide_seh), ([7u64; 8], 0));

    // The non-unwinding longjmp must survive many faults through the same buffer, a fault
    // nested a few frames below the guard, and a fault after a clean call.
    #[inline(never)]
    fn deep(n: u32) -> i32 {
        if n == 0 {
            // SAFETY: none -- deliberate fault.
            unsafe { std::ptr::read_volatile(0xdead_beef_0000usize as *const i32) }
        } else {
            deep(n - 1) + 1
        }
    }
    for i in 0..10 {
        let (r, seh) = guard::guarded(|| deep(5), -1);
        assert_eq!((r, seh), (-1, 0xC000_0005), "fault {i}");
        guard::clear_fault_latch_for_test();
        let (r, seh) = guard::guarded(|| 1, -1);
        assert_eq!((r, seh), (1, 0), "clean call after fault {i}");
    }

    println!("PASS: the process survived a real access violation inside guard::guarded");
}
