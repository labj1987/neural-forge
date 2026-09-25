//! A self-implemented, in-process `NVSDK_NGX_Parameter` object -- used only when
//! neither `nvngx_dlssnr.dll` (the snippet) nor `nvngx.dll` (Core) hand back a
//! working parameter block of their own.
//!
//! Real evidence this exists for, not speculation: a real side-by-side comparison on
//! `lordnikon` (2026-09-11) against upstream's own compiled helper binary (recovered
//! from its official GitHub release, never its source -- same "shape not expression"
//! rule as everywhere else in this project) showed upstream hitting the *exact same*
//! `0xbad00002` (`FAIL_PLATFORM_ERROR`) from Core's `VULKAN_Init_with_ProjectID`/
//! `VULKAN_Init_Ext`/`AllocateParameters` that this crate does, on this same machine,
//! with this same real `nvngx.dll`. Upstream's own log then shows: `AllocateParameters`
//! failing on Core, the snippet not exporting it at all, and immediately after --
//! `[params] using own NVSDK_NGX_Parameter implementation`, followed by a passing
//! round-trip self-test and a real, successful `VULKAN_CreateFeature(18) -> 0x1`.
//! NGX's own `VULKAN_Init_Ext`/`CreateFeature` never actually require a parameter
//! block *allocated by the DLL itself* -- they just need a pointer matching the real
//! `NVSDK_NGX_Parameter` vtable shape, which anyone can construct. Rejecting Core's own
//! allocator is apparently an expected, survivable condition in this exact environment,
//! not something either implementation is meant to treat as fatal.
//!
//! The object layout matches [`abi::NgxParameterObj`]: a vtable pointer as the first
//! (and, for this implementation, only C++-visible) field, backed by a real
//! `HashMap` behind a `Mutex` for storage -- safe because C++ virtual dispatch only ever touches the
//! `this` pointer opaquely through the vtable; nothing on the DLL side assumes
//! anything about this object's layout beyond that first pointer.

use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::sync::{Mutex, MutexGuard};

use crate::abi::{self, NgxParameter, NgxParameterObj, NgxParameterVtable, NgxResult};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    Ptr(*mut c_void),
    U64(u64),
    F32(f32),
    F64(f64),
    U32(u32),
    I32(i32),
}

// SAFETY: every `Value` variant is either a plain number or a pointer this object
// never dereferences itself -- it only ever hands pointers back out to the same
// NGX call sequence that stored them, exactly as `NgxSnippet` already assumes for its
// own raw-pointer fields.
unsafe impl Send for Value {}

/// A value read back as another type, the way NGX's own parameter object converts: any
/// numeric type reads as any other (`as` conversion), and a pointer reads only as a pointer or
/// as the 64-bit integer holding it (NGX keeps both in one 64-bit slot).
trait FromValue: Sized {
    fn from_value(v: Value) -> Option<Self>;
}

macro_rules! numeric_from_value {
    ($($t:ty),*) => {$(
        impl FromValue for $t {
            fn from_value(v: Value) -> Option<Self> {
                Some(match v {
                    Value::U64(x) => x as $t,
                    Value::F32(x) => x as $t,
                    Value::F64(x) => x as $t,
                    Value::U32(x) => x as $t,
                    Value::I32(x) => x as $t,
                    Value::Ptr(_) => return None,
                })
            }
        }
    )*};
}
numeric_from_value!(f32, f64, u32, i32);

impl FromValue for u64 {
    fn from_value(v: Value) -> Option<Self> {
        Some(match v {
            Value::U64(x) => x,
            Value::F32(x) => x as u64,
            Value::F64(x) => x as u64,
            Value::U32(x) => u64::from(x),
            Value::I32(x) => x as u64,
            Value::Ptr(p) => p as u64,
        })
    }
}

impl FromValue for *mut c_void {
    fn from_value(v: Value) -> Option<Self> {
        match v {
            Value::Ptr(p) => Some(p),
            Value::U64(x) => Some(x as usize as *mut c_void),
            _ => None,
        }
    }
}

/// `Mutex`, not `RefCell`: the DLL may call into this object from a thread of its own, and a
/// `RefCell` borrow conflict is a panic (an abort, with this workspace's `panic = "abort"`).
/// The lock is only ever held for one map operation, never across a call out.
#[repr(C)]
struct SelfParam {
    vtable: *const NgxParameterVtable,
    store: Mutex<HashMap<String, Value>>,
}

/// # Safety
/// `name` must be a valid, NUL-terminated C string for the duration of this call —
/// guaranteed by every real NGX vtable call's own contract (the string is only read
/// synchronously, never retained past the call).
unsafe fn key(name: *const i8) -> String {
    // SAFETY: contract above.
    unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
}

/// # Safety
/// `this` must be a live `*mut SelfParam` -- guaranteed by every call here originating
/// from a vtable slot invoked against a pointer this module itself allocated.
unsafe fn store<'a>(this: *mut c_void) -> MutexGuard<'a, HashMap<String, Value>> {
    // SAFETY: contract above; `this` is exactly the `SelfParam` this module allocated,
    // cast back to what it really is.
    let store = unsafe { &(*this.cast::<SelfParam>()).store };
    // A poisoned lock only means an earlier holder panicked (which aborts here anyway).
    store.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// # Safety
/// `this`/`name` as for [`store`]/[`key`].
unsafe fn set(this: *mut c_void, name: *const i8, value: Value) {
    // SAFETY: forwarded.
    let k = unsafe { key(name) };
    unsafe { store(this) }.insert(k, value);
}

/// # Safety
/// `this`/`name` as for [`store`]/[`key`]; `out` valid to write a `T` through.
unsafe fn get<T: FromValue>(this: *mut c_void, name: *const i8, out: *mut T) -> NgxResult {
    // SAFETY: forwarded.
    let k = unsafe { key(name) };
    let stored = unsafe { store(this) }.get(&k).copied();
    match stored {
        None => abi::result::FAIL_INVALID_PARAMETER,
        Some(v) => match T::from_value(v) {
            Some(x) => {
                // SAFETY: `out` is a valid out-pointer per every real NGX getter's own contract.
                unsafe { out.write(x) };
                abi::result::SUCCESS
            }
            None => abi::result::FAIL_INCOMPATIBLE_TYPES,
        },
    }
}

unsafe extern "system" fn set_ptr(this: *mut c_void, name: *const i8, value: *mut c_void) {
    // SAFETY: every vtable slot's arguments meet `set`/`get`'s contracts (see above).
    unsafe { set(this, name, Value::Ptr(value)) }
}
unsafe extern "system" fn set_u64(this: *mut c_void, name: *const i8, value: u64) {
    unsafe { set(this, name, Value::U64(value)) }
}
unsafe extern "system" fn set_f32(this: *mut c_void, name: *const i8, value: f32) {
    unsafe { set(this, name, Value::F32(value)) }
}
unsafe extern "system" fn set_f64(this: *mut c_void, name: *const i8, value: f64) {
    unsafe { set(this, name, Value::F64(value)) }
}
unsafe extern "system" fn set_u32(this: *mut c_void, name: *const i8, value: u32) {
    unsafe { set(this, name, Value::U32(value)) }
}
unsafe extern "system" fn set_i32(this: *mut c_void, name: *const i8, value: i32) {
    unsafe { set(this, name, Value::I32(value)) }
}

unsafe extern "system" fn get_ptr(this: *mut c_void, name: *const i8, out: *mut *mut c_void) -> NgxResult {
    unsafe { get(this, name, out) }
}
unsafe extern "system" fn get_u64(this: *mut c_void, name: *const i8, out: *mut u64) -> NgxResult {
    unsafe { get(this, name, out) }
}
unsafe extern "system" fn get_f32(this: *mut c_void, name: *const i8, out: *mut f32) -> NgxResult {
    unsafe { get(this, name, out) }
}
unsafe extern "system" fn get_f64(this: *mut c_void, name: *const i8, out: *mut f64) -> NgxResult {
    unsafe { get(this, name, out) }
}
unsafe extern "system" fn get_u32(this: *mut c_void, name: *const i8, out: *mut u32) -> NgxResult {
    unsafe { get(this, name, out) }
}
unsafe extern "system" fn get_i32(this: *mut c_void, name: *const i8, out: *mut i32) -> NgxResult {
    unsafe { get(this, name, out) }
}
/// Slots 9/10/13 in the real vtable are reserved padding — never called for real, but
/// present so every later slot's offset matches the real interface.
unsafe extern "system" fn get_reserved(_this: *mut c_void, _name: *const i8, _out: *mut c_void) -> NgxResult {
    abi::result::FAIL_NOT_IMPLEMENTED
}
unsafe extern "system" fn reset(this: *mut c_void) {
    unsafe { store(this) }.clear();
}

static VTABLE: NgxParameterVtable = NgxParameterVtable {
    set_ptr,
    set_u64,
    set_f32,
    set_f64,
    set_u32,
    set_i32,
    get_f64,
    get_u64,
    get_ptr,
    get_reserved9: get_reserved,
    get_reserved10: get_reserved,
    get_i32,
    get_u32,
    get_reserved13: get_reserved,
    get_f32,
    reset,
};

/// Allocates a new self-implemented parameter block, matching the real
/// `NVSDK_NGX_Parameter` vtable shape closely enough that `nvngx_dlssnr.dll` accepts a
/// pointer to it wherever a real `NgxParameter` is expected. Pair with [`destroy`],
/// never with a DLL's own `DestroyParameters` export -- this object isn't one of its
/// allocations.
pub fn allocate() -> NgxParameter {
    let boxed = Box::new(SelfParam { vtable: &VTABLE, store: Mutex::new(HashMap::new()) });
    Box::into_raw(boxed).cast::<NgxParameterObj>()
}

/// # Safety
/// `params` must be a pointer previously returned by [`allocate`] and not already
/// destroyed.
pub unsafe fn destroy(params: NgxParameter) {
    // SAFETY: contract above -- reconstructs exactly the `Box<SelfParam>` `allocate`
    // leaked, as the same concrete type it was allocated as.
    drop(unsafe { Box::from_raw(params.cast::<SelfParam>()) });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_params(f: impl FnOnce(NgxParameter)) {
        let p = allocate();
        f(p);
        // SAFETY: allocated just above, destroyed once.
        unsafe { destroy(p) };
    }

    #[test]
    fn round_trips_every_type_through_the_vtable() {
        with_params(|p| unsafe {
            let n = c"k".as_ptr();
            abi::ngx_set_u32(p, n, 0x5a5a);
            let mut u = 0u32;
            assert_eq!(abi::ngx_get_u32(p, n, &mut u), abi::result::SUCCESS);
            assert_eq!(u, 0x5a5a);

            abi::ngx_set_f32(p, n, 1.25);
            let mut f = 0f32;
            assert_eq!(abi::ngx_get_f32(p, n, &mut f), abi::result::SUCCESS);
            assert_eq!(f, 1.25);

            abi::ngx_set_u64(p, n, 0x1_0000_0001);
            let mut q = 0u64;
            assert_eq!(abi::ngx_get_u64(p, n, &mut q), abi::result::SUCCESS);
            assert_eq!(q, 0x1_0000_0001);

            abi::ngx_set_i32(p, n, -7);
            let mut i = 0i32;
            assert_eq!(abi::ngx_get_i32(p, n, &mut i), abi::result::SUCCESS);
            assert_eq!(i, -7);

            let mut x = 5u8;
            abi::ngx_set_ptr(p, n, std::ptr::from_mut(&mut x).cast());
            let mut back: *mut c_void = std::ptr::null_mut();
            assert_eq!(abi::ngx_get_ptr(p, n, &mut back), abi::result::SUCCESS);
            assert_eq!(back, std::ptr::from_mut(&mut x).cast());
        });
    }

    #[test]
    fn converts_between_numeric_types_like_ngx() {
        with_params(|p| unsafe {
            let n = c"k".as_ptr();
            abi::ngx_set_u32(p, n, 3);
            let mut f = 0f32;
            assert_eq!(abi::ngx_get_f32(p, n, &mut f), abi::result::SUCCESS);
            assert_eq!(f, 3.0);
            abi::ngx_set_f32(p, n, 2.75);
            let mut u = 0u32;
            assert_eq!(abi::ngx_get_u32(p, n, &mut u), abi::result::SUCCESS);
            assert_eq!(u, 2);
            let mut back: *mut c_void = std::ptr::null_mut();
            assert_eq!(abi::ngx_get_ptr(p, n, &mut back), abi::result::FAIL_INCOMPATIBLE_TYPES);
            assert_eq!(abi::ngx_get_u32(p, c"missing".as_ptr(), &mut u), abi::result::FAIL_INVALID_PARAMETER);
        });
    }
}
