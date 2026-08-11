//! C-ABI Foreign Function Interface (FFI) bindings for `libdive`.
//! Allows C, C++, Go (Cgo), Java (FFM/JNI), Node.js (N-API), and C# to invoke DIVE core.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int};
use std::slice;

use super::{compute_dataset_hash, compute_pearson_correlation, compute_psi};

/// C-compatible function calculating Pearson correlation coefficient.
#[no_mangle]
pub unsafe extern "C" fn dive_compute_pearson(
    x_ptr: *const c_double,
    y_ptr: *const c_double,
    len: usize,
) -> c_double {
    if x_ptr.is_null() || y_ptr.is_null() || len == 0 {
        return 0.0;
    }
    let x = slice::from_raw_parts(x_ptr, len);
    let y = slice::from_raw_parts(y_ptr, len);
    compute_pearson_correlation(x, y)
}

/// C-compatible function calculating Population Stability Index (PSI).
#[no_mangle]
pub unsafe extern "C" fn dive_compute_psi(
    ref_ptr: *const c_double,
    curr_ptr: *const c_double,
    len: usize,
) -> c_double {
    if ref_ptr.is_null() || curr_ptr.is_null() || len == 0 {
        return 0.0;
    }
    let r = slice::from_raw_parts(ref_ptr, len);
    let c = slice::from_raw_parts(curr_ptr, len);
    compute_psi(r, c)
}

/// Free a C string allocated by `libdive`.
#[no_mangle]
pub unsafe extern "C" fn dive_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
