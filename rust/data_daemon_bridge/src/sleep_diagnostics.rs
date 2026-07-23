//! Native sleep boundary used only by opt-in performance-test diagnostics.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type DiagnosticNativeSleepResult = (
    u64,
    u64,
    u64,
    i32,
    Option<u32>,
    Option<u32>,
    Option<u32>,
    Option<i32>,
);

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeSleepResult {
    wait_start_ns: u64,
    wait_return_ns: u64,
    wait_error: i32,
    cpu_before: u32,
    cpu_after: u32,
    qos_class: u32,
    qos_relative_priority: i32,
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn nc_diagnostic_sleep_ns(
        requested_ns: u64,
        sleep_id: u64,
        label: *const libc::c_char,
        correlation_id: *const libc::c_char,
        anomaly_threshold_ns: u64,
        result: *mut NativeSleepResult,
    ) -> i32;

    fn nc_diagnostic_sleep_anomaly(
        sleep_id: u64,
        label: *const libc::c_char,
        correlation_id: *const libc::c_char,
        requested_ns: u64,
        actual_ns: u64,
        overshoot_ns: i64,
        native_thread_id: u64,
    );
}

#[cfg(target_os = "macos")]
fn monotonic_ns() -> u64 {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `value` is writable and CLOCK_MONOTONIC is supported on macOS.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) };
    if result == 0 {
        (value.tv_sec as u64 * 1_000_000_000) + value.tv_nsec as u64
    } else {
        0
    }
}

/// Release the GIL, perform a native timed wait, and expose the three boundary
/// timestamps: syscall entry, syscall return, and interpreter/GIL resumption.
#[pyfunction]
#[pyo3(signature = (seconds, sleep_id, label, correlation_id, anomaly_threshold_ms))]
pub(crate) fn diagnostic_native_sleep(
    py: Python<'_>,
    seconds: f64,
    sleep_id: u64,
    label: &str,
    correlation_id: &str,
    anomaly_threshold_ms: f64,
) -> PyResult<DiagnosticNativeSleepResult> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(PyValueError::new_err(
            "seconds must be finite and non-negative",
        ));
    }
    if !anomaly_threshold_ms.is_finite() || anomaly_threshold_ms < 0.0 {
        return Err(PyValueError::new_err(
            "anomaly_threshold_ms must be finite and non-negative",
        ));
    }

    #[cfg(target_os = "macos")]
    {
        use std::ffi::CString;

        let label =
            CString::new(label).map_err(|_| PyValueError::new_err("label must not contain NUL"))?;
        let correlation_id = CString::new(correlation_id)
            .map_err(|_| PyValueError::new_err("correlation_id must not contain NUL"))?;
        let requested_ns = (seconds * 1_000_000_000.0).round() as u64;
        let anomaly_threshold_ns = (anomaly_threshold_ms * 1_000_000.0).round() as u64;
        let native = py.detach(|| {
            let mut result = NativeSleepResult::default();
            // SAFETY: both C strings and `result` live through the synchronous
            // call; the C helper neither retains nor aliases these pointers.
            unsafe {
                nc_diagnostic_sleep_ns(
                    requested_ns,
                    sleep_id,
                    label.as_ptr(),
                    correlation_id.as_ptr(),
                    anomaly_threshold_ns,
                    &mut result,
                )
            };
            result
        });
        let interpreter_resume_ns = monotonic_ns();
        let unavailable_cpu = u32::MAX;
        Ok((
            native.wait_start_ns,
            native.wait_return_ns,
            interpreter_resume_ns,
            native.wait_error,
            (native.cpu_before != unavailable_cpu).then_some(native.cpu_before),
            (native.cpu_after != unavailable_cpu).then_some(native.cpu_after),
            Some(native.qos_class),
            Some(native.qos_relative_priority),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (py, sleep_id, label, correlation_id, anomaly_threshold_ms);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "native sleep diagnostics are only available on macOS",
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (
    sleep_id,
    label,
    correlation_id,
    requested_ns,
    actual_ns,
    overshoot_ns,
    native_thread_id
))]
pub(crate) fn diagnostic_signpost_anomaly(
    sleep_id: u64,
    label: &str,
    correlation_id: &str,
    requested_ns: u64,
    actual_ns: u64,
    overshoot_ns: i64,
    native_thread_id: u64,
) -> PyResult<()> {
    #[cfg(target_os = "macos")]
    {
        use std::ffi::CString;

        let label =
            CString::new(label).map_err(|_| PyValueError::new_err("label must not contain NUL"))?;
        let correlation_id = CString::new(correlation_id)
            .map_err(|_| PyValueError::new_err("correlation_id must not contain NUL"))?;
        // SAFETY: the C helper consumes both string pointers synchronously.
        unsafe {
            nc_diagnostic_sleep_anomaly(
                sleep_id,
                label.as_ptr(),
                correlation_id.as_ptr(),
                requested_ns,
                actual_ns,
                overshoot_ns,
                native_thread_id,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (
        sleep_id,
        label,
        correlation_id,
        requested_ns,
        actual_ns,
        overshoot_ns,
        native_thread_id,
    );
    Ok(())
}
