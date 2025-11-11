//! Minimal Rust acceleration hooks (optional).
//! Build with `maturin`/`setuptools-rust` in a full toolchain. Provided as a
//! future optimization path; Python SDK works without compiling this crate.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_numpy::ndarray::{PyArray1, PyArray2};

/// Simple matrix-vector multiply: y = U * x
#[pyfunction]
fn matvec<'py>(
    py: Python<'py>,
    unitary: &'py PyArray2<Complex64>,
    state: &'py PyArray1<Complex64>,
) -> PyResult<&'py PyArray1<Complex64>> {
    let u: Array2<Complex64> = unitary.readonly().as_array().to_owned();
    let x: Array1<Complex64> = state.readonly().as_array().to_owned();
    let y = u.dot(&x);
    Ok(PyArray1::from_owned_array(py, y))
}

#[pymodule]
fn bio_q_accel(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(matvec, m)?)?;
    Ok(())
}


