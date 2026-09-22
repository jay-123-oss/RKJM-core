//! wave_brain_core: Bare-Metal Associative Reasoning Engine for RKMJ-Core.
//! Features:
//!   - 16-byte aligned TokenEmbedding (PolarityMask S & SalienceMask H)
//!   - 4-State Gray Code synaptic storage with Morris counter dither & Schmitt hysteresis
//!   - L1-resident Anchor File (128-token static ring buffer)
//!   - Directional non-commutative vector symbolic binding
//!   - Zero-copy C-FFI & PyO3 Python bindings

pub mod embedding;
pub mod synapse;
pub mod anchor;
pub mod reasoning;

pub use embedding::TokenEmbedding;
pub use synapse::{Synapse, GrayState};
pub use anchor::AnchorFile;
pub use reasoning::{ReasoningLoop, directional_bind, directional_unbind};

// -----------------------------------------------------------------------------
// C-FFI Bridge for Zero-Copy Buffer Passing
// -----------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn rkmj_token_embedding_create(polarity: u64, salience: u16) -> TokenEmbedding {
    TokenEmbedding::new(polarity, salience)
}

#[no_mangle]
pub unsafe extern "C" fn rkmj_token_embedding_similarity(
    a: *const TokenEmbedding,
    b: *const TokenEmbedding,
) -> f32 {
    if a.is_null() || b.is_null() {
        return 0.0;
    }
    (*a).similarity(&*b)
}

#[no_mangle]
pub extern "C" fn rkmj_anchor_file_create() -> *mut AnchorFile {
    Box::into_raw(Box::new(AnchorFile::new()))
}

#[no_mangle]
pub unsafe extern "C" fn rkmj_anchor_file_push(
    af: *mut AnchorFile,
    token: TokenEmbedding,
) {
    if !af.is_null() {
        (*af).push(token);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rkmj_anchor_file_query_nearest(
    af: *const AnchorFile,
    query: *const TokenEmbedding,
    out_idx: *mut usize,
) -> f32 {
    if af.is_null() || query.is_null() {
        return -1.0;
    }
    let (idx, sim) = (*af).query_nearest(&*query);
    if !out_idx.is_null() {
        *out_idx = idx;
    }
    sim
}

#[no_mangle]
pub unsafe extern "C" fn rkmj_anchor_file_free(af: *mut AnchorFile) {
    if !af.is_null() {
        drop(Box::from_raw(af));
    }
}

// -----------------------------------------------------------------------------
// PyO3 Bindings (Optional `python` feature)
// -----------------------------------------------------------------------------

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyTokenEmbedding {
    pub inner: TokenEmbedding,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTokenEmbedding {
    #[new]
    pub fn new(polarity: u64, salience: u16) -> Self {
        Self {
            inner: TokenEmbedding::new(polarity, salience),
        }
    }

    #[staticmethod]
    pub fn from_floats(activations: Vec<f32>) -> Self {
        Self {
            inner: TokenEmbedding::from_floats(&activations),
        }
    }

    pub fn similarity(&self, other: &PyTokenEmbedding) -> f32 {
        self.inner.similarity(&other.inner)
    }

    pub fn hamming_distance(&self, other: &PyTokenEmbedding) -> u32 {
        self.inner.hamming_distance(&other.inner)
    }

    #[getter]
    pub fn polarity(&self) -> u64 {
        self.inner.polarity
    }

    #[getter]
    pub fn salience(&self) -> u16 {
        self.inner.salience
    }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyAnchorFile {
    pub inner: AnchorFile,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAnchorFile {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: AnchorFile::new(),
        }
    }

    pub fn push(&mut self, token: &PyTokenEmbedding) {
        self.inner.push(token.inner);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn query_nearest(&self, query: &PyTokenEmbedding) -> (usize, f32) {
        self.inner.query_nearest(&query.inner)
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn wave_brain_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTokenEmbedding>()?;
    m.add_class::<PyAnchorFile>()?;
    Ok(())
}
