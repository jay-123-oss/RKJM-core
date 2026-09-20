//! wave_brain_core: Unsupervised context learning and symbolic loops for RKMJ-Core.

pub struct ContextMemory {
    pub capacity: usize,
    pub states: Vec<Vec<f32>>,
}

impl ContextMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            states: Vec::with_capacity(capacity),
        }
    }

    pub fn push_state(&mut self, state: Vec<f32>) {
        if self.states.len() >= self.capacity {
            self.states.remove(0);
        }
        self.states.push(state);
    }

    pub fn compute_coherence(&self) -> f32 {
        if self.states.is_empty() {
            return 0.0;
        }
        let total_norm: f32 = self.states.iter()
            .map(|s| s.iter().map(|x| x * x).sum::<f32>().sqrt())
            .sum();
        total_norm / self.states.len() as f32
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass]
pub struct PyContextMemory {
    inner: ContextMemory,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyContextMemory {
    #[new]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: ContextMemory::new(capacity),
        }
    }

    pub fn push_state(&mut self, state: Vec<f32>) {
        self.inner.push_state(state);
    }

    pub fn compute_coherence(&self) -> f32 {
        self.inner.compute_coherence()
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn wave_brain_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyContextMemory>()?;
    Ok(())
}
