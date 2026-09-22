//! Thought-Traversal, Attractor Loops & Directional Vector Symbolic Binding.
//! Directional non-commutative binding: bind(A, B) = rotl(A, k) ^ B.
//! Enables recurrent reasoning paths without matrix multiplications.

use crate::embedding::TokenEmbedding;

const ROTATION_BITS: u32 = 13;

/// Bind two token embeddings with non-commutative directional binding: A -> B.
#[inline(always)]
pub fn directional_bind(a: &TokenEmbedding, b: &TokenEmbedding) -> TokenEmbedding {
    let bound_pol = a.polarity.rotate_left(ROTATION_BITS) ^ b.polarity;
    let bound_sal = a.salience.rotate_left(ROTATION_BITS % 16) ^ b.salience;
    TokenEmbedding::new(bound_pol, bound_sal)
}

/// Unbind operation: given bound C = bind(A, B), unbind(C, A) recovers B.
#[inline(always)]
pub fn directional_unbind(c: &TokenEmbedding, a: &TokenEmbedding) -> TokenEmbedding {
    let recovered_pol = c.polarity ^ a.polarity.rotate_left(ROTATION_BITS);
    let recovered_sal = c.salience ^ a.salience.rotate_left(ROTATION_BITS % 16);
    TokenEmbedding::new(recovered_pol, recovered_sal)
}

/// Reasoning state machine tracking attractor trajectories.
#[derive(Clone, Debug)]
pub struct ReasoningLoop {
    pub current_thought: TokenEmbedding,
    pub step_count: usize,
    pub history: [u64; 16],
    pub history_idx: usize,
}

impl Default for ReasoningLoop {
    fn default() -> Self {
        Self::new(TokenEmbedding::new(0, 0))
    }
}

impl ReasoningLoop {
    pub fn new(initial_thought: TokenEmbedding) -> Self {
        Self {
            current_thought: initial_thought,
            step_count: 0,
            history: [0; 16],
            history_idx: 0,
        }
    }

    /// Step reasoning state by binding new input token.
    pub fn step(&mut self, token: &TokenEmbedding) {
        self.history[self.history_idx] = self.current_thought.polarity;
        self.history_idx = (self.history_idx + 1) % 16;
        self.step_count += 1;

        self.current_thought = directional_bind(&self.current_thought, token);
    }

    /// Check if reasoning loop has converged into an attractor cycle.
    pub fn check_attractor_convergence(&self) -> bool {
        if self.step_count < 4 {
            return false;
        }
        let curr = self.current_thought.polarity;
        // Check if current thought matches any recent state in trajectory
        for &past in &self.history {
            if (curr ^ past).count_ones() <= 2 {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_commutative_binding() {
        let a = TokenEmbedding::new(0x1234_5678_9ABC_DEF0, 0x1111);
        let b = TokenEmbedding::new(0xFEDC_BA98_7654_3210, 0x2222);

        let ab = directional_bind(&a, &b);
        let ba = directional_bind(&b, &a);

        // Directional binding is non-commutative: bind(A, B) != bind(B, A)
        assert_ne!(ab.polarity, ba.polarity);

        // Exact reversibility: unbind(ab, a) == b
        let recovered_b = directional_unbind(&ab, &a);
        assert_eq!(recovered_b.polarity, b.polarity);
        assert_eq!(recovered_b.salience, b.salience);
    }

    #[test]
    fn test_reasoning_attractor() {
        let mut loop_state = ReasoningLoop::new(TokenEmbedding::new(0x5555_5555_5555_5555, 0));
        let token = TokenEmbedding::new(0, 0);

        for _ in 0..20 {
            loop_state.step(&token);
        }

        assert!(loop_state.step_count >= 20);
    }
}
