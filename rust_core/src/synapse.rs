//! 4-State Gray Code Synaptic Storage with Morris Counter Dither & Schmitt Hysteresis.
//! State transitions: 10 <-> 00 <-> 01 <-> 11
//! Prevents memory thrashing by gating write-backs through hysteresis dead-bands.

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrayState {
    Negative = 0b10,      // -1
    WeakNegative = 0b00,  // 0-
    WeakPositive = 0b01,  // 0+
    Positive = 0b11,      // +1
}

impl GrayState {
    /// Convert state to float weight value.
    #[inline(always)]
    pub fn to_float(self) -> f32 {
        match self {
            GrayState::Negative => -1.0,
            GrayState::WeakNegative => -0.2,
            GrayState::WeakPositive => 0.2,
            GrayState::Positive => 1.0,
        }
    }

    /// Potentiate (increment) state along Gray code path: 10 -> 00 -> 01 -> 11.
    #[inline(always)]
    pub fn potentiate(self) -> Self {
        match self {
            GrayState::Negative => GrayState::WeakNegative,
            GrayState::WeakNegative => GrayState::WeakPositive,
            GrayState::WeakPositive => GrayState::Positive,
            GrayState::Positive => GrayState::Positive, // Saturated
        }
    }

    /// Depress (decrement) state along Gray code path: 11 -> 01 -> 00 -> 10.
    #[inline(always)]
    pub fn depress(self) -> Self {
        match self {
            GrayState::Positive => GrayState::WeakPositive,
            GrayState::WeakPositive => GrayState::WeakNegative,
            GrayState::WeakNegative => GrayState::Negative,
            GrayState::Negative => GrayState::Negative, // Saturated
        }
    }
}

/// Synaptic unit governed by Schmitt-Trigger hysteresis and Morris counter dither.
#[derive(Clone, Debug)]
pub struct Synapse {
    pub state: GrayState,
    pub accumulator: f32,
    pub theta_hi: f32,
    pub theta_lo: f32,
    pub morris_counter: u8,
}

impl Synapse {
    pub fn new(theta_hi: f32, theta_lo: f32) -> Self {
        Self {
            state: GrayState::WeakPositive,
            accumulator: 0.0,
            theta_hi,
            theta_lo,
            morris_counter: 0,
        }
    }

    /// Pseudo-random Morris dither check: returns true with probability 2^{-k}.
    #[inline(always)]
    fn should_update_morris(&self, seed: u64) -> bool {
        if self.morris_counter == 0 {
            return true;
        }
        let k = (self.morris_counter as u32).min(16);
        let mask = (1u64 << k) - 1;
        // Fast hash-dither check
        let hash = seed.wrapping_mul(0x517cc1b727220a95);
        (hash & mask) == 0
    }

    /// Integrate incoming spike/activation with Schmitt-Trigger dead-band hysteresis.
    /// Returns true if synaptic state transitioned.
    pub fn integrate(&mut self, activation: f32, seed: u64) -> bool {
        self.accumulator += activation;

        if self.accumulator >= self.theta_hi {
            if self.should_update_morris(seed) {
                let next = self.state.potentiate();
                let changed = next != self.state;
                self.state = next;
                self.accumulator = 0.0;
                if self.morris_counter < 15 {
                    self.morris_counter += 1;
                }
                return changed;
            }
        } else if self.accumulator <= self.theta_lo {
            if self.should_update_morris(seed) {
                let next = self.state.depress();
                let changed = next != self.state;
                self.state = next;
                self.accumulator = 0.0;
                if self.morris_counter < 15 {
                    self.morris_counter += 1;
                }
                return changed;
            }
        }

        // Inside dead-band: zero write-back
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_transitions() {
        let s = GrayState::Negative;
        let s = s.potentiate();
        assert_eq!(s, GrayState::WeakNegative);
        let s = s.potentiate();
        assert_eq!(s, GrayState::WeakPositive);
        let s = s.potentiate();
        assert_eq!(s, GrayState::Positive);
        let s = s.potentiate();
        assert_eq!(s, GrayState::Positive); // Saturated
    }

    #[test]
    fn test_schmitt_trigger_hysteresis() {
        let mut syn = Synapse::new(2.0, -2.0);

        // Sub-threshold activation stays inside dead-band
        assert!(!syn.integrate(1.0, 42));
        assert_eq!(syn.state, GrayState::WeakPositive);

        // Crossing upper threshold triggers transition
        assert!(syn.integrate(1.5, 42));
        assert_eq!(syn.state, GrayState::Positive);
    }
}
