//! 16-Byte Aligned TokenEmbedding Data Structure.
//! Stores binary sign polarity mask (S) and salience mask (H) with zero padding waste.

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenEmbedding {
    /// 64-bit Sign Polarity Mask (1 if x >= 0, 0 if x < 0)
    pub polarity: u64,
    /// 16-bit Salience / Feature Importance Mask
    pub salience: u16,
    /// 6-byte Padding to ensure exact 16-byte alignment
    pub _pad: [u8; 6],
}

impl TokenEmbedding {
    pub const ALIGNMENT: usize = 16;
    pub const SIZE: usize = 16;

    /// Create a new TokenEmbedding from raw masks.
    #[inline(always)]
    pub fn new(polarity: u64, salience: u16) -> Self {
        Self {
            polarity,
            salience,
            _pad: [0; 6],
        }
    }

    /// Construct TokenEmbedding from floating point activation slice.
    /// - Polarity (u64): signs of the first 64 floats (or folded).
    /// - Salience (u16): top magnitude features above mean absolute value.
    pub fn from_floats(activations: &[f32]) -> Self {
        let mut polarity: u64 = 0;
        let n = activations.len().min(64);

        for (i, &val) in activations.iter().take(n).enumerate() {
            if val >= 0.0 {
                polarity |= 1u64 << i;
            }
        }

        // Compute mean absolute value for salience thresholding
        let mean_abs: f32 = if !activations.is_empty() {
            activations.iter().map(|x| x.abs()).sum::<f32>() / (activations.len() as f32)
        } else {
            0.0
        };

        let mut salience: u16 = 0;
        let sal_n = activations.len().min(16);
        for (i, &val) in activations.iter().take(sal_n).enumerate() {
            if val.abs() >= mean_abs {
                salience |= 1u16 << i;
            }
        }

        Self::new(polarity, salience)
    }

    /// Compute Hamming distance between polarity masks (0 to 64).
    #[inline(always)]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.polarity ^ other.polarity).count_ones()
    }

    /// Compute bitwise salience overlap (0 to 16).
    #[inline(always)]
    pub fn salience_overlap(&self, other: &Self) -> u32 {
        (self.salience & other.salience).count_ones()
    }

    /// Fast normalized associative similarity in range [-1.0, 1.0].
    #[inline(always)]
    pub fn similarity(&self, other: &Self) -> f32 {
        let diff = self.hamming_distance(other);
        let agreement = 64i32 - 2 * (diff as i32);
        (agreement as f32) / 64.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_memory_layout_and_alignment() {
        assert_eq!(size_of::<TokenEmbedding>(), 16);
        assert_eq!(align_of::<TokenEmbedding>(), 16);
    }

    #[test]
    fn test_similarity_metric() {
        let e1 = TokenEmbedding::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF);
        let e2 = TokenEmbedding::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF);
        assert_eq!(e1.similarity(&e2), 1.0);

        let e3 = TokenEmbedding::new(0x0000_0000_0000_0000, 0xFFFF);
        assert_eq!(e1.similarity(&e3), -1.0);
    }
}
