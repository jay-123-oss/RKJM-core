//! L1-Resident Anchor File: 128-Token Circular Ring Buffer.
//! Total memory footprint: 128 * 16 bytes = 2 KB (fits completely in L1 Data Cache).
//! Zero heap reallocations, O(1) insertion, and fast parallel associative lookup.

use crate::embedding::TokenEmbedding;

pub const ANCHOR_CAPACITY: usize = 128;

#[repr(C, align(64))]
pub struct AnchorFile {
    /// 128 Contiguous 16-byte TokenEmbeddings (2048 bytes total)
    buffer: [TokenEmbedding; ANCHOR_CAPACITY],
    /// Circular write index [0, 127]
    head: usize,
    /// Number of valid tokens stored [0, 128]
    len: usize,
}

impl Default for AnchorFile {
    fn default() -> Self {
        Self::new()
    }
}

impl AnchorFile {
    pub fn new() -> Self {
        Self {
            buffer: [TokenEmbedding::new(0, 0); ANCHOR_CAPACITY],
            head: 0,
            len: 0,
        }
    }

    /// Push token into circular ring buffer with zero allocation.
    #[inline(always)]
    pub fn push(&mut self, token: TokenEmbedding) {
        self.buffer[self.head] = token;
        self.head = (self.head + 1) % ANCHOR_CAPACITY;
        if self.len < ANCHOR_CAPACITY {
            self.len += 1;
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get token at relative backward offset (0 = most recent, 1 = previous, ...).
    pub fn get_recent(&self, offset: usize) -> Option<&TokenEmbedding> {
        if offset >= self.len {
            return None;
        }
        let idx = (self.head + ANCHOR_CAPACITY - 1 - offset) % ANCHOR_CAPACITY;
        Some(&self.buffer[idx])
    }

    /// Fast associative lookup: find token index and similarity score of best match.
    pub fn query_nearest(&self, query: &TokenEmbedding) -> (usize, f32) {
        if self.len == 0 {
            return (0, -1.0);
        }

        let mut best_sim = -2.0f32;
        let mut best_idx = 0;

        for i in 0..self.len {
            let sim = self.buffer[i].similarity(query);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        (best_idx, best_sim)
    }

    /// Reset anchor file.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_behavior() {
        let mut af = AnchorFile::new();
        assert_eq!(af.len(), 0);

        for i in 0..150 {
            let token = TokenEmbedding::new(i as u64, (i % 65536) as u16);
            af.push(token);
        }

        assert_eq!(af.len(), ANCHOR_CAPACITY);
        // Most recent token should have polarity 149
        assert_eq!(af.get_recent(0).unwrap().polarity, 149);
        assert_eq!(af.get_recent(1).unwrap().polarity, 148);
    }

    #[test]
    fn test_associative_lookup() {
        let mut af = AnchorFile::new();
        let target = TokenEmbedding::new(0xAAAA_AAAA_AAAA_AAAA, 0x1234);
        af.push(TokenEmbedding::new(0x0000_0000_0000_0000, 0x0000));
        af.push(target);
        af.push(TokenEmbedding::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF));

        let (idx, sim) = af.query_nearest(&target);
        assert_eq!(idx, 1);
        assert_eq!(sim, 1.0);
    }
}
