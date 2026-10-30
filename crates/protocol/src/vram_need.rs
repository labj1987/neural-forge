//! What the network needs in VRAM at a size, measured: shared by `doctor`'s VRAM check and the layer's VRAM guard.

/// `(width, height, MiB)`: the device-local memory the full network's open and build at that size take on the test
/// machine (weights and activations, the activations shared by lifetime: docs/NATIVE_BACKEND.md, "Activation
/// memory").
pub const NETWORK_VRAM_MIB: [(u32, u32, u64); 4] = [(1485, 836, 377), (2228, 1253, 607), (2560, 1440, 729), (3840, 2160, 1415)];

/// The small network takes about this much less than the full one at the same size.
pub const SMALL_LESS_MIB: u64 = 45;

/// The full network's need at `width` x `height`: the smallest measured size that covers it, or the largest.
pub fn network_need_mib(width: u32, height: u32) -> (u32, u32, u64) {
    let pixels = u64::from(width) * u64::from(height);
    NETWORK_VRAM_MIB.iter().copied().find(|&(w, h, _)| u64::from(w) * u64::from(h) >= pixels).unwrap_or(NETWORK_VRAM_MIB[NETWORK_VRAM_MIB.len() - 1])
}
