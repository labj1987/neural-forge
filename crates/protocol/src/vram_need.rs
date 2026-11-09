//! What the network needs in VRAM at a size, measured: shared by `doctor`'s VRAM check and the layer's VRAM guard.

/// `(width, height, MiB)`: the device-local memory the full network's open and build at that size take on the test
/// machine (weights and activations, the activations shared by lifetime: docs/NATIVE_BACKEND.md, "Activation
/// memory").
pub const NETWORK_VRAM_MIB: [(u32, u32, u64); 4] = [(1485, 836, 377), (2228, 1253, 607), (2560, 1440, 729), (3840, 2160, 1415)];

/// The small network takes about this much less than the full one at the same size (45-46 MiB at each size of
/// [`NETWORK_VRAM_MIB`]: the rig's `what_each_network_takes_at_the_tables_sizes`, 2026-11-09).
pub const SMALL_LESS_MIB: u64 = 45;

/// The full network's need at a size, and the two measured points of [`NETWORK_VRAM_MIB`] it comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkNeed {
    /// MiB, rounded to the nearest.
    pub mib: u64,
    /// The measured `(width, height, MiB)` points it is interpolated between (or extrapolated from, outside them),
    /// smaller first. Both are the same point when the size is one of them.
    pub from: [(u32, u32, u64); 2],
}

impl NetworkNeed {
    /// What it is based on, for a person: "measured at WxH" or "between (beyond) the WxH and WxH measurements".
    pub fn basis(&self) -> String {
        let [(w0, h0, m0), (w1, h1, m1)] = self.from;
        if self.from[0] == self.from[1] {
            format!("measured {m0} MiB at {w0}x{h0}")
        } else {
            format!("from the {m0} MiB measured at {w0}x{h0} and the {m1} MiB at {w1}x{h1}, linear in the pixel count")
        }
    }
}

/// The full network's need at `width` x `height`: linear in the pixel count (the measured points are close to
/// linear), between the two measured sizes that bracket it, extrapolated from the two smallest below them and from
/// the two largest above them (frames beyond 4K, DLAA on a wide display), never below 0.
pub fn network_need(width: u32, height: u32) -> NetworkNeed {
    let pixels = |w: u32, h: u32| u64::from(w) * u64::from(h);
    let px = pixels(width, height);
    if let Some(&point) = NETWORK_VRAM_MIB.iter().find(|&&(w, h, _)| pixels(w, h) == px) {
        return NetworkNeed { mib: point.2, from: [point, point] };
    }
    let last = NETWORK_VRAM_MIB.len() - 1;
    // The first measured point above the size picks the segment; none above: the last segment.
    let upper = NETWORK_VRAM_MIB.iter().position(|&(w, h, _)| pixels(w, h) > px).unwrap_or(last).max(1);
    let (a, b) = (NETWORK_VRAM_MIB[upper - 1], NETWORK_VRAM_MIB[upper]);
    let (pa, pb) = (pixels(a.0, a.1) as f64, pixels(b.0, b.1) as f64);
    let mib = a.2 as f64 + (b.2 as f64 - a.2 as f64) * (px as f64 - pa) / (pb - pa);
    NetworkNeed { mib: mib.round().max(0.0) as u64, from: [a, b] }
}

/// [`network_need`]'s MiB.
pub fn network_need_mib(width: u32, height: u32) -> u64 {
    network_need(width, height).mib
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_need_is_linear_in_pixels_between_and_beyond_the_measured_sizes() {
        // The measured sizes themselves.
        for (w, h, mib) in NETWORK_VRAM_MIB {
            assert_eq!(network_need(w, h), NetworkNeed { mib, from: [(w, h, mib); 2] });
        }
        // Between two: 1920x1080 lies 54 % of the way from 1485x836 to 2228x1253 in pixels, not at the next size.
        assert_eq!(network_need(1920, 1080), NetworkNeed { mib: 500, from: [NETWORK_VRAM_MIB[0], NETWORK_VRAM_MIB[1]] });
        // One pixel row past a measured size is next to its figure, not the next size's.
        assert_eq!(network_need_mib(1486, 836), 377);
        // Ultrawide 3440x1440: 27 % of the way from 2560x1440 to 3840x2160, far from the 4K figure.
        assert_eq!(network_need(3440, 1440), NetworkNeed { mib: 918, from: [NETWORK_VRAM_MIB[2], NETWORK_VRAM_MIB[3]] });
        // Beyond 4K: the last two points' slope carries on (8K is four times 4K's pixels).
        assert_eq!(network_need(7680, 4320), NetworkNeed { mib: 5119, from: [NETWORK_VRAM_MIB[2], NETWORK_VRAM_MIB[3]] });
        assert!(network_need_mib(5120, 2160) > 1415);
        // Below the smallest: the first two points' slope, never below 0.
        assert_eq!(network_need(1280, 720), NetworkNeed { mib: 330, from: [NETWORK_VRAM_MIB[0], NETWORK_VRAM_MIB[1]] });
        assert_eq!(network_need_mib(0, 0), 193);
        assert!(network_need_mib(u32::MAX, u32::MAX) > 1415, "no overflow");
        // Monotonic in the pixel count.
        let sizes = [(0, 0), (640, 360), (1280, 720), (1485, 836), (1920, 1080), (2560, 1440), (3440, 1440), (3840, 2160), (7680, 4320)];
        assert!(sizes.windows(2).all(|p| network_need_mib(p[0].0, p[0].1) <= network_need_mib(p[1].0, p[1].1)));
        // What doctor shows.
        assert_eq!(network_need(2560, 1440).basis(), "measured 729 MiB at 2560x1440");
        assert_eq!(network_need(3440, 1440).basis(), "from the 729 MiB measured at 2560x1440 and the 1415 MiB at 3840x2160, linear in the pixel count");
    }
}
