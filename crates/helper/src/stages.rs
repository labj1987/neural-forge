//! Per-request stage timing for the helper's periodic `[frame] stages` line: each answered
//! request contributes one row of wall-clock milliseconds per stage, and every `window` rows the
//! medians are formatted and the window starts over. Cheap enough for every request (a push into
//! a preallocated vector); the only I/O is the one line per window.
//!
//! Pure arithmetic, no Win32 or Vulkan, so it builds and tests natively.

/// A window of per-request stage timings.
pub struct StageWindow<const N: usize> {
    names: [&'static str; N],
    rows: Vec<[f32; N]>,
    window: usize,
}

impl<const N: usize> StageWindow<N> {
    pub fn new(names: [&'static str; N], window: usize) -> Self {
        let window = window.max(1);
        Self { names, rows: Vec::with_capacity(window), window }
    }

    /// Books one request's stages (milliseconds). Returns the summary line's body once the window
    /// is full (and clears it), `None` otherwise.
    pub fn push(&mut self, row: [f32; N]) -> Option<String> {
        self.rows.push(row);
        if self.rows.len() < self.window {
            return None;
        }
        let line = self.summary();
        self.rows.clear();
        Some(line)
    }

    /// `n=<rows> name=<median> ...` over the rows booked so far, milliseconds to two decimals.
    pub fn summary(&self) -> String {
        let mut out = format!("n={}", self.rows.len());
        for (i, name) in self.names.iter().enumerate() {
            let column: Vec<f32> = self.rows.iter().map(|r| r[i]).collect();
            out.push_str(&format!(" {name}={:.2}", median(&column)));
        }
        out
    }
}

/// The median (upper of the two middle values for an even count); 0 for no values. NaN sorts last.
pub fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Milliseconds of a duration, as `f32`.
pub fn ms(d: std::time::Duration) -> f32 {
    (d.as_secs_f64() * 1000.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_window_yields_per_stage_medians_and_starts_over() {
        let mut w = StageWindow::new(["a", "b"], 3);
        assert_eq!(w.push([1.0, 10.0]), None);
        assert_eq!(w.push([3.0, 30.0]), None);
        assert_eq!(w.push([2.0, 20.0]).as_deref(), Some("n=3 a=2.00 b=20.00"));
        // The next window starts empty.
        assert_eq!(w.summary(), "n=0 a=0.00 b=0.00");
        assert_eq!(w.push([5.0, 0.5]), None);
        assert_eq!(w.summary(), "n=1 a=5.00 b=0.50");
    }

    #[test]
    fn median_handles_empty_even_and_nan() {
        assert_eq!(median(&[]), 0.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 3.0);
        assert_eq!(median(&[f32::NAN, 1.0, 2.0]), 2.0);
        assert_eq!(ms(std::time::Duration::from_micros(1500)), 1.5);
    }
}
