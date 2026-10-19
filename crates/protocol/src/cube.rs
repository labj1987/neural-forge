//! The look-up table of the colour of the edit (docs/NATIVE_BACKEND.md, "The colour of the edit"): a 3D
//! `.cube` file (Adobe/Resolve's text format) as the layer uploads it, and where it is named: `edit_lut`
//! in `config.ini` (a path; empty or missing for none). The layer reads the file again when
//! `ShmHeader::edit_lut_seq` moves on, so a GUI or CLI that changes the key bumps it.

/// The `config.ini` key naming the `.cube` file.
pub const CONFIG_KEY: &str = "edit_lut";

/// The table sizes taken: 2 to 65 points a side (17, 33 and 65 are the usual ones).
pub const MIN_SIZE: usize = 2;
pub const MAX_SIZE: usize = 65;

/// A parsed 3D table: `size`^3 RGB entries, red fastest, then green, then blue (the format's order), for
/// inputs from `domain_min` to `domain_max` per channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Cube {
    pub size: usize,
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
    pub table: Vec<[f32; 3]>,
}

impl Cube {
    /// The identity table of `size` points a side over 0-1.
    pub fn identity(size: usize) -> Self {
        let n = size.max(MIN_SIZE);
        let s = (n - 1) as f32;
        let table = (0..n * n * n).map(|i| [(i % n) as f32 / s, ((i / n) % n) as f32 / s, (i / (n * n)) as f32 / s]).collect();
        Self { size: n, domain_min: [0.0; 3], domain_max: [1.0; 3], table }
    }

    /// The table's value at `rgb` by trilinear interpolation (the shaders' lookup, on the CPU).
    pub fn sample(&self, rgb: [f32; 3]) -> [f32; 3] {
        let n = self.size;
        let mut base = [0usize; 3];
        let mut frac = [0f32; 3];
        for c in 0..3 {
            let span = self.domain_max[c] - self.domain_min[c];
            let t = ((rgb[c] - self.domain_min[c]) / span * (n - 1) as f32).clamp(0.0, (n - 1) as f32);
            base[c] = (t.floor() as usize).min(n - 2);
            frac[c] = t - base[c] as f32;
        }
        let at = |r: usize, g: usize, b: usize| self.table[r + g * n + b * n * n];
        let mut out = [0f32; 3];
        for (corner, w) in (0..8).map(|k| {
            let (dr, dg, db) = (k & 1, (k >> 1) & 1, (k >> 2) & 1);
            let w = (if dr == 1 { frac[0] } else { 1.0 - frac[0] }) * (if dg == 1 { frac[1] } else { 1.0 - frac[1] }) * (if db == 1 { frac[2] } else { 1.0 - frac[2] });
            (at(base[0] + dr, base[1] + dg, base[2] + db), w)
        }) {
            for c in 0..3 {
                out[c] += corner[c] * w;
            }
        }
        out
    }
}

/// Parses a `.cube` file's text: `LUT_3D_SIZE`, optional `DOMAIN_MIN`/`DOMAIN_MAX` and `TITLE`, `#`
/// comments, then exactly size^3 lines of three numbers. A 1D table (`LUT_1D_SIZE`) is refused, as is a
/// size outside [`MIN_SIZE`]..=[`MAX_SIZE`], a non-finite value or an empty domain.
pub fn parse(text: &str) -> Result<Cube, String> {
    let mut size = None;
    let mut domain_min = [0f32; 3];
    let mut domain_max = [1f32; 3];
    let mut table = Vec::new();
    let three = |rest: &str, line: usize| -> Result<[f32; 3], String> {
        let v: Vec<f32> = rest.split_whitespace().map(str::parse::<f32>).collect::<Result<_, _>>().map_err(|e| format!("line {line}: {e}"))?;
        match v.as_slice() {
            [a, b, c] if v.iter().all(|x| x.is_finite()) => Ok([*a, *b, *c]),
            _ => Err(format!("line {line}: expected three finite numbers")),
        }
    };
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let n = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (word, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        match word {
            "TITLE" => {}
            "LUT_1D_SIZE" => return Err("a 1D table: only 3D tables (LUT_3D_SIZE) are taken".into()),
            "LUT_3D_SIZE" => {
                let s: usize = rest.trim().parse().map_err(|e| format!("line {n}: LUT_3D_SIZE: {e}"))?;
                if !(MIN_SIZE..=MAX_SIZE).contains(&s) {
                    return Err(format!("LUT_3D_SIZE {s} is outside {MIN_SIZE} to {MAX_SIZE}"));
                }
                size = Some(s);
                table.reserve(s * s * s);
            }
            "DOMAIN_MIN" => domain_min = three(rest, n)?,
            "DOMAIN_MAX" => domain_max = three(rest, n)?,
            "LUT_1D_INPUT_RANGE" | "LUT_3D_INPUT_RANGE" => {
                let v = three(&format!("{rest} 0"), n).map_err(|_| format!("line {n}: expected two numbers"))?;
                domain_min = [v[0]; 3];
                domain_max = [v[1]; 3];
            }
            _ if word.starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '+' || c == '.') => table.push(three(line, n)?),
            // Keywords of other tools (LUT_IN_VIDEO_RANGE, ...) are skipped.
            _ => {}
        }
    }
    let Some(size) = size else { return Err("no LUT_3D_SIZE line".into()) };
    if table.len() != size * size * size {
        return Err(format!("{} entries where LUT_3D_SIZE {size} needs {}", table.len(), size * size * size));
    }
    if (0..3).any(|c| domain_max[c] <= domain_min[c]) {
        return Err("DOMAIN_MAX must be above DOMAIN_MIN in every channel".into());
    }
    Ok(Cube { size, domain_min, domain_max, table })
}

/// Reads and parses the file at `path`.
pub fn load(path: &str) -> Result<Cube, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    parse(&text).map_err(|e| format!("{path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(cube: &Cube) -> String {
        let mut t = format!("TITLE \"test\"\n# a comment\nLUT_3D_SIZE {}\n", cube.size);
        for e in &cube.table {
            t.push_str(&format!("{} {} {}\n", e[0], e[1], e[2]));
        }
        t
    }

    #[test]
    fn a_cube_file_round_trips_and_its_identity_samples_to_its_input() {
        let id = Cube::identity(17);
        let parsed = parse(&text(&id)).expect("parses");
        assert_eq!(parsed.size, 17);
        assert_eq!(parsed.table.len(), 17 * 17 * 17);
        for rgb in [[0.0, 0.0, 0.0], [0.3, 0.6, 0.9], [1.0, 1.0, 1.0], [0.51, 0.02, 0.77]] {
            let s = parsed.sample(rgb);
            for c in 0..3 {
                assert!((s[c] - rgb[c]).abs() < 1e-5, "{rgb:?} -> {s:?}");
            }
        }
    }

    #[test]
    fn a_bad_cube_file_is_refused_with_a_reason() {
        assert!(parse("LUT_1D_SIZE 16\n").unwrap_err().contains("1D"));
        assert!(parse("LUT_3D_SIZE 66\n").unwrap_err().contains("outside"));
        assert!(parse("LUT_3D_SIZE 2\n0 0 0\n").unwrap_err().contains("needs 8"));
        assert!(parse("0 0 0\n").unwrap_err().contains("LUT_3D_SIZE"));
        let mut t = text(&Cube::identity(2));
        t.push_str("DOMAIN_MIN 1 0 0\nDOMAIN_MAX 1 1 1\n");
        assert!(parse(&t).unwrap_err().contains("DOMAIN"));
        assert!(parse("LUT_3D_SIZE 2\n0 0 nan\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n").is_err());
    }

    #[test]
    fn the_domain_scales_the_lookup() {
        let mut c = Cube::identity(2);
        c.domain_min = [0.0; 3];
        c.domain_max = [2.0; 3];
        // Input 1.0 is half way along a 0-2 domain: the identity table's 0.5.
        assert!((c.sample([1.0, 1.0, 1.0])[0] - 0.5).abs() < 1e-6);
    }
}
