//! Shared PTX building blocks for the DLSS-NR Swin kernels (fragment-level, exact arithmetic).
//!
//! Fragment conventions (mma.sync m16n8k32 e4m3 -> f16, and m16n8k16 f16):
//!   lane l: g = l >> 2 (row), t = l & 3.
//!   C/D tile (16 x 8): [c0, c1] f16x2 regs; c0 = (row g, cols 2t, 2t+1), c1 = (row g+8, same cols).
//!   A (16 x 32 e4m3): [a0..a3] b32; a0 = (row g, k 4t..4t+3), a1 = (row g+8, k 4t..), a2 = (row g, 16+4t..), a3 = (row g+8, 16+4t..).
//!   B (32 x 8 e4m3): [b0, b1]; b0 = (k 4t..4t+3, col g), b1 = (k 16+4t.., col g).
//!   A f16 (16 x 16): a0 = (row g, k 2t..2t+1), a1 = (row g+8, ..), a2 = (row g, 8+2t..), a3 = (row g+8, 8+2t..).
//!   B f16 (16 x 8): b0 = (k 2t..2t+1, col g), b1 = (k 8+2t.., col g).
//!
//! A 16 x 32 f16 accumulator (four C tiles j = 0..3) holds in lane (g, t) the columns 8j + 2t, 8j + 2t + 1
//! of rows g and g + 8: the coopmat "element 8h + 2j + i" layout of the GLSL kernels.
//!
//! A tile or fragment is a `Vec<String>` of register names (two for a C tile, four for an A fragment).

use std::fmt::Display;

use crate::ptxgen::Ptx;

pub fn mma_e4(p: &Ptx, d: &[String], a: &[String], b: &[String], c: &[String]) {
    p.emit(format!(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {{{}, {}}}, {{{}, {}, {}, {}}}, {{{}, {}}}, {{{}, {}}};",
        d[0], d[1], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1]
    ));
}

pub fn mma_f16(p: &Ptx, d: &[String], a: &[String], b: &[String], c: &[String]) {
    p.emit(format!(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {{{}, {}}}, {{{}, {}, {}, {}}}, {{{}, {}}}, {{{}, {}}};",
        d[0], d[1], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1]
    ));
}

pub fn ldm4(p: &Ptx, addr: impl Display) -> Vec<String> {
    let r = p.regs("b32", 4);
    p.emit(format!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{}, {}, {}, {}}}, [{addr}];", r[0], r[1], r[2], r[3]));
    r
}

pub fn ldm2(p: &Ptx, addr: impl Display) -> Vec<String> {
    let r = p.regs("b32", 2);
    p.emit(format!("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{}, {}}}, [{addr}];", r[0], r[1]));
    r
}

pub fn zero_tile(p: &Ptx) -> Vec<String> {
    let d = p.regs("b32", 2);
    p.emit(format!("mov.b32 {}, 0;", d[0]));
    p.emit(format!("mov.b32 {}, 0;", d[1]));
    d
}

pub fn imm(p: &Ptx, value: impl Display) -> String {
    let r = p.reg("b32");
    p.emit(format!("mov.b32 {r}, {value};"));
    r
}

pub fn f16x2_op(p: &Ptx, op: &str, a: impl Display, b: impl Display) -> String {
    let r = p.reg("b32");
    p.emit(format!("{op}.f16x2 {r}, {a}, {b};"));
    r
}

pub fn hadd2(p: &Ptx, a: impl Display, b: impl Display) -> String {
    f16x2_op(p, "add.rn", a, b)
}
pub fn hmul2(p: &Ptx, a: impl Display, b: impl Display) -> String {
    f16x2_op(p, "mul.rn", a, b)
}
pub fn hmax2(p: &Ptx, a: impl Display, b: impl Display) -> String {
    f16x2_op(p, "max", a, b)
}
pub fn hmin2(p: &Ptx, a: impl Display, b: impl Display) -> String {
    f16x2_op(p, "min", a, b)
}

pub fn hfma2(p: &Ptx, a: impl Display, b: impl Display, c: impl Display) -> String {
    let r = p.reg("b32");
    p.emit(format!("fma.rn.f16x2 {r}, {a}, {b}, {c};"));
    r
}

pub fn hadd(p: &Ptx, a: impl Display, b: impl Display) -> String {
    let r = p.reg("f16");
    p.emit(format!("add.rn.f16 {r}, {a}, {b};"));
    r
}

pub fn unpack16(p: &Ptx, r: impl Display) -> (String, String) {
    let lo = p.reg("f16");
    let hi = p.reg("f16");
    p.emit(format!("mov.b32 {{{lo}, {hi}}}, {r};"));
    (lo, hi)
}

pub fn pack16(p: &Ptx, lo: impl Display, hi: impl Display) -> String {
    let r = p.reg("b32");
    p.emit(format!("mov.b32 {r}, {{{lo}, {hi}}};"));
    r
}

/// f16x2 -> two E4M3 codes (saturating, NaN -> 0x7f) in a .b16 register.
pub fn cvt_e4x2(p: &Ptx, v: impl Display) -> String {
    let h = p.reg("b16");
    p.emit(format!("cvt.rn.satfinite.e4m3x2.f16x2 {h}, {v};"));
    h
}

/// Two E4M3 codes (.b16) -> f16x2 (exact widening; NaN codes -> NaN).
pub fn e4x2_to_f16x2(p: &Ptx, h: impl Display) -> String {
    let r = p.reg("b32");
    p.emit(format!("cvt.rn.f16x2.e4m3x2 {r}, {h};"));
    r
}

pub fn b16_to_b32(p: &Ptx, h: impl Display) -> String {
    let r = p.reg("b32");
    p.emit(format!("cvt.u32.u16 {r}, {h};"));
    r
}

pub fn b32_to_b16(p: &Ptx, r: impl Display) -> String {
    let h = p.reg("b16");
    p.emit(format!("cvt.u16.u32 {h}, {r};"));
    h
}

/// f16x2: NaN lanes -> 0, everything else unchanged (max/min return the non-NaN operand).
pub fn nanzero(p: &Ptx, v: impl Display, zero: impl Display) -> String {
    let pos = hmax2(p, &v, &zero);
    let neg = hmin2(p, &v, &zero);
    hadd2(p, pos, neg)
}

pub fn shfl_idx(p: &Ptx, v: impl Display, src: impl Display) -> String {
    let r = p.reg("b32");
    p.emit(format!("shfl.sync.idx.b32 {r}, {v}, {src}, 0x1f, 0xffffffff;"));
    r
}

pub fn shfl_xor(p: &Ptx, v: impl Display, mask: impl Display) -> String {
    let r = p.reg("b32");
    p.emit(format!("shfl.sync.bfly.b32 {r}, {v}, {mask}, 0x1f, 0xffffffff;"));
    r
}

pub fn prmt(p: &Ptx, a: impl Display, b: impl Display, sel: impl Display) -> String {
    let r = p.reg("b32");
    p.emit(format!("prmt.b32 {r}, {a}, {b}, {sel};"));
    r
}

pub struct SiluConsts {
    pub c4: String,
    pub cm4: String,
    pub c_a: String,
    pub c_b: String,
    pub c_c: String,
}

impl SiluConsts {
    pub fn new(p: &Ptx) -> Self {
        Self {
            c4: imm(p, "0x44004400"),  // 4.0
            cm4: imm(p, "0xC400C400"), // -4.0
            c_a: imm(p, "0xAB28AB28"), // -0.055908203125
            c_b: imm(p, "0x37283728"), // 0.447265625
            c_c: imm(p, "0x3B283B28"), // 0.89453125
        }
    }
}

/// siluPair: clamp to [-4, 4], inner = fma(-0.0559, |bounded|, 0.4473), poly = fma(bounded, inner, 0.8945), v * poly.
pub fn silu(p: &Ptx, v: impl Display, k: &SiluConsts) -> String {
    let bounded = hmax2(p, &v, &k.cm4);
    let bounded = hmin2(p, bounded, &k.c4);
    let ab = p.reg("b32");
    p.emit(format!("abs.f16x2 {ab}, {bounded};"));
    let inner = hfma2(p, &k.c_a, ab, &k.c_b);
    let poly = hfma2(p, bounded, inner, &k.c_c);
    hmul2(p, v, poly)
}

/// Per-lane constants for the fragment shuffles.
pub struct LaneInfo {
    pub lane: String,
    pub g: String,
    pub t: String,
    pub quad: String,
    pub src_a: String,
    pub src_b: String,
    pub sel: String,
    pub quad_lanes: Vec<String>,
}

impl LaneInfo {
    pub fn new(p: &Ptx, lane: &str) -> Self {
        let g = p.shr32(lane, 2);
        let t = p.and32(lane, 3);
        let quad = p.and32(lane, 28);
        // acc_to_a sources: lane (g, 2 (t & 1)) and (g, 2 (t & 1) + 1); selector picks the low pairs (t >> 1 == 0) or the high pairs
        let tl = p.and32(lane, 1);
        let src_a = p.add32(&quad, p.shl32(tl, 1));
        let src_b = p.add32(&src_a, 1);
        let th = p.and32(p.shr32(lane, 1), 1);
        let p_hi = p.setp("ne.u32", th, 0);
        let sel = p.selp32(p_hi, imm(p, "0x7632"), imm(p, "0x5410"));
        let quad_lanes = vec![quad.clone(), p.add32(&quad, 1), p.add32(&quad, 2), p.add32(&quad, 3)];
        Self { lane: lane.to_string(), g, t, quad, src_a, src_b, sel, quad_lanes }
    }
}

/// Four C tiles (16 x 32 f16) -> the E4M3 A fragment of the same 16 x 32 (NaN -> 0 when `zero` is given).
pub fn acc_to_a(p: &Ptx, tiles: &[Vec<String>], li: &LaneInfo, zero: Option<&str>) -> Vec<String> {
    let mut q = Vec::new();
    for tile in tiles.iter().take(4) {
        let mut row = Vec::new();
        for v in tile.iter().take(2) {
            let v = match zero {
                Some(z) => nanzero(p, v, z),
                None => v.clone(),
            };
            row.push(cvt_e4x2(p, v));
        }
        q.push(row);
    }
    let mut a = Vec::new();
    for pair in 0..2 {
        // k 0..15 (tiles 0, 1) then k 16..31 (tiles 2, 3)
        for h in 0..2 {
            let w = pack16(p, &q[2 * pair][h], &q[2 * pair + 1][h]);
            let s0 = shfl_idx(p, &w, &li.src_a);
            let s1 = shfl_idx(p, &w, &li.src_b);
            a.push(prmt(p, s0, s1, &li.sel));
        }
    }
    // order: a0 (row g, k 0..15), a1 (row g+8, k 0..15), a2 (row g, k 16..31), a3 (row g+8, k 16..31)
    a
}

/// 16 x 32 f16 accumulator -> f16 A fragment of k16 step `step` (cols 16 step ..): a register renaming.
pub fn acc_f16_to_a(tiles: &[Vec<String>], step: usize) -> Vec<String> {
    vec![tiles[2 * step][0].clone(), tiles[2 * step][1].clone(), tiles[2 * step + 1][0].clone(), tiles[2 * step + 1][1].clone()]
}

/// Cosine normalization of a 16 x 32 accumulator (per row half), reference association order:
/// r[c] = fma(v[c], v[c], v[c+16]^2) (c < 16), s8 = r[c] + r[c+8], s4 (lane ^ 2), s2 (lane ^ 1), s1 = s2.x + s2.y,
/// norm = f16(rsqrt(f32(s1))); v * norm (* scale); NaN -> 0.
/// guarded: NaN can only come from 0 * inf (a zero row: s1 == 0), so the NaN -> 0 pass runs under a warp-uniform
/// branch taken only when some lane's row sum is zero (image borders).
pub fn normalize(p: &Ptx, tiles: &[Vec<String>], _li: &LaneInfo, zero: &str, scale2: Option<&str>, guarded: bool) -> Vec<Vec<String>> {
    let mut out = vec![vec![String::new(), String::new()]; 4];
    let any_zero = p.reg("pred");
    p.emit(format!("setp.ne.u32 {any_zero}, 0, 0;")); // false
    for h in 0..2 {
        let (p0, p1, p2, p3) = (&tiles[0][h], &tiles[1][h], &tiles[2][h], &tiles[3][h]);
        let hs0 = hmul2(p, p2, p2);
        let r0 = hfma2(p, p0, p0, hs0);
        let hs1 = hmul2(p, p3, p3);
        let r1 = hfma2(p, p1, p1, hs1);
        let s8 = hadd2(p, r0, r1);
        let s8o = shfl_xor(p, &s8, 2);
        let s4 = hadd2(p, s8, s8o);
        let s4o = shfl_xor(p, &s4, 1);
        let s2 = hadd2(p, s4, s4o);
        let (lo, hi) = unpack16(p, s2);
        let s1 = hadd(p, lo, hi);
        let f = p.reg("f32");
        p.emit(format!("cvt.f32.f16 {f}, {s1};"));
        if guarded {
            let pz = p.reg("pred");
            p.emit(format!("setp.eq.f32 {pz}, {f}, 0f00000000;"));
            p.emit(format!("or.pred {any_zero}, {any_zero}, {pz};"));
        }
        let rs = p.reg("f32");
        p.emit(format!("rsqrt.approx.f32 {rs}, {f};"));
        let norm = p.reg("f16");
        p.emit(format!("cvt.rn.f16.f32 {norm}, {rs};"));
        let norm2 = pack16(p, &norm, &norm);
        for j in 0..4 {
            let mut n = hmul2(p, &tiles[j][h], &norm2);
            if let Some(s) = scale2 {
                n = hmul2(p, n, s);
            }
            out[j][h] = if guarded { n } else { nanzero(p, n, zero) };
        }
    }
    if guarded {
        let vote = p.reg("pred");
        p.emit(format!("vote.sync.any.pred {vote}, {any_zero}, 0xffffffff;"));
        let skip = p.label("NZ");
        p.emit(format!("@!{vote} bra {skip};"));
        for tile in &out {
            for v in tile {
                let clean = nanzero(p, v, zero);
                p.emit(format!("mov.b32 {v}, {clean};"));
            }
        }
        p.emit(format!("{skip}:"));
    }
    out
}

pub struct ExpConsts {
    pub a: String,
    pub b: String,
    pub lo: String,
    pub hi: String,
    pub m: String,
    pub x: String,
}

impl ExpConsts {
    pub fn new(p: &Ptx) -> Self {
        let a = p.reg("f32");
        p.emit(format!("mov.f32 {a}, 0f3D380000;")); // 0.044921875
        let b = p.reg("f32");
        p.emit(format!("mov.f32 {b}, 0f3FA68000;")); // 1.30078125
        let lo = imm(p, "0x3C203C20"); // 1.03125
        let hi = imm(p, "0x3E473E47"); // 1.5693359375
        let m = imm(p, "0xffe0ffe0");
        let x = imm(p, "0x80008000");
        Self { a, b, lo, hi, m, x }
    }
}

/// The exponential approximation on a packed pair: f32 affine, f16 clamp, (bits << 5 & 0xffe0) ^ 0x8000 per half.
pub fn exp_weight(p: &Ptx, s: impl Display, k: &ExpConsts) -> String {
    let (lo, hi) = unpack16(p, s);
    let flo = p.reg("f32");
    p.emit(format!("cvt.f32.f16 {flo}, {lo};"));
    let fhi = p.reg("f32");
    p.emit(format!("cvt.f32.f16 {fhi}, {hi};"));
    p.emit(format!("fma.rn.f32 {flo}, {flo}, {}, {};", k.a, k.b));
    p.emit(format!("fma.rn.f32 {fhi}, {fhi}, {}, {};", k.a, k.b));
    let h = p.reg("b32");
    p.emit(format!("cvt.rn.f16x2.f32 {h}, {fhi}, {flo};"));
    let h = hmax2(p, h, &k.lo);
    let h = hmin2(p, h, &k.hi);
    let r = p.reg("b32");
    p.emit(format!("shl.b32 {r}, {h}, 5;"));
    let d = p.reg("b32");
    p.emit(format!("lop3.b32 {d}, {r}, {}, {}, 0x6A;", k.m, k.x)); // (a & b) ^ c
    d
}

/// 16 x 64 score accumulator (8 C tiles) -> attention weights in the same layout (the specified softmax reduction order).
pub fn softmax(p: &Ptx, s_tiles: &[Vec<String>], li: &LaneInfo, k: &ExpConsts) -> Vec<Vec<String>> {
    let mut w = vec![vec![String::new(), String::new()]; 8];
    for h in 0..2 {
        let e: Vec<String> = (0..8).map(|m| exp_weight(p, &s_tiles[m][h], k)).collect();
        let bp: Vec<String> = (0..4).map(|j| hadd2(p, &e[2 * j], &e[2 * j + 1])).collect();
        let t01 = hadd2(p, &bp[0], &bp[1]);
        let t012 = hadd2(p, t01, &bp[2]);
        let tp = hadd2(p, t012, &bp[3]);
        let tq: Vec<String> = li.quad_lanes.iter().map(|src| shfl_idx(p, &tp, src)).collect();
        let parts: Vec<(String, String)> = tq.iter().map(|t| unpack16(p, t)).collect();
        let e01 = hadd(p, &parts[0].0, &parts[1].0);
        let e012 = hadd(p, e01, &parts[2].0);
        let even = hadd(p, e012, &parts[3].0);
        let o01 = hadd(p, &parts[0].1, &parts[1].1);
        let o012 = hadd(p, o01, &parts[2].1);
        let odd = hadd(p, o012, &parts[3].1);
        let total = hadd(p, even, odd);
        let f = p.reg("f32");
        p.emit(format!("cvt.f32.f16 {f}, {total};"));
        let rc = p.reg("f32");
        p.emit(format!("rcp.rn.f32 {rc}, {f};"));
        let rec = p.reg("f16");
        p.emit(format!("cvt.rn.f16.f32 {rec}, {rc};"));
        let rec2 = pack16(p, &rec, &rec);
        for m in 0..8 {
            w[m][h] = hmul2(p, &e[m], &rec2);
        }
    }
    w
}

/// natural token (0..63) -> physical (tiled) index: (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3).
pub fn tiled_token_regs(p: &Ptx, token: impl Display) -> String {
    let x = p.and32(&token, 7);
    let y = p.shr32(&token, 3);
    let a = p.shl32(p.shr32(&y, 2), 5);
    let b = p.shl32(p.shr32(&x, 2), 4);
    let c = p.shl32(p.and32(&y, 3), 2);
    let d = p.and32(&x, 3);
    p.add32(p.add32(a, b), p.add32(c, d))
}

/// Exact unsigned division of n (< 2^24) by d (> 0) via the f32 reciprocal d_rcp = 1 / d (rcp.rn), corrected by
/// one remainder check each way: returns (q, r).
pub fn fast_divmod(p: &Ptx, n: impl Display, d: impl Display, d_rcp: impl Display) -> (String, String) {
    let f = p.reg("f32");
    p.emit(format!("cvt.rn.f32.u32 {f}, {n};"));
    let qf = p.reg("f32");
    p.emit(format!("mul.rn.f32 {qf}, {f}, {d_rcp};"));
    let q = p.reg("b32");
    p.emit(format!("cvt.rzi.u32.f32 {q}, {qf};"));
    let r = p.reg("b32");
    p.emit(format!("mul.lo.u32 {r}, {q}, {d};"));
    p.emit(format!("sub.u32 {r}, {n}, {r};"));
    let p_neg = p.setp("lt.s32", &r, 0); // r < 0 as signed: q one too large
    let p_big = p.setp("ge.u32", &r, &d); // r >= d: q one too small (only when r >= 0)
    p.emit(format!("@{p_neg} sub.u32 {q}, {q}, 1;"));
    p.emit(format!("@{p_neg} add.u32 {r}, {r}, {d};"));
    let p_big2 = p.reg("pred");
    p.emit(format!("and.pred {p_big2}, {p_big}, !{p_neg};"));
    p.emit(format!("@{p_big2} add.u32 {q}, {q}, 1;"));
    p.emit(format!("@{p_big2} sub.u32 {r}, {r}, {d};"));
    (q, r)
}

pub fn rcp_f32(p: &Ptx, d: impl Display) -> String {
    let f = p.reg("f32");
    p.emit(format!("cvt.rn.f32.u32 {f}, {d};"));
    let r = p.reg("f32");
    p.emit(format!("rcp.rn.f32 {r}, {f};"));
    r
}

/// Four C tiles (16 x 32 f16) -> the E4M3 A fragment through a per-warp [16][32 B] chunk-swizzled staging tile:
/// c_base[c] = tile + g * 32 + ((c ^ ((g >> 2) & 1)) * 16) + 2 t for chunk c (the C-layout store addresses),
/// a_lane = tile + aRowL * 32 + (aChunkL ^ ((aRowL >> 2) & 1)) * 16 (the ldmatrix address). NaN -> 0 when `zero`.
pub fn acc_to_a_smem(p: &Ptx, tiles: &[Vec<String>], c_base: &[String], a_lane: impl Display, zero: Option<&str>) -> Vec<String> {
    p.emit("bar.warp.sync 0xffffffff;"); // the previous fragment load from the tile has completed
    for (j, tile) in tiles.iter().take(4).enumerate() {
        for (h, v) in tile.iter().take(2).enumerate() {
            let v = match zero {
                Some(z) => nanzero(p, v, z),
                None => v.clone(),
            };
            let hc = cvt_e4x2(p, v);
            p.emit(format!("st.shared.b16 [{}+{}], {hc};", c_base[j >> 1], 8 * (j & 1) + 256 * h));
        }
    }
    p.emit("bar.warp.sync 0xffffffff;");
    ldm4(p, a_lane)
}

// ---- cross-kernel dependency counters (barrier-free chaining of consecutive launches)
// Counters are u32 in a per-frame zeroed buffer. Producers: fence.acq_rel.gpu + bar.sync + thread 0 red.release.gpu.add.
// Consumers: every warp polls its counters (lanes in parallel) with ld.acquire.gpu until all reach the expected count.

/// ns: polling backoff ceiling
pub const SLEEP_MAX: u32 = 256;
/// A wait this long is a failed scheduling assumption (docs/execution.md), not a slow frame.
pub const WAIT_LIMIT_NS: u64 = 1_000_000_000;

/// The count `sync_wait` waits for: a register, or a function emitting the expected count for the counter index
/// register.
pub enum Expected<'a> {
    Reg(&'a str),
    Per(&'a dyn Fn(&str) -> String),
}

/// Spin until counters [first .. last] (at most 32) at base64 (u64 byte address of counter 0) are all >= expected
/// (a register, or a function emitting the expected count for the counter index register). `guard` (pred) skips the
/// wait when false. With `warp` given, only warp 0 polls (exponential backoff 128 .. 256 ns) and a bar.sync
/// releases the workgroup; otherwise every warp polls for itself.
/// A wait that lasts WAIT_LIMIT_NS gives up instead of hanging: it sets bit 31 of the counters it is stuck on (which
/// releases every other waiter on them), adds one to the u32 at error64 and, if it is the first, records the low
/// 32 bits of a stuck counter's address at error64 + 4. Every other wait still spinning then gives up too, so the
/// frame completes one limit later and the host reports it.
#[allow(clippy::too_many_arguments)]
pub fn sync_wait(p: &Ptx, base64: &str, first: &str, last: &str, expected: Expected, lane: &str, guard: Option<&str>, warp: Option<&str>, error64: &str) {
    let skip = p.label("NOWAIT");
    if let Some(g) = guard {
        p.emit(format!("@!{g} bra {skip};"));
    }
    let mut join_l = None;
    if let Some(w) = warp {
        let p_w0 = p.setp("ne.u32", w, 0);
        let l = p.label("WJOIN");
        p.emit(format!("@{p_w0} bra {l};"));
        join_l = Some(l);
    }
    let idx = p.add32(first, lane);
    let p_mine = p.setp("le.u32", &idx, last);
    let expected = match expected {
        Expected::Reg(r) => r.to_string(),
        Expected::Per(f) => f(&idx),
    };
    let addr = p.add64(base64, p.widen(p.shl32(&idx, 2)));
    let sleep = p.reg("b32");
    p.emit(format!("mov.u32 {sleep}, 128;"));
    let start = p.reg("b64");
    p.emit(format!("mov.u64 {start}, %globaltimer;"));
    let lp = p.label("WAIT");
    p.emit(format!("{lp}:"));
    let v = p.reg("b32");
    p.emit(format!("mov.u32 {v}, 0xffffffff;"));
    p.emit(format!("@{p_mine} ld.acquire.gpu.global.u32 {v}, [{addr}];"));
    let p_ok = p.setp("ge.u32", &v, &expected);
    let p_all = p.reg("pred");
    p.emit(format!("vote.sync.all.pred {p_all}, {p_ok}, 0xffffffff;"));
    let done = p.label("WAITED");
    p.emit(format!("@{p_all} bra {done};"));
    p.emit(format!("nanosleep.u32 {sleep};"));
    p.emit(format!("shl.b32 {sleep}, {sleep}, 1;"));
    p.emit(format!("min.u32 {sleep}, {sleep}, {SLEEP_MAX};"));
    let elapsed = p.reg("b64");
    p.emit(format!("mov.u64 {elapsed}, %globaltimer;"));
    p.emit(format!("sub.u64 {elapsed}, {elapsed}, {start};"));
    let p_late = p.setp("gt.u64", &elapsed, WAIT_LIMIT_NS);
    let gave_up = p.reg("b32");
    p.emit(format!("ld.relaxed.gpu.global.u32 {gave_up}, [{error64}];")); // once one wait gave up, all do
    let p_abandon = p.setp("ne.u32", &gave_up, 0);
    let p_give_up = p.reg("pred");
    p.emit(format!("or.pred {p_give_up}, {p_late}, {p_abandon};"));
    p.emit(format!("@!{p_give_up} bra {lp};"));
    let p_stuck = p.reg("pred");
    p.emit(format!("and.pred {p_stuck}, {p_mine}, !{p_ok};"));
    p.emit(format!("@{p_stuck} red.relaxed.gpu.global.or.b32 [{addr}], 0x80000000;"));
    let p_lane0 = p.setp("eq.u32", lane, 0);
    p.emit(format!("@{p_lane0} red.relaxed.gpu.global.add.u32 [{error64}], 1;"));
    let addr_low = p.reg("b32");
    p.emit(format!("cvt.u32.u64 {addr_low}, {addr};"));
    let first_ = p.reg("b32");
    p.emit(format!("@{p_stuck} atom.relaxed.gpu.global.cas.b32 {first_}, [{error64}+4], 0, {addr_low};"));
    p.emit(format!("{done}:"));
    if let Some(l) = join_l {
        p.emit(format!("{l}:"));
        p.emit("bar.sync 0;");
    }
    p.emit(format!("{skip}:"));
}

/// After this workgroup's global stores: release them and increment the counters at addrs (u64 regs) by one.
/// All threads must reach this point (bar.sync inside).
pub fn sync_signal(p: &Ptx, addrs: &[&str], tid: &str, guard: Option<&str>) {
    p.emit("fence.acq_rel.gpu;");
    p.emit("bar.sync 0;");
    let mut p_t0 = p.setp("eq.u32", tid, 0);
    if let Some(g) = guard {
        let p_g = p.reg("pred");
        p.emit(format!("and.pred {p_g}, {p_t0}, {g};"));
        p_t0 = p_g;
    }
    for a in addrs {
        p.emit(format!("@{p_t0} red.release.gpu.global.add.u32 [{a}], 1;"));
    }
}

/// Pixel-row bands (8 rows) touched by token rows [row_first, row_last]: (b0, b1).
pub fn band_range(p: &Ptx, row_first: impl Display, row_last: impl Display, width: impl Display) -> (String, String) {
    let y0 = p.reg("b32");
    p.emit(format!("div.u32 {y0}, {row_first}, {width};"));
    let y1 = p.reg("b32");
    p.emit(format!("div.u32 {y1}, {row_last}, {width};"));
    (p.shr32(y0, 3), p.shr32(y1, 3))
}

/// Number of producer row groups (of `group_rows` token rows each; a register, 64 when None) intersecting pixel-row
/// band `band` (rows [8 band, 8 band + 8) of `width` tokens), times `mul` (the producer's signals per row group,
/// e.g. its column-group count). The producer signals every band its rows touch once per workgroup, so this count
/// must be exactly its workgroup size: a smaller group size here would release the consumer early (the C=256
/// FFN runs 48-row workgroups).
pub fn band_expected(p: &Ptx, band: &str, width: &str, rows: &str, mul: Option<&str>, group_rows: Option<&str>) -> String {
    let t0 = p.mul32(p.shl32(band, 3), width);
    let t1 = p.sub32(p.add32(&t0, p.shl32(width, 3)), 1);
    let rows_minus1 = p.sub32(rows, 1);
    let p_clamp = p.setp("gt.u32", &t1, &rows_minus1);
    let t1c = p.selp32(p_clamp, &rows_minus1, &t1);
    let (g0, g1) = match group_rows {
        None => (p.shr32(&t0, 6), p.shr32(&t1c, 6)),
        Some(gr) => {
            let g0 = p.reg("b32");
            p.emit(format!("div.u32 {g0}, {t0}, {gr};"));
            let g1 = p.reg("b32");
            p.emit(format!("div.u32 {g1}, {t1c}, {gr};"));
            (g0, g1)
        }
    };
    let n = p.add32(p.sub32(g1, g0), 1);
    match mul {
        None => n,
        Some(m) => p.mul32(n, m),
    }
}

// ---- ViT variants of the same publication rules (different constants, different orderings)

/// expWeightPairVit on a packed pair: fma.rn.f16x2 (0.0895, 1.709), clamp [1.4395, 1.9775], ((bits << 4) + 0x4000) & 0xffff per half.
pub fn vit_exp(p: &Ptx, s: impl Display, k: &VitExpConsts) -> String {
    let a = hfma2(p, s, &k.c1, &k.c2);
    let a = hmax2(p, a, &k.lo);
    let a = hmin2(p, a, &k.hi);
    let lo = p.reg("b32");
    p.emit(format!("shl.b32 {lo}, {a}, 4;"));
    p.emit(format!("add.u32 {lo}, {lo}, 0x4000;"));
    p.emit(format!("and.b32 {lo}, {lo}, 0xffff;"));
    let hi = p.reg("b32");
    p.emit(format!("shr.u32 {hi}, {a}, 12;"));
    p.emit(format!("and.b32 {hi}, {hi}, 0xfff0;"));
    p.emit(format!("add.u32 {hi}, {hi}, 0x4000;"));
    p.emit(format!("and.b32 {hi}, {hi}, 0xffff;"));
    p.emit(format!("shl.b32 {hi}, {hi}, 16;"));
    let r = p.reg("b32");
    p.emit(format!("or.b32 {r}, {lo}, {hi};"));
    r
}

pub struct VitExpConsts {
    pub c1: String,
    pub c2: String,
    pub lo: String,
    pub hi: String,
}

impl VitExpConsts {
    pub fn new(p: &Ptx) -> Self {
        Self {
            c1: imm(p, "0x2DBB2DBB"), // f16(0.08953946828842163)
            c2: imm(p, "0x3ED63ED6"), // f16(1.7093614339828491)
            lo: imm(p, "0x3DC23DC2"), // 1.439453125
            hi: imm(p, "0x3FE93FE9"), // 1.9775390625
        }
    }
}

/// inverseNormVit on 32 halves (16 packed words): r[m] = f16(f32(v[m])^2 + f32(f16(v[m+8]^2))) per component,
/// s8/s4/s2 in f16x2, s1 = s2.x + s2.y, f16(rsqrt(f32(s1))) broadcast to a pair.
pub fn vit_norm(p: &Ptx, w: &[String]) -> String {
    let mut r = Vec::new();
    for m in 0..8 {
        let hs = hmul2(p, &w[m + 8], &w[m + 8]);
        let (lo, hi) = unpack16(p, &w[m]);
        let flo = p.reg("f32");
        p.emit(format!("cvt.f32.f16 {flo}, {lo};"));
        let fhi = p.reg("f32");
        p.emit(format!("cvt.f32.f16 {fhi}, {hi};"));
        p.emit(format!("mul.rn.f32 {flo}, {flo}, {flo};"));
        p.emit(format!("mul.rn.f32 {fhi}, {fhi}, {fhi};"));
        let (hlo, hhi) = unpack16(p, hs);
        let glo = p.reg("f32");
        p.emit(format!("cvt.f32.f16 {glo}, {hlo};"));
        let ghi = p.reg("f32");
        p.emit(format!("cvt.f32.f16 {ghi}, {hhi};"));
        p.emit(format!("add.rn.f32 {flo}, {flo}, {glo};"));
        p.emit(format!("add.rn.f32 {fhi}, {fhi}, {ghi};"));
        let d = p.reg("b32");
        p.emit(format!("cvt.rn.f16x2.f32 {d}, {fhi}, {flo};"));
        r.push(d);
    }
    let s8: Vec<String> = (0..4).map(|m| hadd2(p, &r[m], &r[m + 4])).collect();
    let s4a = hadd2(p, &s8[0], &s8[2]);
    let s4b = hadd2(p, &s8[1], &s8[3]);
    let s2 = hadd2(p, s4a, s4b);
    let (lo, hi) = unpack16(p, s2);
    let s1 = hadd(p, lo, hi);
    let f = p.reg("f32");
    p.emit(format!("cvt.f32.f16 {f}, {s1};"));
    let rs = p.reg("f32");
    p.emit(format!("rsqrt.approx.f32 {rs}, {f};"));
    let n = p.reg("f16");
    p.emit(format!("cvt.rn.f16.f32 {n}, {rs};"));
    pack16(p, &n, &n)
}

/// scaleQuantizeVit: 32 halves times the norm (queries: then f16(sqrt 32), then the learned scale), E4 codes (8 words).
/// A zero row has norm = inf and 0 * inf = NaN: the GLSL conversion yields 0 for it (nanzero before the cvt).
pub fn vit_quantize(p: &Ptx, w: &[String], norm2: &str, query: bool, hs2: &str, learned2: &str, zero: &str) -> Vec<String> {
    let mut codes = Vec::new();
    for i in 0..8 {
        let mut a = hmul2(p, &w[2 * i], norm2);
        let mut b = hmul2(p, &w[2 * i + 1], norm2);
        if query {
            a = hmul2(p, hmul2(p, a, hs2), learned2);
            b = hmul2(p, hmul2(p, b, hs2), learned2);
        }
        let a = nanzero(p, a, zero);
        let b = nanzero(p, b, zero);
        codes.push(pack16(p, cvt_e4x2(p, a), cvt_e4x2(p, b)));
    }
    codes
}
