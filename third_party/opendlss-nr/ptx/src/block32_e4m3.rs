//! Fused 32-channel Swin block (blocks 0-4, 66-70) in PTX: the fused_block32.comp dataflow at the fragment level.
//!
//! One workgroup = 4 warps = one 8x8 window (warp w: tokens 16w .. 16w+15 = window rows 2w, 2w+1), persistent over
//! windows. Per warp everything is a 16-row fragment: FFN (x W1 -> SiLU -> E4 -> W2 + skip * ffnScale, in four k32
//! quarters with W1's columns permuted so the hidden C fragment packs straight into the W2 A fragment), q/k/v
//! projections, cosine normalization, K rows / transposed V into shared memory (physical token order), S = Q K^T +
//! prior, the specified softmax, P V, projection + ffn * attnScale, E4 output. Every arithmetic step is that of the
//! GLSL kernel (same instructions on the same fragment elements); only the data movement differs.
//!
//! Variants (flags): PRE (block 0: f32 features x f16 adapter in the input path), POST (block 70: learned post
//! blend of block-0 E4 and the 2x upsampled half-res E4), UPRES (block 66: f16 low-res projection + E4 skip *
//! scale), HEAD (RGBA head from the f16 result), POOL (2x2 box pool of the f16 result to E4), OUT_E4.
//!
//! Arguments: `<flags> [maxRegs]`      flags: sum of E4=2 PRE=8 POST=16 HEAD=32 POOL=64 UPRES=128

use std::collections::HashMap;

use crate::ptxgen::Ptx;
use crate::swin::*;

pub const F_OUT_E4: usize = 2;
pub const F_PRE: usize = 8;
pub const F_POST: usize = 16;
pub const F_HEAD: usize = 32;
pub const F_POOL: usize = 64;
pub const F_UPRES: usize = 128;

// shared memory map (bytes)
pub const W1S: usize = 0;
pub const W2S: usize = 4096;
pub const WQKVS: usize = 8192;
pub const WPROJS: usize = 11264;
/// K [64 physical][48 B]
pub const KROWS: usize = 12288;
/// Vt [32 dim][80 B]
pub const VTS: usize = 15360;
/// per warp 1280 B: f16 tile [16][80 B]; the E4 tile [16][32 B] aliases its start
pub const STG: usize = 17920;
pub const STG_WARP: usize = 1280;
/// adapter [32 n][16 k] f16 / head [16 n][32 k] f16 (1 KB)
pub const WF16S: usize = STG + 4 * STG_WARP;
pub const SHARED_BYTES: usize = WF16S + 1024;
pub const THREADS: usize = 128;
/// accumulator -> A fragment through the per-warp staging tile (else shuffles)
pub const A2A_SMEM: usize = 0;

/// The per-window geometry registers.
struct Geometry {
    window_x: String,
    window_y: String,
    valid: String,
    pixel: String,
    low_pixel: Option<String>,
    wy: String,
    wr0: String,
    wr1: String,
}

/// The kernel's name and PTX.
pub fn generate(flags: usize, max_regs: Option<u32>) -> (String, String) {
    let (pre, post, upres) = (flags & F_PRE != 0, flags & F_POST != 0, flags & F_UPRES != 0);
    let (head, pool, out_e4) = (flags & F_HEAD != 0, flags & F_POOL != 0, flags & F_OUT_E4 != 0);
    assert!(pre as usize + post as usize + upres as usize <= 1);
    let name = format!("block32_e4m3_f{flags}");
    let p = Ptx::new();
    let params = [
        ("u64", "pState"),
        ("u64", "pLow"),
        ("u64", "pW1"),
        ("u64", "pW2"),
        ("u64", "pWqkv"),
        ("u64", "pWproj"),
        ("u64", "pAux"),
        ("u64", "pPrior"),
        ("u64", "pOutE4"),
        ("u64", "pOut2"),
        ("u64", "pWf16"),
        ("u32", "width"),
        ("u32", "height"),
        ("u32", "shiftX"),
        ("u32", "shiftY"),
        ("u32", "windowsX"),
        ("u32", "windowCount"),
        ("u32", "auxFfnHalf"),
        ("u32", "auxAttnHalf"),
        ("u32", "scaleWord"),
        ("u32", "auxInputHalf"),
        ("u32", "auxAdapterHalf"),
        ("u32", "lowWidth"),
        ("u64", "pWait"),
        ("u32", "waitExpected"),
        ("u32", "waitShiftY"),
        ("u32", "waitScale"),
        ("u64", "pSignal"),
        ("u64", "pError"),
    ];
    // chaining (pWait / pSignal = 0: off): wait on the producer block's window rows covering this window's pixel rows
    // (waitScale 0: same resolution, 1: producer at 2x (pool), 2: producer at half (upres / post)), signal this block's
    p.entry(&name, &params, SHARED_BYTES, Some(THREADS as u32), max_regs, false);
    let pm: HashMap<&str, String> = params.iter().map(|&(t, n)| (n, if t == "u64" { p.load_param_u64(n) } else { p.load_param_u32(n) })).collect();
    let tid = p.special("tid.x");
    let ctaid = p.special("ctaid.x");
    let nctaid = p.special("nctaid.x");
    let lane = p.and32(&tid, 31);
    let warp = p.shr32(&tid, 5);
    let li = LaneInfo::new(&p, &lane);
    let (g, t) = (&li.g, &li.t);
    let smem = p.shared_addr(p.imm32(0));
    let zero = imm(&p, 0);
    let silu_k = SiluConsts::new(&p);
    let exp_k = ExpConsts::new(&p);

    // ---- stage the block weights once: [rows][32 B] tiles, chunk ^= (row >> 2) & 1
    let stage_e4_rows = |src: &str, dst_base: usize, chunks: usize| {
        for i in 0..chunks.div_ceil(THREADS) {
            let c = p.add32(&tid, i * THREADS);
            let pv = if !chunks.is_multiple_of(THREADS) { Some(p.setp("lt.u32", &c, chunks)) } else { None };
            let row = p.shr32(&c, 1);
            let chunk = p.and32(&c, 1);
            let pch = p.xor32(&chunk, p.and32(p.shr32(&row, 2), 1));
            let dst = p.add32(p.add32(&smem, dst_base), p.add32(p.shl32(&row, 5), p.shl32(&pch, 4)));
            let gaddr = p.add64(src, p.widen(p.shl32(&c, 4)));
            let r = p.regs("b32", 4);
            let pred = match &pv {
                Some(pv) => format!("@{pv} "),
                None => String::new(),
            };
            p.emit(format!("{pred}ld.global.nc.v4.b32 {{{}, {}, {}, {}}}, [{gaddr}];", r[0], r[1], r[2], r[3]));
            p.emit(format!("{pred}st.shared.v4.b32 [{dst}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
        }
    };
    stage_e4_rows(&pm["pW1"], W1S, 256);
    stage_e4_rows(&pm["pW2"], W2S, 256);
    stage_e4_rows(&pm["pWqkv"], WQKVS, 192);
    stage_e4_rows(&pm["pWproj"], WPROJS, 64);
    if pre {
        // adapter f16 [16 k][32 n] (global) -> shared [32 n][16 k] with the 32-byte-row chunk swizzle
        let n = p.shr32(&tid, 2);
        let kq = p.and32(&tid, 3);
        let mut hs = Vec::new();
        for i in 0..4 {
            let k = p.add32(p.shl32(&kq, 2), i);
            let addr = p.add64(&pm["pWf16"], p.widen(p.shl32(p.add32(p.shl32(&k, 5), &n), 1)));
            let h = p.reg("b16");
            p.emit(format!("ld.global.nc.b16 {h}, [{addr}];"));
            hs.push(h);
        }
        let w0 = p.reg("b32");
        p.emit(format!("mov.b32 {w0}, {{{}, {}}};", hs[0], hs[1]));
        let w1 = p.reg("b32");
        p.emit(format!("mov.b32 {w1}, {{{}, {}}};", hs[2], hs[3]));
        let pch = p.xor32(p.shr32(&kq, 1), p.and32(p.shr32(&n, 2), 1));
        let dst = p.add32(p.add32(&smem, WF16S), p.add32(p.shl32(&n, 5), p.add32(p.shl32(&pch, 4), p.shl32(p.and32(&kq, 1), 3))));
        p.emit(format!("st.shared.v2.b32 [{dst}], {{{w0}, {w1}}};"));
    }
    if head {
        // head f16 [32 k][16 n] (global, padded N = 16) -> shared [16 n][32 k] (64-byte rows)
        let n = p.shr32(&tid, 3);
        let kq = p.and32(&tid, 7);
        let mut hs = Vec::new();
        for i in 0..4 {
            let k = p.add32(p.shl32(&kq, 2), i);
            let addr = p.add64(&pm["pWf16"], p.widen(p.shl32(p.add32(p.shl32(&k, 4), &n), 1)));
            let h = p.reg("b16");
            p.emit(format!("ld.global.nc.b16 {h}, [{addr}];"));
            hs.push(h);
        }
        let w0 = p.reg("b32");
        p.emit(format!("mov.b32 {w0}, {{{}, {}}};", hs[0], hs[1]));
        let w1 = p.reg("b32");
        p.emit(format!("mov.b32 {w1}, {{{}, {}}};", hs[2], hs[3]));
        let dst = p.add32(p.add32(&smem, WF16S), p.add32(p.shl32(&n, 6), p.shl32(&kq, 3)));
        p.emit(format!("st.shared.v2.b32 [{dst}], {{{w0}, {w1}}};"));
    }

    // ---- per-lane constants
    // skip scale pairs: accumulator tile j of either row half holds columns 8j + 2t, +1
    let aux_base = &pm["pAux"];
    let aux_pairs = |half_offset_reg: &str| -> Vec<String> {
        let mut regs = Vec::new();
        for j in 0..4 {
            let off = p.add32(p.shl32(half_offset_reg, 1), p.add32(p.shl32(t, 2), p.imm32(16 * j)));
            let addr = p.add64(aux_base, p.widen(&off));
            let r = p.reg("b32");
            p.emit(format!("ld.global.nc.b32 {r}, [{addr}];"));
            regs.push(r);
        }
        regs
    };
    let aux_ffn = aux_pairs(&pm["auxFfnHalf"]);
    let aux_attn = aux_pairs(&pm["auxAttnHalf"]);
    let scale_addr = p.add64(aux_base, p.widen(p.shl32(&pm["scaleWord"], 2)));
    let scale_f = p.reg("f32");
    p.emit(format!("ld.global.nc.f32 {scale_f}, [{scale_addr}];"));
    let scale_h = p.reg("f16");
    p.emit(format!("cvt.rn.f16.f32 {scale_h}, {scale_f};"));
    let scale2 = pack16(&p, &scale_h, &scale_h);
    // load lane: row = lane >> 1 of this warp's 16 tokens, part = lane & 1 (16 bytes of the E4 row)
    let ld_row = p.shr32(&lane, 1);
    let ld_part = p.and32(&lane, 1);
    let ld_tok_x = p.and32(&ld_row, 7);
    let ld_tok_y = p.add32(p.shl32(&warp, 1), p.shr32(&lane, 4));
    let ld_sw = p.and32(p.shr32(&ld_row, 2), 1);
    let stg_warp = p.add32(p.add32(&smem, STG), p.mul32(&warp, STG_WARP));
    let stg_ld_e4 = p.add32(&stg_warp, p.add32(p.shl32(&ld_row, 5), p.shl32(p.xor32(&ld_part, &ld_sw), 4))); // this lane's 16-byte E4 chunk
    let stg_ld_f16 = p.add32(&stg_warp, p.add32(p.mul32(&ld_row, 80), p.shl32(&ld_part, 5))); // this lane's 32-byte f16 chunk
                                                                                              // ldmatrix lane addressing for [16 rows][32 B] tiles (A) and [n rows][32 B] tiles (B), both swizzled
    let l8 = p.and32(&lane, 7);
    let lm = p.shr32(&lane, 3);
    let a_row_l = p.add32(&l8, p.shl32(p.and32(&lm, 1), 3));
    let a_chunk_l = p.shr32(&lane, 4);
    let a_pch = p.xor32(&a_chunk_l, p.and32(p.shr32(&a_row_l, 2), 1));
    let a_lane_off = p.add32(p.shl32(&a_row_l, 5), p.shl32(&a_pch, 4));
    let stg_a = p.add32(&stg_warp, &a_lane_off);
    let b_nl = p.add32(&l8, p.shl32(p.shr32(&lane, 4), 3));
    let b_chunk_l = p.and32(&lm, 1);
    let b_pch = p.xor32(&b_chunk_l, p.and32(p.shr32(&b_nl, 2), 1));
    let b_lane_off = p.add32(p.shl32(&b_nl, 5), p.shl32(&b_pch, 4));
    let w1_b = p.add32(p.add32(&smem, W1S), &b_lane_off);
    let w2_b = p.add32(p.add32(&smem, W2S), &b_lane_off);
    let wqkv_b = p.add32(p.add32(&smem, WQKVS), &b_lane_off);
    let wproj_b = p.add32(p.add32(&smem, WPROJS), &b_lane_off);
    let wf16_b = p.add32(p.add32(&smem, WF16S), &b_lane_off);
    // C-layout b16 loads from the E4 staging tile: row g (+8), byte column 8j + 2t -> chunk j >> 1 (swizzled), +8 (j & 1)
    let sw_g = p.and32(p.shr32(g, 2), 1);
    let c_base: Vec<String> =
        (0..2).map(|c| p.add32(&stg_warp, p.add32(p.shl32(g, 5), p.add32(p.shl32(p.xor32(p.imm32(c), &sw_g), 4), p.shl32(t, 1))))).collect();
    // C-layout b32 loads from the f16 staging tile: row g (+8) at 80 B, word 4j + t
    let c_base_f16 = p.add32(&stg_warp, p.add32(p.mul32(g, 80), p.shl32(t, 2)));
    // K rows / Vt publication: this lane's tokens g and g + 8 -> physical indices
    let phys: Vec<String> = (0..2).map(|h| tiled_token_regs(&p, p.add32(p.shl32(&warp, 4), p.add32(g, p.imm32(8 * h))))).collect();
    let k_pub: Vec<String> = (0..2).map(|h| p.add32(p.add32(&smem, KROWS), p.add32(p.mul32(&phys[h], 48), p.shl32(t, 1)))).collect();
    let v_pub: Vec<String> = (0..2).map(|h| p.add32(p.add32(&smem, VTS), p.add32(p.mul32(t, 160), &phys[h]))).collect();
    // ldmatrix addressing for K [64][48 B] (n = key rows) and Vt [32][80 B] (n = dim rows)
    let k_lane = p.add32(p.add32(&smem, KROWS), p.add32(p.mul32(&b_nl, 48), p.shl32(&b_chunk_l, 4)));
    let v_lane = p.add32(p.add32(&smem, VTS), p.add32(p.mul32(&b_nl, 80), p.shl32(&b_chunk_l, 4)));
    // prior rows for queries 16 warp + g (+8): [64][64] f16, keys 8m + 2t
    let prior_lane = p.add64(&pm["pPrior"], p.widen(p.add32(p.shl32(p.add32(p.shl32(&warp, 4), g), 7), p.shl32(t, 2))));
    let quarter2 = imm(&p, "0x34003400"); // 0.25, 0.25
    let mut blend_a: Vec<String> = Vec::new();
    let mut blend_b: Vec<String> = Vec::new();
    if post || upres {
        for i in 0..8 {
            let c = p.add32(p.shl32(&ld_part, 4), p.imm32(2 * i));
            let a = p.add64(aux_base, p.widen(p.shl32(p.add32(&pm["auxInputHalf"], &c), 1)));
            let r = p.reg("b32");
            p.emit(format!("ld.global.nc.b32 {r}, [{a}];"));
            blend_a.push(r);
            if post {
                let b = p.add64(aux_base, p.widen(p.shl32(p.add32(&pm["auxAdapterHalf"], &c), 1)));
                let r = p.reg("b32");
                p.emit(format!("ld.global.nc.b32 {r}, [{b}];"));
                blend_b.push(r);
            }
        }
    }
    // the attention prior rows of this warp's 16 queries never change: 16 registers for the whole kernel
    let mut prior_regs: Vec<[String; 2]> = Vec::new();
    for m in 0..8 {
        let c0 = p.reg("b32");
        p.emit(format!("ld.global.nc.b32 {c0}, [{prior_lane}+{}];", 16 * m));
        let c1 = p.reg("b32");
        p.emit(format!("ld.global.nc.b32 {c1}, [{prior_lane}+{}];", 1024 + 16 * m));
        prior_regs.push([c0, c1]);
    }
    p.emit("bar.sync 0;"); // weights staged

    // ================= persistent window loop (the next window's input words are prefetched into registers)
    let geometry = |window_reg: &str| -> Geometry {
        let wx = p.reg("b32");
        p.emit(format!("rem.u32 {wx}, {window_reg}, {};", pm["windowsX"]));
        let wy = p.reg("b32");
        p.emit(format!("div.u32 {wy}, {window_reg}, {};", pm["windowsX"]));
        let window_x = p.sub32(p.shl32(&wx, 3), &pm["shiftX"]);
        let window_y = p.sub32(p.shl32(&wy, 3), &pm["shiftY"]);
        let x = p.add32(&window_x, &ld_tok_x);
        let y = p.add32(&window_y, &ld_tok_y);
        let pvx = p.setp("lt.u32", &x, &pm["width"]);
        let pvy = p.setp("lt.u32", &y, &pm["height"]);
        let valid = p.reg("pred");
        p.emit(format!("and.pred {valid}, {pvx}, {pvy};"));
        let pixel = p.mad32(&y, &pm["width"], &x);
        let low_pixel = if post || upres { Some(p.mad32(p.shr32(&y, 1), &pm["lowWidth"], p.shr32(&x, 1))) } else { None };
        // producer window rows: this window's pixel rows [max(8 wy - shiftY, 0), min(+7, height - 1)] mapped to the
        // producer's resolution (x2 / x0.5), then (y + producer shiftY) >> 3
        let y_top = &window_y;
        let p_neg = p.setp("gt.s32", p.imm32(0), y_top);
        let y0 = p.selp32(&p_neg, p.imm32(0), y_top);
        let y1 = p.add32(y_top, 7);
        let hm1 = p.sub32(&pm["height"], 1);
        let p_big = p.setp("gt.u32", &y1, &hm1);
        let y1 = p.selp32(&p_big, &hm1, &y1);
        let p_s1 = p.setp("eq.u32", &pm["waitScale"], 1);
        let p_s2 = p.setp("eq.u32", &pm["waitScale"], 2);
        let y0 = p.selp32(&p_s1, p.shl32(&y0, 1), p.selp32(&p_s2, p.shr32(&y0, 1), &y0));
        let y1 = p.selp32(&p_s1, p.add32(p.shl32(&y1, 1), p.imm32(1)), p.selp32(&p_s2, p.shr32(&y1, 1), &y1));
        let wr0 = p.shr32(p.add32(&y0, &pm["waitShiftY"]), 3);
        let wr1 = p.shr32(p.add32(&y1, &pm["waitShiftY"]), 3);
        Geometry { window_x, window_y, valid, pixel, low_pixel, wy, wr0, wr1 }
    };

    let p_wait_on = p.setp("ne.u64", &pm["pWait"], 0);
    let p_signal_on = p.setp("ne.u64", &pm["pSignal"], 0);
    let wait_window = |gm: &Geometry, guard: Option<&str>| {
        let mut g2 = p_wait_on.clone();
        if let Some(guard) = guard {
            g2 = p.reg("pred");
            p.emit(format!("and.pred {g2}, {p_wait_on}, {guard};"));
        }
        sync_wait(&p, &pm["pWait"], &gm.wr0, &gm.wr1, Expected::Reg(&pm["waitExpected"]), &lane, Some(&g2), Some(&warp), &pm["pError"]);
    };

    // produced: written by a chained (barrier-free) launch -> L2 path; else the non-coherent (texture) path.
    let load_v4 = |base: &str, byte_off: &str, pred: &str, produced: bool| -> Vec<String> {
        let r = p.regs("b32", 4);
        for q in &r {
            p.emit(format!("mov.b32 {q}, 0;"));
        }
        let addr = p.add64(base, p.widen(byte_off));
        let hint = if produced { "cg" } else { "nc" };
        p.emit(format!("@{pred} ld.global.{hint}.v4.b32 {{{}, {}, {}, {}}}, [{addr}];", r[0], r[1], r[2], r[3]));
        r
    };

    let issue_loads = |gm: &Geometry, pred: &str| -> Vec<String> {
        let pixel = &gm.pixel;
        if pre {
            // features [tokens][16] f32: 64 B per token, this lane's 8 floats at part * 32
            let off = p.add32(p.shl32(pixel, 6), p.shl32(&ld_part, 5));
            let mut a = load_v4(&pm["pState"], &off, pred, false);
            let off = p.add32(p.add32(p.shl32(pixel, 6), p.shl32(&ld_part, 5)), p.imm32(16));
            let b = load_v4(&pm["pState"], &off, pred, false);
            a.extend(b);
            return a;
        }
        // the state is produced by the previous chained block except for the post block (block 0's output) and the
        // upres block (the skip from the encoder); the low-resolution input of the post block is chained
        let off = p.add32(p.shl32(pixel, 5), p.shl32(&ld_part, 4));
        let mut regs = load_v4(&pm["pState"], &off, pred, !(post || upres));
        let low_pixel = gm.low_pixel.as_ref();
        if post {
            let off = p.add32(p.shl32(low_pixel.unwrap(), 5), p.shl32(&ld_part, 4));
            regs.extend(load_v4(&pm["pLow"], &off, pred, true));
        } else if upres {
            let off = p.add32(p.shl32(low_pixel.unwrap(), 6), p.shl32(&ld_part, 5));
            regs.extend(load_v4(&pm["pLow"], &off, pred, false));
            let off = p.add32(p.add32(p.shl32(low_pixel.unwrap(), 6), p.shl32(&ld_part, 5)), p.imm32(16));
            regs.extend(load_v4(&pm["pLow"], &off, pred, false));
        }
        regs
    };

    let window = p.reg("b32");
    p.emit(format!("mov.u32 {window}, {ctaid};"));
    let p_w = p.setp("lt.u32", &window, &pm["windowCount"]);
    let end_l = p.label("END");
    let loop_l = p.label("WINDOW");
    p.emit(format!("@!{p_w} bra {end_l};"));
    let gm0 = geometry(&window);
    wait_window(&gm0, None);
    let first = issue_loads(&gm0, &gm0.valid);
    let carried = p.regs("b32", first.len());
    for (c, f) in carried.iter().zip(first.iter()) {
        p.emit(format!("mov.b32 {c}, {f};"));
    }
    p.emit(format!("{loop_l}:"));
    let gm = geometry(&window);
    let (window_x, window_y, valid, pixel) = (&gm.window_x, &gm.window_y, &gm.valid, &gm.pixel);
    let inputs = &carried;
    // prefetch the next window
    let next_window = p.add32(&window, &nctaid);
    let p_next = p.setp("lt.u32", &next_window, &pm["windowCount"]);
    let gm_n = geometry(&next_window);
    wait_window(&gm_n, Some(&p_next));
    let p_load_n = p.reg("pred");
    p.emit(format!("and.pred {p_load_n}, {}, {p_next};", gm_n.valid));
    let nxt = issue_loads(&gm_n, &p_load_n);

    let st_v4 = |addr: &str, r: &[String]| {
        p.emit(format!("st.shared.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
    };

    let warp_sync = || {
        p.emit("bar.warp.sync 0xffffffff;");
    };

    // four b32 words (16 E4 codes) -> eight .b16 code pairs in order.
    let e4_word_pairs = |words: &[String]| -> Vec<String> {
        let mut out = Vec::new();
        for w in words {
            let lo = p.reg("b16");
            let hi = p.reg("b16");
            p.emit(format!("mov.b32 {{{lo}, {hi}}}, {w};"));
            out.push(lo);
            out.push(hi);
        }
        out
    };

    // ---- Phase F inputs: x (E4 A fragment) and cInit (f16 C tiles = skip)
    let mut c_init: Vec<Vec<String>> = vec![vec![String::new(), String::new()]; 4];
    let xa: Vec<String>;
    if pre {
        let (f0, f1) = (&inputs[0..4], &inputs[4..8]);
        let mut hw = Vec::new();
        for (a, b) in [(&f0[0], &f0[1]), (&f0[2], &f0[3]), (&f1[0], &f1[1]), (&f1[2], &f1[3])] {
            let r = p.reg("b32");
            p.emit(format!("cvt.rn.f16x2.f32 {r}, {b}, {a};"));
            hw.push(r);
        }
        st_v4(&stg_ld_e4, &hw);
        warp_sync();
        let fa = ldm4(&p, &stg_a);
        let mut adapted: Vec<Vec<String>> = Vec::new();
        for i2 in 0..2 {
            let r = ldm4(&p, p.add32(&wf16_b, i2 * 512));
            for jj in 0..2 {
                let d = zero_tile(&p);
                mma_f16(&p, &d, &fa, if jj == 0 { &r[0..2] } else { &r[2..4] }, &d);
                adapted.push(d);
            }
        }
        for j in 0..4 {
            c_init[j] = vec![adapted[j][0].clone(), adapted[j][1].clone()];
        }
        xa = acc_to_a(&p, &adapted, &li, Some(&zero)); // adapter output: keep the exact NaN -> 0 (f32 features)
        warp_sync();
    } else if post || upres {
        let xw = &inputs[0..4];
        let sk = if post { &inputs[4..8] } else { &inputs[0..0] };
        let (sk0, sk1) = if post { (&inputs[0..0], &inputs[0..0]) } else { (&inputs[4..8], &inputs[8..12]) };
        let ad_pairs = e4_word_pairs(xw);
        let up_pairs = if post { e4_word_pairs(sk) } else { Vec::new() };
        let mut raw: Vec<String> = Vec::new();
        for i in 0..8 {
            let ad = nanzero(&p, e4x2_to_f16x2(&p, &ad_pairs[i]), &zero);
            if post {
                let up = nanzero(&p, e4x2_to_f16x2(&p, &up_pairs[i]), &zero);
                let input_value = hmul2(&p, &up, &blend_a[i]);
                raw.push(hfma2(&p, &ad, &blend_b[i], &input_value));
            } else {
                let projected = if i < 4 { &sk0[i] } else { &sk1[i - 4] };
                raw.push(hfma2(&p, &ad, &blend_a[i], projected));
            }
        }
        st_v4(&stg_ld_f16, &raw[0..4]);
        st_v4(&p.add32(&stg_ld_f16, 16), &raw[4..8]);
        warp_sync();
        for j in 0..4 {
            for h in 0..2 {
                let r = p.reg("b32");
                p.emit(format!("ld.shared.b32 {r}, [{c_base_f16}+{}];", 16 * j + 640 * h));
                c_init[j][h] = r;
            }
        }
        warp_sync(); // the E4 tile aliases the f16 tile
        let codes: Vec<String> = (0..8).map(|i| cvt_e4x2(&p, nanzero(&p, &raw[i], &zero))).collect();
        let cw: Vec<String> = (0..4).map(|i| pack16(&p, &codes[2 * i], &codes[2 * i + 1])).collect();
        st_v4(&stg_ld_e4, &cw);
        warp_sync();
        xa = ldm4(&p, &stg_a);
        warp_sync();
    } else {
        let xw = &inputs[0..4];
        st_v4(&stg_ld_e4, xw);
        warp_sync();
        xa = ldm4(&p, &stg_a);
        for j in 0..4 {
            for h in 0..2 {
                let hcode = p.reg("b16");
                p.emit(format!("ld.shared.b16 {hcode}, [{}+{}];", c_base[j >> 1], 8 * (j & 1) + 256 * h));
                c_init[j][h] = e4x2_to_f16x2(&p, &hcode);
            }
        }
        warp_sync();
    }
    // scaled skip seeds the W2 accumulators
    let ffn: Vec<Vec<String>> = (0..4).map(|j| (0..2).map(|h| hmul2(&p, &c_init[j][h], &aux_ffn[j])).collect()).collect();

    // ---- FFN: four quarters of 32 hidden units (W1 columns permuted), W2 chained in K order
    for qq in 0..4 {
        let mut hid: Vec<Vec<String>> = Vec::new();
        for i2 in 0..2 {
            let r = ldm4(&p, p.add32(&w1_b, (2 * qq + i2) * 512));
            for jj in 0..2 {
                let d = zero_tile(&p);
                mma_e4(&p, &d, &xa, if jj == 0 { &r[0..2] } else { &r[2..4] }, &d);
                hid.push(d);
            }
        }
        for d in hid.iter_mut() {
            d[0] = silu(&p, &d[0], &silu_k);
            d[1] = silu(&p, &d[1], &silu_k);
        }
        let q: Vec<Vec<String>> = hid.iter().map(|d| (0..2).map(|h| b16_to_b32(&p, cvt_e4x2(&p, &d[h]))).collect()).collect();
        let a2 = vec![
            prmt(&p, &q[0][0], &q[1][0], "0x5410"),
            prmt(&p, &q[0][1], &q[1][1], "0x5410"),
            prmt(&p, &q[2][0], &q[3][0], "0x5410"),
            prmt(&p, &q[2][1], &q[3][1], "0x5410"),
        ];
        for i2 in 0..2 {
            let r = ldm4(&p, p.add32(&w2_b, qq * 1024 + i2 * 512));
            for jj in 0..2 {
                let j = 2 * i2 + jj;
                mma_e4(&p, &ffn[j], &a2, if jj == 0 { &r[0..2] } else { &r[2..4] }, &ffn[j]);
            }
        }
    }
    let ffn_q = if A2A_SMEM != 0 { acc_to_a_smem(&p, &ffn, &c_base, &stg_a, None) } else { acc_to_a(&p, &ffn, &li, None) };

    // ---- q, k, v
    let mut qkv: Vec<Vec<Vec<String>>> = Vec::new();
    for m in 0..3 {
        let mut acc: Vec<Vec<String>> = Vec::new();
        for i2 in 0..2 {
            let r = ldm4(&p, p.add32(&wqkv_b, (2 * m + i2) * 512));
            for jj in 0..2 {
                let d = zero_tile(&p);
                mma_e4(&p, &d, &ffn_q, if jj == 0 { &r[0..2] } else { &r[2..4] }, &d);
                acc.push(d);
            }
        }
        qkv.push(acc);
    }
    let qn = normalize(&p, &qkv[0], &li, &zero, Some(&scale2), true);
    let kn = normalize(&p, &qkv[1], &li, &zero, None, true);
    let mq = if A2A_SMEM != 0 { acc_to_a_smem(&p, &qn, &c_base, &stg_a, None) } else { acc_to_a(&p, &qn, &li, None) };
    for j in 0..4 {
        for h in 0..2 {
            let hc = cvt_e4x2(&p, &kn[j][h]);
            p.emit(format!("st.shared.b16 [{}+{}], {hc};", k_pub[h], 8 * j));
            let vc = cvt_e4x2(&p, &qkv[2][j][h]);
            let vhi = p.reg("b16");
            p.emit(format!("shr.b16 {vhi}, {vc}, 8;"));
            p.emit(format!("st.shared.u8 [{}+{}], {vc};", v_pub[h], 640 * j));
            p.emit(format!("st.shared.u8 [{}+{}], {vhi};", v_pub[h], 640 * j + 80));
        }
    }
    p.emit("bar.sync 0;"); // K / V rows of all four warps visible

    // ---- attention: S = prior + Q K^T (physical key order), softmax, P V
    let mut s: Vec<Vec<String>> = Vec::new();
    for m in 0..8 {
        let c0 = p.reg("b32");
        p.emit(format!("mov.b32 {c0}, {};", prior_regs[m][0]));
        let c1 = p.reg("b32");
        p.emit(format!("mov.b32 {c1}, {};", prior_regs[m][1]));
        s.push(vec![c0, c1]);
    }
    for m2 in 0..4 {
        let r = ldm4(&p, p.add32(&k_lane, m2 * 768));
        mma_e4(&p, &s[2 * m2], &mq, &r[0..2], &s[2 * m2]);
        mma_e4(&p, &s[2 * m2 + 1], &mq, &r[2..4], &s[2 * m2 + 1]);
    }
    let w = softmax(&p, &s, &li, &exp_k);
    let pa = if A2A_SMEM != 0 {
        vec![acc_to_a_smem(&p, &w[0..4], &c_base, &stg_a, None), acc_to_a_smem(&p, &w[4..8], &c_base, &stg_a, None)]
    } else {
        vec![acc_to_a(&p, &w[0..4], &li, None), acc_to_a(&p, &w[4..8], &li, None)]
    };
    let o: Vec<Vec<String>> = (0..4).map(|_| zero_tile(&p)).collect();
    for step in 0..2 {
        for d2 in 0..2 {
            let r = ldm4(&p, p.add32(&v_lane, d2 * 1280 + step * 32));
            mma_e4(&p, &o[2 * d2], &pa[step], &r[0..2], &o[2 * d2]);
            mma_e4(&p, &o[2 * d2 + 1], &pa[step], &r[2..4], &o[2 * d2 + 1]);
        }
    }
    let attended = if A2A_SMEM != 0 { acc_to_a_smem(&p, &o, &c_base, &stg_a, None) } else { acc_to_a(&p, &o, &li, None) };
    let result: Vec<Vec<String>> = (0..4).map(|j| (0..2).map(|h| hmul2(&p, &ffn[j][h], &aux_attn[j])).collect()).collect();
    for i2 in 0..2 {
        let r = ldm4(&p, p.add32(&wproj_b, i2 * 512));
        for jj in 0..2 {
            let j = 2 * i2 + jj;
            mma_e4(&p, &result[j], &attended, if jj == 0 { &r[0..2] } else { &r[2..4] }, &result[j]);
        }
    }

    // ---- outputs
    let stg_out = p.add32(&stg_warp, p.add32(p.shl32(g, 5), p.shl32(t, 1))); // E4 output tile [16][32 B] (plain rows)
    let stg_read = p.add32(&stg_warp, p.add32(p.shl32(&ld_row, 5), p.shl32(&ld_part, 4)));
    if out_e4 {
        for j in 0..4 {
            for h in 0..2 {
                let hc = cvt_e4x2(&p, &result[j][h]);
                p.emit(format!("st.shared.b16 [{stg_out}+{}], {hc};", 8 * j + 256 * h));
            }
        }
        warp_sync();
        let r = p.regs("b32", 4);
        p.emit(format!("ld.shared.v4.b32 {{{}, {}, {}, {}}}, [{stg_read}];", r[0], r[1], r[2], r[3]));
        let addr = p.add64(&pm["pOutE4"], p.widen(p.add32(p.shl32(pixel, 5), p.shl32(&ld_part, 4))));
        p.emit(format!("@{valid} st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
        warp_sync();
    }
    if pool {
        // ds_fp8: window rows 2w, 2w+1 (rows g, g+8 of the tile) pool to half-resolution pixels (windowX + 2i, windowY + 2w):
        // ((a + b) + (c + d)) * 0.25 in halves; lanes with even g hold pixel i = g >> 1.
        let pool_stg = p.add32(&stg_warp, 512);
        let g_even = p.setp("eq.u32", p.and32(g, 1), 0);
        for j in 0..4 {
            let (a, c) = (&result[j][0], &result[j][1]);
            let b = shfl_xor(&p, a, 4);
            let d = shfl_xor(&p, c, 4);
            let top = hadd2(&p, a, &b);
            let bottom = hadd2(&p, c, &d);
            let sm = hadd2(&p, &top, &bottom);
            let v = hmul2(&p, &sm, &quarter2);
            let hc = cvt_e4x2(&p, &v);
            let addr = p.add32(&pool_stg, p.add32(p.shl32(p.shr32(g, 1), 5), p.add32(p.shl32(t, 1), p.imm32(8 * j))));
            p.emit(format!("@{g_even} st.shared.b16 [{addr}], {hc};"));
        }
        warp_sync();
        let p_lane8 = p.setp("lt.u32", &lane, 8);
        let i = p.shr32(&lane, 1);
        let part = p.and32(&lane, 1);
        let r = p.regs("b32", 4);
        let raddr = p.add32(&pool_stg, p.add32(p.shl32(&i, 5), p.shl32(&part, 4)));
        p.emit(format!("@{p_lane8} ld.shared.v4.b32 {{{}, {}, {}, {}}}, [{raddr}];", r[0], r[1], r[2], r[3]));
        let sx = p.add32(window_x, p.shl32(&i, 1));
        let sy = p.add32(window_y, p.shl32(&warp, 1));
        let sx1 = p.add32(&sx, 1);
        let sy1 = p.add32(&sy, 1);
        let pvx = p.setp("lt.u32", &sx1, &pm["width"]);
        let pvy = p.setp("lt.u32", &sy1, &pm["height"]);
        let pv = p.reg("pred");
        p.emit(format!("and.pred {pv}, {pvx}, {pvy};"));
        p.emit(format!("and.pred {pv}, {pv}, {p_lane8};"));
        let pnx = p.setp("ge.s32", &sx, 0);
        let pny = p.setp("ge.s32", &sy, 0);
        p.emit(format!("and.pred {pv}, {pv}, {pnx};"));
        p.emit(format!("and.pred {pv}, {pv}, {pny};"));
        let low = p.mad32(p.shr32(&sy, 1), &pm["lowWidth"], p.shr32(&sx, 1));
        let addr = p.add64(&pm["pOut2"], p.widen(p.add32(p.shl32(&low, 5), p.shl32(&part, 4))));
        p.emit(format!("@{pv} st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
        warp_sync();
    }
    if head {
        // RGBA head: two k16 steps of the f16 result x head weights [16 n][32 k] (n8 tile 0 holds columns 0..3)
        let hd = zero_tile(&p);
        let h_lane = p.add32(p.add32(&smem, WF16S), p.add32(p.shl32(&l8, 6), p.shl32(p.and32(&lm, 1), 4)));
        for step in 0..2 {
            let b = ldm2(&p, p.add32(&h_lane, step * 32));
            mma_f16(&p, &hd, &acc_f16_to_a(&result, step), &b, &hd);
        }
        let p_t2 = p.setp("lt.u32", t, 2);
        for h in 0..2 {
            let xo = p.add32(window_x, g);
            let yo = p.add32(window_y, p.add32(p.shl32(&warp, 1), p.imm32(h)));
            let pvx = p.setp("lt.u32", &xo, &pm["width"]);
            let pvy = p.setp("lt.u32", &yo, &pm["height"]);
            let pv = p.reg("pred");
            p.emit(format!("and.pred {pv}, {pvx}, {pvy};"));
            p.emit(format!("and.pred {pv}, {pv}, {p_t2};"));
            let (lo, hi) = unpack16(&p, &hd[h]);
            let f0 = p.reg("f32");
            p.emit(format!("cvt.f32.f16 {f0}, {lo};"));
            let f1 = p.reg("f32");
            p.emit(format!("cvt.f32.f16 {f1}, {hi};"));
            let px = p.mad32(&yo, &pm["width"], &xo);
            let addr = p.add64(&pm["pOut2"], p.widen(p.add32(p.shl32(&px, 4), p.shl32(t, 3))));
            p.emit(format!("@{pv} st.global.v2.f32 [{addr}], {{{f0}, {f1}}};"));
        }
    }
    p.emit("fence.acq_rel.gpu;");
    p.emit("bar.sync 0;"); // K / V and staging are reused by the next window; the window's stores are released
    let p_sig = p.reg("pred");
    let p_t0 = p.setp("eq.u32", &tid, 0);
    p.emit(format!("and.pred {p_sig}, {p_signal_on}, {p_t0};"));
    let sig_addr = p.add64(&pm["pSignal"], p.widen(p.shl32(&gm.wy, 2)));
    p.emit(format!("@{p_sig} red.release.gpu.global.add.u32 [{sig_addr}], 1;"));
    for (c, n) in carried.iter().zip(nxt.iter()) {
        p.emit(format!("mov.b32 {c}, {n};"));
    }
    p.emit(format!("mov.u32 {window}, {next_window};"));
    p.emit(format!("@{p_next} bra {loop_l};"));
    p.emit(format!("{end_l}:"));
    p.emit("ret;");
    (name, p.finish())
}

/// `<flags> [maxRegs]`
pub fn cli(args: &[&str]) -> String {
    let flags = crate::req(args, 0);
    let max_regs = if args.len() > 1 { Some(crate::reg(args[1])) } else { None };
    generate(flags, max_regs).1
}
