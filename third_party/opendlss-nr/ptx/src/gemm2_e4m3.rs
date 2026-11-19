//! Exact FP8 GEMM with the block epilogues in PTX (the gemm_fp8.comp shapes of the expert stages).
//!
//! D[rows][N] = A[rows][K] (E4, at inputColumnBase of inputStride) x W (E4, k32-tile-major [K/32][Nmatrix][32] at
//! weightColumnOffset), f16 accumulation chained over K in order (mma.sync m16n8k32, two-group F13 arithmetic).
//! Flags:
//!   RES   (seed): the accumulator starts from round_f16(E4 residual * f16 scale) - residual at the output location,
//!         scale = aux halves at auxHalfOffset + column. The scaled residual seeds the C fragments rather than
//!         being added in the epilogue: one rounding instead of two.
//!   SILU  : cubic SiLU on the f16 result.
//!   E4    : E4M3 output (NaN -> 0); F16: f16 output (both: dual).
//! Workgroup: 4 warps x 16 rows = 64 rows, 64 columns (8 n8 tiles per warp), cp.async ring of A [64][32] + W [64][32]
//! per k32 step, ldmatrix fragments, grid (N / 64, ceil(rows / 64)).
//!
//! Arguments: `<K> <flags> [maxregs] [stages] [ksub] [ws]`     flags: RES=1 SILU=2 E4=4 F16=8

use std::collections::HashMap;

use crate::ptxgen::Ptx;
use crate::swin::*;

pub const F_RES: usize = 1;
pub const F_SILU: usize = 2;
pub const F_E4: usize = 4;
pub const F_F16: usize = 8;
pub const THREADS: usize = 128;
pub const BM: usize = 64;
pub const BN: usize = 64;
pub const STAGES: usize = 3;
pub const STAGE_BYTES: usize = (BM + BN) * 32;

/// ws: warp-specialized copies - warp 4 issues every cp.async (the LSU accepts ~16 B/cycle per SM and a
/// cp.async issue blocks the issuing warp), warps 0..3 only run ldmatrix + MMA; the stages hand over through
/// named barriers FULL(slot) / EMPTY(slot). The launch has 160 threads (the PTX carries `// threads 160`).
pub fn generate(k: usize, flags: usize, max_regs: Option<u32>, stages_wanted: usize, ksub: usize, ws: bool) -> (String, String) {
    assert!(k.is_multiple_of(32 * ksub));
    let steps = k / (32 * ksub); // pipeline stages of ksub k32 tiles each
    let stages = stages_wanted.min(steps);
    assert!(!ws || stages >= 2);
    let stage_bytes = ksub * (BM + BN) * 32;
    let (res, silu_f, out_e4, out_f16) = (flags & F_RES != 0, flags & F_SILU != 0, flags & F_E4 != 0, flags & F_F16 != 0);
    assert!(out_e4 || out_f16);
    let name = format!("gemm2_e4m3_K{k}_f{flags}") + &if ksub != 1 { format!("_s{ksub}") } else { String::new() };
    let mut ress = stages * stage_bytes; // residual E4 tile [64][64 B] (4 KB), then the E4 output tile
    if out_f16 && (out_e4 || res) {
        ress = ress.max(BM * 128); // past the f16 tile [64][128 B] the ring becomes (a one-stage ring at K=32 is smaller)
    }
    let shared_bytes = ress + 4096;
    let nt = if ws { THREADS + 32 } else { THREADS }; // + the copy warp
    let p = Ptx::new();
    if ws {
        p.header_line(format!("// threads {nt}"));
    }
    let params = [
        ("u64", "pA"),
        ("u64", "pW"),
        ("u64", "pRes"),
        ("u64", "pAux"),
        ("u64", "pOut"),
        ("u64", "pOut16"),
        ("u32", "rows"),
        ("u32", "inputStride"),
        ("u32", "inputColumnBase"),
        ("u32", "Nmatrix"),
        ("u32", "weightColumnOffset"),
        ("u32", "outputStride"),
        ("u32", "outputColumnOffset"),
        ("u32", "auxHalfOffset"),
        ("u64", "pWaitRows"),
        ("u32", "waitExpected"),
        ("u32", "waitShiftY"),
        ("u64", "pSignal"),
        ("u32", "width"),
        ("u64", "pWaitBands"),
        ("u32", "waitMul"),
        ("u32", "waitGroupRows"),
        ("u64", "pError"),
    ]; // chaining (0 = off)
    p.entry(&name, &params, shared_bytes, Some(nt as u32), max_regs, false);
    let pm: HashMap<&str, String> = params.iter().map(|&(t, n)| (n, if t == "u64" { p.load_param_u64(n) } else { p.load_param_u32(n) })).collect();
    let tid = p.special("tid.x");
    let col_group = p.special("ctaid.x");
    let row_group = p.special("ctaid.y");
    let barc = "bar.sync 0, 128;"; // the four MMA warps only
    let lane = p.and32(&tid, 31);
    let warp = p.shr32(&tid, 5);
    let li = LaneInfo::new(&p, &lane);
    let (g, t) = (&li.g, &li.t);
    let smem = p.shared_addr(p.imm32(0));
    let zero = imm(&p, 0);
    let block_row = p.shl32(&row_group, 6);
    let col_base = p.shl32(&col_group, 6); // output column of this workgroup's 64 columns
    let rows_minus1 = p.sub32(&pm["rows"], 1);
    // ---- chaining: wait for the window rows covering this workgroup's rows (the attention output), signal row bands
    let row_last = p.add32(&block_row, p.imm32(BM - 1));
    let p_last_big = p.setp("gt.u32", &row_last, &rows_minus1);
    let row_last = p.selp32(&p_last_big, &rows_minus1, &row_last);
    let y_first = p.reg("b32");
    p.emit(format!("div.u32 {y_first}, {block_row}, {};", pm["width"]));
    let y_last = p.reg("b32");
    p.emit(format!("div.u32 {y_last}, {row_last}, {};", pm["width"]));
    let p_wait_rows_on = p.setp("ne.u64", &pm["pWaitRows"], 0);
    let wy0 = p.shr32(p.add32(&y_first, &pm["waitShiftY"]), 3);
    let wy1 = p.shr32(p.add32(&y_last, &pm["waitShiftY"]), 3);
    sync_wait(&p, &pm["pWaitRows"], &wy0, &wy1, Expected::Reg(&pm["waitExpected"]), &lane, Some(&p_wait_rows_on), Some(&warp), &pm["pError"]);
    let band_first = p.shr32(&y_first, 3);
    let band_last = p.shr32(&y_last, 3);
    let p_wait_bands_on = p.setp("ne.u64", &pm["pWaitBands"], 0);
    sync_wait(
        &p,
        &pm["pWaitBands"],
        &band_first,
        &band_last,
        Expected::Per(&|b: &str| band_expected(&p, b, &pm["width"], &pm["rows"], Some(pm["waitMul"].as_str()), Some(pm["waitGroupRows"].as_str()))),
        &lane,
        Some(&p_wait_bands_on),
        Some(&warp),
        &pm["pError"],
    );

    // ---- cp.async chunks: 256 per stage (A rows 0..63 x 2 chunks, then W rows) = 2 per thread (8 per lane of the
    // copy warp); chunk ^= (row >> 2) & 1
    let mut copies: Vec<(String, String, String)> = Vec::new();
    let copy_lane = p.and32(&tid, 31);
    for i in 0..(if ws { 8 } else { 2 }) {
        let c = if ws { p.add32(&copy_lane, i * 32) } else { p.add32(&tid, i * THREADS) };
        let is_a = p.setp("lt.u32", &c, BM * 2);
        let row = p.shr32(&c, 1);
        let chunk = p.and32(&c, 1);
        let pch = p.xor32(&chunk, p.and32(p.shr32(&row, 2), 1));
        let dst_a = p.add32(p.shl32(&row, 5), p.shl32(&pch, 4));
        let g_row = p.add32(&block_row, &row);
        let g_row_c = p.selp32(p.setp("lt.u32", &g_row, &pm["rows"]), &g_row, &rows_minus1);
        let a_off = p.add32(p.mul32(&g_row_c, &pm["inputStride"]), p.add32(&pm["inputColumnBase"], p.shl32(&chunk, 4)));
        let src_a = p.add64(&pm["pA"], p.widen(&a_off));
        let c_b = p.sub32(&c, BM * 2);
        let n = p.shr32(&c_b, 1);
        let chunk_b = p.and32(&c_b, 1);
        let pch_b = p.xor32(&chunk_b, p.and32(p.shr32(&n, 2), 1));
        let dst_b = p.add32(BM * 32, p.add32(p.shl32(&n, 5), p.shl32(&pch_b, 4)));
        let w_row = p.add32(&pm["weightColumnOffset"], p.add32(&col_base, &n));
        let src_b = p.add64(&pm["pW"], p.widen(p.add32(p.shl32(&w_row, 5), p.shl32(&chunk_b, 4))));
        let dst = p.selp32(&is_a, &dst_a, &dst_b);
        let src = p.reg("b64");
        p.emit(format!("selp.b64 {src}, {src_a}, {src_b}, {is_a};"));
        let adv = p.selp32(&is_a, p.imm32(32), p.shl32(&pm["Nmatrix"], 5)); // per k32 step: A +32 B, W +Nmatrix*32 B
        copies.push((dst, src, adv));
    }

    let issue_stage = |stage_index: usize, step: usize| {
        for sub in 0..ksub {
            for (dst, src, adv) in &copies {
                let saddr = p.add32(p.add32(&smem, dst), p.imm32(stage_index * stage_bytes + sub * (BM + BN) * 32));
                let gaddr = p.mad_wide(adv, p.imm32(step * ksub + sub), src);
                p.emit(format!("cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;"));
            }
        }
        p.emit("cp.async.commit_group;");
    };

    let l8 = p.and32(&lane, 7);
    let lm = p.shr32(&lane, 3);
    let a_row_l = p.add32(&l8, p.shl32(p.and32(&lm, 1), 3));
    let a_chunk_l = p.shr32(&lane, 4);
    let a_pch = p.xor32(&a_chunk_l, p.and32(p.shr32(&a_row_l, 2), 1));
    let a_lane = p.add32(&smem, p.add32(p.shl32(p.add32(&a_row_l, p.shl32(&warp, 4)), 5), p.shl32(&a_pch, 4)));
    let b_n_l = p.add32(&l8, p.shl32(p.shr32(&lane, 4), 3));
    let b_chunk_l = p.and32(&lm, 1);
    let b_pch = p.xor32(&b_chunk_l, p.and32(p.shr32(&b_n_l, 2), 1));
    let b_lane = p.add32(&smem, p.add32(BM * 32, p.add32(p.shl32(&b_n_l, 5), p.shl32(&b_pch, 4))));

    // ---- output rows of this warp: g and g + 8
    let out_row: Vec<String> = (0..2).map(|h| p.add32(&block_row, p.add32(p.shl32(&warp, 4), p.add32(g, p.imm32(8 * h))))).collect();
    let _row_ok: Vec<String> = (0..2).map(|h| p.setp("lt.u32", &out_row[h], &pm["rows"])).collect();
    let out_col = p.add32(&pm["outputColumnOffset"], p.add32(&col_base, p.shl32(t, 1))); // + 8 j
    let _out_idx: Vec<String> = (0..2).map(|h| p.add32(p.mul32(&out_row[h], &pm["outputStride"]), &out_col)).collect(); // element index

    // tile row / chunk of this thread for the 64 x 64 B residual / output tiles: 256 chunks, 2 per thread (the copy
    // warp: 8 per lane for the residual)
    let tile_chunks = |count: usize, base: &str| -> Vec<(String, String, String)> {
        let mut out = Vec::new();
        for i in 0..count {
            let c = p.add32(base, i * (256 / count));
            let row = p.shr32(&c, 2);
            let chunk = p.and32(&c, 3);
            let g_row = p.add32(&block_row, &row);
            let ok = p.setp("lt.u32", &g_row, &pm["rows"]);
            let g_off = p.add32(p.mul32(&g_row, &pm["outputStride"]), p.add32(&pm["outputColumnOffset"], p.add32(&col_base, p.shl32(&chunk, 4))));
            out.push((p.add32(p.shl32(&row, 6), p.shl32(&chunk, 4)), g_off, ok));
        }
        out
    };
    let tile_chunks_v = tile_chunks(2, &tid);
    let res_chunks = if ws { tile_chunks(8, &copy_lane) } else { tile_chunks_v.clone() };

    let issue_residual = || {
        // residual tile through shared memory (16-byte copies), then the C-layout pairs
        for (s_off, g_off, ok) in &res_chunks {
            let saddr = p.add32(p.add32(&smem, s_off), p.imm32(ress));
            let gaddr = p.add64(&pm["pRes"], p.widen(g_off));
            p.emit(format!("@{ok} cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;"));
        }
    };

    let mut acc: Vec<Vec<String>> = Vec::new();
    let mut scales: Vec<String> = Vec::new();
    if res {
        let aux_addr = p.add64(&pm["pAux"], p.widen(p.shl32(p.add32(&pm["auxHalfOffset"], p.add32(&col_base, p.shl32(t, 1))), 1)));
        for j in 0..8 {
            let r = p.reg("b32");
            p.emit(format!("ld.global.nc.b32 {r}, [{aux_addr}+{}];", 16 * j));
            scales.push(r);
        }
    }

    let seed_from_residual = || -> Vec<Vec<String>> {
        let mut acc = Vec::new();
        let c_res = p.add32(p.add32(&smem, ress), p.add32(p.shl32(p.add32(p.shl32(&warp, 4), g), 6), p.shl32(t, 1)));
        for j in 0..8 {
            let mut d = Vec::new();
            for h in 0..2 {
                let hc = p.reg("b16");
                p.emit(format!("ld.shared.b16 {hc}, [{c_res}+{}];", 8 * j + 512 * h));
                d.push(hmul2(&p, e4x2_to_f16x2(&p, &hc), &scales[j]));
            }
            acc.push(d);
        }
        acc
    };

    if ws {
        // ---- the copy warp: residual + stage 0 as group 0, stages 1 .. S-2, then one stage per step; FULL(slot)
        // once the group landed, EMPTY(slot) before a slot is refilled. Barrier ids: FULL 1.., EMPTY 1 + stages ..
        let full = |slot: usize| 1 + slot;
        let empty = |slot: usize| 1 + stages + slot;
        let p_producer = p.setp("ge.u32", &tid, 128);
        let prod_l = p.label("PRODUCER");
        let epi_l = p.label("EPILOGUE");
        p.emit(format!("@{p_producer} bra {prod_l};"));
        // ---- the MMA warps
        if !res {
            acc = (0..8).map(|_| zero_tile(&p)).collect();
        }
        for step in 0..steps {
            let stage_index = step % stages;
            p.emit(format!("bar.sync {}, {nt};", full(stage_index)));
            if step == 0 && res {
                acc = seed_from_residual();
            }
            for sub in 0..ksub {
                let sub_off = stage_index * stage_bytes + sub * (BM + BN) * 32;
                let a = ldm4(&p, p.add32(&a_lane, sub_off));
                let rr: Vec<Vec<String>> = (0..4).map(|i2| ldm4(&p, p.add32(&b_lane, sub_off + i2 * 512))).collect();
                for i2 in 0..4 {
                    let r = &rr[i2];
                    mma_e4(&p, &acc[2 * i2], &a, &r[0..2], &acc[2 * i2]);
                    mma_e4(&p, &acc[2 * i2 + 1], &a, &r[2..4], &acc[2 * i2 + 1]);
                }
            }
            if step + stages < steps {
                // the slot is refilled with stage step + stages
                p.emit(format!("bar.arrive {}, {nt};", empty(stage_index)));
            }
        }
        p.emit(format!("bra {epi_l};"));
        // ---- the copy warp
        p.emit(format!("{prod_l}:"));
        if res {
            issue_residual();
        }
        issue_stage(0, 0); // commits group 0
        for s in 1..stages.saturating_sub(1) {
            issue_stage(s, s);
        }
        for step in 0..steps {
            let j = step + stages - 1;
            if j < steps {
                if j >= stages {
                    p.emit(format!("bar.sync {}, {nt};", empty(j % stages)));
                }
                issue_stage(j % stages, j);
            }
            let committed = j.min(steps - 1) + 1; // groups 0 .. min(j, steps - 1)
            p.emit(format!("cp.async.wait_group {};", committed - 1 - step)); // group `step` landed
            p.emit(format!("bar.arrive {}, {nt};", full(step % stages)));
        }
        p.emit("ret;");
        p.emit(format!("{epi_l}:"));
    } else {
        if res {
            issue_residual();
            p.emit("cp.async.commit_group;");
        }
        for s in 0..(stages - 1).max(1) {
            // stages == 1 (K = 32): the only step is issued here
            issue_stage(s, s);
        }
        // accumulator seed: round_f16(E4 residual * scale) at the output location
        if res {
            p.emit(format!("cp.async.wait_group {};", stages - 1)); // the residual group (the oldest) has landed
            p.emit("bar.sync 0;");
            acc = seed_from_residual();
        } else {
            acc = (0..8).map(|_| zero_tile(&p)).collect();
        }
    }

    for step in 0..(if !ws { steps } else { 0 }) {
        let stage_index = step % stages;
        p.emit(format!("cp.async.wait_group {};", stages.saturating_sub(2)));
        p.emit("bar.sync 0;");
        if stages > 1 && step + stages - 1 < steps {
            issue_stage((step + stages - 1) % stages, step + stages - 1);
        } else if stages > 1 {
            p.emit("cp.async.commit_group;");
        }
        for sub in 0..ksub {
            let sub_off = stage_index * stage_bytes + sub * (BM + BN) * 32;
            let a = ldm4(&p, p.add32(&a_lane, sub_off));
            let rr: Vec<Vec<String>> = (0..4).map(|i2| ldm4(&p, p.add32(&b_lane, sub_off + i2 * 512))).collect();
            for i2 in 0..4 {
                let r = &rr[i2];
                mma_e4(&p, &acc[2 * i2], &a, &r[0..2], &acc[2 * i2]);
                mma_e4(&p, &acc[2 * i2 + 1], &a, &r[2..4], &acc[2 * i2 + 1]);
            }
        }
    }
    // ---- epilogue
    if silu_f {
        let k = SiluConsts::new(&p);
        for j in 0..8 {
            for h in 0..2 {
                acc[j][h] = silu(&p, &acc[j][h], &k);
            }
        }
    }
    p.emit(barc); // every MMA warp finished the last stage (the ring becomes the f16 output tile)
    let c_out = p.add32(p.add32(&smem, ress), p.add32(p.shl32(p.add32(p.shl32(&warp, 4), g), 6), p.shl32(t, 1)));
    if out_f16 {
        // f16 tile [64][128 B] in the ring area, then 16-byte stores (512 chunks, 4 per thread)
        let c_out16 = p.add32(&smem, p.add32(p.shl32(p.add32(p.shl32(&warp, 4), g), 7), p.shl32(t, 2)));
        for j in 0..8 {
            for h in 0..2 {
                p.emit(format!("st.shared.b32 [{c_out16}+{}], {};", 16 * j + 1024 * h, acc[j][h]));
            }
        }
    }
    if out_e4 {
        for j in 0..8 {
            for h in 0..2 {
                let hc = cvt_e4x2(&p, nanzero(&p, &acc[j][h], &zero));
                p.emit(format!("st.shared.b16 [{c_out}+{}], {hc};", 8 * j + 512 * h));
            }
        }
    }
    p.emit(barc);
    if out_f16 {
        for i in 0..4 {
            let c = p.add32(&tid, i * THREADS);
            let row = p.shr32(&c, 3);
            let chunk = p.and32(&c, 7);
            let g_row = p.add32(&block_row, &row);
            let ok = p.setp("lt.u32", &g_row, &pm["rows"]);
            let g_off = p.add32(p.mul32(&g_row, &pm["outputStride"]), p.add32(&pm["outputColumnOffset"], p.add32(&col_base, p.shl32(&chunk, 3))));
            let r = p.regs("b32", 4);
            let la = p.add32(&smem, p.add32(p.shl32(&row, 7), p.shl32(&chunk, 4)));
            p.emit(format!("ld.shared.v4.b32 {{{}, {}, {}, {}}}, [{la}];", r[0], r[1], r[2], r[3]));
            let addr = p.add64(&pm["pOut16"], p.widen(p.shl32(&g_off, 1)));
            p.emit(format!("@{ok} st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
        }
    }
    if out_e4 {
        for (s_off, g_off, ok) in &tile_chunks_v {
            let r = p.regs("b32", 4);
            let la = p.add32(p.add32(&smem, s_off), p.imm32(ress));
            p.emit(format!("ld.shared.v4.b32 {{{}, {}, {}, {}}}, [{la}];", r[0], r[1], r[2], r[3]));
            let addr = p.add64(&pm["pOut"], p.widen(g_off));
            p.emit(format!("@{ok} st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
        }
    }
    let p_signal_on = p.setp("ne.u64", &pm["pSignal"], 0);
    p.emit("fence.acq_rel.gpu;");
    p.emit(barc);
    let p_t0 = p.reg("pred");
    let p_eq = p.setp("eq.u32", &tid, 0);
    p.emit(format!("and.pred {p_t0}, {p_signal_on}, {p_eq};"));
    let b_iter = p.reg("b32");
    p.emit(format!("mov.u32 {b_iter}, {band_first};"));
    let sig_loop = p.label("SIG");
    let sig_done = p.label("SIGDONE");
    p.emit(format!("@!{p_t0} bra {sig_done};"));
    p.emit(format!("{sig_loop}:"));
    let sig_addr = p.add64(&pm["pSignal"], p.widen(p.shl32(&b_iter, 2)));
    p.emit(format!("red.release.gpu.global.add.u32 [{sig_addr}], 1;"));
    p.emit(format!("add.u32 {b_iter}, {b_iter}, 1;"));
    let p_more_b = p.setp("le.u32", &b_iter, &band_last);
    p.emit(format!("@{p_more_b} bra {sig_loop};"));
    p.emit(format!("{sig_done}:"));
    p.emit("ret;");
    (name, p.finish())
}

/// `<K> <flags> [maxregs] [stages] [ksub] [ws]`
pub fn cli(args: &[&str]) -> String {
    let (k, flags) = (crate::req(args, 0), crate::req(args, 1));
    let max_regs = if args.len() > 2 { Some(crate::reg(args[2])) } else { None };
    let stages_wanted = if args.len() > 3 { crate::req(args, 3) } else { STAGES };
    let ksub = if args.len() > 4 { crate::req(args, 4) } else { 1 };
    let ws = args.len() > 5 && crate::req(args, 5) != 0;
    generate(k, flags, max_regs, stages_wanted, ksub, ws).1
}
