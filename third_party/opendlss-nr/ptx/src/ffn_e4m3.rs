//! Expert FFN + W3 of one block in a single PTX kernel: E4( E4(SiLU(x W1_e)) W2_e ) for every expert e of a row
//! group, then E4( skip * scale + narrow W3 ) with the narrow activations staged in shared memory.
//!
//! Workgroup: E experts x ROWT row tiles of 16 = 4E * ROWT / 4 ... warps = E * ROWT (warp w: expert / column group
//! e = w / ROWT, row tile q = w % ROWT), rows per workgroup = 16 ROWT, grid = ceil(rows / (16 ROWT)).
//!   phase 1: hidden_e = x W1_e over K = C in k32 steps; every expert's W1 tile of the step sits in one ring stage
//!            (E x 4 KB, chunk-swizzled rows), A fragments straight from the state rows (2 steps in flight).
//!   phase 2: SiLU, E4, W2_e (the W2 tiles land in the free ring stage during the last W1 step; the hidden C fragment
//!            packs into the W2 A fragment through the permuted W1 columns), narrow tile [16 ROWT][C + 16] E4 in shared.
//!   phase 3: out[rows][32 e ..] = E4( skip * aux + narrow W3[:, 32 e ..] ), W3 tiles ring-staged per k32 step, the
//!            narrow rows read with ldmatrix (padded stride: conflict-free), output through the narrow region.
//! Arithmetic identical to mlp_e4m3 + gemm2_e4m3 (verified bit-exact). Dynamic shared memory (up to ~83 KB).
//!
//! Arguments: `<C> <ROWT> [maxregs] [proj]`

use std::collections::HashMap;

use crate::ptxgen::Ptx;
use crate::swin::*;

pub const HIDDEN: usize = 128;

/// The kernel's name, PTX and shared memory size.
/// proj: the previous block's projection runs first (phase 0): state = E4(ffnPrev * auxAttn + attended Wproj),
/// kept in shared memory as the FFN input and skip (the state never touches global memory).
pub fn generate(chan: usize, rowt: usize, max_regs: Option<u32>, proj: bool) -> (String, String, usize) {
    let experts = chan / 32;
    let steps = chan / 32;
    let warps = experts * rowt;
    let threads = 32 * warps;
    let rows_per_wg = 16 * rowt;
    let stage_bytes = experts * 4096 + 2048; // W1 tiles of one k32 step for all experts ([e][128 n][32 B]) + the A tile [64][32 B]
    let a_off = experts * 4096; // A tile offset within a W1 stage (chunk-swizzled rows, L2 copies)
    let ring = 0;
    let narrow = 2 * stage_bytes;
    let narrow_stride = chan + 16;
    let shared_bytes = narrow + rows_per_wg * narrow_stride;
    let w3_stage_bytes = chan * 32; // W3 tiles of one k32 step ([C n][32 B])
    let wp_stage_bytes = chan * 32 + 2048; // Wproj tiles + the attended A tile (phase 0 ring inside W1 stage 0)
    assert!(2 * wp_stage_bytes <= stage_bytes);
    let name = format!("ffn_e4m3_C{chan}_R{rowt}") + if proj { "_proj" } else { "" };
    let p = Ptx::new();
    let params = [
        ("u64", "pA"),
        ("u64", "pW1"),
        ("u64", "pW2"),
        ("u64", "pW3"),
        ("u64", "pAux"),
        ("u64", "pOut"),
        ("u32", "rows"),
        ("u32", "auxHalfOffset"),
        ("u64", "pAtt"),
        ("u64", "pFfnPrev"),
        ("u64", "pWproj"),
        ("u32", "auxAttnHalf"),
        ("u64", "pAuxPrev"),
        ("u64", "pStateOut"),
        ("u32", "storeState"),
        ("u64", "pWaitRows"),
        ("u32", "waitExpected"),
        ("u32", "waitShiftY"),
        ("u64", "pWaitBands"),
        ("u64", "pSignal"),
        ("u32", "width"),
        ("u32", "waitMul"),
        ("u32", "waitGroupRows"),
        ("u64", "pError"),
    ]; // chaining (0 = off): wait on the previous block's window rows / state row bands (waitMul signals per row group of waitGroupRows rows), signal this block's FFN row bands
    p.entry(&name, &params, shared_bytes, Some(threads as u32), max_regs, true);
    let pm: HashMap<&str, String> = params.iter().map(|&(t, n)| (n, if t == "u64" { p.load_param_u64(n) } else { p.load_param_u32(n) })).collect();
    let tid = p.special("tid.x");
    let ctaid = p.special("ctaid.x");
    let lane = p.and32(&tid, 31);
    let warp = p.shr32(&tid, 5);
    let li = LaneInfo::new(&p, &lane);
    let (g, t) = (&li.g, &li.t);
    let smem = p.shared_addr(p.imm32(0));
    imm(&p, 0);
    let e = p.reg("b32");
    let q = p.reg("b32");
    if rowt & (rowt - 1) == 0 {
        p.emit(format!("shr.u32 {e}, {warp}, {};", rowt.ilog2()));
        p.emit(format!("and.b32 {q}, {warp}, {};", rowt - 1));
    } else {
        p.emit(format!("div.u32 {e}, {warp}, {rowt};"));
        p.emit(format!("rem.u32 {q}, {warp}, {rowt};"));
    }
    let block_row = p.mul32(&ctaid, rows_per_wg);
    let rows_minus1 = p.sub32(&pm["rows"], 1);
    // this warp's rows g, g + 8 (clamped for the loads, never stored past the end)
    let mut row_ok: Vec<String> = Vec::new();
    let mut a_row: Vec<String> = Vec::new();
    let mut row_idx: Vec<String> = Vec::new();
    for h in 0..2 {
        let r = p.add32(&block_row, p.add32(p.shl32(&q, 4), p.add32(g, p.imm32(8 * h))));
        row_idx.push(r.clone());
        row_ok.push(p.setp("lt.u32", &r, &pm["rows"]));
        let rc = p.selp32(&row_ok[h], &r, &rows_minus1);
        a_row.push(p.add64(&pm["pA"], p.widen(p.add32(p.mul32(&rc, chan), p.shl32(t, 2)))));
    }
    let stg_addr: Vec<String> = (0..2).map(|s| p.add32(&smem, ring + s * stage_bytes)).collect();
    // ---- chaining: wait for the producers of this workgroup's rows
    let row_last = p.add32(&block_row, p.imm32(rows_per_wg - 1));
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
    let p_wait_bands_on = p.setp("ne.u64", &pm["pWaitBands"], 0);
    let b0 = p.shr32(&y_first, 3);
    let b1 = p.shr32(&y_last, 3);
    sync_wait(
        &p,
        &pm["pWaitBands"],
        &b0,
        &b1,
        Expected::Per(&|b: &str| band_expected(&p, b, &pm["width"], &pm["rows"], Some(&pm["waitMul"]), Some(&pm["waitGroupRows"]))),
        &lane,
        Some(&p_wait_bands_on),
        Some(&warp),
        &pm["pError"],
    );

    // ---- W1 ring copies: E x 256 chunks per stage, 2 per thread: c = tid + threads i -> expert c >> 8, row (c & 255) >> 1, half c & 1
    let mut w1_copies: Vec<(String, String)> = Vec::new();
    let copies_per_thread = (256 * experts).div_ceil(threads);
    let copy_exact = copies_per_thread * threads == 256 * experts;
    let mut w1_preds: Vec<Option<String>> = Vec::new();
    for i in 0..copies_per_thread {
        let c = p.add32(&tid, i * threads);
        w1_preds.push(if copy_exact { None } else { Some(p.setp("lt.u32", &c, 256 * experts)) });
        let ce = p.shr32(&c, 8);
        let cc = p.and32(&c, 255);
        let n = p.shr32(&cc, 1);
        let half = p.and32(&cc, 1);
        let pch = p.xor32(&half, p.and32(p.shr32(&n, 2), 1));
        let dst = p.add32(p.shl32(&ce, 12), p.add32(p.shl32(&n, 5), p.shl32(&pch, 4)));
        // W1 permuted tile-major [e][steps][128][32]: expert e step s row n at ((e steps + s) 128 + n) 32
        let src = p.add64(&pm["pW1"], p.widen(p.add32(p.mul32(&ce, steps * 4096), p.add32(p.shl32(&n, 5), p.shl32(&half, 4)))));
        w1_copies.push((dst, src));
    }
    // (issue_w1 is defined after issue_a below: it calls it)
    // W2 tiles [e][4 kt][32 n][32 B] (E x 4 KB): same chunk mapping (row = kt * 32 + n within the expert)
    let issue_w2 = |stage: usize| {
        for i in 0..copies_per_thread {
            let c = p.add32(&tid, i * threads);
            let pv = if copy_exact { None } else { Some(p.setp("lt.u32", &c, 256 * experts)) };
            let ce = p.shr32(&c, 8);
            let cc = p.and32(&c, 255);
            let row = p.shr32(&cc, 1);
            let half = p.and32(&cc, 1);
            let pch = p.xor32(&half, p.and32(p.shr32(&row, 2), 1));
            let dst = p.add32(&stg_addr[stage], p.add32(p.shl32(&ce, 12), p.add32(p.shl32(&row, 5), p.shl32(&pch, 4))));
            let src = p.add64(&pm["pW2"], p.widen(p.add32(p.shl32(&ce, 12), p.add32(p.shl32(&row, 5), p.shl32(&half, 4)))));
            let pred = match &pv {
                Some(pv) => format!("@{pv} "),
                None => String::new(),
            };
            p.emit(format!("{pred}cp.async.cg.shared.global [{dst}], [{src}], 16;"));
        }
        p.emit("cp.async.commit_group;");
    };
    // W3 ring copies: C x 2 chunks per stage (in the ring region, stage s at RING + s * w3StageBytes)
    let w3_chunks = chan * 2;
    let mut w3_copies: Vec<(String, String, Option<String>)> = Vec::new();
    for i in 0..w3_chunks.div_ceil(threads) {
        let c = p.add32(&tid, i * threads);
        let pv = if !w3_chunks.is_multiple_of(threads) { Some(p.setp("lt.u32", &c, w3_chunks)) } else { None };
        let n = p.shr32(&c, 1);
        let half = p.and32(&c, 1);
        let pch = p.xor32(&half, p.and32(p.shr32(&n, 2), 1));
        let dst = p.add32(p.shl32(&n, 5), p.shl32(&pch, 4));
        let src = p.add64(&pm["pW3"], p.widen(p.add32(p.shl32(&n, 5), p.shl32(&half, 4))));
        w3_copies.push((dst, src, pv));
    }
    let issue_w3 = |stage: usize, step: usize| {
        for (dst, src, pv) in &w3_copies {
            let saddr = p.add32(p.add32(&smem, ring + stage * w3_stage_bytes), dst);
            let gaddr = p.add64(src, p.widen(p.imm32(step * chan * 32)));
            let pred = match pv {
                Some(pv) => format!("@{pv} "),
                None => String::new(),
            };
            p.emit(format!("{pred}cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;"));
        }
        p.emit("cp.async.commit_group;");
    };

    // ---- fragment addressing
    let l8 = p.and32(&lane, 7);
    let lm = p.shr32(&lane, 3);
    let b_nl = p.add32(&l8, p.shl32(p.shr32(&lane, 4), 3));
    let b_chunk_l = p.and32(&lm, 1);
    let b_pch = p.xor32(&b_chunk_l, p.and32(p.shr32(&b_nl, 2), 1));
    let b_lane_off = p.add32(p.shl32(&b_nl, 5), p.shl32(&b_pch, 4));
    let w1_b: Vec<String> = (0..2).map(|s| p.add32(&stg_addr[s], p.add32(p.shl32(&e, 12), &b_lane_off))).collect(); // this expert's tiles
                                                                                                                    // A tile copies: 128 chunks (64 rows x 2 halves) per step, threads < 128: row = tid >> 1 (clamped to the last row)
    let a_copy_on = p.setp("lt.u32", &tid, 128);
    let a_row_c = p.shr32(&tid, 1);
    let a_half_c = p.and32(&tid, 1);
    let a_row_g = p.add32(&block_row, &a_row_c);
    let a_row_g = p.selp32(p.setp("lt.u32", &a_row_g, &pm["rows"]), &a_row_g, &rows_minus1);
    let a_dst_off = p.add32(p.shl32(&a_row_c, 5), p.shl32(p.xor32(&a_half_c, p.and32(p.shr32(&a_row_c, 2), 1)), 4));
    let a_src_off = p.widen(p.add32(p.mul32(&a_row_g, chan), p.shl32(&a_half_c, 4)));
    let issue_a = |stage_addr: &str, a_off: usize, base64: &str, step: usize| {
        let saddr = p.add32(stage_addr, p.add32(&a_dst_off, p.imm32(a_off)));
        let gaddr = p.add64(base64, p.add64(&a_src_off, p.widen(p.imm32(32 * step))));
        p.emit(format!("@{a_copy_on} cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;"));
    };
    let issue_w1 = |stage: usize, step: usize| {
        for ((dst, src), pv) in w1_copies.iter().zip(w1_preds.iter()) {
            let saddr = p.add32(&stg_addr[stage], dst);
            let gaddr = p.add64(src, p.widen(p.imm32(step * 4096)));
            let pred = match pv {
                Some(pv) => format!("@{pv} "),
                None => String::new(),
            };
            p.emit(format!("{pred}cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;"));
        }
        if !proj {
            issue_a(&stg_addr[stage], a_off, &pm["pA"], step);
        }
        p.emit("cp.async.commit_group;");
    };
    let a_row_l = p.add32(&l8, p.shl32(p.and32(&lm, 1), 3));
    let a_chunk_l = p.shr32(&lane, 4);
    let a_frag_off = p.add32(p.shl32(p.add32(&a_row_l, p.shl32(&q, 4)), 5), p.shl32(p.xor32(&a_chunk_l, p.and32(p.shr32(&a_row_l, 2), 1)), 4));
    let narrow_base = p.add32(&smem, narrow);
    let narrow_a = p.add32(&narrow_base, p.add32(p.mul32(p.add32(p.shl32(&q, 4), &a_row_l), narrow_stride), p.shl32(&a_chunk_l, 4)));
    let w3_b: Vec<String> = (0..2).map(|s| p.add32(p.add32(&smem, ring + s * w3_stage_bytes), p.add32(p.shl32(&e, 10), &b_lane_off))).collect(); // columns 32 e ..
                                                                                                                                                 // residual seed for phase 3: E4 state (row, 32 e + 8 j + 2 t) pairs and the aux scales
    let mut seed_regs: Vec<(usize, usize, String)> = Vec::new();
    let aux_addr = p.add64(&pm["pAux"], p.widen(p.shl32(p.add32(&pm["auxHalfOffset"], p.add32(p.shl32(&e, 5), p.shl32(t, 1))), 1)));
    let mut scales: Vec<String> = Vec::new();
    for j in 0..4 {
        let r = p.reg("b32");
        p.emit(format!("ld.global.nc.b32 {r}, [{aux_addr}+{}];", 16 * j));
        scales.push(r);
    }
    // the state tile (proj): [rowsPerWg][C + 16] E4 aliasing the narrow region (dead before phase 2 writes it)
    let state_out = p.add32(&narrow_base, p.add32(p.mul32(p.add32(p.shl32(&q, 4), g), narrow_stride), p.add32(p.shl32(&e, 5), p.shl32(t, 1))));
    if !proj {
        for h in 0..2 {
            let res_addr =
                p.add64(&pm["pA"], p.widen(p.add32(p.mul32(p.selp32(&row_ok[h], &row_idx[h], &rows_minus1), chan), p.add32(p.shl32(&e, 5), p.shl32(t, 1)))));
            for j in 0..4 {
                let hc = p.reg("b16");
                p.emit(format!("ld.global.cg.b16 {hc}, [{res_addr}+{}];", 8 * j));
                seed_regs.push((j, h, hc));
            }
        }
    }

    let chunks_per_row = chan / 16;
    // The [rowsPerWg][C] E4 tile in the narrow region -> global [rows][C] rows (16-byte chunks, one per thread).
    let store_tile = |dst_base: &str| {
        for i in 0..(rows_per_wg * chunks_per_row).div_ceil(threads) {
            let c = p.add32(&tid, i * threads);
            let pv = if !(rows_per_wg * chunks_per_row).is_multiple_of(threads) { Some(p.setp("lt.u32", &c, rows_per_wg * chunks_per_row)) } else { None };
            let row = p.reg("b32");
            let chunk = p.reg("b32");
            p.emit(format!("div.u32 {row}, {c}, {chunks_per_row};"));
            p.emit(format!("rem.u32 {chunk}, {c}, {chunks_per_row};"));
            let g_row = p.add32(&block_row, &row);
            let ok = p.setp("lt.u32", &g_row, &pm["rows"]);
            if let Some(pv) = &pv {
                p.emit(format!("and.pred {ok}, {ok}, {pv};"));
            }
            let r = p.regs("b32", 4);
            let src = p.add32(&narrow_base, p.add32(p.mul32(&row, narrow_stride), p.shl32(&chunk, 4)));
            p.emit(format!("ld.shared.v4.b32 {{{}, {}, {}, {}}}, [{src}];", r[0], r[1], r[2], r[3]));
            let addr = p.add64(dst_base, p.widen(p.add32(p.mul32(&g_row, chan), p.shl32(&chunk, 4))));
            p.emit(format!("@{ok} st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
        }
    };

    let mut w1_first = 0;
    if proj {
        // ================= phase 0: state = E4(ffnPrev * auxAttn + attended Wproj[:, 32 e ..]); Wproj ring in stage 0, W1 step 0 -> stage 1
        w1_first = 1;
        let aux_attn_addr = p.add64(&pm["pAuxPrev"], p.widen(p.shl32(p.add32(&pm["auxAttnHalf"], p.add32(p.shl32(&e, 5), p.shl32(t, 1))), 1)));
        let mut scales_attn: Vec<String> = Vec::new();
        for j in 0..4 {
            let r = p.reg("b32");
            p.emit(format!("ld.global.nc.b32 {r}, [{aux_attn_addr}+{}];", 16 * j));
            scales_attn.push(r);
        }
        let mut pacc: Vec<Vec<String>> = (0..4).map(|_| vec![String::new(), String::new()]).collect();
        for h in 0..2 {
            let prev_addr = p.add64(
                &pm["pFfnPrev"],
                p.widen(p.add32(p.mul32(p.selp32(&row_ok[h], &row_idx[h], &rows_minus1), chan), p.add32(p.shl32(&e, 5), p.shl32(t, 1)))),
            );
            for j in 0..4 {
                let hc = p.reg("b16");
                p.emit(format!("ld.global.cg.b16 {hc}, [{prev_addr}+{}];", 8 * j));
                pacc[j][h] = hmul2(&p, e4x2_to_f16x2(&p, &hc), &scales_attn[j]);
            }
        }
        // Wproj tiles share the W3 copy geometry: chunk offsets relative to the matrix base
        let mut wp_copies: Vec<(String, String, Option<String>)> = Vec::new();
        for i in 0..w3_chunks.div_ceil(threads) {
            let c = p.add32(&tid, i * threads);
            let pv = if !w3_chunks.is_multiple_of(threads) { Some(p.setp("lt.u32", &c, w3_chunks)) } else { None };
            let n = p.shr32(&c, 1);
            let half = p.and32(&c, 1);
            let pch = p.xor32(&half, p.and32(p.shr32(&n, 2), 1));
            let dst = p.add32(p.shl32(&n, 5), p.shl32(&pch, 4));
            let src = p.add64(&pm["pWproj"], p.widen(p.add32(p.shl32(&n, 5), p.shl32(&half, 4))));
            wp_copies.push((dst, src, pv));
        }
        let wp_b: Vec<String> = (0..2).map(|st| p.add32(p.add32(&smem, ring + st * wp_stage_bytes), p.add32(p.shl32(&e, 10), &b_lane_off))).collect();
        let issue_wproj = |stage: usize, step: usize| {
            for (dst, src, pv) in &wp_copies {
                let saddr = p.add32(p.add32(&smem, ring + stage * wp_stage_bytes), dst);
                let gaddr = p.add64(src, p.widen(p.imm32(step * chan * 32)));
                let pred = match pv {
                    Some(pv) => format!("@{pv} "),
                    None => String::new(),
                };
                p.emit(format!("{pred}cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;"));
            }
            let stage_addr = p.add32(&smem, ring + stage * wp_stage_bytes);
            issue_a(&stage_addr, chan * 32, &pm["pAtt"], step);
            p.emit("cp.async.commit_group;");
        };
        issue_wproj(0, 0);
        issue_w1(1, 0); // W1 step 0 lands in stage 1 during phase 0
        for step in 0..steps {
            let stage = step & 1;
            p.emit("cp.async.wait_group 0;");
            p.emit("bar.sync 0;");
            if step + 1 < steps {
                issue_wproj(stage ^ 1, step + 1);
            } else {
                p.emit("cp.async.commit_group;");
            }
            let a = ldm4(&p, p.add32(p.add32(&smem, ring + stage * wp_stage_bytes + chan * 32), &a_frag_off));
            for i2 in 0..2 {
                let r = ldm4(&p, p.add32(&wp_b[stage], i2 * 512));
                mma_e4(&p, &pacc[2 * i2], &a, &r[0..2], &pacc[2 * i2]);
                mma_e4(&p, &pacc[2 * i2 + 1], &a, &r[2..4], &pacc[2 * i2 + 1]);
            }
        }
        // publish the state tile: E4 codes (the FFN input and skip)
        for j in 0..4 {
            for h in 0..2 {
                let hc = cvt_e4x2(&p, &pacc[j][h]);
                p.emit(format!("st.shared.b16 [{state_out}+{}], {hc};", 8 * j + 8 * narrow_stride * h));
                seed_regs.push((j, h, hc));
            }
        }
        p.emit("bar.sync 0;");
        // W1 step 1 into stage 0 (the Wproj ring is dead)
        if steps > 1 {
            issue_w1(0, 1);
        }
        // optional materialization of the block state (parity captures)
        let p_store = p.setp("ne.u32", &pm["storeState"], 0);
        let skip_store = p.label("NOSTATE");
        p.emit(format!("@!{p_store} bra {skip_store};"));
        store_tile(&pm["pStateOut"]);
        p.emit(format!("{skip_store}:"));
    }

    // ================= phase 1: hidden = x W1_e
    if !proj {
        issue_w1(0, 0);
    }
    let mut hid: Vec<Vec<String>> = (0..HIDDEN / 8).map(|_| zero_tile(&p)).collect();
    for step in 0..steps {
        let stage = (step + w1_first) & 1;
        p.emit("cp.async.wait_group 0;");
        p.emit("bar.sync 0;");
        if step + 1 < steps {
            if !proj || step >= 1 {
                issue_w1(stage ^ 1, step + 1);
            }
        } else {
            issue_w2(stage ^ 1);
        }
        let a = if proj {
            ldm4(&p, p.add32(&narrow_a, 32 * step)) // the state tile in shared
        } else {
            ldm4(&p, p.add32(&stg_addr[stage], p.add32(&a_frag_off, p.imm32(a_off))))
        };
        for i2 in 0..8 {
            let r = ldm4(&p, p.add32(&w1_b[stage], i2 * 512));
            mma_e4(&p, &hid[2 * i2], &a, &r[0..2], &hid[2 * i2]);
            mma_e4(&p, &hid[2 * i2 + 1], &a, &r[2..4], &hid[2 * i2 + 1]);
        }
    }
    // ================= phase 2: SiLU, E4, W2_e -> narrow tile
    p.emit("cp.async.wait_group 0;");
    p.emit("bar.sync 0;");
    let w2_stage = (steps + w1_first) & 1;
    let k = SiluConsts::new(&p);
    for d in hid.iter_mut() {
        d[0] = silu(&p, &d[0], &k);
        d[1] = silu(&p, &d[1], &k);
    }
    let narrow_t: Vec<Vec<String>> = (0..4).map(|_| zero_tile(&p)).collect();
    for kt in 0..4 {
        let qcodes: Vec<Vec<String>> = (0..4).map(|i| (0..2).map(|h| b16_to_b32(&p, cvt_e4x2(&p, &hid[4 * kt + i][h]))).collect()).collect();
        let a2 = {
            let a0 = prmt(&p, &qcodes[0][0], &qcodes[1][0], "0x5410");
            let a1 = prmt(&p, &qcodes[0][1], &qcodes[1][1], "0x5410");
            let a2_ = prmt(&p, &qcodes[2][0], &qcodes[3][0], "0x5410");
            let a3 = prmt(&p, &qcodes[2][1], &qcodes[3][1], "0x5410");
            vec![a0, a1, a2_, a3]
        };
        for i2 in 0..2 {
            let r = ldm4(&p, p.add32(&w1_b[w2_stage], kt * 1024 + i2 * 512));
            mma_e4(&p, &narrow_t[2 * i2], &a2, &r[0..2], &narrow_t[2 * i2]);
            mma_e4(&p, &narrow_t[2 * i2 + 1], &a2, &r[2..4], &narrow_t[2 * i2 + 1]);
        }
    }
    let narrow_out = p.add32(&narrow_base, p.add32(p.mul32(p.add32(p.shl32(&q, 4), g), narrow_stride), p.add32(p.shl32(&e, 5), p.shl32(t, 1))));
    for j in 0..4 {
        for h in 0..2 {
            let hc = cvt_e4x2(&p, &narrow_t[j][h]);
            p.emit(format!("st.shared.b16 [{narrow_out}+{}], {hc};", 8 * j + 8 * narrow_stride * h));
        }
    }
    // W3 stage 0 into the ring (both W1/W2 stages are dead after this barrier)
    p.emit("bar.sync 0;");
    issue_w3(0, 0);
    // ================= phase 3: out = skip * aux + narrow W3[:, 32 e ..]
    let mut acc: Vec<Vec<String>> = (0..4).map(|_| vec![String::new(), String::new()]).collect();
    for (j, h, hc) in &seed_regs {
        acc[*j][*h] = hmul2(&p, e4x2_to_f16x2(&p, hc), &scales[*j]);
    }
    for step in 0..steps {
        let stage = step & 1;
        p.emit("cp.async.wait_group 0;");
        p.emit("bar.sync 0;");
        if step + 1 < steps {
            issue_w3(stage ^ 1, step + 1);
        }
        let a = ldm4(&p, p.add32(&narrow_a, 32 * step));
        for i2 in 0..2 {
            let r = ldm4(&p, p.add32(&w3_b[stage], i2 * 512));
            mma_e4(&p, &acc[2 * i2], &a, &r[0..2], &acc[2 * i2]);
            mma_e4(&p, &acc[2 * i2 + 1], &a, &r[2..4], &acc[2 * i2 + 1]);
        }
    }
    // ---- output through the narrow region (every warp finished reading it), then 16-byte stores: one chunk per thread
    p.emit("bar.sync 0;");
    for j in 0..4 {
        for h in 0..2 {
            let hc = cvt_e4x2(&p, &acc[j][h]);
            p.emit(format!("st.shared.b16 [{narrow_out}+{}], {hc};", 8 * j + 8 * narrow_stride * h));
        }
    }
    p.emit("bar.sync 0;");
    store_tile(&pm["pOut"]);
    // ---- chaining: release the FFN rows (bands b0 .. b1 of this workgroup)
    let p_signal_on = p.setp("ne.u64", &pm["pSignal"], 0);
    p.emit("fence.acq_rel.gpu;");
    p.emit("bar.sync 0;");
    let p_t0 = p.reg("pred");
    let p_tid0 = p.setp("eq.u32", &tid, 0);
    p.emit(format!("and.pred {p_t0}, {p_signal_on}, {p_tid0};"));
    let b_iter = p.reg("b32");
    p.emit(format!("mov.u32 {b_iter}, {b0};"));
    let sig_loop = p.label("SIG");
    let sig_done = p.label("SIGDONE");
    p.emit(format!("@!{p_t0} bra {sig_done};"));
    p.emit(format!("{sig_loop}:"));
    let sig_addr = p.add64(&pm["pSignal"], p.widen(p.shl32(&b_iter, 2)));
    p.emit(format!("red.release.gpu.global.add.u32 [{sig_addr}], 1;"));
    p.emit(format!("add.u32 {b_iter}, {b_iter}, 1;"));
    let p_more_b = p.setp("le.u32", &b_iter, &b1);
    p.emit(format!("@{p_more_b} bra {sig_loop};"));
    p.emit(format!("{sig_done}:"));
    p.emit("ret;");
    let _ = &a_row;
    (name, p.finish(), shared_bytes)
}

/// `<C> <ROWT> [maxregs] [proj]`
pub fn cli(args: &[&str]) -> String {
    let (chan, rowt) = (crate::req(args, 0), crate::req(args, 1));
    let max_regs = args.get(2).map(|a| crate::reg(a));
    let proj = args.get(3).is_some_and(|a| crate::int(a) != 0);
    generate(chan, rowt, max_regs, proj).1
}
