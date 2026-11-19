//! QKV projection + cosine normalization + 8x8 window attention for (window, head) items in PTX
//! (the qkv_attention.comp dataflow at the fragment level; see block32_e4m3.rs for the block-level version).
//!
//! Workgroup = 4 warps, persistent over items (item = head * windowCount + window, stride = grid); warp q owns the
//! window's tokens 16q .. 16q+15.
//!   q/k/v = A(window rows of the E4 block state, K = C) x Wqkv[head columns] over K in k32 steps: the A fragments come
//!   straight from global memory (ADEPTH k32 steps in flight in registers, zero outside the image); the head's 96
//!   weight columns per k32 step are staged in a cp.async ring (3 KB per step).
//!   normalize q (learned scale) and k; K rows / transposed V of this head into shared (physical token order);
//!   S = prior[head] + Q K^T; the specified softmax; P V; E4 output at columns head * 32 of the attended tensor.
//! NaN handling: the normalization's NaN -> 0 (zero rows at the image border) runs under a warp vote; V and the
//! attended tile are converted directly (finite data: every state is a saturated publication).
//!
//! Arguments: `<C> [maxregs]`

use std::collections::HashMap;

use crate::ptxgen::Ptx;
use crate::swin::*;

pub const THREADS: usize = 128;
/// cp.async ring depth (steps in flight = STAGES - 1)
pub const STAGES: usize = 2;
/// A operand load cache hint (produced by the chained FFN launch: no L1 / texture path)
pub const ALD: &str = "cg";
/// k32 steps of A fragments in flight
pub const ADEPTH: usize = 2;
/// one k32 step of the head's B tiles: [96 n][32 B] (chunk-swizzled rows)
pub const B_BYTES: usize = 96 * 32;
/// + the window's A tile [64 tokens][32 B] (chunk-swizzled rows, zero outside the image)
pub const STEP_BYTES: usize = B_BYTES + 64 * 32;
/// K [64 physical][32 B] swizzled (2 KB)
pub const KROWS: usize = STAGES * STEP_BYTES;
/// Vt [32 dim][64 B] swizzled (chunk ^= (dim >> 1) & 3)
pub const VTS: usize = KROWS + 2048;
/// per warp 512 B output staging
pub const STG: usize = VTS + 2048;
pub const SHARED_BYTES: usize = STG + 4 * 512;

/// The geometry of one item (and, with the band fields empty, the registers carried to the next iteration).
struct Gm {
    head: String,
    window_x: String,
    window_y: String,
    wy: String,
    a_src: String,
    a_size: String,
    w_head: String,
    band0: String,
    band1: String,
}

pub fn generate(c: usize, max_regs: Option<u32>) -> (String, String) {
    assert!(c.is_multiple_of(32));
    let steps = c / 32;
    let stages = STAGES.min(steps);
    let name = format!("qkv_e4m3_K{c}");
    let p = Ptx::new();
    let params = [
        ("u64", "pState"),
        ("u64", "pWqkv"),
        ("u64", "pPrior"),
        ("u64", "pAux"),
        ("u64", "pOut"),
        ("u32", "width"),
        ("u32", "height"),
        ("u32", "shiftX"),
        ("u32", "shiftY"),
        ("u32", "windowsX"),
        ("u32", "scaleWord"),
        ("u32", "windowCount"),
        ("u32", "itemCount"),
        ("u64", "pWait"),
        ("u64", "pSignal"),
        ("u32", "waitMul"),
        ("u32", "waitGroupRows"),
        ("u64", "pError"),
    ]; // chaining: this block's FFN row bands (wait, waitMul signals per row group of waitGroupRows rows), window rows (signal); 0 = off
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
    let exp_k = ExpConsts::new(&p);

    // ---- lane constants
    let l8 = p.and32(&lane, 7);
    let lm = p.shr32(&lane, 3);
    let b_nl = p.add32(&l8, p.shl32(p.shr32(&lane, 4), 3));
    let b_chunk_l = p.and32(&lm, 1);
    let b_pch = p.xor32(&b_chunk_l, p.and32(p.shr32(&b_nl, 2), 1));
    let b_lane_off = p.add32(p.shl32(&b_nl, 5), p.shl32(&b_pch, 4));
    let ring_b = p.add32(&smem, &b_lane_off);
    let k_lane = p.add32(p.add32(&smem, KROWS), &b_lane_off);
    let v_swz = p.and32(p.shr32(&b_nl, 1), 3);
    let v_lane: Vec<String> =
        (0..2).map(|st| p.add32(p.add32(&smem, VTS), p.add32(p.shl32(&b_nl, 6), p.shl32(p.xor32(p.xor32(&b_chunk_l, p.imm32(2 * st)), &v_swz), 4)))).collect();
    // K rows / Vt publication of this warp's tokens g, g + 8 (physical order)
    let phys: Vec<String> = (0..2).map(|h| tiled_token_regs(&p, p.add32(p.shl32(&warp, 4), p.add32(g, p.imm32(8 * h))))).collect();
    let mut k_base: Vec<Vec<String>> = Vec::new();
    let mut v_base: Vec<String> = Vec::new();
    for h in 0..2 {
        let swz = p.and32(p.shr32(&phys[h], 2), 1);
        k_base.push(
            (0..2)
                .map(|c| p.add32(p.add32(&smem, KROWS), p.add32(p.shl32(&phys[h], 5), p.add32(p.shl32(p.xor32(p.imm32(c), &swz), 4), p.shl32(t, 1)))))
                .collect(),
        );
        v_base.push(p.add32(p.add32(&smem, VTS), p.add32(p.shl32(t, 7), p.add32(p.shl32(p.xor32(p.shr32(&phys[h], 4), t), 4), p.and32(&phys[h], 15)))));
    }
    // A tile chunk of this thread (one 16-byte copy per step): token = tid >> 1, half = tid & 1
    let a_tok = p.shr32(&tid, 1);
    let a_half = p.and32(&tid, 1);
    let a_tok_x = p.and32(&a_tok, 7);
    let a_tok_y = p.shr32(&a_tok, 3);
    let a_dst = p.add32(&smem, p.add32(p.imm32(B_BYTES), p.add32(p.shl32(&a_tok, 5), p.shl32(p.xor32(&a_half, p.and32(p.shr32(&a_tok, 2), 1)), 4))));
    let a_row_l = p.add32(&l8, p.shl32(p.and32(&lm, 1), 3));
    let a_chunk_l = p.shr32(&lane, 4);
    let a_pch = p.xor32(&a_chunk_l, p.and32(p.shr32(&a_row_l, 2), 1));
    let a_lane = p.add32(&smem, p.add32(p.imm32(B_BYTES), p.add32(p.shl32(p.add32(&a_row_l, p.shl32(&warp, 4)), 5), p.shl32(&a_pch, 4))));
    // B ring copies: 192 chunks per step, 2 per thread; chunk c: row n = c >> 1, half = c & 1, chunk ^= (n >> 2) & 1
    let mut copies: Vec<(String, String, String)> = Vec::new();
    for i in 0..2 {
        let c = p.add32(&tid, i * THREADS);
        let pv = p.setp("lt.u32", &c, 192);
        let n = p.shr32(&c, 1);
        let half = p.and32(&c, 1);
        let pch = p.xor32(&half, p.and32(p.shr32(&n, 2), 1));
        let dst = p.add32(&smem, p.add32(p.shl32(&n, 5), p.shl32(&pch, 4)));
        copies.push((dst, p.shl32(&c, 4), pv));
    }
    // output staging / store lane
    let stg_warp = p.add32(p.add32(&smem, STG), p.shl32(&warp, 9));
    let stg_out = p.add32(&stg_warp, p.add32(p.shl32(g, 5), p.shl32(t, 1)));
    let ld_row = p.shr32(&lane, 1);
    let ld_part = p.and32(&lane, 1);
    let stg_read = p.add32(&stg_warp, p.add32(p.shl32(&ld_row, 5), p.shl32(&ld_part, 4)));
    let prior_row_off = p.add32(p.shl32(p.add32(p.shl32(&warp, 4), g), 7), p.shl32(t, 2)); // + head * 8192

    // ================= persistent item loop (the next item's first weight stage and A fragments are prefetched
    // during the current item's attention phase)
    let heads = c / 32;
    let head_shift = heads.ilog2() as usize;
    assert!(1 << head_shift == heads);
    let windows_x_rcp = rcp_f32(&p, &pm["windowsX"]);
    let geometry = |item_reg: &str| -> Gm {
        // item = window * heads + head (heads is a power of two); window -> (wy, wx) by the f32 reciprocal
        let head = p.and32(item_reg, heads - 1);
        let window = p.shr32(item_reg, head_shift);
        let (wy, wx) = fast_divmod(&p, &window, &pm["windowsX"], &windows_x_rcp);
        let window_x = p.sub32(p.shl32(&wx, 3), &pm["shiftX"]);
        let window_y = p.sub32(p.shl32(&wy, 3), &pm["shiftY"]);
        // this thread's A chunk: token tid >> 1 (pixel windowX + (token & 7), windowY + (token >> 3)), half tid & 1
        let xa = p.add32(&window_x, &a_tok_x);
        let ya = p.add32(&window_y, &a_tok_y);
        let pvx = p.setp("lt.u32", &xa, &pm["width"]);
        let pvy = p.setp("lt.u32", &ya, &pm["height"]);
        let pv = p.reg("pred");
        p.emit(format!("and.pred {pv}, {pvx}, {pvy};"));
        let pixel = p.mad32(&ya, &pm["width"], &xa);
        let off = p.selp32(&pv, p.add32(p.mul32(&pixel, c), p.shl32(&a_half, 4)), p.imm32(0));
        let a_src = p.add64(&pm["pState"], p.widen(&off));
        let a_size = p.selp32(&pv, p.imm32(16), p.imm32(0));
        let w_head = p.add64(&pm["pWqkv"], p.widen(p.mul32(&head, 96 * 32)));
        // pixel-row bands of the window: rows [max(8 wy - shiftY, 0), min(+7, height - 1)]
        let y_top = &window_y;
        let p_neg = p.setp("gt.s32", p.imm32(0), y_top);
        let y0 = p.selp32(&p_neg, p.imm32(0), y_top);
        let y1 = p.add32(y_top, 7);
        let hm1 = p.sub32(&pm["height"], 1);
        let p_big = p.setp("gt.u32", &y1, &hm1);
        let y1 = p.selp32(&p_big, &hm1, &y1);
        let band0 = p.shr32(&y0, 3);
        let band1 = p.shr32(&y1, 3);
        Gm { head, window_x: window_x.clone(), window_y: window_y.clone(), wy, a_src, a_size, w_head, band0, band1 }
    };

    let rows_total = p.mul32(&pm["width"], &pm["height"]);
    let p_wait_on = p.setp("ne.u64", &pm["pWait"], 0);
    let p_signal_on = p.setp("ne.u64", &pm["pSignal"], 0);
    let wait_item = |gm: &Gm, guard: Option<&str>| {
        let mut g2 = p_wait_on.clone();
        if let Some(gd) = guard {
            g2 = p.reg("pred");
            p.emit(format!("and.pred {g2}, {p_wait_on}, {gd};"));
        }
        sync_wait(
            &p,
            &pm["pWait"],
            &gm.band0,
            &gm.band1,
            Expected::Per(&|b: &str| band_expected(&p, b, &pm["width"], &rows_total, Some(pm["waitMul"].as_str()), Some(pm["waitGroupRows"].as_str()))),
            &lane,
            Some(&g2),
            Some(&warp),
            &pm["pError"],
        );
    };

    let issue_stage = |gm: &Gm, stage_index: usize, step: usize, guard: Option<&str>| {
        // the head's weight tiles (L1-cached, constant) and the window's A tile (L2 only: produced by the chained FFN)
        for (dst, src_off, pv) in &copies {
            let saddr = p.add32(dst, p.imm32(stage_index * STEP_BYTES));
            let gaddr = p.add64(&gm.w_head, p.widen(p.add32(src_off, p.imm32(step * 3 * c * 32))));
            let mut pred = pv.clone();
            if let Some(gd) = guard {
                pred = p.reg("pred");
                p.emit(format!("and.pred {pred}, {pv}, {gd};"));
            }
            p.emit(format!("@{pred} cp.async.ca.shared.global [{saddr}], [{gaddr}], 16;"));
        }
        let saddr = p.add32(&a_dst, p.imm32(stage_index * STEP_BYTES));
        let gaddr = p.add64(&gm.a_src, p.widen(p.imm32(32 * step)));
        let pred = match guard {
            Some(gd) => format!("@{gd} "),
            None => String::new(),
        };
        p.emit(format!("{pred}cp.async.cg.shared.global [{saddr}], [{gaddr}], 16, {};", gm.a_size));
        p.emit("cp.async.commit_group;");
    };

    let carry_regs = || -> Gm {
        Gm {
            head: p.reg("b32"),
            window_x: p.reg("b32"),
            window_y: p.reg("b32"),
            wy: p.reg("b32"),
            a_src: p.reg("b64"),
            a_size: p.reg("b32"),
            w_head: p.reg("b64"),
            band0: String::new(),
            band1: String::new(),
        }
    };

    let carry_assign = |dst: &Gm, gm: &Gm| {
        for (d, s) in [(&dst.head, &gm.head), (&dst.window_x, &gm.window_x), (&dst.window_y, &gm.window_y), (&dst.wy, &gm.wy), (&dst.a_size, &gm.a_size)] {
            p.emit(format!("mov.b32 {d}, {s};"));
        }
        p.emit(format!("mov.b64 {}, {};", dst.a_src, gm.a_src));
        p.emit(format!("mov.b64 {}, {};", dst.w_head, gm.w_head));
    };

    let item = p.reg("b32");
    p.emit(format!("mov.u32 {item}, {ctaid};"));
    let p_item = p.setp("lt.u32", &item, &pm["itemCount"]);
    let end_l = p.label("END");
    let loop_l = p.label("ITEM");
    p.emit(format!("@!{p_item} bra {end_l};"));
    let gm0 = geometry(&item);
    wait_item(&gm0, None);
    for j in 0..stages - 1 {
        issue_stage(&gm0, j, j, None);
    }
    let carried = carry_regs();
    carry_assign(&carried, &gm0);
    let s_keep: Vec<Vec<String>> = (0..8).map(|_| vec![p.reg("b32"), p.reg("b32")]).collect(); // the item's scores for the attention tail
    for m in 0..8 {
        p.emit(format!("mov.b32 {}, 0;", s_keep[m][0]));
        p.emit(format!("mov.b32 {}, 0;", s_keep[m][1]));
    }

    // softmax, P V, E4 output of the item whose scores are in sKeep and geometry in gmT (straight-line code).
    let emit_tail = |gm_t: &Gm| {
        let w = softmax(&p, &s_keep, &li, &exp_k);
        let pa = [acc_to_a(&p, &w[0..4], &li, None), acc_to_a(&p, &w[4..8], &li, None)];
        let o: Vec<Vec<String>> = (0..4).map(|_| zero_tile(&p)).collect();
        for step in 0..2 {
            for d2 in 0..2 {
                let r = ldm4(&p, p.add32(&v_lane[step], d2 * 1024));
                mma_e4(&p, &o[2 * d2], &pa[step], &r[0..2], &o[2 * d2]);
                mma_e4(&p, &o[2 * d2 + 1], &pa[step], &r[2..4], &o[2 * d2 + 1]);
            }
        }
        for j in 0..4 {
            for h in 0..2 {
                let hc = cvt_e4x2(&p, &o[j][h]);
                p.emit(format!("st.shared.b16 [{stg_out}+{}], {hc};", 8 * j + 256 * h));
            }
        }
        p.emit("bar.warp.sync 0xffffffff;");
        let r = p.regs("b32", 4);
        p.emit(format!("ld.shared.v4.b32 {{{}, {}, {}, {}}}, [{stg_read}];", r[0], r[1], r[2], r[3]));
        let tok_x = p.add32(&gm_t.window_x, p.and32(&ld_row, 7));
        let tok_y = p.add32(&gm_t.window_y, p.add32(p.shl32(&warp, 1), p.shr32(&lane, 4)));
        let pvx2 = p.setp("lt.u32", &tok_x, &pm["width"]);
        let pvy2 = p.setp("lt.u32", &tok_y, &pm["height"]);
        let pv = p.reg("pred");
        p.emit(format!("and.pred {pv}, {pvx2}, {pvy2};"));
        let pixel = p.mad32(&tok_y, &pm["width"], &tok_x);
        let addr = p.add64(&pm["pOut"], p.widen(p.add32(p.mul32(&pixel, c), p.add32(p.shl32(&gm_t.head, 5), p.shl32(&ld_part, 4)))));
        p.emit(format!("@{pv} st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
    };

    let signal_item = |gm_t: &Gm| {
        // release the attended rows of the item: window row counter (fence + barrier + one red)
        p.emit("fence.acq_rel.gpu;");
        p.emit("bar.sync 0;");
        let p_sig = p.reg("pred");
        let eq = p.setp("eq.u32", &tid, 0);
        p.emit(format!("and.pred {p_sig}, {p_signal_on}, {eq};"));
        let sig_addr = p.add64(&pm["pSignal"], p.widen(p.shl32(&gm_t.wy, 2)));
        p.emit(format!("@{p_sig} red.release.gpu.global.add.u32 [{sig_addr}], 1;"));
    };

    p.emit(format!("{loop_l}:"));
    let gm = &carried;
    let head = &gm.head;
    // prior rows of this head and the head scale: issued first
    let prior_lane = p.add64(&pm["pPrior"], p.widen(p.add32(p.shl32(head, 13), &prior_row_off)));
    let mut prior_regs: Vec<Vec<String>> = Vec::new();
    for m in 0..8 {
        let c0 = p.reg("b32");
        p.emit(format!("ld.global.nc.b32 {c0}, [{prior_lane}+{}];", 16 * m));
        let c1 = p.reg("b32");
        p.emit(format!("ld.global.nc.b32 {c1}, [{prior_lane}+{}];", 1024 + 16 * m));
        prior_regs.push(vec![c0, c1]);
    }
    let scale_addr = p.add64(&pm["pAux"], p.widen(p.shl32(p.add32(&pm["scaleWord"], head), 2)));
    let scale_f = p.reg("f32");
    p.emit(format!("ld.global.nc.f32 {scale_f}, [{scale_addr}];"));
    let acc: Vec<Vec<String>> = (0..12).map(|_| zero_tile(&p)).collect(); // q tiles 0..3, k 4..7, v 8..11
    for step in 0..steps {
        let stage_index = step % stages;
        if step + stages - 1 < steps {
            issue_stage(gm, (step + stages - 1) % stages, step + stages - 1, None);
        } else {
            p.emit("cp.async.commit_group;");
        }
        p.emit(format!("cp.async.wait_group {};", stages - 1));
        p.emit("bar.sync 0;");
        let a = ldm4(&p, p.add32(&a_lane, stage_index * STEP_BYTES));
        let base = p.add32(&ring_b, stage_index * STEP_BYTES);
        for i2 in 0..6 {
            let r = ldm4(&p, p.add32(&base, i2 * 512));
            mma_e4(&p, &acc[2 * i2], &a, &r[0..2], &acc[2 * i2]);
            mma_e4(&p, &acc[2 * i2 + 1], &a, &r[2..4], &acc[2 * i2 + 1]);
        }
        if step + 1 < steps {
            p.emit("bar.sync 0;"); // the stage is free for the next copy
        }
    }
    let scale_h = p.reg("f16");
    p.emit(format!("cvt.rn.f16.f32 {scale_h}, {scale_f};"));
    let scale2 = pack16(&p, &scale_h, &scale_h);
    let qn = normalize(&p, &acc[0..4], &li, &zero, Some(&scale2), true);
    let kn = normalize(&p, &acc[4..8], &li, &zero, None, true);
    let mq = acc_to_a(&p, &qn, &li, None);
    for h in 0..2 {
        for j in 0..4 {
            let hc = cvt_e4x2(&p, &kn[j][h]);
            p.emit(format!("st.shared.b16 [{}+{}], {hc};", k_base[h][j >> 1], 8 * (j & 1)));
            let vc = cvt_e4x2(&p, &acc[8 + j][h]);
            let vhi = p.reg("b16");
            p.emit(format!("shr.b16 {vhi}, {vc}, 8;"));
            p.emit(format!("st.shared.u8 [{}+{}], {vc};", v_base[h], 512 * j));
            p.emit(format!("st.shared.u8 [{}+{}], {vhi};", v_base[h], 512 * j + 64));
        }
    }
    p.emit("bar.sync 0;");
    // prefetch the next item: its first weight stage (the ring is free) and A fragments
    let next_item = p.add32(&item, &nctaid);
    let p_next = p.setp("lt.u32", &next_item, &pm["itemCount"]);
    let gm_n = geometry(&next_item);
    wait_item(&gm_n, Some(&p_next));
    for j in 0..stages - 1 {
        issue_stage(&gm_n, j, j, Some(&p_next));
    }
    let s: Vec<Vec<String>> = (0..8).map(|m| vec![prior_regs[m][0].clone(), prior_regs[m][1].clone()]).collect();
    for m2 in 0..4 {
        let r = ldm4(&p, p.add32(&k_lane, m2 * 512));
        mma_e4(&p, &s[2 * m2], &mq, &r[0..2], &s[2 * m2]);
        mma_e4(&p, &s[2 * m2 + 1], &mq, &r[2..4], &s[2 * m2 + 1]);
    }
    for m in 0..8 {
        p.emit(format!("mov.b32 {}, {};", s_keep[m][0], s[m][0]));
        p.emit(format!("mov.b32 {}, {};", s_keep[m][1], s[m][1]));
    }
    emit_tail(gm);
    signal_item(gm);
    carry_assign(&carried, &gm_n);
    p.emit(format!("mov.u32 {item}, {next_item};"));
    p.emit(format!("@{p_next} bra {loop_l};"));
    p.emit(format!("{end_l}:"));
    p.emit("ret;");
    (name, p.finish())
}

/// `<C> [maxregs]`
pub fn cli(args: &[&str]) -> String {
    let c = crate::req(args, 0);
    let max_regs = args.get(1).map(|a| crate::reg(a));
    generate(c, max_regs).1
}
