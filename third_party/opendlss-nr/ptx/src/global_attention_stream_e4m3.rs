//! Global ViT cosine attention for any token count (high resolutions), in two PTX passes with the arithmetic of
//! global_attention_e4m3 (bit-identical per key block, blocks accumulated in key order):
//!
//! 1. global_normalize_e4m3: one thread per (token, head): Q normalized (times f16(sqrt 32) and the learned scale),
//!    K normalized, V as E4 codes, written per head so the attention streams contiguous tiles:
//!    Q [head][padded][32 B] at 0, K [head][padded][32 B] at heads * padded * 32,
//!    V^T [head][32 dims][padded] (32-key permutation) at 2 * heads * padded * 32.
//!    Tokens >= `tokens` (padding) are zero codes. Grid (padded / 64, heads), 64 threads.
//!
//! 2. global_attention_stream_e4m3: one workgroup per (head, 64-query block), four warps of 16 query rows; the
//!    key blocks (K tile 2 KB + V^T tile 32 x 80 B) stream through a STAGES-deep cp.async ring; per block the same
//!    S = Q K^T, ViT exponential, the specified block sums, unnormalized P V as the resident kernel; padding correction,
//!    f32 reciprocal and the E4 output afterwards. Grid (heads, padded / 64), 128 threads, static shared memory.
//!
//! Chaining: the normalize waits for the QKV GEMM's signals and signals once per workgroup; the attention waits for
//! the normalize workgroups and signals once per workgroup (the projection GEMM waits for heads * padded / 64).
//!
//! Arguments: `normalize|attention [stages]`

use std::collections::HashMap;

use crate::ptxgen::Ptx;
use crate::swin::*;

/// [64][32 B] swizzled
pub const QTILE: usize = 0;
pub const STAGE0: usize = 2048;
/// [64 keys][32 B] swizzled
pub const K_BYTES: usize = 2048;
/// 64 keys + 16 pad: (80 / 4) mod 32 = 20, conflict-free ldmatrix rows
pub const VT_STRIDE: usize = 80;
pub const VT_BYTES: usize = 32 * VT_STRIDE;
pub const STAGE_BYTES: usize = K_BYTES + VT_BYTES;

/// Physical key position within the V^T row: (k & ~31) | (k & 16) | (((k & 7) >> 1) << 2) | (k & 1) | (((k & 15) >> 3) << 1).
pub fn key_position(p: &Ptx, token: &str) -> String {
    p.or32(
        p.or32(p.and32(token, p.imm32(0xffffffe0_u32 | 16)), p.shl32(p.shr32(p.and32(token, 7), 1), 2)),
        p.or32(p.and32(token, 1), p.shl32(p.shr32(p.and32(token, 15), 3), 1)),
    )
}

/// The normalize pass: its name, PTX and (static) shared memory size.
pub fn generate_normalize() -> (String, String, usize) {
    let name = "global_normalize_e4m3".to_string();
    let p = Ptx::new();
    let params = [
        ("u64", "pQkv"),
        ("u64", "pAux"),
        ("u64", "pOut"),
        ("u32", "tokens"),
        ("u32", "padded"),
        ("u32", "heads"),
        ("u32", "scaleWordOffset"),
        ("u64", "pWait"),
        ("u32", "waitExpected"),
        ("u64", "pSignal"),
        ("u64", "pError"),
    ];
    p.entry(&name, &params, 0, Some(64), None, false);
    let pm: HashMap<&str, String> = params.iter().map(|&(t, n)| (n, if t == "u64" { p.load_param_u64(n) } else { p.load_param_u32(n) })).collect();
    let tid = p.special("tid.x");
    let tb = p.special("ctaid.x");
    let head = p.special("ctaid.y");
    let lane = p.and32(&tid, 31);
    let warp = p.shr32(&tid, 5);
    let zero32 = p.imm32(0);
    let token = p.add32(p.shl32(&tb, 6), &tid);
    let valid = p.setp("lt.u32", &token, &pm["tokens"]);
    let p_wait_on = p.setp("ne.u64", &pm["pWait"], "0");
    sync_wait(&p, &pm["pWait"], &zero32, &zero32, Expected::Reg(&pm["waitExpected"]), &lane, Some(&p_wait_on), Some(&warp), &pm["pError"]);
    let row_bytes = p.mul32(&pm["heads"], p.imm32(192));
    let head_byte = p.mul32(&head, p.imm32(192));
    let learned = p.reg("f32");
    let learned_addr = p.add64(&pm["pAux"], p.widen(p.shl32(p.add32(&pm["scaleWordOffset"], &head), 2)));
    p.emit(format!("ld.global.nc.f32 {learned}, [{learned_addr}];"));
    let learned_h = p.reg("f16");
    p.emit(format!("cvt.rn.f16.f32 {learned_h}, {learned};"));
    let learned2 = pack16(&p, &learned_h, &learned_h);
    let hs2 = imm(&p, "0x45A845A8");
    let src = p.add64(&pm["pQkv"], p.widen(p.add32(p.mul32(&token, &row_bytes), &head_byte)));

    let load_row = |part: usize| -> Vec<String> {
        let mut w = Vec::new();
        for c in 0..4 {
            let r = p.regs("b32", 4);
            for x in &r {
                p.emit(format!("mov.b32 {x}, 0;"));
            }
            p.emit(format!("@{valid} ld.global.cg.v4.b32 {{{}, {}, {}, {}}}, [{src}+{}];", r[0], r[1], r[2], r[3], 64 * part + 16 * c));
            w.extend(r);
        }
        w
    };
    let wq = load_row(0);
    let wk = load_row(1);
    let wv = load_row(2);
    let qc = vit_quantize(&p, &wq, &vit_norm(&p, &wq), true, &hs2, &learned2, &zero32);
    let kc = vit_quantize(&p, &wk, &vit_norm(&p, &wk), false, &hs2, &learned2, &zero32);
    let vc: Vec<String> = (0..8).map(|i| pack16(&p, cvt_e4x2(&p, &wv[2 * i]), cvt_e4x2(&p, &wv[2 * i + 1]))).collect();
    let qc: Vec<String> = qc.iter().map(|c| p.selp32(&valid, c, &zero32)).collect();
    let kc: Vec<String> = kc.iter().map(|c| p.selp32(&valid, c, &zero32)).collect();
    let vc: Vec<String> = vc.iter().map(|c| p.selp32(&valid, c, &zero32)).collect();
    let head_rows = p.mul32(&head, &pm["padded"]); // head * padded
    let row_index = p.add32(&head_rows, &token);
    let plane_bytes = p.mul32(p.mul32(&pm["heads"], &pm["padded"]), p.imm32(32));
    let q_addr = p.add64(&pm["pOut"], p.widen(p.shl32(&row_index, 5)));
    let k_addr = p.add64(&q_addr, p.widen(&plane_bytes));
    for c in 0..2 {
        p.emit(format!("st.global.v4.b32 [{q_addr}+{}], {{{}, {}, {}, {}}};", 16 * c, qc[4 * c], qc[4 * c + 1], qc[4 * c + 2], qc[4 * c + 3]));
        p.emit(format!("st.global.v4.b32 [{k_addr}+{}], {{{}, {}, {}, {}}};", 16 * c, kc[4 * c], kc[4 * c + 1], kc[4 * c + 2], kc[4 * c + 3]));
    }
    // V^T: byte (dim) of the token at [head][dim][position]
    let v_base = p.add64(&pm["pOut"], p.widen(p.shl32(&plane_bytes, 1)));
    let v_addr = p.add64(&v_base, p.widen(p.add32(p.mul32(&head_rows, p.imm32(32)), key_position(&p, &token))));
    for i in 0..8 {
        for b in 0..4 {
            let byte = p.reg("b32");
            if b == 0 {
                p.emit(format!("mov.b32 {byte}, {};", vc[i]));
            } else {
                p.emit(format!("shr.u32 {byte}, {}, {};", vc[i], 8 * b));
            }
            let addr = p.add64(&v_addr, p.widen(p.mul32(&pm["padded"], p.imm32(4 * i + b))));
            p.emit(format!("st.global.u8 [{addr}], {byte};"));
        }
    }
    let p_signal_on = p.setp("ne.u64", &pm["pSignal"], "0");
    sync_signal(&p, &[&pm["pSignal"]], &tid, Some(&p_signal_on));
    p.emit("ret;");
    (name, p.finish(), 0)
}

/// The attention pass: its name, PTX and (static) shared memory size.
pub fn generate_attention(stages: usize) -> (String, String, usize) {
    let name = "global_attention_stream_e4m3".to_string();
    let shared_bytes = STAGE0 + stages * STAGE_BYTES;
    let p = Ptx::new();
    let params = [
        ("u64", "pNorm"),
        ("u64", "pOut"),
        ("u32", "tokens"),
        ("u32", "padded"),
        ("u32", "heads"),
        ("u64", "pWait"),
        ("u32", "waitExpected"),
        ("u64", "pSignal"),
        ("u64", "pError"),
    ];
    p.entry(&name, &params, shared_bytes, Some(128), None, false);
    let pm: HashMap<&str, String> = params.iter().map(|&(t, n)| (n, if t == "u64" { p.load_param_u64(n) } else { p.load_param_u32(n) })).collect();
    let tid = p.special("tid.x");
    let head = p.special("ctaid.x");
    let qb = p.special("ctaid.y");
    let lane = p.and32(&tid, 31);
    let warp = p.shr32(&tid, 5);
    let li = LaneInfo::new(&p, &lane);
    let (g, t) = (&li.g, &li.t);
    let smem = p.shared_addr(p.imm32(0));
    let zero32 = p.imm32(0);
    let channels = p.mul32(&pm["heads"], p.imm32(32));
    let blocks = p.shr32(&pm["padded"], 6);
    let p_wait_on = p.setp("ne.u64", &pm["pWait"], "0");
    sync_wait(&p, &pm["pWait"], &zero32, &zero32, Expected::Reg(&pm["waitExpected"]), &lane, Some(&p_wait_on), Some(&warp), &pm["pError"]);
    // ---- source addresses (this head)
    let head_rows = p.mul32(&head, &pm["padded"]);
    let plane_bytes = p.mul32(p.mul32(&pm["heads"], &pm["padded"]), p.imm32(32));
    let q_rows_addr = p.add64(&pm["pNorm"], p.widen(p.shl32(p.add32(&head_rows, p.shl32(&qb, 6)), 5))); // Q rows of the query block
    let k_plane = p.add64(&pm["pNorm"], p.widen(&plane_bytes));
    let k_head = p.add64(&k_plane, p.widen(p.shl32(&head_rows, 5))); // K rows of the head
    let v_head = p.add64(p.add64(&pm["pNorm"], p.widen(p.shl32(&plane_bytes, 1))), p.widen(p.mul32(&head_rows, p.imm32(32)))); // V^T rows
                                                                                                                               // per-thread copy assignments: K / Q chunk (row t >> 1, chunk t & 1, swizzled), V^T chunk (dim t >> 2, chunk t & 3)
    let c_row = p.shr32(&tid, 1);
    let c_chunk = p.and32(&tid, 1);
    let c_swz = p.and32(p.shr32(&c_row, 2), 1);
    let k_dst_off = p.add32(p.shl32(&c_row, 5), p.shl32(p.xor32(&c_chunk, &c_swz), 4)); // within a K / Q tile
    let k_src_off = p.widen(p.add32(p.shl32(&c_row, 5), p.shl32(&c_chunk, 4))); // within a 64-row block
    let v_dim = p.shr32(&tid, 2);
    let v_chunk = p.and32(&tid, 3);
    let v_dst_off = p.add32(p.mul32(&v_dim, p.imm32(VT_STRIDE)), p.shl32(&v_chunk, 4));
    let v_src = p.add64(&v_head, p.widen(p.add32(p.mul32(&v_dim, &pm["padded"]), p.shl32(&v_chunk, 4)))); // + 64 * block
    let k_src = p.add64(&k_head, &k_src_off); // + 2048 * block
                                              // Q tile
    let q_dst = p.add32(&smem, p.add32(p.imm32(QTILE), &k_dst_off));
    let q_src = p.add64(&q_rows_addr, &k_src_off);
    p.emit(format!("cp.async.cg.shared.global [{q_dst}], [{q_src}], 16;"));

    // cp.async of key block `block` (register) into stage `stage` (literal), guarded by block < blocks; one group.
    let issue = |block: &str, stage: usize| {
        let ok = p.setp("lt.u32", block, &blocks);
        let k_s = p.add64(&k_src, p.widen(p.shl32(block, 11)));
        let v_s = p.add64(&v_src, p.widen(p.shl32(block, 6)));
        let k_d = p.add32(&smem, p.add32(p.imm32(STAGE0 + stage * STAGE_BYTES), &k_dst_off));
        let v_d = p.add32(&smem, p.add32(p.imm32(STAGE0 + stage * STAGE_BYTES + K_BYTES), &v_dst_off));
        p.emit(format!("@{ok} cp.async.cg.shared.global [{k_d}], [{k_s}], 16;"));
        p.emit(format!("@{ok} cp.async.cg.shared.global [{v_d}], [{v_s}], 16;"));
        p.emit("cp.async.commit_group;");
    };
    for s in 0..stages - 1 {
        issue(&p.imm32(s), s);
    }
    // ---- fragments
    let l8 = p.and32(&lane, 7);
    let lm = p.shr32(&lane, 3);
    let a_row_l = p.add32(&l8, p.shl32(p.and32(&lm, 1), 3));
    let a_chunk_l = p.shr32(&lane, 4);
    let a_pch_l = p.xor32(&a_chunk_l, p.and32(p.shr32(&a_row_l, 2), 1));
    let a_lane = p.add32(&smem, p.add32(p.imm32(QTILE), p.add32(p.shl32(p.add32(&a_row_l, p.shl32(&warp, 4)), 5), p.shl32(&a_pch_l, 4))));
    let b_nl = p.add32(&l8, p.shl32(p.shr32(&lane, 4), 3));
    let b_chunk_l = p.and32(&lm, 1);
    let b_pch_l = p.xor32(&b_chunk_l, p.and32(p.shr32(&b_nl, 2), 1));
    let k_lane = p.add32(&smem, p.add32(p.imm32(STAGE0), p.add32(p.shl32(&b_nl, 5), p.shl32(&b_pch_l, 4))));
    let v_lane = p.add32(&smem, p.add32(p.imm32(STAGE0 + K_BYTES), p.add32(p.mul32(&b_nl, p.imm32(VT_STRIDE)), p.shl32(&b_chunk_l, 4))));
    let k = VitExpConsts::new(&p);
    let o: Vec<Vec<String>> = (0..4).map(|_| zero_tile(&p)).collect();
    let total = [p.reg("f16"), p.reg("f16")];
    for tot in &total {
        p.emit(format!("mov.b16 {tot}, 0;"));
    }
    let mut mq: Option<Vec<String>> = None;
    // ---- the key-block loop, unrolled `stages` times so every stage address is a literal
    let b = p.reg("b32");
    p.emit(format!("mov.u32 {b}, 0;"));
    let loop_l = p.label("BLOCKS");
    let done_l = p.label("BLOCKSDONE");
    p.emit(format!("{loop_l}:"));
    for s in 0..stages {
        let p_b = p.setp("lt.u32", &b, &blocks);
        p.emit(format!("@!{p_b} bra {done_l};"));
        p.emit(format!("cp.async.wait_group {};", stages - 2));
        p.emit("bar.sync 0;");
        // refill the stage consumed last iteration with block b + stages - 1
        issue(&p.add32(&b, p.imm32(stages - 1)), (s + stages - 1) % stages);
        // Q fragment (the tile arrived with the first group)
        let mq = mq.get_or_insert_with(|| ldm4(&p, &a_lane));
        let st = s * STAGE_BYTES;
        let s_tiles: Vec<Vec<String>> = (0..8).map(|_| zero_tile(&p)).collect();
        for m2 in 0..4 {
            let r = ldm4(&p, format!("{k_lane}+{}", st + m2 * 512));
            mma_e4(&p, &s_tiles[2 * m2], mq, &r[0..2], &s_tiles[2 * m2]);
            mma_e4(&p, &s_tiles[2 * m2 + 1], mq, &r[2..4], &s_tiles[2 * m2 + 1]);
        }
        let e: Vec<Vec<String>> = (0..8).map(|j| (0..2).map(|h| vit_exp(&p, &s_tiles[j][h], &k)).collect()).collect();
        for (h, tot) in total.iter().enumerate() {
            let bp: Vec<String> = (0..4).map(|j| hadd2(&p, &e[2 * j][h], &e[2 * j + 1][h])).collect();
            let tp = hadd2(&p, hadd2(&p, hadd2(&p, &bp[0], &bp[1]), &bp[2]), &bp[3]);
            let tq: Vec<String> = li.quad_lanes.iter().map(|src| shfl_idx(&p, &tp, src)).collect();
            let parts: Vec<(String, String)> = tq.iter().map(|x| unpack16(&p, x)).collect();
            let even = hadd(&p, hadd(&p, hadd(&p, &parts[0].0, &parts[1].0), &parts[2].0), &parts[3].0);
            let odd = hadd(&p, hadd(&p, hadd(&p, &parts[0].1, &parts[1].1), &parts[2].1), &parts[3].1);
            let block_sum = hadd(&p, even, odd);
            p.emit(format!("add.rn.f16 {tot}, {tot}, {block_sum};"));
        }
        for kb in 0..2 {
            let c: Vec<Vec<String>> = (0..4).map(|j| (0..2).map(|h| cvt_e4x2(&p, &e[4 * kb + j][h])).collect()).collect();
            let pa = vec![pack16(&p, &c[0][0], &c[1][0]), pack16(&p, &c[0][1], &c[1][1]), pack16(&p, &c[2][0], &c[3][0]), pack16(&p, &c[2][1], &c[3][1])];
            for d2 in 0..2 {
                let r = ldm4(&p, format!("{v_lane}+{}", st + kb * 32 + d2 * 16 * VT_STRIDE));
                mma_e4(&p, &o[2 * d2], &pa, &r[0..2], &o[2 * d2]);
                mma_e4(&p, &o[2 * d2 + 1], &pa, &r[2..4], &o[2 * d2 + 1]);
            }
        }
        p.emit(format!("add.u32 {b}, {b}, 1;"));
    }
    p.emit(format!("bra {loop_l};"));
    p.emit(format!("{done_l}:"));
    p.emit("cp.async.wait_group 0;");
    // ---- padding correction, reciprocal, output (as the resident kernel)
    let padding = p.sub32(&pm["padded"], &pm["tokens"]);
    let p_pad = p.setp("ne.u32", &padding, &zero32);
    let exp_zero = vit_exp(&p, &zero32, &k);
    let (ez, _) = unpack16(&p, exp_zero);
    let ezf = p.reg("f32");
    p.emit(format!("cvt.f32.f16 {ezf}, {ez};"));
    let padf = p.reg("f32");
    p.emit(format!("cvt.rn.f32.u32 {padf}, {padding};"));
    let corr_f = p.reg("f32");
    p.emit(format!("mul.rn.f32 {corr_f}, {ezf}, {padf};"));
    let corr = p.reg("f16");
    p.emit(format!("cvt.rn.f16.f32 {corr}, {corr_f};"));
    for tot in &total {
        let corrected = p.reg("f16");
        p.emit(format!("sub.rn.f16 {corrected}, {tot}, {corr};"));
        p.emit(format!("@{p_pad} mov.b16 {tot}, {corrected};"));
    }
    for (h, tot) in total.iter().enumerate() {
        let f = p.reg("f32");
        p.emit(format!("cvt.f32.f16 {f}, {tot};"));
        let rc = p.reg("f32");
        p.emit(format!("rcp.rn.f32 {rc}, {f};"));
        let rh = p.reg("f16");
        p.emit(format!("cvt.rn.f16.f32 {rh}, {rc};"));
        let rf = p.reg("f32");
        p.emit(format!("cvt.f32.f16 {rf}, {rh};"));
        let token = p.add32(p.shl32(&qb, 6), p.add32(p.shl32(&warp, 4), p.add32(g, p.imm32(8 * h))));
        let ok = p.setp("lt.u32", &token, &pm["tokens"]);
        let out_addr = p.add64(&pm["pOut"], p.widen(p.add32(p.mul32(&token, &channels), p.add32(p.shl32(&head, 5), p.shl32(t, 1)))));
        for j in 0..4 {
            let (lo, hi) = unpack16(&p, &o[j][h]);
            let flo = p.reg("f32");
            p.emit(format!("cvt.f32.f16 {flo}, {lo};"));
            p.emit(format!("mul.rn.f32 {flo}, {flo}, {rf};"));
            let fhi = p.reg("f32");
            p.emit(format!("cvt.f32.f16 {fhi}, {hi};"));
            p.emit(format!("mul.rn.f32 {fhi}, {fhi}, {rf};"));
            let v = p.reg("b32");
            p.emit(format!("cvt.rn.f16x2.f32 {v}, {fhi}, {flo};"));
            let hc = cvt_e4x2(&p, v);
            p.emit(format!("@{ok} st.global.b16 [{out_addr}+{}], {hc};", 8 * j));
        }
    }
    let p_signal_on = p.setp("ne.u64", &pm["pSignal"], "0");
    sync_signal(&p, &[&pm["pSignal"]], &tid, Some(&p_signal_on));
    p.emit("ret;");
    (name, p.finish(), shared_bytes)
}

/// `normalize|attention [stages]`
pub fn cli(args: &[&str]) -> String {
    let stages = args.get(1).map_or(4, |s| crate::int(s));
    if crate::req_str(args, 0) == "normalize" { generate_normalize() } else { generate_attention(stages) }.1
}
