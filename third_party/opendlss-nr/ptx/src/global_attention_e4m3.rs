//! Global ViT cosine attention (global_attention.comp) for one (head, 64-query block): the head's K/V rows of all
//! padded tokens are normalized / quantized into shared memory (K rows [token][32 B], V transposed [dim][token] with the
//! keys of every 32-group permuted so that the exp'd score fragments pack straight into the P·V A operand), the four
//! warps normalize their 16 query rows, then per 64-key block: S = Q K^T, the ViT exponential (f16 fma, clamp, bit
//! shift), the specified block sum ((b0 + b1) + b2) + b3 / ((t0 + t1) + t2) + t3 / even + odd, running total per row,
//! unnormalized E4 weights times V accumulated over the blocks in key order; the padding correction and the f32
//! reciprocal (rounded to half) are applied afterwards. Arithmetic identical to the GLSL kernel.
//!
//! Chaining: waits for `waitExpected` signals at pWait (the QKV GEMM), signals pSignal once per workgroup.
//!
//! Arguments: `<paddedTokens>`

use std::collections::HashMap;

use crate::ptxgen::Ptx;
use crate::swin::*;

pub const THREADS: usize = 128;
/// [64][32 B] swizzled (chunk ^= (row >> 2) & 1)
pub const QROWS: usize = 0;
/// [padded][32 B] swizzled
pub const KROWS: usize = 2048;

/// The kernel's name, PTX and shared memory size.
pub fn generate(padded: usize) -> (String, String, usize) {
    assert!(padded.is_multiple_of(64) && padded <= 256);
    let blocks = padded / 64;
    let vt = KROWS + padded * 32; // [32 dims][VT_STRIDE B]
    let vt_stride = padded + 16; // (stride / 4) mod 32 = 4 or 20: conflict-free ldmatrix rows
    let shared_bytes = vt + 32 * vt_stride;
    let name = format!("global_attention_e4m3_p{padded}");
    let p = Ptx::new();
    let params = [
        ("u64", "pQkv"),
        ("u64", "pAux"),
        ("u64", "pOut"),
        ("u32", "tokens"),
        ("u32", "heads"),
        ("u32", "scaleWordOffset"),
        ("u64", "pWait"),
        ("u32", "waitExpected"),
        ("u64", "pSignal"),
        ("u64", "pError"),
    ];
    p.entry(&name, &params, shared_bytes, Some(THREADS as u32), None, true);
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
    let row_bytes = p.mul32(&pm["heads"], p.imm32(192)); // f16 [heads * 96] per token
    let head_byte = p.mul32(&head, p.imm32(192));
    // per-head learned scale (f32 -> f16), f16(sqrt 32) = 0x45A8
    let learned = p.reg("f32");
    let learned_addr = p.add64(&pm["pAux"], p.widen(p.shl32(p.add32(&pm["scaleWordOffset"], &head), 2)));
    p.emit(format!("ld.global.nc.f32 {learned}, [{learned_addr}];"));
    let learned_h = p.reg("f16");
    p.emit(format!("cvt.rn.f16.f32 {learned_h}, {learned};"));
    let learned2 = pack16(&p, &learned_h, &learned_h);
    let hs2 = imm(&p, "0x45A845A8");
    // ---- chain wait (the QKV rows are produced by the chained GEMM)
    let p_wait_on = p.setp("ne.u64", &pm["pWait"], "0");
    sync_wait(&p, &pm["pWait"], &zero32, &zero32, Expected::Reg(&pm["waitExpected"]), &lane, Some(&p_wait_on), Some(&warp), &pm["pError"]);

    // 16 words of f16 (part 0 = Q, 1 = K, 2 = V of the head) for `token` (zeros when not valid).
    let load_row = |token: &str, valid: &str, part: usize| -> Vec<String> {
        let addr = p.add64(&pm["pQkv"], p.widen(p.add32(p.mul32(token, &row_bytes), p.add32(&head_byte, p.imm32(64 * part)))));
        let mut w = Vec::new();
        for c in 0..4 {
            let r = p.regs("b32", 4);
            for x in &r {
                p.emit(format!("mov.b32 {x}, 0;"));
            }
            p.emit(format!("@{valid} ld.global.cg.v4.b32 {{{}, {}, {}, {}}}, [{addr}+{}];", r[0], r[1], r[2], r[3], 16 * c));
            w.extend(r);
        }
        w
    };

    // 8 words of E4 codes -> a [row][32 B] swizzled tile at `base`.
    let store_row16 = |base: &str, row: &str, codes: &[String]| {
        let swz = p.and32(p.shr32(row, 2), 1);
        for c in 0..2 {
            let addr = p.add32(base, p.add32(p.shl32(row, 5), p.shl32(p.xor32(p.imm32(c), &swz), 4)));
            p.emit(format!("st.shared.v4.b32 [{addr}], {{{}, {}, {}, {}}};", codes[4 * c], codes[4 * c + 1], codes[4 * c + 2], codes[4 * c + 3]));
        }
    };

    // ---- Q of this workgroup's 64 tokens (threads 0..63)
    let p_q = p.setp("lt.u32", &tid, p.imm32(64));
    let q_token = p.add32(p.shl32(&qb, 6), &tid);
    let q_valid = p.reg("pred");
    p.emit(format!("setp.lt.and.u32 {q_valid}, {q_token}, {}, {p_q};", pm["tokens"]));
    let q_l = p.label("QDONE");
    p.emit(format!("@!{p_q} bra {q_l};"));
    let wq = load_row(&q_token, &q_valid, 0);
    let qc = vit_quantize(&p, &wq, &vit_norm(&p, &wq), true, &hs2, &learned2, &zero32);
    let qc: Vec<String> = qc.iter().map(|c| p.selp32(&q_valid, c, &zero32)).collect();
    store_row16(&p.add32(&smem, p.imm32(QROWS)), &tid, &qc);
    p.emit(format!("{q_l}:"));
    // ---- K / V of all padded tokens: thread t handles tokens t, t + 128; V transposed with the 32-group permutation
    // physical position of key k: (k & ~31) | (k & 16) | (((k & 7) >> 1) << 2) | (k & 1) | (((k & 15) >> 3) << 1)
    for local0 in (0..padded).step_by(THREADS) {
        let token = p.add32(&tid, p.imm32(local0));
        let p_in = if local0 + THREADS > padded { Some(p.setp("lt.u32", &token, p.imm32(padded))) } else { None };
        let valid = p.reg("pred");
        match &p_in {
            Some(pi) => p.emit(format!("setp.lt.and.u32 {valid}, {token}, {}, {pi};", pm["tokens"])),
            None => p.emit(format!("setp.lt.u32 {valid}, {token}, {};", pm["tokens"])),
        }
        let skip_l = p.label("KVSKIP");
        if let Some(pi) = &p_in {
            p.emit(format!("@!{pi} bra {skip_l};"));
        }
        let wk = load_row(&token, &valid, 1);
        let wv = load_row(&token, &valid, 2);
        let kc = vit_quantize(&p, &wk, &vit_norm(&p, &wk), false, &hs2, &learned2, &zero32);
        let kc: Vec<String> = kc.iter().map(|c| p.selp32(&valid, c, &zero32)).collect();
        store_row16(&p.add32(&smem, p.imm32(KROWS)), &token, &kc);
        let vc: Vec<String> = (0..8).map(|i| pack16(&p, cvt_e4x2(&p, &wv[2 * i]), cvt_e4x2(&p, &wv[2 * i + 1]))).collect();
        let vc: Vec<String> = vc.iter().map(|c| p.selp32(&valid, c, &zero32)).collect();
        let ppos = p.or32(
            p.or32(p.and32(&token, p.imm32(0xffffffe0_u32 | 16)), p.shl32(p.shr32(p.and32(&token, 7), 1), 2)),
            p.or32(p.and32(&token, 1), p.shl32(p.shr32(p.and32(&token, 15), 3), 1)),
        );
        let v_addr = p.add32(&smem, p.add32(p.imm32(vt), ppos));
        for i in 0..8 {
            // word i holds dims 4 i .. 4 i + 3
            for b in 0..4 {
                let byte = p.reg("b32");
                if b == 0 {
                    p.emit(format!("mov.b32 {byte}, {};", vc[i]));
                } else {
                    p.emit(format!("shr.u32 {byte}, {}, {};", vc[i], 8 * b));
                }
                p.emit(format!("st.shared.u8 [{v_addr}+{}], {byte};", (4 * i + b) * vt_stride));
            }
        }
        p.emit(format!("{skip_l}:"));
    }
    p.emit("bar.sync 0;");
    // ---- fragments: Q A (this warp's 16 rows), K B rows, V^T B rows
    let l8 = p.and32(&lane, 7);
    let lm = p.shr32(&lane, 3);
    let a_row_l = p.add32(&l8, p.shl32(p.and32(&lm, 1), 3));
    let a_chunk_l = p.shr32(&lane, 4);
    let a_pch_l = p.xor32(&a_chunk_l, p.and32(p.shr32(&a_row_l, 2), 1));
    let a_lane = p.add32(&smem, p.add32(p.imm32(QROWS), p.add32(p.shl32(p.add32(&a_row_l, p.shl32(&warp, 4)), 5), p.shl32(&a_pch_l, 4))));
    let mq = ldm4(&p, &a_lane);
    let b_nl = p.add32(&l8, p.shl32(p.shr32(&lane, 4), 3));
    let b_chunk_l = p.and32(&lm, 1);
    let b_pch_l = p.xor32(&b_chunk_l, p.and32(p.shr32(&b_nl, 2), 1));
    let k_lane = p.add32(&smem, p.add32(p.imm32(KROWS), p.add32(p.shl32(&b_nl, 5), p.shl32(&b_pch_l, 4))));
    let v_lane = p.add32(&smem, p.add32(p.imm32(vt), p.add32(p.mul32(&b_nl, p.imm32(vt_stride)), p.shl32(&b_chunk_l, 4))));
    let k = VitExpConsts::new(&p);
    let o: Vec<Vec<String>> = (0..4).map(|_| zero_tile(&p)).collect(); // 16 x 32 output: 4 n-tiles of 8 dims
    let mut total = [String::new(), String::new()]; // f16 per row half (g, g + 8)
    for b in 0..blocks {
        let s: Vec<Vec<String>> = (0..8).map(|_| zero_tile(&p)).collect();
        for m2 in 0..4 {
            let r = ldm4(&p, format!("{k_lane}+{}", b * 2048 + m2 * 512));
            mma_e4(&p, &s[2 * m2], &mq, &r[0..2], &s[2 * m2]);
            mma_e4(&p, &s[2 * m2 + 1], &mq, &r[2..4], &s[2 * m2 + 1]);
        }
        let e: Vec<Vec<String>> = (0..8).map(|j| (0..2).map(|h| vit_exp(&p, &s[j][h], &k)).collect()).collect();
        for h in 0..2 {
            let bp: Vec<String> = (0..4).map(|j| hadd2(&p, &e[2 * j][h], &e[2 * j + 1][h])).collect();
            let tp = hadd2(&p, hadd2(&p, hadd2(&p, &bp[0], &bp[1]), &bp[2]), &bp[3]);
            let tq: Vec<String> = li.quad_lanes.iter().map(|src| shfl_idx(&p, &tp, src)).collect();
            let parts: Vec<(String, String)> = tq.iter().map(|x| unpack16(&p, x)).collect();
            let even = hadd(&p, hadd(&p, hadd(&p, &parts[0].0, &parts[1].0), &parts[2].0), &parts[3].0);
            let odd = hadd(&p, hadd(&p, hadd(&p, &parts[0].1, &parts[1].1), &parts[2].1), &parts[3].1);
            let block_sum = hadd(&p, even, odd);
            total[h] = if b == 0 { block_sum } else { hadd(&p, &total[h], block_sum) };
            // total = 0 + blockSum is exact
        }
        // P (E4 weights) as the A operand: k32 group kb = n-tiles 4 kb .. 4 kb + 3 (the V tile's key permutation)
        for kb in 0..2 {
            let c: Vec<Vec<String>> = (0..4).map(|j| (0..2).map(|h| cvt_e4x2(&p, &e[4 * kb + j][h])).collect()).collect();
            let pa = vec![pack16(&p, &c[0][0], &c[1][0]), pack16(&p, &c[0][1], &c[1][1]), pack16(&p, &c[2][0], &c[3][0]), pack16(&p, &c[2][1], &c[3][1])];
            for d2 in 0..2 {
                let r = ldm4(&p, format!("{v_lane}+{}", b * 64 + kb * 32 + d2 * 16 * vt_stride));
                mma_e4(&p, &o[2 * d2], &pa, &r[0..2], &o[2 * d2]);
                mma_e4(&p, &o[2 * d2 + 1], &pa, &r[2..4], &o[2 * d2 + 1]);
            }
        }
    }
    // ---- padding correction, reciprocal, output
    let padding = p.sub32(p.imm32(padded), &pm["tokens"]);
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

/// `<paddedTokens>`
pub fn cli(args: &[&str]) -> String {
    generate(crate::int(args[0])).1
}
