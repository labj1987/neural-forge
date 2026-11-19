//! Tall-tile FP8 GEMM for the few-row stages (ViT: 192 tokens): 192 x 64 output tiles (12 warps), partitions and
//! split-K whose partial sums are added in partition order, which is the association the unsplit chain has.
//!
//! D = A[rows][K] x W (E4, k32-tile-major [K/32][Nmatrix][32]), f16 accumulation chained in K order per partition:
//! total = ((seed + chain_0) + chain_1) + ... where chain_p is the MMA chain over partition p (PARTITION k values) and
//! seed = round_f16(E4 residual * scale) (flag RES; split 0 only). Split-K: workgroup z runs partitions
//! [z * partsPerSplit, (z + 1) * partsPerSplit) and stores its f16 total to the partial buffer slice z; the reduce
//! kernel sums the slices in order and publishes (that is `total = total + acc` of gemm_fp8 / gemm_reduce).
//! Without splits the epilogue publishes directly: SILU, E4 and / or F16 outputs.
//! The weight tile (2 KB) is read once per row group: for 192-row problems the operand traffic is the weights once
//! plus A once per column group.
//!
//! Arguments: `<K> <flags> <partitionSteps> <splits> [maxregs] [stages] [prefetch]`   flags: RES=1 SILU=2 E4=4 F16=8

use std::collections::HashMap;

use crate::ptxgen::Ptx;
use crate::swin::*;

pub const F_RES: usize = 1;
pub const F_SILU: usize = 2;
pub const F_E4: usize = 4;
pub const F_F16: usize = 8;
pub const BM: usize = 192;
pub const BN: usize = 64;
pub const WARPS: usize = BM / 16;
pub const THREADS: usize = 32 * WARPS;
pub const STAGES: usize = 3;
/// 8 KB
pub const STAGE_BYTES: usize = (BM + BN) * 32;

/// weight k32 steps prefetched into L2 ahead of the cp.async issue (DRAM latency hiding)
pub const PREFETCH: usize = 8;

/// The kernel's name, PTX and shared memory size.
pub fn generate(
    k_dim: usize,
    flags: usize,
    part_steps: usize,
    splits: usize,
    max_regs: Option<u32>,
    stages_wanted: usize,
    prefetch: usize,
) -> (String, String, usize) {
    assert!(k_dim.is_multiple_of(32));
    let steps = k_dim / 32;
    let part_steps = if part_steps == 0 { steps } else { part_steps };
    assert!(steps.is_multiple_of(part_steps) && (splits == 1 || steps == part_steps * splits)); // split-K: one partition per split
    let steps_per_split = steps / splits;
    let stages = stages_wanted.min(steps_per_split);
    let (res, silu_f, out_e4, out_f16) = (flags & F_RES != 0, flags & F_SILU != 0, flags & F_E4 != 0, flags & F_F16 != 0);
    let name = format!("gemmt_e4m3_K{k_dim}_f{flags}_p{part_steps}_s{splits}");
    let ring = 0;
    let _outs = 0; // the output tile aliases the ring after the loop
    let has_f16 = splits > 1 || out_f16;
    let e4_off = if has_f16 { BM * 128 } else { 0 }; // the E4 tile [192][64 B] follows the f16 tile [192][128 B] or aliases the ring
    let shared_bytes = (stages * STAGE_BYTES).max(if has_f16 { BM * 128 } else { 0 }) + if out_e4 && splits == 1 && has_f16 { BM * 64 } else { 0 };
    assert!(shared_bytes >= e4_off + BM * 64 || !(out_e4 && splits == 1));
    let p = Ptx::new();
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
        ("u32", "splitStride"),
    ];
    p.entry(&name, &params, shared_bytes, Some(THREADS as u32), max_regs, true);
    let pm: HashMap<&str, String> = params.iter().map(|&(t, n)| (n, if t == "u64" { p.load_param_u64(n) } else { p.load_param_u32(n) })).collect();
    let tid = p.special("tid.x");
    let col_group = p.special("ctaid.x");
    let row_group = p.special("ctaid.y");
    let split = p.special("ctaid.z");
    let lane = p.and32(&tid, 31);
    let warp = p.shr32(&tid, 5);
    let li = LaneInfo::new(&p, &lane);
    let (g, t) = (&li.g, &li.t);
    let smem = p.shared_addr(p.imm32(0));
    let zero = imm(&p, 0);
    let block_row = p.mul32(&row_group, BM);
    let col_base = p.shl32(&col_group, 6);
    let rows_minus1 = p.sub32(&pm["rows"], 1);
    let k_step0 = p.mul32(&split, steps_per_split); // first k32 step of this split

    // ---- cp.async chunks per stage: A 384 (one per thread), W 128 (threads < 128); chunk ^= (row >> 2) & 1
    let a_row = p.shr32(&tid, 1);
    let a_chunk = p.and32(&tid, 1);
    let a_pch = p.xor32(&a_chunk, p.and32(p.shr32(&a_row, 2), 1));
    let a_dst = p.add32(p.shl32(&a_row, 5), p.shl32(&a_pch, 4));
    let g_row = p.add32(&block_row, &a_row);
    let g_row_c = p.selp32(p.setp("lt.u32", &g_row, &pm["rows"]), &g_row, &rows_minus1);
    let a_src = p.add64(&pm["pA"], p.widen(p.add32(p.mul32(&g_row_c, &pm["inputStride"]), p.add32(&pm["inputColumnBase"], p.shl32(&a_chunk, 4)))));
    let p_b = p.setp("lt.u32", &tid, 128);
    let n = p.shr32(&tid, 1);
    let b_chunk = p.and32(&tid, 1);
    let b_pch_c = p.xor32(&b_chunk, p.and32(p.shr32(&n, 2), 1));
    let b_dst = p.add32(BM * 32, p.add32(p.shl32(&n, 5), p.shl32(&b_pch_c, 4)));
    let w_row = p.add32(&pm["weightColumnOffset"], p.add32(&col_base, &n));
    let b_src = p.add64(&pm["pW"], p.widen(p.add32(p.shl32(&w_row, 5), p.shl32(&b_chunk, 4))));
    let w_step = p.shl32(&pm["Nmatrix"], 5);
    // L2 prefetch of the weight tile `prefetch` steps ahead of its cp.async: the 2 KB tile is 16 lines, threads 0..15
    let p_pf = p.setp("lt.u32", &tid, 16);
    let pf_base = p.add64(&pm["pW"], p.widen(p.add32(p.shl32(p.add32(&pm["weightColumnOffset"], &col_base), 5), p.shl32(&tid, 7))));

    let prefetch_step = |step_local: usize| {
        if prefetch == 0 || step_local >= steps_per_split {
            return;
        }
        let step_g = p.add32(&k_step0, p.imm32(step_local));
        let wide = p.mad_wide(&w_step, &step_g, &pf_base);
        p.emit(format!("@{p_pf} prefetch.global.L2 [{wide}];"));
    };

    let issue_stage = |stage_index: usize, step_local: usize| {
        let step_g = p.add32(&k_step0, p.imm32(step_local));
        let sa = p.add32(p.add32(&smem, &a_dst), p.imm32(ring + stage_index * STAGE_BYTES));
        let ga = p.add64(&a_src, p.widen(p.shl32(&step_g, 5)));
        p.emit(format!("cp.async.cg.shared.global [{sa}], [{ga}], 16;"));
        let sb = p.add32(p.add32(&smem, &b_dst), p.imm32(ring + stage_index * STAGE_BYTES));
        let gb = p.mad_wide(&w_step, &step_g, &b_src);
        p.emit(format!("@{p_b} cp.async.cg.shared.global [{sb}], [{gb}], 16;"));
        p.emit("cp.async.commit_group;");
        prefetch_step(step_local + prefetch);
    };

    let l8 = p.and32(&lane, 7);
    let lm = p.shr32(&lane, 3);
    let a_row_l = p.add32(&l8, p.shl32(p.and32(&lm, 1), 3));
    let a_chunk_l = p.shr32(&lane, 4);
    let a_pch_l = p.xor32(&a_chunk_l, p.and32(p.shr32(&a_row_l, 2), 1));
    let a_lane = p.add32(&smem, p.add32(p.shl32(p.add32(&a_row_l, p.shl32(&warp, 4)), 5), p.shl32(&a_pch_l, 4)));
    let b_nl = p.add32(&l8, p.shl32(p.shr32(&lane, 4), 3));
    let b_chunk_l = p.and32(&lm, 1);
    let b_pch_l = p.xor32(&b_chunk_l, p.and32(p.shr32(&b_nl, 2), 1));
    let b_lane = p.add32(&smem, p.add32(BM * 32, p.add32(p.shl32(&b_nl, 5), p.shl32(&b_pch_l, 4))));

    let out_row: Vec<String> = (0..2).map(|h| p.add32(&block_row, p.add32(p.shl32(&warp, 4), p.add32(g, p.imm32(8 * h))))).collect();
    let row_ok: Vec<String> = (0..2).map(|h| p.setp("lt.u32", &out_row[h], &pm["rows"])).collect();
    let out_col = p.add32(&pm["outputColumnOffset"], p.add32(&col_base, p.shl32(t, 1)));
    let out_idx: Vec<String> = (0..2).map(|h| p.add32(p.mul32(&out_row[h], &pm["outputStride"]), &out_col)).collect();

    for s in (stages - 1).max(1)..(stages - 1).max(1) + prefetch {
        prefetch_step(s);
    }
    for s in 0..(stages - 1).max(1) {
        issue_stage(s, s);
    }
    // ---- accumulators: partition chain `acc`, running `total`; the seed only in split 0's first partition
    let acc: Vec<Vec<String>> = (0..8).map(|_| zero_tile(&p)).collect();
    if res {
        let p_seed = p.setp("eq.u32", &split, 0);
        let aux_addr = p.add64(&pm["pAux"], p.widen(p.shl32(p.add32(&pm["auxHalfOffset"], p.add32(&col_base, p.shl32(t, 1))), 1)));
        let mut scales: Vec<String> = Vec::new();
        for j in 0..8 {
            let r = p.reg("b32");
            p.emit(format!("ld.global.nc.b32 {r}, [{aux_addr}+{}];", 16 * j));
            scales.push(r);
        }
        let res_addr: Vec<String> = (0..2).map(|h| p.add64(&pm["pRes"], p.widen(&out_idx[h]))).collect();
        for j in 0..8 {
            for h in 0..2 {
                let hc = p.reg("b16");
                p.emit(format!("mov.b16 {hc}, 0;"));
                let pr = p.reg("pred");
                p.emit(format!("and.pred {pr}, {}, {p_seed};", row_ok[h]));
                p.emit(format!("@{pr} ld.global.cg.b16 {hc}, [{}+{}];", res_addr[h], 8 * j));
                let v = hmul2(&p, e4x2_to_f16x2(&p, &hc), &scales[j]);
                p.emit(format!("mov.b32 {}, {v};", acc[j][h]));
            }
        }
    }
    let mut total: Vec<Vec<String>> = (0..8).map(|_| zero_tile(&p)).collect();
    for step in 0..steps_per_split {
        let stage_index = step % stages;
        p.emit(format!("cp.async.wait_group {};", stages.saturating_sub(2)));
        p.emit("bar.sync 0;");
        if stages > 1 && step + stages - 1 < steps_per_split {
            issue_stage((step + stages - 1) % stages, step + stages - 1);
        } else if stages > 1 {
            p.emit("cp.async.commit_group;");
        }
        let a = ldm4(&p, p.add32(&a_lane, stage_index * STAGE_BYTES));
        for i2 in 0..4 {
            let r = ldm4(&p, p.add32(&b_lane, stage_index * STAGE_BYTES + i2 * 512));
            mma_e4(&p, &acc[2 * i2], &a, &r[0..2], &acc[2 * i2]);
            mma_e4(&p, &acc[2 * i2 + 1], &a, &r[2..4], &acc[2 * i2 + 1]);
        }
        // partition boundary: total = (first partition of this split ? acc : total + acc); acc = 0
        if (step + 1).is_multiple_of(part_steps) {
            let first_part = (step + 1) / part_steps == 1;
            for j in 0..8 {
                for h in 0..2 {
                    if first_part {
                        p.emit(format!("mov.b32 {}, {};", total[j][h], acc[j][h]));
                    } else {
                        let s2 = hadd2(&p, &total[j][h], &acc[j][h]);
                        p.emit(format!("mov.b32 {}, {s2};", total[j][h]));
                    }
                    p.emit(format!("mov.b32 {}, 0;", acc[j][h]));
                }
            }
        }
    }
    // ---- epilogue (the ring is dead after this barrier: the output tile aliases it)
    if silu_f && splits == 1 {
        let k = SiluConsts::new(&p);
        for j in 0..8 {
            for h in 0..2 {
                total[j][h] = silu(&p, &total[j][h], &k);
            }
        }
    }
    p.emit("bar.sync 0;");
    if splits > 1 || out_f16 {
        // f16 tile [192][128 B] then 16-byte stores (1536 chunks, 4 per thread)
        let c_out16 = p.add32(&smem, p.add32(p.shl32(p.add32(p.shl32(&warp, 4), g), 7), p.shl32(t, 2)));
        for j in 0..8 {
            for h in 0..2 {
                p.emit(format!("st.shared.b32 [{c_out16}+{}], {};", 16 * j + 1024 * h, total[j][h]));
            }
        }
    }
    if out_e4 && splits == 1 {
        let c_out = p.add32(&smem, p.add32(p.imm32(e4_off), p.add32(p.shl32(p.add32(p.shl32(&warp, 4), g), 6), p.shl32(t, 1))));
        for j in 0..8 {
            for h in 0..2 {
                let hc = cvt_e4x2(&p, &total[j][h]);
                p.emit(format!("st.shared.b16 [{c_out}+{}], {hc};", 8 * j + 512 * h));
            }
        }
    }
    p.emit("bar.sync 0;");
    if splits > 1 || out_f16 {
        let mut dst16 = pm["pOut16"].clone();
        if splits > 1 {
            dst16 = p.add64(&dst16, p.widen(p.shl32(p.mul32(&split, &pm["splitStride"]), 1)));
        }
        for i in 0..4 {
            let c = p.add32(&tid, i * THREADS);
            let row = p.shr32(&c, 3);
            let chunk = p.and32(&c, 7);
            let g_row_o = p.add32(&block_row, &row);
            let ok = p.setp("lt.u32", &g_row_o, &pm["rows"]);
            let r = p.regs("b32", 4);
            let ld_addr = p.add32(&smem, p.add32(p.shl32(&row, 7), p.shl32(&chunk, 4)));
            p.emit(format!("ld.shared.v4.b32 {{{}, {}, {}, {}}}, [{ld_addr}];", r[0], r[1], r[2], r[3]));
            // splits > 1: the host passes outputStride = N, outputColumnOffset = 0 (the partial slice layout)
            let g_off = p.add32(p.mul32(&g_row_o, &pm["outputStride"]), p.add32(&pm["outputColumnOffset"], p.add32(&col_base, p.shl32(&chunk, 3))));
            let addr = p.add64(&dst16, p.widen(p.shl32(&g_off, 1)));
            p.emit(format!("@{ok} st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
        }
    }
    if out_e4 && splits == 1 {
        for i in 0..2 {
            // 768 chunks of 16 B: 2 per thread
            let c = p.add32(&tid, i * THREADS);
            let row = p.shr32(&c, 2);
            let chunk = p.and32(&c, 3);
            let g_row_o = p.add32(&block_row, &row);
            let ok = p.setp("lt.u32", &g_row_o, &pm["rows"]);
            let r = p.regs("b32", 4);
            let ld_addr = p.add32(&smem, p.add32(p.imm32(e4_off), p.add32(p.shl32(&row, 6), p.shl32(&chunk, 4))));
            p.emit(format!("ld.shared.v4.b32 {{{}, {}, {}, {}}}, [{ld_addr}];", r[0], r[1], r[2], r[3]));
            let g_off = p.add32(p.mul32(&g_row_o, &pm["outputStride"]), p.add32(&pm["outputColumnOffset"], p.add32(&col_base, p.shl32(&chunk, 4))));
            let addr = p.add64(&pm["pOut"], p.widen(&g_off));
            p.emit(format!("@{ok} st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
        }
    }
    p.emit("ret;");
    let _ = &zero;
    (name, p.finish(), shared_bytes)
}

/// `<K> <flags> <partitionSteps> <splits> [maxregs] [stages] [prefetch]`
pub fn cli(args: &[&str]) -> String {
    let (k_dim, flags, part_steps, splits) = (crate::int(args[0]), crate::int(args[1]), crate::int(args[2]), crate::int(args[3]));
    let max_regs = args.get(4).map(|a| crate::int(a) as u32);
    let stages_wanted = args.get(5).map_or(STAGES, |a| crate::int(a));
    let prefetch = args.get(6).map_or(PREFETCH, |a| crate::int(a));
    generate(k_dim, flags, part_steps, splits, max_regs, stages_wanted, prefetch).1
}
