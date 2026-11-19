//! ViT FP8 GEMM (the few-row stages: 192 tokens): 192 x 128 output tiles, 8 warps of 48 x 64 (3 m-tiles x 8 n-tiles,
//! 24 MMAs per 7 ldmatrix; two warps per sub-partition so one warp's loads hide behind the other's MMAs), one f16 MMA
//! chain per workgroup (the whole K, or one partition per split-K workgroup).
//!
//! D = A[rows][K] x W (E4, k32-tile-major [K/32][Nmatrix][32]); total = ((seed + chain_0) + chain_1) + ... where chain_z
//! is split z's chain and seed = round_f16(E4 residual * scale) in split 0 (flag RES). Split-K partials are stored in
//! fragment order (one 128-byte line per warp store) and summed in slice order by the last-arriving workgroup of the
//! column group (a per-tile counter, zeroed per frame), which then publishes: SILU, E4 and / or F16 outputs. That is
//! gemm_fp8's `total = total + acc` and gemm_reduce's slice order.
//!
//! Operand staging: an A ring (stages x 6 KB; the activations are L2-resident) and a deeper weight ring (bstages x 4 KB;
//! the DRAM stream), one cp.async group per k32 step; the fragments of the next step are loaded while the current
//! step's MMAs run (the ring keeps two complete stages). Chaining: the weight staging (and an optional L2 prefetch)
//! starts before the wait for `waitExpected` signals at pWait; pSignal is incremented once per published column group.
//!
//! Arguments: `<K> <flags> <splits> [stages] [bstages] [prefetch] [maxregs] [prefetch0] [BM]`
//!   flags: RES=1 SILU=2 E4=4 F16=8

use std::cell::Cell;
use std::collections::HashMap;

use crate::ptxgen::Ptx;
use crate::swin::*;

pub const F_RES: usize = 1;
pub const F_SILU: usize = 2;
pub const F_E4: usize = 4;
pub const F_F16: usize = 8;
pub const BN: usize = 128;
/// 4 KB per k32 step
pub const B_BYTES: usize = BN * 32;
pub const STAGES: usize = 6;
pub const BSTAGES: usize = 8;
pub const PREFETCH: usize = 0;
/// weight tiles prefetched into L2 in the prologue beyond the staged ones (measured: hurts)
pub const PREFETCH0: usize = 0;

/// The kernel's name, PTX and shared memory size.
pub fn generate(
    k: usize,
    flags: usize,
    splits: usize,
    stages: usize,
    bstages: usize,
    prefetch: usize,
    max_regs: Option<u32>,
    prefetch0: usize,
    bm: usize,
) -> (String, String, usize) {
    assert!(k.is_multiple_of(32) && (k / 32).is_multiple_of(splits));
    assert!(bm.is_multiple_of(48) && matches!(bm / 48, 1 | 2 | 4));
    let row_qs = bm / 48; // 48-row warp quarters
    let (warps, threads) = (2 * row_qs, 64 * row_qs);
    let a_bytes = bm * 32; // per k32 step
    let res_bytes = bm * BN; // the residual tile [BM][128 B] (E4)
    let partial_wg_bytes = warps * 48 * 128; // the f16 partial in fragment order: [warp][48 tiles][32 lanes] x 4 B
    let steps = k / 32 / splits; // k32 steps of this workgroup's chain
    assert!(steps >= 3, "the fragment pipeline needs three stages");
    let stages = 3.max(stages.min(steps));
    let bstages = stages.max(bstages.min(steps));
    let lead = bstages - stages; // extra weight lead (steps)
    let (res, silu_f, out_e4, out_f16) = (flags & F_RES != 0, flags & F_SILU != 0, flags & F_E4 != 0, flags & F_F16 != 0);
    let name = format!("gemmv_e4m3_K{k}_f{flags}_s{splits}") + &if bm != 192 { format!("_m{bm}") } else { String::new() };
    let (a_ring, b_ring) = (0, stages * a_bytes);
    let res_off = b_ring + bstages * B_BYTES;
    let ring_bytes = res_off + if res { res_bytes } else { 0 };
    let e4_off = if out_e4 && out_f16 { bm * 256 } else { 0 }; // the E4 tile beside the f16 tile only for dual outputs
    let epilogue_bytes = (if out_f16 { bm * 256 } else { 0 }).max(e4_off + if out_e4 { bm * 128 } else { 0 });
    let shared_bytes = ring_bytes.max(epilogue_bytes);
    let p = Ptx::new();
    let params = [
        ("u64", "pA"),
        ("u64", "pW"),
        ("u64", "pRes"),
        ("u64", "pAux"),
        ("u64", "pOut"),
        ("u64", "pOut16"),
        ("u64", "pPartial"),
        ("u64", "pCount"),
        ("u32", "rows"),
        ("u32", "inputStride"),
        ("u32", "inputColumnBase"),
        ("u32", "Nmatrix"),
        ("u32", "weightColumnOffset"),
        ("u32", "outputStride"),
        ("u32", "outputColumnOffset"),
        ("u32", "auxHalfOffset"),
        ("u64", "pWait"),
        ("u32", "waitExpected"),
        ("u64", "pSignal"),
        ("u64", "pError"),
    ];
    p.entry(&name, &params, shared_bytes, Some(threads as u32), max_regs, true);
    let pm: HashMap<&str, String> = params.iter().map(|&(t, n)| (n, if t == "u64" { p.load_param_u64(n) } else { p.load_param_u32(n) })).collect();
    let tid = p.special("tid.x");
    let col_group = p.special("ctaid.x");
    let row_group = p.special("ctaid.y");
    let split = p.special("ctaid.z");
    let col_groups = p.special("nctaid.x");
    let row_groups = p.special("nctaid.y");
    let lane = p.and32(&tid, 31);
    let warp = p.shr32(&tid, 5);
    let row_q = p.and32(&warp, row_qs - 1);
    let col_h = p.shr32(
        &warp,
        match row_qs {
            1 => 0,
            2 => 1,
            _ => 2,
        },
    ); // the warp's 48-row quarter and 64-column half
    let li = LaneInfo::new(&p, &lane);
    let (g, t) = (&li.g, &li.t);
    let smem = p.shared_addr(p.imm32(0));
    let block_row = p.mul32(&row_group, p.imm32(bm));
    let col_base = p.shl32(&col_group, p.imm32(7));
    let warp_col = p.add32(&col_base, p.shl32(&col_h, p.imm32(6))); // first output column of this warp
    let rows_minus1 = p.sub32(&pm["rows"], p.imm32(1));
    let k_step0 = p.mul32(&split, p.imm32(steps)); // first k32 step of this workgroup's chain
    let zero32 = p.imm32(0);

    // ---- cp.async chunks per step: A 2 BM (chunks tid + THREADS i), W 256; a row is two 16-byte chunks,
    // chunk ^= (row >> 2) & 1 (conflict-free ldmatrix)
    let a_chunk = p.and32(&tid, 1);
    let mut a_dst: Vec<String> = Vec::new();
    let mut a_src: Vec<String> = Vec::new();
    let mut a_pred: Vec<Option<String>> = Vec::new();
    for i in 0..(2 * bm).div_ceil(threads) {
        a_pred.push(if threads * (i + 1) > 2 * bm { Some(p.setp("lt.u32", &tid, p.imm32(2 * bm - threads * i))) } else { None });
        let a_row = p.add32(p.shr32(&tid, 1), p.imm32(threads / 2 * i));
        let a_pch = p.xor32(&a_chunk, p.and32(p.shr32(&a_row, 2), 1));
        a_dst.push(p.add32(&smem, p.add32(p.shl32(&a_row, 5), p.shl32(&a_pch, 4))));
        let g_row = p.add32(&block_row, &a_row);
        let g_row_c = p.selp32(p.setp("lt.u32", &g_row, &pm["rows"]), &g_row, &rows_minus1);
        let base = p.add64(&pm["pA"], p.widen(p.add32(p.mul32(&g_row_c, &pm["inputStride"]), p.add32(&pm["inputColumnBase"], p.shl32(&a_chunk, 4)))));
        a_src.push(p.add64(&base, p.widen(p.shl32(&k_step0, 5)))); // + step * 32 per k32 step
    }
    let w_step = p.shl32(&pm["Nmatrix"], 5);
    let w_step64 = p.widen(&w_step);
    let mut b_dst: Vec<String> = Vec::new();
    let mut b_ptr: Vec<String> = Vec::new();
    for i in 0..(256 / threads) {
        let c = p.add32(&tid, p.imm32(threads * i));
        let n = p.shr32(&c, 1);
        let b_chunk = p.and32(&c, 1);
        let b_pch = p.xor32(&b_chunk, p.and32(p.shr32(&n, 2), 1));
        b_dst.push(p.add32(&smem, p.add32(p.imm32(b_ring), p.add32(p.shl32(&n, 5), p.shl32(&b_pch, 4)))));
        let w_row = p.add32(&pm["weightColumnOffset"], p.add32(&col_base, &n));
        let b_src = p.mad_wide(&w_step, &k_step0, p.add64(&pm["pW"], p.widen(p.add32(p.shl32(&w_row, 5), p.shl32(&b_chunk, 4)))));
        let r = p.reg("b64");
        p.emit(format!("mov.b64 {r}, {b_src};"));
        b_ptr.push(r); // the next weight tile to stage (advances by wStep)
    }
    let p_pf = p.setp("lt.u32", &tid, p.imm32(32)); // the 4 KB weight tile is 32 lines
    let pf_base = p.add64(&pm["pW"], p.widen(p.add32(p.shl32(p.add32(&pm["weightColumnOffset"], &col_base), 5), p.shl32(&tid, 7))));
    let pf_ptr = p.reg("b64");
    let pf_steps = p.add32(&k_step0, p.imm32(lead + stages - 1));
    p.emit(format!("mad.wide.u32 {pf_ptr}, {w_step}, {pf_steps}, {pf_base};"));
    let b_issued = Cell::new(0usize);
    let pf_issued = Cell::new(lead + stages - 1); // prefetching starts beyond the staged prologue tiles

    let issue_a = |step_local: usize| {
        let slot = step_local % stages;
        for i in 0..a_dst.len() {
            let guard = match &a_pred[i] {
                Some(pr) => format!("@{pr} "),
                None => String::new(),
            };
            p.emit(format!("{guard}cp.async.cg.shared.global [{}+{}], [{}+{}], 16;", a_dst[i], a_ring + slot * a_bytes, a_src[i], 32 * step_local));
        }
    };

    let issue_b = |step_local: usize| {
        assert!(step_local == b_issued.get());
        b_issued.set(b_issued.get() + 1);
        let slot = step_local % bstages;
        for i in 0..b_dst.len() {
            p.emit(format!("cp.async.cg.shared.global [{}+{}], [{}], 16;", b_dst[i], slot * B_BYTES, b_ptr[i]));
            p.emit(format!("add.u64 {}, {}, {w_step64};", b_ptr[i], b_ptr[i]));
        }
    };

    let prefetch_b = |step_local: usize| {
        if step_local >= steps {
            return;
        }
        assert!(step_local == pf_issued.get());
        p.emit(format!("@{p_pf} prefetch.global.L2 [{pf_ptr}];"));
        p.emit(format!("add.u64 {pf_ptr}, {pf_ptr}, {w_step64};"));
        pf_issued.set(pf_issued.get() + 1);
    };

    // ---- ldmatrix lane addresses (A: 16 rows x 32 B per m-tile of this warp's quarter; B: 16 columns per x4 of
    // this warp's half)
    let l8 = p.and32(&lane, 7);
    let lm = p.shr32(&lane, 3);
    let a_row_l = p.add32(&l8, p.shl32(p.and32(&lm, 1), 3));
    let a_chunk_l = p.shr32(&lane, 4);
    let a_pch_l = p.xor32(&a_chunk_l, p.and32(p.shr32(&a_row_l, 2), 1));
    let a_lane = p.add32(&smem, p.add32(p.shl32(p.add32(&a_row_l, p.mul32(&row_q, p.imm32(48))), 5), p.shl32(&a_pch_l, 4)));
    let b_n_l = p.add32(&l8, p.shl32(p.shr32(&lane, 4), 3));
    let b_chunk_l = p.and32(&lm, 1);
    let b_pch_l = p.xor32(&b_chunk_l, p.and32(p.shr32(&b_n_l, 2), 1));
    let b_lane = p.add32(&smem, p.add32(p.imm32(b_ring), p.add32(p.shl32(p.add32(&b_n_l, p.shl32(&col_h, 6)), 5), p.shl32(&b_pch_l, 4))));
    let load_a = |step_local: usize| -> Vec<Vec<String>> {
        let slot = step_local % stages;
        (0..3).map(|m| ldm4(&p, format!("{a_lane}+{}", a_ring + slot * a_bytes + 512 * m))).collect()
    };

    let load_b = |step_local: usize, i2: usize| -> Vec<String> {
        let slot = step_local % bstages;
        ldm4(&p, format!("{b_lane}+{}", slot * B_BYTES + 512 * i2))
    };

    // output rows of this warp's tiles: 48 rowQ + 16 m + g + 8 h (row & 7 = g)
    let out_row: Vec<Vec<String>> =
        (0..3).map(|m| (0..2).map(|h| p.add32(&block_row, p.add32(p.mul32(&row_q, p.imm32(48)), p.add32(g, p.imm32(16 * m + 8 * h))))).collect()).collect();
    let _row_ok: Vec<Vec<String>> = (0..3).map(|m| (0..2).map(|h| p.setp("lt.u32", &out_row[m][h], &pm["rows"])).collect()).collect();

    // ---- cp.async groups: every step commits the weight group B(s + stages - 1 + lead) and then the A group
    // A(s + stages - 1); wait_group 2 stages - 6 at the top of step s completes A(s + 1) and everything older, which
    // includes B(s + 1) (committed `lead` steps earlier). The weights thus lead by stages - 2 + lead steps.
    // Weight staging and L2 prefetch start before the dependency wait (the weights do not depend on the producer).
    for s in (lead + stages - 1)..(lead + stages - 1 + prefetch0 + prefetch) {
        // the staged tiles need no prefetch
        prefetch_b(s);
    }
    for s in 0..lead.min(steps) {
        issue_b(s);
    }
    p.emit("cp.async.commit_group;");
    let p_wait_on = p.setp("ne.u64", &pm["pWait"], 0);
    sync_wait(&p, &pm["pWait"], &zero32, &zero32, Expected::Reg(&pm["waitExpected"]), &lane, Some(&p_wait_on), Some(&warp), &pm["pError"]);
    let mut scales: Vec<String> = Vec::new();
    let mut p_seed = String::new();
    if res {
        // split 0 stages the residual tile [192][128 B] (16-byte chunk ^= row & 7) at RES_OFF, within group 0
        p_seed = p.setp("eq.u32", &split, &zero32);
        for i in 0..(bm * 8 / threads) {
            // BM x 8 chunks
            let c = p.add32(&tid, p.imm32(threads * i));
            let row = p.shr32(&c, 3);
            let chunk = p.and32(&c, 7);
            let g_row = p.add32(&block_row, &row);
            let ok = p.reg("pred");
            p.emit(format!("setp.lt.and.u32 {ok}, {g_row}, {}, {p_seed};", pm["rows"]));
            let src = p.add64(
                &pm["pRes"],
                p.widen(p.add32(p.mul32(&g_row, &pm["outputStride"]), p.add32(&pm["outputColumnOffset"], p.add32(&col_base, p.shl32(&chunk, 4))))),
            );
            let dst = p.add32(&smem, p.add32(p.imm32(res_off), p.add32(p.shl32(&row, 7), p.shl32(p.xor32(&chunk, p.and32(&row, 7)), 4))));
            p.emit(format!("@{ok} cp.async.cg.shared.global [{dst}], [{src}], 16;"));
        }
        let aux_addr = p.add64(&pm["pAux"], p.widen(p.shl32(p.add32(&pm["auxHalfOffset"], p.add32(&warp_col, p.shl32(t, 1))), 1)));
        for j in 0..8 {
            let r = p.reg("b32");
            p.emit(format!("ld.global.nc.b32 {r}, [{aux_addr}+{}];", 16 * j));
            scales.push(r);
        }
    }
    for s in 0..stages - 1 {
        if s + lead < steps {
            issue_b(s + lead);
        }
        p.emit("cp.async.commit_group;");
        issue_a(s);
        p.emit("cp.async.commit_group;");
    }
    let mut acc: Vec<Vec<Vec<String>>> = (0..3).map(|_| (0..8).map(|_| zero_tile(&p)).collect()).collect();
    // ---- main loop: step s consumes stage s (fragments loaded during step s - 1), stages s and s + 1 are complete
    // after the barrier; the issue refills the slots read at step s - 1
    let mut a_frag: Vec<Vec<String>> = Vec::new();
    let mut b_frag: Vec<String> = Vec::new();
    let mut a_next: Vec<Vec<String>> = Vec::new();
    let mut b_next: Vec<String> = Vec::new();
    for s in 0..steps {
        p.emit(format!("cp.async.wait_group {};", 2 * stages - 6));
        p.emit("bar.sync 0;");
        if s == 0 {
            if res {
                // the chain seed round_f16(residual * scale) from the staged tile (split 0 only)
                let res_lane = p.add32(&smem, p.add32(p.imm32(res_off), p.shl32(t, 1)));
                let mut codes: Vec<Vec<Vec<String>>> = Vec::new();
                for m in 0..3 {
                    let mut cm = Vec::new();
                    for h in 0..2 {
                        let row_l = p.add32(p.mul32(&row_q, p.imm32(48)), p.add32(g, p.imm32(16 * m + 8 * h)));
                        let row_off = p.add32(&res_lane, p.shl32(&row_l, 7));
                        let mut ch = Vec::new();
                        for j in 0..8 {
                            let chunk = p.add32(p.shl32(p.xor32(p.add32(p.shl32(&col_h, 2), p.imm32(j >> 1)), g), 4), p.imm32(8 * (j & 1)));
                            let hc = p.reg("b16");
                            let la = p.add32(&row_off, &chunk);
                            p.emit(format!("ld.shared.b16 {hc}, [{la}];"));
                            ch.push(hc);
                        }
                        cm.push(ch);
                    }
                    codes.push(cm);
                }
                for m in 0..3 {
                    for h in 0..2 {
                        for j in 0..8 {
                            let v = hmul2(&p, e4x2_to_f16x2(&p, &codes[m][h][j]), &scales[j]);
                            p.emit(format!("@{p_seed} mov.b32 {}, {v};", acc[m][j][h]));
                        }
                    }
                }
            }
            a_frag = load_a(0);
            b_frag = load_b(0, 0);
        }
        let j = s + stages - 1;
        // the step's copies right after the barrier
        if j + lead < steps {
            issue_b(j + lead);
        }
        if prefetch != 0 {
            prefetch_b(j + lead + prefetch0 + prefetch);
        }
        p.emit("cp.async.commit_group;");
        if j < steps {
            issue_a(j);
        }
        p.emit("cp.async.commit_group;");
        for i2 in 0..4 {
            if i2 < 3 {
                b_next = load_b(s, i2 + 1);
            } else if s + 1 < steps {
                a_next = load_a(s + 1);
                b_next = load_b(s + 1, 0);
            }
            for m in 0..3 {
                mma_e4(&p, &acc[m][2 * i2], &a_frag[m], &b_frag[0..2], &acc[m][2 * i2]);
                mma_e4(&p, &acc[m][2 * i2 + 1], &a_frag[m], &b_frag[2..4], &acc[m][2 * i2 + 1]);
            }
            b_frag = b_next.clone();
            if i2 == 3 && s + 1 < steps {
                a_frag = a_next.clone();
            }
        }
    }
    p.emit("bar.sync 0;"); // the ring is dead: the staging tiles alias it
    let end_l = p.label("END");

    let mut p_t0 = String::new();
    let mut count_addr = String::new();
    let mut slice_bytes = String::new();
    let mut lane_base = String::new();
    if splits > 1 {
        // the partial in fragment order: [split][rowGroup][colGroup][warp][m, j, h][lane] f16x2 (a line per store)
        let tile_index = p.add32(&col_group, p.mul32(&row_group, &col_groups));
        let tiles = p.mul32(&row_groups, &col_groups);
        slice_bytes = p.mul32(&tiles, p.imm32(partial_wg_bytes));
        let tile_base = p.mad_wide(p.add32(p.mul32(&tile_index, p.imm32(warps)), &warp), p.imm32(48 * 128), &pm["pPartial"]);
        lane_base = p.add64(&tile_base, p.widen(p.shl32(&lane, 2))); // slice 0 of this tile / warp / lane
        let own_base = p.mad_wide(&slice_bytes, &split, &lane_base);
        for m in 0..3 {
            for j in 0..8 {
                for h in 0..2 {
                    p.emit(format!("st.global.b32 [{own_base}+{}], {};", (m * 16 + j * 2 + h) * 128, acc[m][j][h]));
                }
            }
        }
        p.emit("fence.acq_rel.gpu;");
        p.emit("bar.sync 0;");
        p_t0 = p.setp("eq.u32", &tid, &zero32);
        count_addr = p.add64(&pm["pCount"], p.widen(p.shl32(&tile_index, 2)));
    }
    if splits > 1 {
        // last arrival of the column group reduces (threadfence reduction)
        let old = p.reg("b32");
        p.emit(format!("mov.u32 {old}, 0;"));
        p.emit(format!("@{p_t0} atom.acq_rel.gpu.global.add.u32 {old}, [{count_addr}], 1;"));
        p.emit(format!("@{p_t0} st.shared.u32 [{smem}], {old};"));
        p.emit("bar.sync 0;");
        p.emit(format!("ld.shared.u32 {old}, [{smem}];"));
        let p_last = p.setp("eq.u32", &old, p.imm32(splits - 1));
        p.emit(format!("@!{p_last} bra {end_l};"));
        p.emit("fence.acq_rel.gpu;");
        let mut total: Vec<Vec<Vec<String>>> = (0..3).map(|_| (0..8).map(|_| vec![String::new(), String::new()]).collect()).collect();
        for z in 0..splits {
            let p_is = p.setp("eq.u32", &split, p.imm32(z));
            let p_not = p.reg("pred");
            p.emit(format!("not.pred {p_not}, {p_is};"));
            let z_base = p.mad_wide(&slice_bytes, p.imm32(z), &lane_base);
            let mut loaded: HashMap<(usize, usize, usize), String> = HashMap::new();
            for m in 0..3 {
                // the slice's 48 loads first (independent), then the in-order sum
                for j in 0..8 {
                    for h in 0..2 {
                        let v = p.reg("b32");
                        p.emit(format!("mov.b32 {v}, 0;"));
                        p.emit(format!("@{p_not} ld.global.cg.b32 {v}, [{z_base}+{}];", (m * 16 + j * 2 + h) * 128));
                        loaded.insert((m, j, h), v);
                    }
                }
            }
            for m in 0..3 {
                for j in 0..8 {
                    for h in 0..2 {
                        let v = p.selp32(&p_is, &acc[m][j][h], &loaded[&(m, j, h)]);
                        total[m][j][h] = if z == 0 { v } else { hadd2(&p, &total[m][j][h], &v) };
                    }
                }
            }
        }
        acc = total;
    }
    // ---- publish
    if silu_f {
        let kc = SiluConsts::new(&p);
        for m in 0..3 {
            for j in 0..8 {
                for h in 0..2 {
                    acc[m][j][h] = silu(&p, &acc[m][j][h], &kc);
                }
            }
        }
    }
    if out_f16 {
        // f16 tile [192][256 B], 16-byte chunk ^= row & 7 (= g for this warp's rows): conflict-free C stores
        for m in 0..3 {
            for h in 0..2 {
                let row_l = p.add32(p.mul32(&row_q, p.imm32(48)), p.add32(g, p.imm32(16 * m + 8 * h)));
                let row_off = p.add32(&smem, p.add32(p.shl32(&row_l, 8), p.shl32(t, 2)));
                for j in 0..8 {
                    let chunk = p.shl32(p.xor32(p.add32(p.shl32(&col_h, 3), p.imm32(j)), g), 4);
                    let sa = p.add32(&row_off, &chunk);
                    p.emit(format!("st.shared.b32 [{sa}], {};", acc[m][j][h]));
                }
            }
        }
    }
    if out_e4 {
        // E4 tile [192][128 B], 16-byte chunk ^= row & 7
        for m in 0..3 {
            for h in 0..2 {
                let row_l = p.add32(p.mul32(&row_q, p.imm32(48)), p.add32(g, p.imm32(16 * m + 8 * h)));
                let row_off = p.add32(&smem, p.add32(p.imm32(e4_off), p.add32(p.shl32(&row_l, 7), p.shl32(t, 1))));
                for j in 0..8 {
                    let chunk = p.add32(p.shl32(p.xor32(p.add32(p.shl32(&col_h, 2), p.imm32(j >> 1)), g), 4), p.imm32(8 * (j & 1)));
                    let hc = cvt_e4x2(&p, &acc[m][j][h]);
                    let sa = p.add32(&row_off, &chunk);
                    p.emit(format!("st.shared.b16 [{sa}], {hc};"));
                }
            }
        }
    }
    p.emit("bar.sync 0;");
    if out_f16 {
        for i in 0..(bm * 16 / threads) {
            // BM x 16 chunks of 16 B
            let c = p.add32(&tid, p.imm32(threads * i));
            let row = p.shr32(&c, 4);
            let chunk = p.and32(&c, 15);
            let phys = p.add32(&smem, p.add32(p.shl32(&row, 8), p.shl32(p.xor32(&chunk, p.and32(&row, 7)), 4)));
            let r = p.regs("b32", 4);
            p.emit(format!("ld.shared.v4.b32 {{{}, {}, {}, {}}}, [{phys}];", r[0], r[1], r[2], r[3]));
            let g_row_o = p.add32(&block_row, &row);
            let ok = p.setp("lt.u32", &g_row_o, &pm["rows"]);
            let off = p.add32(p.mul32(&g_row_o, &pm["outputStride"]), p.add32(&pm["outputColumnOffset"], p.add32(&col_base, p.shl32(&chunk, 3))));
            let addr = p.add64(&pm["pOut16"], p.widen(p.shl32(&off, 1)));
            p.emit(format!("@{ok} st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
        }
    }
    if out_e4 {
        for i in 0..(bm * 8 / threads) {
            // BM x 8 chunks of 16 B
            let c = p.add32(&tid, p.imm32(threads * i));
            let row = p.shr32(&c, 3);
            let chunk = p.and32(&c, 7);
            let phys = p.add32(&smem, p.add32(p.imm32(e4_off), p.add32(p.shl32(&row, 7), p.shl32(p.xor32(&chunk, p.and32(&row, 7)), 4))));
            let r = p.regs("b32", 4);
            p.emit(format!("ld.shared.v4.b32 {{{}, {}, {}, {}}}, [{phys}];", r[0], r[1], r[2], r[3]));
            let g_row_o = p.add32(&block_row, &row);
            let ok = p.setp("lt.u32", &g_row_o, &pm["rows"]);
            let off = p.add32(p.mul32(&g_row_o, &pm["outputStride"]), p.add32(&pm["outputColumnOffset"], p.add32(&col_base, p.shl32(&chunk, 4))));
            let addr = p.add64(&pm["pOut"], p.widen(&off));
            p.emit(format!("@{ok} st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", r[0], r[1], r[2], r[3]));
        }
    }
    let p_signal_on = p.setp("ne.u64", &pm["pSignal"], 0);
    sync_signal(&p, &[pm["pSignal"].as_str()], &tid, Some(&p_signal_on));
    p.emit(format!("{end_l}:"));
    p.emit("ret;");
    (name, p.finish(), shared_bytes)
}

/// `<K> <flags> <splits> [stages] [bstages] [prefetch] [maxregs] [prefetch0] [BM]`
pub fn cli(args: &[&str]) -> String {
    let (k, flags, splits) = (crate::req(args, 0), crate::req(args, 1), crate::req(args, 2));
    let stages = if args.len() > 3 { crate::req(args, 3) } else { STAGES };
    let bstages = if args.len() > 4 { crate::req(args, 4) } else { BSTAGES };
    let prefetch = if args.len() > 5 { crate::req(args, 5) } else { PREFETCH };
    let max_regs = if args.len() > 6 && crate::req(args, 6) != 0 { Some(crate::reg(args[6])) } else { None };
    let prefetch0 = if args.len() > 7 { crate::req(args, 7) } else { PREFETCH0 };
    let bm = if args.len() > 8 { crate::req(args, 8) } else { 192 };
    generate(k, flags, splits, stages, bstages, prefetch, max_regs, prefetch0, bm).1
}
