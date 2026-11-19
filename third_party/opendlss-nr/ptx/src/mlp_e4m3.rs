//! Exact Ada FP8 two-layer MLP in PTX: D(E4) = E4( E4(SiLU(A * W1)) * W2 ) per expert.
//!
//! Native arithmetic: mma.sync.m16n8k32 e4m3 with f16 accumulation chained over K in order
//! (verified bit-exact against the coopmat path). The hidden activation never leaves the
//! registers: W1's output columns are permuted (host side, within each 16-unit group) so that
//! the C fragment a lane holds after the first layer IS its A fragment for the second layer: just
//! two f16x2 -> e4m3x2 conversions and one byte permute per A register, no shuffles. The
//! within-16 relabeling is exact because the hardware sums each 16-product group without an
//! order dependence.
//!
//! Layouts: A E4 [rows][inputStride] (broadcast to every expert); W1 tile-major
//! [e][K/32][128][32] with the column permutation; W2 tile-major [e][4][32][32]
//! (natural hidden order); output E4 [rows][outputStride] at outputColumnOffset + e * 32.
//! Workgroup: 128 threads = 4 warps x 16 rows (BM = 64); grid (experts, rowGroups).
//! Params: A B1 B2 D(u64) rows inputStride inputColumnBase outputStride outputColumnOffset (u32).
//!
//! Arguments: `<K> [stages] [ksub] [mt] [maxregs] [hidden] [nout]`

use crate::ptxgen::Ptx;
use crate::swin::*;

pub const HIDDEN: usize = 128;
pub const NOUT: usize = 32;
pub const THREADS: usize = 128;
pub const STAGES: usize = 3;
/// k32 tiles per pipeline stage (1 or 2)
pub const KSUB: usize = 1;
/// m16 tiles per warp (register blocking: B fragments serve MT multiplies)
pub const MT: usize = 1;

/// W1' column c holds hidden unit u(c): lane t's C columns {2t, 2t+1, 8+2t, 9+2t} (+16, +32j) hold
/// units {4t .. 4t+3} (+16, +32j), so the C fragment packs straight into the A fragment.
pub fn hidden_permutation(hidden: usize) -> Vec<usize> {
    let mut perm = vec![0; hidden];
    for block in 0..hidden / 32 {
        for half in 0..2 {
            for t in 0..4 {
                for i in 0..2 {
                    perm[block * 32 + half * 16 + 2 * t + i] = block * 32 + half * 16 + 4 * t + i;
                    perm[block * 32 + half * 16 + 8 + 2 * t + i] = block * 32 + half * 16 + 4 * t + 2 + i;
                }
            }
        }
    }
    perm
}

fn pick(pair: &(String, String), half: usize) -> &String {
    if half == 0 {
        &pair.0
    } else {
        &pair.1
    }
}

pub fn generate(k: usize, stages: usize, ksub: usize, mt: usize, max_regs: Option<u32>, hidden: usize, nout: usize) -> (String, String) {
    assert!(k.is_multiple_of(32 * ksub));
    let bm = 64 * mt; // rows per workgroup (4 warps x 16 mt)
    let steps = k / (32 * ksub); // pipeline stages (each ksub k32 tiles)
    let a_bytes = bm * 32 * ksub; // A tiles [ksub][64][32] (one k32 tile after another)
    let w1_bytes = hidden * 32 * ksub; // W1 tiles [ksub][128 n][32 k]
    let stage_bytes = a_bytes + w1_bytes;
    let stages_ = stages;
    let w2_bytes = (hidden / 32) * nout * 32; // W2 [HIDDEN/32 k32 tiles][NOUT n][32 k], staged once
    let shared_bytes = stages_ * stage_bytes + w2_bytes;
    let name = format!(
        "mlp_e4m3_K{k}{}{}",
        if mt != 1 { format!("_M{mt}") } else { String::new() },
        if (hidden, nout) != (128, 32) { format!("_H{hidden}_N{nout}") } else { String::new() }
    );
    let p = Ptx::new();
    let params = [
        ("u64", "pA"),
        ("u64", "pW1"),
        ("u64", "pW2"),
        ("u64", "pD"),
        ("u32", "rows"),
        ("u32", "inputStride"),
        ("u32", "inputColumnBase"),
        ("u32", "outputStride"),
        ("u32", "outputColumnOffset"),
    ];
    p.entry(&name, &params, shared_bytes, Some(THREADS as u32), max_regs, false);
    let a = p.load_param_u64("pA");
    let w1 = p.load_param_u64("pW1");
    let w2 = p.load_param_u64("pW2");
    let d = p.load_param_u64("pD");
    let rows = p.load_param_u32("rows");
    let input_stride = p.load_param_u32("inputStride");
    let input_column_base = p.load_param_u32("inputColumnBase");
    let output_stride = p.load_param_u32("outputStride");
    let output_column_offset = p.load_param_u32("outputColumnOffset");
    let tid = p.special("tid.x");
    let expert = p.special("ctaid.x");
    let row_group = p.special("ctaid.y");
    let lane = p.and32(&tid, 31);
    let warp = p.shr32(&tid, 5);
    let block_row = p.mul32(&row_group, bm);
    let smem = p.shared_addr(p.imm32(0));
    let w2_smem = p.add32(&smem, stages_ * stage_bytes);
    // ---- per-thread cp.async chunks per stage: 128 threads x 3 chunks = 384 = (64 + 128) rows x 2 chunks.
    // Chunk id c: rows 0..63 of A (c < 128) then W1 rows (c - 128 >> 1). Swizzle: chunk ^= (row >> 2) & 1.
    let rows_minus1 = p.sub32(&rows, 1);
    let w1_expert = p.mad_wide(&expert, steps * hidden * 32, &w1); // this expert's W1 tiles
    let mut copies: Vec<(String, String, String)> = Vec::new();
    let chunks_per_tile = (bm + hidden) * 2;
    let per_tile = chunks_per_tile / THREADS;
    assert!(chunks_per_tile.is_multiple_of(THREADS));
    for t in 0..per_tile * ksub {
        // chunk t of this thread: k32 tile tileIdx = t // perTile, then within the tile A rows (cid < 2 BM) or W1 rows
        let cid = p.add32(&tid, (t % per_tile) * THREADS);
        let tile_idx = t / per_tile;
        let is_a = p.setp("lt.u32", &cid, bm * 2);
        let row = p.shr32(&cid, 1);
        let chunk = p.and32(&cid, 1);
        let swz = p.and32(p.shr32(&row, 2), 1);
        let pchunk = p.xor32(&chunk, &swz);
        let dst_a = p.add32(p.mul32(&row, 32), p.add32(p.mul32(&pchunk, 16), p.imm32(tile_idx * bm * 32)));
        let g_row = p.add32(&block_row, &row);
        let g_row_c = p.selp32(p.setp("lt.u32", &g_row, &rows), &g_row, &rows_minus1);
        let a_off = p.add32(p.mul32(&g_row_c, &input_stride), p.add32(&input_column_base, p.add32(p.mul32(&chunk, 16), p.imm32(tile_idx * 32))));
        let src_a = p.add64(&a, p.widen(&a_off));
        let cid_b = p.sub32(&cid, bm * 2);
        let n = p.shr32(&cid_b, 1);
        let chunk_b = p.and32(&cid_b, 1);
        let swz_b = p.and32(p.shr32(&n, 2), 1);
        let pchunk_b = p.xor32(&chunk_b, &swz_b);
        let dst_b = p.add32(a_bytes + tile_idx * hidden * 32, p.add32(p.mul32(&n, 32), p.mul32(&pchunk_b, 16)));
        let src_b = p.add64(&w1_expert, p.widen(p.add32(p.mul32(&n, 32), p.add32(p.mul32(&chunk_b, 16), p.imm32(tile_idx * hidden * 32)))));
        let dst = p.selp32(&is_a, &dst_a, &dst_b);
        let src = p.reg("b64");
        p.emit(format!("selp.b64 {src}, {src_a}, {src_b}, {is_a};"));
        let adv = p.selp32(&is_a, p.imm32(32 * ksub), p.imm32(hidden * 32 * ksub)); // per stage: A +32*ksub B, W1 +4 KB*ksub
        copies.push((dst, src, adv));
    }

    let issue_stage = |stage_off_reg: &str, step_reg: &str| {
        for (dst, src, adv) in &copies {
            let saddr = p.add32(p.add32(&smem, stage_off_reg), dst);
            let gaddr = p.mad_wide(adv, step_reg, src);
            p.emit(format!("cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;"));
        }
    };

    // ---- W2 [HIDDEN/32][NOUT][32] of this expert, staged once (w2_bytes / 16 chunks), same swizzle per 32-byte row
    let w2_expert = p.mad_wide(&expert, w2_bytes, &w2);
    assert!((w2_bytes / 16).is_multiple_of(THREADS));
    for t in 0..w2_bytes / 16 / THREADS {
        let cid = p.add32(&tid, t * THREADS); // row = cid >> 1 (tile*NOUT + n), chunk = cid & 1
        let row = p.shr32(&cid, 1);
        let chunk = p.and32(&cid, 1);
        let swz = p.and32(p.shr32(&row, 2), 1);
        let pchunk = p.xor32(&chunk, &swz);
        let dst = p.add32(&w2_smem, p.add32(p.mul32(&row, 32), p.mul32(&pchunk, 16)));
        let src = p.add64(&w2_expert, p.widen(p.add32(p.mul32(&row, 32), p.mul32(&chunk, 16))));
        p.emit(format!("cp.async.cg.shared.global [{dst}], [{src}], 16;"));
    }

    // ---- fragments: ldmatrix lane addressing (relative to a stage / tile base)
    let l8 = p.and32(&lane, 7);
    let lm = p.shr32(&lane, 3);
    let a_row_l = p.add32(&l8, p.mul32(p.and32(&lm, 1), 8));
    let a_chunk_l = p.shr32(&lane, 4);
    let a_pchunk_l = p.xor32(&a_chunk_l, p.and32(p.shr32(&a_row_l, 2), 1));
    let a_lane_off = p.add32(p.mul32(p.add32(&a_row_l, p.mul32(&warp, 16 * mt)), 32), p.mul32(&a_pchunk_l, 16)); // + m-tile i * 16 rows * 32
    let b_nl = p.add32(&l8, p.mul32(p.shr32(&lane, 4), 8));
    let b_chunk_l = p.and32(&lm, 1);
    let b_pchunk_l = p.xor32(&b_chunk_l, p.and32(p.shr32(&b_nl, 2), 1));
    let b_lane_off = p.add32(p.mul32(&b_nl, 32), p.mul32(&b_pchunk_l, 16)); // + tile pair q * 16 rows * 32
    let a_base_l = p.add32(&smem, &a_lane_off);
    let b_base_l = p.add32(p.add32(&smem, a_bytes), &b_lane_off);

    // hidden accumulators per m-tile: 16 n8 tiles (128 columns), 2 regs each
    let mut accs: Vec<Vec<(String, String)>> = (0..mt).map(|_| (0..hidden / 8).map(|_| (p.reg("b32"), p.reg("b32"))).collect()).collect();
    for acc in &accs {
        for (d0, d1) in acc {
            p.emit(format!("mov.b32 {d0}, 0;"));
            p.emit(format!("mov.b32 {d1}, 0;"));
        }
    }

    // prologue: stages 0..S-2
    for s in 0..stages_ - 1 {
        if s < steps {
            issue_stage(&p.imm32(s * stage_bytes), &p.imm32(s));
        }
        p.emit("cp.async.commit_group;");
    }
    let step = p.reg("b32");
    p.emit(format!("mov.u32 {step}, 0;"));
    let stage_off = p.reg("b32");
    p.emit(format!("mov.u32 {stage_off}, 0;"));
    let loop_l = p.label("KLOOP");
    p.emit(format!("{loop_l}:"));
    p.emit(format!("cp.async.wait_group {};", stages_ - 2));
    p.emit("bar.sync 0;");
    let a_stage = p.add32(&a_base_l, &stage_off);
    let b_stage = p.add32(&b_base_l, &stage_off);
    for kt in 0..ksub {
        let mut a_regs: Vec<Vec<String>> = Vec::new();
        for i in 0..mt {
            let a = p.regs("b32", 4);
            let a_addr = p.add32(&a_stage, kt * bm * 32 + i * 16 * 32);
            p.emit(format!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{}, {}, {}, {}}}, [{a_addr}];", a[0], a[1], a[2], a[3]));
            a_regs.push(a);
        }
        for q in 0..hidden / 16 {
            let r = p.regs("b32", 4);
            let addr = p.add32(&b_stage, kt * hidden * 32 + q * 16 * 32);
            p.emit(format!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{}, {}, {}, {}}}, [{addr}];", r[0], r[1], r[2], r[3]));
            for i in 0..mt {
                let a = &a_regs[i];
                for j in 0..2 {
                    let (d0, d1) = &accs[i][2 * q + j];
                    let (b0, b1) = if j == 0 { (&r[0], &r[1]) } else { (&r[2], &r[3]) };
                    p.emit(format!(
                        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {{{d0}, {d1}}}, {{{}, {}, {}, {}}}, {{{b0}, {b1}}}, {{{d0}, {d1}}};",
                        a[0], a[1], a[2], a[3]
                    ));
                }
            }
        }
    }
    let next_step = p.add32(&step, stages_ - 1);
    let p_next = p.setp("lt.u32", &next_step, steps);
    let skip = p.label("SKIPCOPY");
    p.emit(format!("@!{p_next} bra {skip};"));
    let prev_off = p.sub32(&stage_off, stage_bytes);
    let p_wrap = p.setp("lt.s32", &stage_off, stage_bytes);
    let wrap_off = p.selp32(&p_wrap, p.imm32((stages_ - 1) * stage_bytes), &prev_off);
    issue_stage(&wrap_off, &next_step);
    p.emit(format!("{skip}:"));
    p.emit("cp.async.commit_group;");
    p.emit(format!("add.u32 {step}, {step}, 1;"));
    let next_off = p.add32(&stage_off, stage_bytes);
    let p_last = p.setp("eq.u32", &next_off, stages_ * stage_bytes);
    p.emit(format!("selp.b32 {stage_off}, 0, {next_off}, {p_last};"));
    let p_done = p.setp("lt.u32", &step, steps);
    p.emit(format!("@{p_done} bra {loop_l};"));
    p.emit("cp.async.wait_group 0;");
    p.emit("bar.sync 0;"); // W2 tiles landed (and every warp finished the last stage)

    // ---- SiLU on the accumulator pairs (swin.silu: the network's one polynomial and its roundings)
    let silu_k = SiluConsts::new(&p);
    for acc in accs.iter_mut() {
        for i in 0..acc.len() {
            let (d0, d1) = acc[i].clone();
            let s0 = silu(&p, &d0, &silu_k);
            let s1 = silu(&p, &d1, &silu_k);
            acc[i] = (s0, s1);
        }
    }
    // ---- C fragment -> A fragments of the second layer: k32 tile kt, register a_i = (rows g/g+8, k 4t.. / 16+4t..)
    //   a0 = row g:   pack(tile 4kt+0 pair, tile 4kt+1 pair)      a2 = row g:   pack(tile 4kt+2, tile 4kt+3)
    //   a1 = row g+8: pack(tile 4kt+0, tile 4kt+1)                a3 = row g+8: pack(tile 4kt+2, tile 4kt+3)
    let e4pair = |v: &str| -> String {
        let h = p.reg("b16");
        p.emit(format!("cvt.rn.satfinite.e4m3x2.f16x2 {h}, {v};"));
        let w = p.reg("b32");
        p.emit(format!("cvt.u32.u16 {w}, {h};"));
        w
    };
    // E4 codes 0x7f / 0xff (NaN) -> 0 per byte, on a 16-bit pair in a b32: the same publication rule as
    // swin.nanzero, applied after the conversion because the codes are already packed here.
    let nan_to_zero_pairs = |w: &str| -> String {
        let lo = p.and32(w, 0x7f);
        let hi = p.and32(p.shr32(w, 8), 0x7f);
        let p_lo = p.setp("eq.u32", &lo, 0x7f);
        let p_hi = p.setp("eq.u32", &hi, 0x7f);
        let m_lo = p.selp32(&p_lo, p.imm32(0xff00), p.imm32(0xffff));
        let m_hi = p.selp32(&p_hi, p.imm32(0x00ff), p.imm32(0xffff));
        p.and32(p.and32(w, &m_lo), &m_hi)
    };
    let mut a_frags: Vec<Vec<Vec<String>>> = Vec::new();
    for acc in &accs {
        let mut a_frag: Vec<Vec<String>> = Vec::new();
        for kt in 0..hidden / 32 {
            let mut regs4: Vec<String> = Vec::new();
            for half in 0..2 {
                for pair in 0..2 {
                    let (t_a, t_b) = (4 * kt + 2 * pair, 4 * kt + 2 * pair + 1);
                    let w_a = nan_to_zero_pairs(&e4pair(pick(&acc[t_a], half)));
                    let w_b = nan_to_zero_pairs(&e4pair(pick(&acc[t_b], half)));
                    let r = p.reg("b32");
                    p.emit(format!("prmt.b32 {r}, {w_a}, {w_b}, 0x5410;")); // bytes: wA.0 wA.1 wB.0 wB.1
                    regs4.push(r);
                }
            }
            // order a0 (g, first pair), a1 (g+8, first pair), a2 (g, second pair), a3 (g+8, second pair)
            a_frag.push(vec![regs4[0].clone(), regs4[2].clone(), regs4[1].clone(), regs4[3].clone()]);
        }
        a_frags.push(a_frag);
    }
    // ---- second layer: acc2[m-tile][4 n8 tiles] over the 4 k32 tiles in order, B from the staged W2 tiles
    let acc2s: Vec<Vec<(String, String)>> = (0..mt).map(|_| (0..nout / 8).map(|_| (p.reg("b32"), p.reg("b32"))).collect()).collect();
    for acc2 in &acc2s {
        for (d0, d1) in acc2 {
            p.emit(format!("mov.b32 {d0}, 0;"));
            p.emit(format!("mov.b32 {d1}, 0;"));
        }
    }
    let w2_lane_base = p.add32(&w2_smem, &b_lane_off);
    for kt in 0..hidden / 32 {
        for q in 0..nout / 16 {
            let r = p.regs("b32", 4);
            let addr = p.add32(&w2_lane_base, kt * nout * 32 + q * 16 * 32);
            p.emit(format!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{}, {}, {}, {}}}, [{addr}];", r[0], r[1], r[2], r[3]));
            for i in 0..mt {
                for j in 0..2 {
                    let (d0, d1) = &acc2s[i][2 * q + j];
                    let (b0, b1) = if j == 0 { (&r[0], &r[1]) } else { (&r[2], &r[3]) };
                    let af = &a_frags[i][kt];
                    p.emit(format!(
                        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {{{d0}, {d1}}}, {{{}, {}, {}, {}}}, {{{b0}, {b1}}}, {{{d0}, {d1}}};",
                        af[0], af[1], af[2], af[3]
                    ));
                }
            }
        }
    }
    // ---- output: E4 pairs (NaN -> 0), rows g / g+8 of each m-tile, columns outputColumnOffset + e*32 + 8j + 2t
    let g = p.shr32(&lane, 2);
    let t = p.and32(&lane, 3);
    let col_base = p.add32(&output_column_offset, p.add32(p.mul32(&expert, nout), p.mul32(&t, 2)));
    for i in 0..mt {
        let row_g = p.add32(p.add32(&block_row, p.mul32(&warp, 16 * mt)), p.add32(&g, p.imm32(i * 16)));
        for half in 0..2 {
            let row = p.add32(&row_g, half * 8);
            let p_row = p.setp("lt.u32", &row, &rows);
            let row_addr = p.mad_wide(&row, &output_stride, &d);
            for j in 0..nout / 8 {
                let w = nan_to_zero_pairs(&e4pair(pick(&acc2s[i][j], half)));
                let h = p.reg("b16");
                p.emit(format!("cvt.u16.u32 {h}, {w};"));
                let addr = p.add64(&row_addr, p.widen(p.add32(&col_base, j * 8)));
                p.emit(format!("@{p_row} st.global.b16 [{addr}], {h};"));
            }
        }
    }
    p.emit("ret;");
    (name, p.finish())
}

/// `<K> [stages] [ksub] [mt] [maxregs] [hidden] [nout]`
pub fn cli(args: &[&str]) -> String {
    let k = crate::int(args[0]);
    let stages = args.get(1).map_or(STAGES, |a| crate::int(a));
    let ksub = args.get(2).map_or(KSUB, |a| crate::int(a));
    let mt = args.get(3).map_or(MT, |a| crate::int(a));
    let max_regs = args.get(4).map(|a| crate::int(a) as u32);
    let hidden = args.get(5).map_or(HIDDEN, |a| crate::int(a));
    let nout = args.get(6).map_or(NOUT, |a| crate::int(a));
    generate(k, stages, ksub, mt, max_regs, hidden, nout).1
}
