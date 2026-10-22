"""Exact Ada FP8 two-layer MLP in PTX: D(E4) = E4( E4(SiLU(A * W1)) * W2 ) per expert.

Native arithmetic: mma.sync.m16n8k32 e4m3 with f16 accumulation chained over K in order
(verified bit-exact against the coopmat path). The hidden activation never leaves the
registers: W1's output columns are permuted (host side, within each 16-unit group) so that
the C fragment a lane holds after the first layer IS its A fragment for the second layer: just
two f16x2 -> e4m3x2 conversions and one byte permute per A register, no shuffles. The
within-16 relabeling is exact because the hardware sums each 16-product group without an
order dependence.

Layouts: A E4 [rows][inputStride] (broadcast to every expert); W1 tile-major
[e][K/32][128][32] with the column permutation; W2 tile-major [e][4][32][32]
(natural hidden order); output E4 [rows][outputStride] at outputColumnOffset + e * 32.
Workgroup: 128 threads = 4 warps x 16 rows (BM = 64); grid (experts, rowGroups).
Params: A B1 B2 D(u64) rows inputStride inputColumnBase outputStride outputColumnOffset (u32).
"""
import sys
from ptxgen import Ptx
from swin import *

HIDDEN = 128
NOUT = 32
THREADS = 128
STAGES = 3
KSUB = 1          # k32 tiles per pipeline stage (1 or 2)
MT = 1            # m16 tiles per warp (register blocking: B fragments serve MT multiplies)


def hidden_permutation(hidden=HIDDEN):
    """W1' column c holds hidden unit u(c): lane t's C columns {2t, 2t+1, 8+2t, 9+2t} (+16, +32j) hold
    units {4t .. 4t+3} (+16, +32j), so the C fragment packs straight into the A fragment."""
    perm = [0] * hidden
    for block in range(hidden // 32):
        for half in range(2):
            for t in range(4):
                for i in range(2):
                    perm[block * 32 + half * 16 + 2 * t + i] = block * 32 + half * 16 + 4 * t + i
                    perm[block * 32 + half * 16 + 8 + 2 * t + i] = block * 32 + half * 16 + 4 * t + 2 + i
    return perm


def generate(K, stages=STAGES, ksub=KSUB, mt=MT, max_regs=None, HIDDEN=HIDDEN, NOUT=NOUT):
    assert K % (32 * ksub) == 0
    BM = 64 * mt                      # rows per workgroup (4 warps x 16 mt)
    steps = K // (32 * ksub)          # pipeline stages (each ksub k32 tiles)
    a_bytes = BM * 32 * ksub          # A tiles [ksub][64][32] (one k32 tile after another)
    w1_bytes = HIDDEN * 32 * ksub     # W1 tiles [ksub][128 n][32 k]
    stage_bytes = a_bytes + w1_bytes
    STAGES_ = stages
    w2_bytes = (HIDDEN // 32) * NOUT * 32   # W2 [HIDDEN/32 k32 tiles][NOUT n][32 k], staged once
    shared_bytes = STAGES_ * stage_bytes + w2_bytes
    name = f"mlp_e4m3_K{K}" + (f"_M{mt}" if mt != 1 else "") + (f"_H{HIDDEN}_N{NOUT}" if (HIDDEN, NOUT) != (128, 32) else "")
    p = Ptx()
    params = [("u64", "pA"), ("u64", "pW1"), ("u64", "pW2"), ("u64", "pD"), ("u32", "rows"), ("u32", "inputStride"),
              ("u32", "inputColumnBase"), ("u32", "outputStride"), ("u32", "outputColumnOffset")]
    p.entry(name, params, shared_bytes, THREADS, max_regs)
    A = p.load_param_u64("pA"); W1 = p.load_param_u64("pW1"); W2 = p.load_param_u64("pW2"); D = p.load_param_u64("pD")
    rows = p.load_param_u32("rows"); inputStride = p.load_param_u32("inputStride")
    inputColumnBase = p.load_param_u32("inputColumnBase"); outputStride = p.load_param_u32("outputStride")
    outputColumnOffset = p.load_param_u32("outputColumnOffset")
    tid = p.special("tid.x"); expert = p.special("ctaid.x"); rowGroup = p.special("ctaid.y")
    lane = p.and32(tid, 31); warp = p.shr32(tid, 5)
    blockRow = p.mul32(rowGroup, BM)
    smem = p.shared_addr(p.imm32(0))
    w2Smem = p.add32(smem, STAGES_ * stage_bytes)
    # ---- per-thread cp.async chunks per stage: 128 threads x 3 chunks = 384 = (64 + 128) rows x 2 chunks.
    # Chunk id c: rows 0..63 of A (c < 128) then W1 rows (c - 128 >> 1). Swizzle: chunk ^= (row >> 2) & 1.
    rowsMinus1 = p.sub32(rows, 1)
    w1Expert = p.mad_wide(expert, steps * HIDDEN * 32, W1)     # this expert's W1 tiles
    copies = []
    chunksPerTile = (BM + HIDDEN) * 2
    perTile = chunksPerTile // THREADS
    assert chunksPerTile % THREADS == 0
    for t in range(perTile * ksub):
        # chunk t of this thread: k32 tile tileIdx = t // perTile, then within the tile A rows (cid < 2 BM) or W1 rows
        cid = p.add32(tid, (t % perTile) * THREADS)
        tileIdx = t // perTile
        isA = p.setp("lt.u32", cid, BM * 2)
        row = p.shr32(cid, 1); chunk = p.and32(cid, 1)
        swz = p.and32(p.shr32(row, 2), 1)
        pchunk = p.xor32(chunk, swz)
        dstA = p.add32(p.mul32(row, 32), p.add32(p.mul32(pchunk, 16), p.imm32(tileIdx * BM * 32)))
        gRow = p.add32(blockRow, row)
        gRowC = p.selp32(p.setp("lt.u32", gRow, rows), gRow, rowsMinus1)
        aOff = p.add32(p.mul32(gRowC, inputStride), p.add32(inputColumnBase, p.add32(p.mul32(chunk, 16), p.imm32(tileIdx * 32))))
        srcA = p.add64(A, p.widen(aOff))
        cidB = p.sub32(cid, BM * 2)
        n = p.shr32(cidB, 1); chunkB = p.and32(cidB, 1)
        swzB = p.and32(p.shr32(n, 2), 1)
        pchunkB = p.xor32(chunkB, swzB)
        dstB = p.add32(a_bytes + tileIdx * HIDDEN * 32, p.add32(p.mul32(n, 32), p.mul32(pchunkB, 16)))
        srcB = p.add64(w1Expert, p.widen(p.add32(p.mul32(n, 32), p.add32(p.mul32(chunkB, 16), p.imm32(tileIdx * HIDDEN * 32)))))
        dst = p.selp32(isA, dstA, dstB)
        src = p.reg("b64"); p.emit(f"selp.b64 {src}, {srcA}, {srcB}, {isA};")
        adv = p.selp32(isA, p.imm32(32 * ksub), p.imm32(HIDDEN * 32 * ksub))   # per stage: A +32*ksub B, W1 +4 KB*ksub
        copies.append((dst, src, adv))

    def issue_stage(stage_off_reg, step_reg):
        for dst, src, adv in copies:
            saddr = p.add32(p.add32(smem, stage_off_reg), dst)
            gaddr = p.mad_wide(adv, step_reg, src)
            p.emit(f"cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;")

    # ---- W2 [HIDDEN/32][NOUT][32] of this expert, staged once (w2_bytes / 16 chunks), same swizzle per 32-byte row
    w2Expert = p.mad_wide(expert, w2_bytes, W2)
    assert (w2_bytes // 16) % THREADS == 0
    for t in range(w2_bytes // 16 // THREADS):
        cid = p.add32(tid, t * THREADS)          # row = cid >> 1 (tile*NOUT + n), chunk = cid & 1
        row = p.shr32(cid, 1); chunk = p.and32(cid, 1)
        swz = p.and32(p.shr32(row, 2), 1)
        pchunk = p.xor32(chunk, swz)
        dst = p.add32(w2Smem, p.add32(p.mul32(row, 32), p.mul32(pchunk, 16)))
        src = p.add64(w2Expert, p.widen(p.add32(p.mul32(row, 32), p.mul32(chunk, 16))))
        p.emit(f"cp.async.cg.shared.global [{dst}], [{src}], 16;")

    # ---- fragments: ldmatrix lane addressing (relative to a stage / tile base)
    l8 = p.and32(lane, 7); lm = p.shr32(lane, 3)
    aRowL = p.add32(l8, p.mul32(p.and32(lm, 1), 8)); aChunkL = p.shr32(lane, 4)
    aPchunkL = p.xor32(aChunkL, p.and32(p.shr32(aRowL, 2), 1))
    aLaneOff = p.add32(p.mul32(p.add32(aRowL, p.mul32(warp, 16 * mt)), 32), p.mul32(aPchunkL, 16))   # + m-tile i * 16 rows * 32
    bNL = p.add32(l8, p.mul32(p.shr32(lane, 4), 8)); bChunkL = p.and32(lm, 1)
    bPchunkL = p.xor32(bChunkL, p.and32(p.shr32(bNL, 2), 1))
    bLaneOff = p.add32(p.mul32(bNL, 32), p.mul32(bPchunkL, 16))      # + tile pair q * 16 rows * 32
    aBaseL = p.add32(smem, aLaneOff)
    bBaseL = p.add32(p.add32(smem, a_bytes), bLaneOff)

    # hidden accumulators per m-tile: 16 n8 tiles (128 columns), 2 regs each
    accs = [[(p.reg("b32"), p.reg("b32")) for _ in range(HIDDEN // 8)] for _ in range(mt)]
    for acc in accs:
        for d0, d1 in acc:
            p.emit(f"mov.b32 {d0}, 0;"); p.emit(f"mov.b32 {d1}, 0;")

    # prologue: stages 0..S-2
    for s in range(STAGES_ - 1):
        if s < steps:
            issue_stage(p.imm32(s * stage_bytes), p.imm32(s))
        p.emit("cp.async.commit_group;")
    step = p.reg("b32"); p.emit(f"mov.u32 {step}, 0;")
    stageOff = p.reg("b32"); p.emit(f"mov.u32 {stageOff}, 0;")
    loop = p.label("KLOOP")
    p.emit(f"{loop}:")
    p.emit(f"cp.async.wait_group {STAGES_ - 2};")
    p.emit("bar.sync 0;")
    aStage = p.add32(aBaseL, stageOff); bStage = p.add32(bBaseL, stageOff)
    for kt in range(ksub):
        aRegs = []
        for i in range(mt):
            a = p.regs("b32", 4)
            aAddr = p.add32(aStage, kt * BM * 32 + i * 16 * 32)
            p.emit(f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a[0]}, {a[1]}, {a[2]}, {a[3]}}}, [{aAddr}];")
            aRegs.append(a)
        for q in range(HIDDEN // 16):
            r = p.regs("b32", 4)
            addr = p.add32(bStage, kt * HIDDEN * 32 + q * 16 * 32)
            p.emit(f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{addr}];")
            for i in range(mt):
                a = aRegs[i]
                for j in range(2):
                    d0, d1 = accs[i][2 * q + j]
                    b0, b1 = (r[0], r[1]) if j == 0 else (r[2], r[3])
                    p.emit(f"mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {{{d0}, {d1}}}, {{{a[0]}, {a[1]}, {a[2]}, {a[3]}}}, {{{b0}, {b1}}}, {{{d0}, {d1}}};")
    nextStep = p.add32(step, STAGES_ - 1)
    pNext = p.setp("lt.u32", nextStep, steps)
    skip = p.label("SKIPCOPY")
    p.emit(f"@!{pNext} bra {skip};")
    prevOff = p.sub32(stageOff, stage_bytes)
    pWrap = p.setp("lt.s32", stageOff, stage_bytes)
    wrapOff = p.selp32(pWrap, p.imm32((STAGES_ - 1) * stage_bytes), prevOff)
    issue_stage(wrapOff, nextStep)
    p.emit(f"{skip}:")
    p.emit("cp.async.commit_group;")
    p.emit(f"add.u32 {step}, {step}, 1;")
    nextOff = p.add32(stageOff, stage_bytes)
    pLast = p.setp("eq.u32", nextOff, STAGES_ * stage_bytes)
    p.emit(f"selp.b32 {stageOff}, 0, {nextOff}, {pLast};")
    pDone = p.setp("lt.u32", step, steps)
    p.emit(f"@{pDone} bra {loop};")
    p.emit("cp.async.wait_group 0;")
    p.emit("bar.sync 0;")   # W2 tiles landed (and every warp finished the last stage)

    # ---- SiLU on the accumulator pairs (swin.silu: the network's one polynomial and its roundings)
    siluK = SiluConsts(p)
    for acc in accs:
        for i, (d0, d1) in enumerate(acc):
            acc[i] = (silu(p, d0, siluK), silu(p, d1, siluK))
    # ---- C fragment -> A fragments of the second layer: k32 tile kt, register a_i = (rows g/g+8, k 4t.. / 16+4t..)
    #   a0 = row g:   pack(tile 4kt+0 pair, tile 4kt+1 pair)      a2 = row g:   pack(tile 4kt+2, tile 4kt+3)
    #   a1 = row g+8: pack(tile 4kt+0, tile 4kt+1)                a3 = row g+8: pack(tile 4kt+2, tile 4kt+3)
    def e4pair(v):
        h = p.reg("b16"); p.emit(f"cvt.rn.satfinite.e4m3x2.f16x2 {h}, {v};")
        w = p.reg("b32"); p.emit(f"cvt.u32.u16 {w}, {h};")
        return w
    def nan_to_zero_pairs(w):
        """E4 codes 0x7f / 0xff (NaN) -> 0 per byte, on a 16-bit pair in a b32: the same publication rule as
        swin.nanzero, applied after the conversion because the codes are already packed here."""
        lo = p.and32(w, 0x7f); hi = p.and32(p.shr32(w, 8), 0x7f)
        pLo = p.setp("eq.u32", lo, 0x7f); pHi = p.setp("eq.u32", hi, 0x7f)
        mLo = p.selp32(pLo, p.imm32(0xff00), p.imm32(0xffff)); mHi = p.selp32(pHi, p.imm32(0x00ff), p.imm32(0xffff))
        return p.and32(p.and32(w, mLo), mHi)
    aFrags = []
    for acc in accs:
        aFrag = []
        for kt in range(HIDDEN // 32):
            regs4 = []
            for half in range(2):
                for pair in range(2):
                    tA, tB = 4 * kt + 2 * pair, 4 * kt + 2 * pair + 1
                    wA = nan_to_zero_pairs(e4pair(acc[tA][half])); wB = nan_to_zero_pairs(e4pair(acc[tB][half]))
                    r = p.reg("b32"); p.emit(f"prmt.b32 {r}, {wA}, {wB}, 0x5410;")   # bytes: wA.0 wA.1 wB.0 wB.1
                    regs4.append(r)
            # order a0 (g, first pair), a1 (g+8, first pair), a2 (g, second pair), a3 (g+8, second pair)
            aFrag.append([regs4[0], regs4[2], regs4[1], regs4[3]])
        aFrags.append(aFrag)
    # ---- second layer: acc2[m-tile][4 n8 tiles] over the 4 k32 tiles in order, B from the staged W2 tiles
    acc2s = [[(p.reg("b32"), p.reg("b32")) for _ in range(NOUT // 8)] for _ in range(mt)]
    for acc2 in acc2s:
        for d0, d1 in acc2:
            p.emit(f"mov.b32 {d0}, 0;"); p.emit(f"mov.b32 {d1}, 0;")
    w2LaneBase = p.add32(w2Smem, bLaneOff)
    for kt in range(HIDDEN // 32):
        for q in range(NOUT // 16):
            r = p.regs("b32", 4)
            addr = p.add32(w2LaneBase, kt * NOUT * 32 + q * 16 * 32)
            p.emit(f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{addr}];")
            for i in range(mt):
                for j in range(2):
                    d0, d1 = acc2s[i][2 * q + j]
                    b0, b1 = (r[0], r[1]) if j == 0 else (r[2], r[3])
                    af = aFrags[i][kt]
                    p.emit(f"mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {{{d0}, {d1}}}, {{{af[0]}, {af[1]}, {af[2]}, {af[3]}}}, {{{b0}, {b1}}}, {{{d0}, {d1}}};")
    # ---- output: E4 pairs (NaN -> 0), rows g / g+8 of each m-tile, columns outputColumnOffset + e*32 + 8j + 2t
    g = p.shr32(lane, 2); t = p.and32(lane, 3)
    colBase = p.add32(outputColumnOffset, p.add32(p.mul32(expert, NOUT), p.mul32(t, 2)))
    for i in range(mt):
        rowG = p.add32(p.add32(blockRow, p.mul32(warp, 16 * mt)), p.add32(g, p.imm32(i * 16)))
        for half in range(2):
            row = p.add32(rowG, half * 8)
            pRow = p.setp("lt.u32", row, rows)
            rowAddr = p.mad_wide(row, outputStride, D)
            for j in range(NOUT // 8):
                w = nan_to_zero_pairs(e4pair(acc2s[i][j][half]))
                h = p.reg("b16"); p.emit(f"cvt.u16.u32 {h}, {w};")
                addr = p.add64(rowAddr, p.widen(p.add32(colBase, j * 8)))
                p.emit(f"@{pRow} st.global.b16 [{addr}], {h};")
    p.emit("ret;")
    return name, p.finish()


if __name__ == "__main__":
    K, out = int(sys.argv[1]), sys.argv[2]
    stages = int(sys.argv[3]) if len(sys.argv) > 3 else STAGES
    ksub = int(sys.argv[4]) if len(sys.argv) > 4 else KSUB
    mt = int(sys.argv[5]) if len(sys.argv) > 5 else MT
    max_regs = int(sys.argv[6]) if len(sys.argv) > 6 else None
    hidden = int(sys.argv[7]) if len(sys.argv) > 7 else HIDDEN
    nout = int(sys.argv[8]) if len(sys.argv) > 8 else NOUT
    name, text = generate(K, stages, ksub, mt, max_regs, hidden, nout)
    with open(out, "w", newline="\n") as f:
        f.write(text)
    print(name)
