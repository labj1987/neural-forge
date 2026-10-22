"""Exact FP8 GEMM with the block epilogues in PTX (the gemm_fp8.comp shapes of the expert stages).

D[rows][N] = A[rows][K] (E4, at inputColumnBase of inputStride) x W (E4, k32-tile-major [K/32][Nmatrix][32] at
weightColumnOffset), f16 accumulation chained over K in order (mma.sync m16n8k32, two-group F13 arithmetic).
Flags:
  RES   (seed): the accumulator starts from round_f16(E4 residual * f16 scale) - residual at the output location,
        scale = aux halves at auxHalfOffset + column. The scaled residual seeds the C fragments rather than
        being added in the epilogue: one rounding instead of two.
  SILU  : cubic SiLU on the f16 result.
  E4    : E4M3 output (NaN -> 0); F16: f16 output (both: dual).
Workgroup: 4 warps x 16 rows = 64 rows, 64 columns (8 n8 tiles per warp), cp.async ring of A [64][32] + W [64][32]
per k32 step, ldmatrix fragments, grid (N / 64, ceil(rows / 64)).

python gemm2_e4m3.py <K> <flags> out.ptx [maxregs]     flags: RES=1 SILU=2 E4=4 F16=8
"""
import sys
from ptxgen import Ptx
from swin import *

F_RES, F_SILU, F_E4, F_F16 = 1, 2, 4, 8
THREADS = 128
BM, BN = 64, 64
STAGES = 3
STAGE_BYTES = (BM + BN) * 32


def generate(K, flags, max_regs=None, stagesWanted=STAGES, ksub=1, ws=False):
    """ws: warp-specialized copies - warp 4 issues every cp.async (the LSU accepts ~16 B/cycle per SM and a
    cp.async issue blocks the issuing warp), warps 0..3 only run ldmatrix + MMA; the stages hand over through
    named barriers FULL(slot) / EMPTY(slot). The launch has 160 threads (the PTX carries `// threads 160`)."""
    assert K % (32 * ksub) == 0
    steps = K // (32 * ksub)               # pipeline stages of ksub k32 tiles each
    stages = min(stagesWanted, steps)
    assert not ws or stages >= 2
    STAGE_BYTES = ksub * (BM + BN) * 32
    res, siluF, outE4, outF16 = bool(flags & F_RES), bool(flags & F_SILU), bool(flags & F_E4), bool(flags & F_F16)
    assert outE4 or outF16
    name = f"gemm2_e4m3_K{K}_f{flags}" + (f"_s{ksub}" if ksub != 1 else "")
    RESS = stages * STAGE_BYTES            # residual E4 tile [64][64 B] (4 KB), then the E4 output tile
    shared_bytes = RESS + 4096
    NT = THREADS + 32 if ws else THREADS   # + the copy warp
    p = Ptx()
    if ws: p.lines.append(f"// threads {NT}")
    params = [("u64", "pA"), ("u64", "pW"), ("u64", "pRes"), ("u64", "pAux"), ("u64", "pOut"), ("u64", "pOut16"),
              ("u32", "rows"), ("u32", "inputStride"), ("u32", "inputColumnBase"), ("u32", "Nmatrix"),
              ("u32", "weightColumnOffset"), ("u32", "outputStride"), ("u32", "outputColumnOffset"), ("u32", "auxHalfOffset"),
              ("u64", "pWaitRows"), ("u32", "waitExpected"), ("u32", "waitShiftY"), ("u64", "pSignal"), ("u32", "width"),
              ("u64", "pWaitBands"), ("u32", "waitMul"), ("u32", "waitGroupRows"), ("u64", "pError")]   # chaining (0 = off)
    p.entry(name, params, shared_bytes, NT, max_regs)
    P = {n: (p.load_param_u64(n) if t == "u64" else p.load_param_u32(n)) for t, n in params}
    tid = p.special("tid.x"); colGroup = p.special("ctaid.x"); rowGroup = p.special("ctaid.y")
    BAR = f"bar.sync 0, {NT};"             # every thread of the workgroup (the copy warp included)
    BARC = "bar.sync 0, 128;"              # the four MMA warps only
    lane = p.and32(tid, 31); warp = p.shr32(tid, 5)
    li = LaneInfo(p, lane)
    g, t = li.g, li.t
    smem = p.shared_addr(p.imm32(0))
    zero = imm(p, 0)
    blockRow = p.shl32(rowGroup, 6)
    colBase = p.shl32(colGroup, 6)                       # output column of this workgroup's 64 columns
    rowsMinus1 = p.sub32(P["rows"], 1)
    # ---- chaining: wait for the window rows covering this workgroup's rows (the attention output), signal row bands
    rowLast = p.add32(blockRow, p.imm32(BM - 1))
    pLastBig = p.setp("gt.u32", rowLast, rowsMinus1)
    rowLast = p.selp32(pLastBig, rowsMinus1, rowLast)
    yFirst = p.reg("b32"); p.emit(f"div.u32 {yFirst}, {blockRow}, {P['width']};")
    yLast = p.reg("b32"); p.emit(f"div.u32 {yLast}, {rowLast}, {P['width']};")
    pWaitRowsOn = p.setp("ne.u64", P["pWaitRows"], 0)
    wy0 = p.shr32(p.add32(yFirst, P["waitShiftY"]), 3); wy1 = p.shr32(p.add32(yLast, P["waitShiftY"]), 3)
    sync_wait(p, P["pWaitRows"], wy0, wy1, P["waitExpected"], lane, pWaitRowsOn, warp, error64=P["pError"])
    bandFirst = p.shr32(yFirst, 3); bandLast = p.shr32(yLast, 3)
    pWaitBandsOn = p.setp("ne.u64", P["pWaitBands"], 0)
    sync_wait(p, P["pWaitBands"], bandFirst, bandLast, lambda b: band_expected(p, b, P["width"], P["rows"], P["waitMul"], P["waitGroupRows"]), lane, pWaitBandsOn, warp, error64=P["pError"])

    # ---- cp.async chunks: 256 per stage (A rows 0..63 x 2 chunks, then W rows) = 2 per thread (8 per lane of the
    # copy warp); chunk ^= (row >> 2) & 1
    copies = []
    copyLane = p.and32(tid, 31)
    for i in range(8 if ws else 2):
        c = p.add32(copyLane, i * 32) if ws else p.add32(tid, i * THREADS)
        isA = p.setp("lt.u32", c, BM * 2)
        row = p.shr32(c, 1); chunk = p.and32(c, 1)
        pch = p.xor32(chunk, p.and32(p.shr32(row, 2), 1))
        dstA = p.add32(p.shl32(row, 5), p.shl32(pch, 4))
        gRow = p.add32(blockRow, row)
        gRowC = p.selp32(p.setp("lt.u32", gRow, P["rows"]), gRow, rowsMinus1)
        aOff = p.add32(p.mul32(gRowC, P["inputStride"]), p.add32(P["inputColumnBase"], p.shl32(chunk, 4)))
        srcA = p.add64(P["pA"], p.widen(aOff))
        cB = p.sub32(c, BM * 2)
        n = p.shr32(cB, 1); chunkB = p.and32(cB, 1)
        pchB = p.xor32(chunkB, p.and32(p.shr32(n, 2), 1))
        dstB = p.add32(BM * 32, p.add32(p.shl32(n, 5), p.shl32(pchB, 4)))
        wRow = p.add32(P["weightColumnOffset"], p.add32(colBase, n))
        srcB = p.add64(P["pW"], p.widen(p.add32(p.shl32(wRow, 5), p.shl32(chunkB, 4))))
        dst = p.selp32(isA, dstA, dstB)
        src = p.reg("b64"); p.emit(f"selp.b64 {src}, {srcA}, {srcB}, {isA};")
        adv = p.selp32(isA, p.imm32(32), p.shl32(P["Nmatrix"], 5))   # per k32 step: A +32 B, W +Nmatrix*32 B
        copies.append((dst, src, adv))

    def issue_stage(stageIndex, step):
        for sub in range(ksub):
            for dst, src, adv in copies:
                saddr = p.add32(p.add32(smem, dst), p.imm32(stageIndex * STAGE_BYTES + sub * (BM + BN) * 32))
                gaddr = p.mad_wide(adv, p.imm32(step * ksub + sub), src)
                p.emit(f"cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;")
        p.emit("cp.async.commit_group;")

    l8 = p.and32(lane, 7); lm = p.shr32(lane, 3)
    aRowL = p.add32(l8, p.shl32(p.and32(lm, 1), 3)); aChunkL = p.shr32(lane, 4)
    aPch = p.xor32(aChunkL, p.and32(p.shr32(aRowL, 2), 1))
    aLane = p.add32(smem, p.add32(p.shl32(p.add32(aRowL, p.shl32(warp, 4)), 5), p.shl32(aPch, 4)))
    bNL = p.add32(l8, p.shl32(p.shr32(lane, 4), 3)); bChunkL = p.and32(lm, 1)
    bPch = p.xor32(bChunkL, p.and32(p.shr32(bNL, 2), 1))
    bLane = p.add32(smem, p.add32(BM * 32, p.add32(p.shl32(bNL, 5), p.shl32(bPch, 4))))

    # ---- output rows of this warp: g and g + 8
    outRow = [p.add32(blockRow, p.add32(p.shl32(warp, 4), p.add32(g, p.imm32(8 * h)))) for h in range(2)]
    rowOk = [p.setp("lt.u32", outRow[h], P["rows"]) for h in range(2)]
    outCol = p.add32(P["outputColumnOffset"], p.add32(colBase, p.shl32(t, 1)))   # + 8 j
    outIdx = [p.add32(p.mul32(outRow[h], P["outputStride"]), outCol) for h in range(2)]   # element index

    # tile row / chunk of this thread for the 64 x 64 B residual / output tiles: 256 chunks, 2 per thread (the copy
    # warp: 8 per lane for the residual)
    def tile_chunks(count, base):
        out = []
        for i in range(count):
            c = p.add32(base, i * (256 // count))
            row = p.shr32(c, 2); chunk = p.and32(c, 3)
            gRow = p.add32(blockRow, row)
            ok = p.setp("lt.u32", gRow, P["rows"])
            gOff = p.add32(p.mul32(gRow, P["outputStride"]), p.add32(P["outputColumnOffset"], p.add32(colBase, p.shl32(chunk, 4))))
            out.append((p.add32(p.shl32(row, 6), p.shl32(chunk, 4)), gOff, ok))
        return out
    tileChunks = tile_chunks(2, tid)
    resChunks = tile_chunks(8, copyLane) if ws else tileChunks

    def issue_residual():
        # residual tile through shared memory (16-byte copies), then the C-layout pairs
        for sOff, gOff, ok in resChunks:
            saddr = p.add32(p.add32(smem, sOff), p.imm32(RESS))
            gaddr = p.add64(P["pRes"], p.widen(gOff))
            p.emit(f"@{ok} cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;")

    acc = []
    if res:
        auxAddr = p.add64(P["pAux"], p.widen(p.shl32(p.add32(P["auxHalfOffset"], p.add32(colBase, p.shl32(t, 1))), 1)))
        scales = []
        for j in range(8):
            r = p.reg("b32"); p.emit(f"ld.global.nc.b32 {r}, [{auxAddr}+{16 * j}];"); scales.append(r)

    def seed_from_residual():
        cRes = p.add32(p.add32(smem, RESS), p.add32(p.shl32(p.add32(p.shl32(warp, 4), g), 6), p.shl32(t, 1)))
        for j in range(8):
            d = []
            for h in range(2):
                hc = p.reg("b16"); p.emit(f"ld.shared.b16 {hc}, [{cRes}+{8 * j + 512 * h}];")
                d.append(hmul2(p, e4x2_to_f16x2(p, hc), scales[j]))
            acc.append(d)

    if ws:
        # ---- the copy warp: residual + stage 0 as group 0, stages 1 .. S-2, then one stage per step; FULL(slot)
        # once the group landed, EMPTY(slot) before a slot is refilled. Barrier ids: FULL 1.., EMPTY 1 + stages ..
        FULL = lambda slot: 1 + slot
        EMPTY = lambda slot: 1 + stages + slot
        pProducer = p.setp("ge.u32", tid, 128)
        prodL = p.label("PRODUCER"); epiL = p.label("EPILOGUE")
        p.emit(f"@{pProducer} bra {prodL};")
        # ---- the MMA warps
        if not res: acc = [zero_tile(p) for _ in range(8)]
        for step in range(steps):
            stageIndex = step % stages
            p.emit(f"bar.sync {FULL(stageIndex)}, {NT};")
            if step == 0 and res: seed_from_residual()
            for sub in range(ksub):
                subOff = stageIndex * STAGE_BYTES + sub * (BM + BN) * 32
                a = ldm4(p, p.add32(aLane, subOff))
                rr = [ldm4(p, p.add32(bLane, subOff + i2 * 512)) for i2 in range(4)]
                for i2 in range(4):
                    r = rr[i2]
                    mma_e4(p, acc[2 * i2], a, (r[0], r[1]), acc[2 * i2])
                    mma_e4(p, acc[2 * i2 + 1], a, (r[2], r[3]), acc[2 * i2 + 1])
            if step + stages < steps:   # the slot is refilled with stage step + stages
                p.emit(f"bar.arrive {EMPTY(stageIndex)}, {NT};")
        p.emit(f"bra {epiL};")
        # ---- the copy warp
        p.emit(f"{prodL}:")
        if res: issue_residual()
        issue_stage(0, 0)                                  # commits group 0
        for s in range(1, stages - 1):
            issue_stage(s, s)
        for step in range(steps):
            j = step + stages - 1
            if j < steps:
                if j >= stages: p.emit(f"bar.sync {EMPTY(j % stages)}, {NT};")
                issue_stage(j % stages, j)
            committed = min(j, steps - 1) + 1              # groups 0 .. min(j, steps - 1)
            p.emit(f"cp.async.wait_group {committed - 1 - step};")   # group `step` landed
            p.emit(f"bar.arrive {FULL(step % stages)}, {NT};")
        p.emit("ret;")
        p.emit(f"{epiL}:")
    else:
      if res:
        issue_residual()
        p.emit("cp.async.commit_group;")
      for s in range(max(stages - 1, 1)):   # stages == 1 (K = 32): the only step is issued here
        issue_stage(s, s)
      # accumulator seed: round_f16(E4 residual * scale) at the output location
      if res:
        p.emit(f"cp.async.wait_group {stages - 1};")   # the residual group (the oldest) has landed
        p.emit("bar.sync 0;")
        seed_from_residual()
      else:
        acc = [zero_tile(p) for _ in range(8)]

    for step in (range(steps) if not ws else []):
        stageIndex = step % stages
        p.emit(f"cp.async.wait_group {max(stages - 2, 0)};")
        p.emit("bar.sync 0;")
        if stages > 1 and step + stages - 1 < steps:
            issue_stage((step + stages - 1) % stages, step + stages - 1)
        elif stages > 1:
            p.emit("cp.async.commit_group;")
        for sub in range(ksub):
            subOff = stageIndex * STAGE_BYTES + sub * (BM + BN) * 32
            a = ldm4(p, p.add32(aLane, subOff))
            rr = [ldm4(p, p.add32(bLane, subOff + i2 * 512)) for i2 in range(4)]
            for i2 in range(4):
                r = rr[i2]
                mma_e4(p, acc[2 * i2], a, (r[0], r[1]), acc[2 * i2])
                mma_e4(p, acc[2 * i2 + 1], a, (r[2], r[3]), acc[2 * i2 + 1])
    # ---- epilogue
    if siluF:
        k = SiluConsts(p)
        for j in range(8):
            for h in range(2):
                acc[j][h] = silu(p, acc[j][h], k)
    p.emit(BARC)   # every MMA warp finished the last stage (the ring becomes the f16 output tile)
    cOut = p.add32(p.add32(smem, RESS), p.add32(p.shl32(p.add32(p.shl32(warp, 4), g), 6), p.shl32(t, 1)))
    if outF16:
        # f16 tile [64][128 B] in the ring area, then 16-byte stores (512 chunks, 4 per thread)
        cOut16 = p.add32(smem, p.add32(p.shl32(p.add32(p.shl32(warp, 4), g), 7), p.shl32(t, 2)))
        for j in range(8):
            for h in range(2):
                p.emit(f"st.shared.b32 [{cOut16}+{16 * j + 1024 * h}], {acc[j][h]};")
    if outE4:
        for j in range(8):
            for h in range(2):
                hc = cvt_e4x2(p, nanzero(p, acc[j][h], zero))
                p.emit(f"st.shared.b16 [{cOut}+{8 * j + 512 * h}], {hc};")
    p.emit(BARC)
    if outF16:
        for i in range(4):
            c = p.add32(tid, i * THREADS)
            row = p.shr32(c, 3); chunk = p.and32(c, 7)
            gRow = p.add32(blockRow, row)
            ok = p.setp("lt.u32", gRow, P["rows"])
            gOff = p.add32(p.mul32(gRow, P["outputStride"]), p.add32(P["outputColumnOffset"], p.add32(colBase, p.shl32(chunk, 3))))
            r = p.regs("b32", 4)
            p.emit(f"ld.shared.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{p.add32(smem, p.add32(p.shl32(row, 7), p.shl32(chunk, 4)))}];")
            addr = p.add64(P["pOut16"], p.widen(p.shl32(gOff, 1)))
            p.emit(f"@{ok} st.global.v4.b32 [{addr}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")
    if outE4:
        for sOff, gOff, ok in tileChunks:
            r = p.regs("b32", 4)
            p.emit(f"ld.shared.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{p.add32(p.add32(smem, sOff), p.imm32(RESS))}];")
            addr = p.add64(P["pOut"], p.widen(gOff))
            p.emit(f"@{ok} st.global.v4.b32 [{addr}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")
    pSignalOn = p.setp("ne.u64", P["pSignal"], 0)
    p.emit("fence.acq_rel.gpu;")
    p.emit(BARC)
    pT0 = p.reg("pred"); p.emit(f"and.pred {pT0}, {pSignalOn}, {p.setp('eq.u32', tid, 0)};")
    bIter = p.reg("b32"); p.emit(f"mov.u32 {bIter}, {bandFirst};")
    sigLoop = p.label("SIG"); sigDone = p.label("SIGDONE")
    p.emit(f"@!{pT0} bra {sigDone};")
    p.emit(f"{sigLoop}:")
    sigAddr = p.add64(P["pSignal"], p.widen(p.shl32(bIter, 2)))
    p.emit(f"red.release.gpu.global.add.u32 [{sigAddr}], 1;")
    p.emit(f"add.u32 {bIter}, {bIter}, 1;")
    pMoreB = p.setp("le.u32", bIter, bandLast)
    p.emit(f"@{pMoreB} bra {sigLoop};")
    p.emit(f"{sigDone}:")
    p.emit("ret;")
    return name, p.finish()


if __name__ == "__main__":
    K, flags, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    max_regs = int(sys.argv[4]) if len(sys.argv) > 4 else None
    stagesWanted = int(sys.argv[5]) if len(sys.argv) > 5 else STAGES
    ksub = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    ws = len(sys.argv) > 7 and int(sys.argv[7]) != 0
    name, text = generate(K, flags, max_regs, stagesWanted, ksub, ws)
    open(out, "w").write(text)
    print(name)
