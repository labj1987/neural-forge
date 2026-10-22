"""Tall-tile FP8 GEMM for the few-row stages (ViT: 192 tokens): 192 x 64 output tiles (12 warps), partitions and
split-K whose partial sums are added in partition order, which is the association the unsplit chain has.

D = A[rows][K] x W (E4, k32-tile-major [K/32][Nmatrix][32]), f16 accumulation chained in K order per partition:
total = ((seed + chain_0) + chain_1) + ... where chain_p is the MMA chain over partition p (PARTITION k values) and
seed = round_f16(E4 residual * scale) (flag RES; split 0 only). Split-K: workgroup z runs partitions
[z * partsPerSplit, (z + 1) * partsPerSplit) and stores its f16 total to the partial buffer slice z; the reduce
kernel sums the slices in order and publishes (that is `total = total + acc` of gemm_fp8 / gemm_reduce).
Without splits the epilogue publishes directly: SILU, E4 and / or F16 outputs.
The weight tile (2 KB) is read once per row group: for 192-row problems the operand traffic is the weights once
plus A once per column group.

python gemmt_e4m3.py <K> <flags> <partitionSteps> <splits> out.ptx [maxregs]   flags: RES=1 SILU=2 E4=4 F16=8
"""
import sys
from ptxgen import Ptx
from swin import *

F_RES, F_SILU, F_E4, F_F16 = 1, 2, 4, 8
BM, BN = 192, 64
WARPS = BM // 16
THREADS = 32 * WARPS
STAGES = 3
STAGE_BYTES = (BM + BN) * 32     # 8 KB


PREFETCH = 8                     # weight k32 steps prefetched into L2 ahead of the cp.async issue (DRAM latency hiding)

def generate(K, flags, partSteps, splits, max_regs=None, stagesWanted=STAGES, prefetch=PREFETCH):
    assert K % 32 == 0
    steps = K // 32
    if partSteps == 0: partSteps = steps
    assert steps % partSteps == 0 and (splits == 1 or steps == partSteps * splits)   # split-K: one partition per split
    stepsPerSplit = steps // splits
    stages = min(stagesWanted, stepsPerSplit)
    res, siluF, outE4, outF16 = bool(flags & F_RES), bool(flags & F_SILU), bool(flags & F_E4), bool(flags & F_F16)
    name = f"gemmt_e4m3_K{K}_f{flags}_p{partSteps}_s{splits}"
    RING = 0
    OUTS = 0                                    # the output tile aliases the ring after the loop
    hasF16 = splits > 1 or outF16
    E4OFF = BM * 128 if hasF16 else 0           # the E4 tile [192][64 B] follows the f16 tile [192][128 B] or aliases the ring
    shared_bytes = max(stages * STAGE_BYTES, BM * 128 if hasF16 else 0) + (BM * 64 if (outE4 and splits == 1 and hasF16) else 0)
    assert shared_bytes >= E4OFF + BM * 64 or not (outE4 and splits == 1)
    p = Ptx()
    params = [("u64", "pA"), ("u64", "pW"), ("u64", "pRes"), ("u64", "pAux"), ("u64", "pOut"), ("u64", "pOut16"),
              ("u32", "rows"), ("u32", "inputStride"), ("u32", "inputColumnBase"), ("u32", "Nmatrix"),
              ("u32", "weightColumnOffset"), ("u32", "outputStride"), ("u32", "outputColumnOffset"), ("u32", "auxHalfOffset"),
              ("u32", "splitStride")]
    p.entry(name, params, shared_bytes, THREADS, max_regs, dynamic_shared=True)
    P = {n: (p.load_param_u64(n) if t == "u64" else p.load_param_u32(n)) for t, n in params}
    tid = p.special("tid.x"); colGroup = p.special("ctaid.x"); rowGroup = p.special("ctaid.y"); split = p.special("ctaid.z")
    lane = p.and32(tid, 31); warp = p.shr32(tid, 5)
    li = LaneInfo(p, lane)
    g, t = li.g, li.t
    smem = p.shared_addr(p.imm32(0))
    zero = imm(p, 0)
    blockRow = p.mul32(rowGroup, BM)
    colBase = p.shl32(colGroup, 6)
    rowsMinus1 = p.sub32(P["rows"], 1)
    kStep0 = p.mul32(split, stepsPerSplit)                       # first k32 step of this split

    # ---- cp.async chunks per stage: A 384 (one per thread), W 128 (threads < 128); chunk ^= (row >> 2) & 1
    aRow = p.shr32(tid, 1); aChunk = p.and32(tid, 1)
    aPch = p.xor32(aChunk, p.and32(p.shr32(aRow, 2), 1))
    aDst = p.add32(p.shl32(aRow, 5), p.shl32(aPch, 4))
    gRow = p.add32(blockRow, aRow)
    gRowC = p.selp32(p.setp("lt.u32", gRow, P["rows"]), gRow, rowsMinus1)
    aSrc = p.add64(P["pA"], p.widen(p.add32(p.mul32(gRowC, P["inputStride"]), p.add32(P["inputColumnBase"], p.shl32(aChunk, 4)))))
    pB = p.setp("lt.u32", tid, 128)
    n = p.shr32(tid, 1); bChunk = p.and32(tid, 1)
    bPchC = p.xor32(bChunk, p.and32(p.shr32(n, 2), 1))
    bDst = p.add32(BM * 32, p.add32(p.shl32(n, 5), p.shl32(bPchC, 4)))
    wRow = p.add32(P["weightColumnOffset"], p.add32(colBase, n))
    bSrc = p.add64(P["pW"], p.widen(p.add32(p.shl32(wRow, 5), p.shl32(bChunk, 4))))
    wStep = p.shl32(P["Nmatrix"], 5)
    # L2 prefetch of the weight tile `prefetch` steps ahead of its cp.async: the 2 KB tile is 16 lines, threads 0..15
    pPf = p.setp("lt.u32", tid, 16)
    pfBase = p.add64(P["pW"], p.widen(p.add32(p.shl32(p.add32(P["weightColumnOffset"], colBase), 5), p.shl32(tid, 7))))

    def prefetch_step(stepLocal):
        if not prefetch or stepLocal >= stepsPerSplit: return
        stepG = p.add32(kStep0, p.imm32(stepLocal))
        p.emit(f"@{pPf} prefetch.global.L2 [{p.mad_wide(wStep, stepG, pfBase)}];")

    def issue_stage(stageIndex, stepLocal):
        stepG = p.add32(kStep0, p.imm32(stepLocal))
        sa = p.add32(p.add32(smem, aDst), p.imm32(RING + stageIndex * STAGE_BYTES))
        ga = p.add64(aSrc, p.widen(p.shl32(stepG, 5)))
        p.emit(f"cp.async.cg.shared.global [{sa}], [{ga}], 16;")
        sb = p.add32(p.add32(smem, bDst), p.imm32(RING + stageIndex * STAGE_BYTES))
        gb = p.mad_wide(wStep, stepG, bSrc)
        p.emit(f"@{pB} cp.async.cg.shared.global [{sb}], [{gb}], 16;")
        p.emit("cp.async.commit_group;")
        prefetch_step(stepLocal + prefetch)

    l8 = p.and32(lane, 7); lm = p.shr32(lane, 3)
    aRowL = p.add32(l8, p.shl32(p.and32(lm, 1), 3)); aChunkL = p.shr32(lane, 4)
    aPchL = p.xor32(aChunkL, p.and32(p.shr32(aRowL, 2), 1))
    aLane = p.add32(smem, p.add32(p.shl32(p.add32(aRowL, p.shl32(warp, 4)), 5), p.shl32(aPchL, 4)))
    bNL = p.add32(l8, p.shl32(p.shr32(lane, 4), 3)); bChunkL = p.and32(lm, 1)
    bPchL = p.xor32(bChunkL, p.and32(p.shr32(bNL, 2), 1))
    bLane = p.add32(smem, p.add32(BM * 32, p.add32(p.shl32(bNL, 5), p.shl32(bPchL, 4))))

    outRow = [p.add32(blockRow, p.add32(p.shl32(warp, 4), p.add32(g, p.imm32(8 * h)))) for h in range(2)]
    rowOk = [p.setp("lt.u32", outRow[h], P["rows"]) for h in range(2)]
    outCol = p.add32(P["outputColumnOffset"], p.add32(colBase, p.shl32(t, 1)))
    outIdx = [p.add32(p.mul32(outRow[h], P["outputStride"]), outCol) for h in range(2)]

    for s in range(max(stages - 1, 1), max(stages - 1, 1) + prefetch):
        prefetch_step(s)
    for s in range(max(stages - 1, 1)):
        issue_stage(s, s)
    # ---- accumulators: partition chain `acc`, running `total`; the seed only in split 0's first partition
    acc = [zero_tile(p) for _ in range(8)]
    if res:
        pSeed = p.setp("eq.u32", split, 0)
        auxAddr = p.add64(P["pAux"], p.widen(p.shl32(p.add32(P["auxHalfOffset"], p.add32(colBase, p.shl32(t, 1))), 1)))
        scales = []
        for j in range(8):
            r = p.reg("b32"); p.emit(f"ld.global.nc.b32 {r}, [{auxAddr}+{16 * j}];"); scales.append(r)
        resAddr = [p.add64(P["pRes"], p.widen(outIdx[h])) for h in range(2)]
        for j in range(8):
            for h in range(2):
                hc = p.reg("b16"); p.emit(f"mov.b16 {hc}, 0;")
                pr = p.reg("pred"); p.emit(f"and.pred {pr}, {rowOk[h]}, {pSeed};")
                p.emit(f"@{pr} ld.global.cg.b16 {hc}, [{resAddr[h]}+{8 * j}];")
                v = hmul2(p, e4x2_to_f16x2(p, hc), scales[j])
                p.emit(f"mov.b32 {acc[j][h]}, {v};")
    total = [zero_tile(p) for _ in range(8)]
    for step in range(stepsPerSplit):
        stageIndex = step % stages
        p.emit(f"cp.async.wait_group {max(stages - 2, 0)};")
        p.emit("bar.sync 0;")
        if stages > 1 and step + stages - 1 < stepsPerSplit:
            issue_stage((step + stages - 1) % stages, step + stages - 1)
        elif stages > 1:
            p.emit("cp.async.commit_group;")
        a = ldm4(p, p.add32(aLane, stageIndex * STAGE_BYTES))
        for i2 in range(4):
            r = ldm4(p, p.add32(bLane, stageIndex * STAGE_BYTES + i2 * 512))
            mma_e4(p, acc[2 * i2], a, (r[0], r[1]), acc[2 * i2])
            mma_e4(p, acc[2 * i2 + 1], a, (r[2], r[3]), acc[2 * i2 + 1])
        # partition boundary: total = (first partition of this split ? acc : total + acc); acc = 0
        if (step + 1) % partSteps == 0:
            firstPart = (step + 1) // partSteps == 1
            for j in range(8):
                for h in range(2):
                    if firstPart:
                        p.emit(f"mov.b32 {total[j][h]}, {acc[j][h]};")
                    else:
                        s2 = hadd2(p, total[j][h], acc[j][h]); p.emit(f"mov.b32 {total[j][h]}, {s2};")
                    p.emit(f"mov.b32 {acc[j][h]}, 0;")
    # ---- epilogue (the ring is dead after this barrier: the output tile aliases it)
    if siluF and splits == 1:
        k = SiluConsts(p)
        for j in range(8):
            for h in range(2):
                total[j][h] = silu(p, total[j][h], k)
    p.emit("bar.sync 0;")
    if splits > 1 or outF16:
        # f16 tile [192][128 B] then 16-byte stores (1536 chunks, 4 per thread)
        cOut16 = p.add32(smem, p.add32(p.shl32(p.add32(p.shl32(warp, 4), g), 7), p.shl32(t, 2)))
        for j in range(8):
            for h in range(2):
                p.emit(f"st.shared.b32 [{cOut16}+{16 * j + 1024 * h}], {total[j][h]};")
    if outE4 and splits == 1:
        cOut = p.add32(smem, p.add32(p.imm32(E4OFF), p.add32(p.shl32(p.add32(p.shl32(warp, 4), g), 6), p.shl32(t, 1))))
        for j in range(8):
            for h in range(2):
                hc = cvt_e4x2(p, total[j][h])
                p.emit(f"st.shared.b16 [{cOut}+{8 * j + 512 * h}], {hc};")
    p.emit("bar.sync 0;")
    if splits > 1 or outF16:
        dst16 = P["pOut16"]
        if splits > 1:
            dst16 = p.add64(dst16, p.widen(p.shl32(p.mul32(split, P["splitStride"]), 1)))
        for i in range(4):
            c = p.add32(tid, i * THREADS)
            row = p.shr32(c, 3); chunk = p.and32(c, 7)
            gRowO = p.add32(blockRow, row)
            ok = p.setp("lt.u32", gRowO, P["rows"])
            r = p.regs("b32", 4)
            p.emit(f"ld.shared.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{p.add32(smem, p.add32(p.shl32(row, 7), p.shl32(chunk, 4)))}];")
            # splits > 1: the host passes outputStride = N, outputColumnOffset = 0 (the partial slice layout)
            gOff = p.add32(p.mul32(gRowO, P["outputStride"]), p.add32(P["outputColumnOffset"], p.add32(colBase, p.shl32(chunk, 3))))
            addr = p.add64(dst16, p.widen(p.shl32(gOff, 1)))
            p.emit(f"@{ok} st.global.v4.b32 [{addr}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")
    if outE4 and splits == 1:
        for i in range(2):   # 768 chunks of 16 B: 2 per thread
            c = p.add32(tid, i * THREADS)
            row = p.shr32(c, 2); chunk = p.and32(c, 3)
            gRowO = p.add32(blockRow, row)
            ok = p.setp("lt.u32", gRowO, P["rows"])
            r = p.regs("b32", 4)
            p.emit(f"ld.shared.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{p.add32(smem, p.add32(p.imm32(E4OFF), p.add32(p.shl32(row, 6), p.shl32(chunk, 4))))}];")
            gOff = p.add32(p.mul32(gRowO, P["outputStride"]), p.add32(P["outputColumnOffset"], p.add32(colBase, p.shl32(chunk, 4))))
            addr = p.add64(P["pOut"], p.widen(gOff))
            p.emit(f"@{ok} st.global.v4.b32 [{addr}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")
    p.emit("ret;")
    return name, p.finish(), shared_bytes


if __name__ == "__main__":
    K, flags, partSteps, splits, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    max_regs = int(sys.argv[6]) if len(sys.argv) > 6 else None
    stagesWanted = int(sys.argv[7]) if len(sys.argv) > 7 else STAGES
    prefetch = int(sys.argv[8]) if len(sys.argv) > 8 else PREFETCH
    name, text, shared = generate(K, flags, partSteps, splits, max_regs, stagesWanted, prefetch)
    open(out, "w").write(text)
    print(name, shared)
