"""ViT FP8 GEMM (the few-row stages: 192 tokens): 192 x 128 output tiles, 8 warps of 48 x 64 (3 m-tiles x 8 n-tiles,
24 MMAs per 7 ldmatrix; two warps per sub-partition so one warp's loads hide behind the other's MMAs), one f16 MMA
chain per workgroup (the whole K, or one partition per split-K workgroup).

D = A[rows][K] x W (E4, k32-tile-major [K/32][Nmatrix][32]); total = ((seed + chain_0) + chain_1) + ... where chain_z
is split z's chain and seed = round_f16(E4 residual * scale) in split 0 (flag RES). Split-K partials are stored in
fragment order (one 128-byte line per warp store) and summed in slice order by the last-arriving workgroup of the
column group (a per-tile counter, zeroed per frame), which then publishes: SILU, E4 and / or F16 outputs. That is
gemm_fp8's `total = total + acc` and gemm_reduce's slice order.

Operand staging: an A ring (stages x 6 KB; the activations are L2-resident) and a deeper weight ring (bstages x 4 KB;
the DRAM stream), one cp.async group per k32 step; the fragments of the next step are loaded while the current
step's MMAs run (the ring keeps two complete stages). Chaining: the weight staging (and an optional L2 prefetch)
starts before the wait for `waitExpected` signals at pWait; pSignal is incremented once per published column group.

python gemmv_e4m3.py <K> <flags> <splits> out.ptx [stages] [bstages] [prefetch] [maxregs] [prefetch0] [BM]
  flags: RES=1 SILU=2 E4=4 F16=8
"""
import sys
from ptxgen import Ptx
from swin import *

F_RES, F_SILU, F_E4, F_F16 = 1, 2, 4, 8
BN = 128
B_BYTES = BN * 32                        # 4 KB per k32 step
STAGES, BSTAGES, PREFETCH = 6, 8, 0
PREFETCH0 = 0                            # weight tiles prefetched into L2 in the prologue beyond the staged ones (measured: hurts)


def generate(K, flags, splits, stages=STAGES, bstages=BSTAGES, prefetch=PREFETCH, max_regs=None, prefetch0=PREFETCH0, BM=192):
    assert K % 32 == 0 and (K // 32) % splits == 0
    assert BM % 48 == 0 and (BM // 48) in (1, 2, 4)
    rowQs = BM // 48                                       # 48-row warp quarters
    WARPS, THREADS = 2 * rowQs, 64 * rowQs
    A_BYTES = BM * 32                                      # per k32 step
    RES_BYTES = BM * BN                                    # the residual tile [BM][128 B] (E4)
    PARTIAL_WG_BYTES = WARPS * 48 * 128                    # the f16 partial in fragment order: [warp][48 tiles][32 lanes] x 4 B
    steps = K // 32 // splits                              # k32 steps of this workgroup's chain
    assert steps >= 3, "the fragment pipeline needs three stages"
    stages = max(3, min(stages, steps))
    bstages = max(stages, min(bstages, steps))
    lead = bstages - stages                                # extra weight lead (steps)
    res, siluF, outE4, outF16 = bool(flags & F_RES), bool(flags & F_SILU), bool(flags & F_E4), bool(flags & F_F16)
    name = f"gemmv_e4m3_K{K}_f{flags}_s{splits}" + (f"_m{BM}" if BM != 192 else "")
    A_RING, B_RING = 0, stages * A_BYTES
    RES_OFF = B_RING + bstages * B_BYTES
    ring_bytes = RES_OFF + (RES_BYTES if res else 0)
    E4OFF = BM * 256 if (outE4 and outF16) else 0          # the E4 tile beside the f16 tile only for dual outputs
    epilogue_bytes = max(BM * 256 if outF16 else 0, E4OFF + (BM * 128 if outE4 else 0))
    shared_bytes = max(ring_bytes, epilogue_bytes)
    p = Ptx()
    params = [("u64", "pA"), ("u64", "pW"), ("u64", "pRes"), ("u64", "pAux"), ("u64", "pOut"), ("u64", "pOut16"),
              ("u64", "pPartial"), ("u64", "pCount"),
              ("u32", "rows"), ("u32", "inputStride"), ("u32", "inputColumnBase"), ("u32", "Nmatrix"),
              ("u32", "weightColumnOffset"), ("u32", "outputStride"), ("u32", "outputColumnOffset"), ("u32", "auxHalfOffset"),
              ("u64", "pWait"), ("u32", "waitExpected"), ("u64", "pSignal"), ("u64", "pError")]
    p.entry(name, params, shared_bytes, THREADS, max_regs, dynamic_shared=True)
    P = {n: (p.load_param_u64(n) if t == "u64" else p.load_param_u32(n)) for t, n in params}
    tid = p.special("tid.x"); colGroup = p.special("ctaid.x"); rowGroup = p.special("ctaid.y"); split = p.special("ctaid.z")
    colGroups = p.special("nctaid.x"); rowGroups = p.special("nctaid.y")
    lane = p.and32(tid, 31); warp = p.shr32(tid, 5)
    rowQ = p.and32(warp, rowQs - 1); colH = p.shr32(warp, {1: 0, 2: 1, 4: 2}[rowQs])   # the warp's 48-row quarter and 64-column half
    li = LaneInfo(p, lane)
    g, t = li.g, li.t
    smem = p.shared_addr(p.imm32(0))
    blockRow = p.mul32(rowGroup, p.imm32(BM))
    colBase = p.shl32(colGroup, p.imm32(7))
    warpCol = p.add32(colBase, p.shl32(colH, p.imm32(6)))     # first output column of this warp
    rowsMinus1 = p.sub32(P["rows"], p.imm32(1))
    kStep0 = p.mul32(split, p.imm32(steps))                   # first k32 step of this workgroup's chain
    zero32 = p.imm32(0)

    # ---- cp.async chunks per step: A 2 BM (chunks tid + THREADS i), W 256; a row is two 16-byte chunks,
    # chunk ^= (row >> 2) & 1 (conflict-free ldmatrix)
    aChunk = p.and32(tid, 1)
    aDst = []; aSrc = []; aPred = []
    for i in range((2 * BM + THREADS - 1) // THREADS):
        aPred.append(p.setp("lt.u32", tid, p.imm32(2 * BM - THREADS * i)) if THREADS * (i + 1) > 2 * BM else None)
        aRow = p.add32(p.shr32(tid, 1), p.imm32(THREADS // 2 * i))
        aPch = p.xor32(aChunk, p.and32(p.shr32(aRow, 2), 1))
        aDst.append(p.add32(smem, p.add32(p.shl32(aRow, 5), p.shl32(aPch, 4))))
        gRow = p.add32(blockRow, aRow)
        gRowC = p.selp32(p.setp("lt.u32", gRow, P["rows"]), gRow, rowsMinus1)
        base = p.add64(P["pA"], p.widen(p.add32(p.mul32(gRowC, P["inputStride"]), p.add32(P["inputColumnBase"], p.shl32(aChunk, 4)))))
        aSrc.append(p.add64(base, p.widen(p.shl32(kStep0, 5))))   # + step * 32 per k32 step
    wStep = p.shl32(P["Nmatrix"], 5)
    wStep64 = p.widen(wStep)
    bDst = []; bPtr = []
    for i in range(256 // THREADS):
        c = p.add32(tid, p.imm32(THREADS * i))
        n = p.shr32(c, 1); bChunk = p.and32(c, 1)
        bPch = p.xor32(bChunk, p.and32(p.shr32(n, 2), 1))
        bDst.append(p.add32(smem, p.add32(p.imm32(B_RING), p.add32(p.shl32(n, 5), p.shl32(bPch, 4)))))
        wRow = p.add32(P["weightColumnOffset"], p.add32(colBase, n))
        bSrc = p.mad_wide(wStep, kStep0, p.add64(P["pW"], p.widen(p.add32(p.shl32(wRow, 5), p.shl32(bChunk, 4)))))
        r = p.reg("b64"); p.emit(f"mov.b64 {r}, {bSrc};"); bPtr.append(r)   # the next weight tile to stage (advances by wStep)
    pPf = p.setp("lt.u32", tid, p.imm32(32))                            # the 4 KB weight tile is 32 lines
    pfBase = p.add64(P["pW"], p.widen(p.add32(p.shl32(p.add32(P["weightColumnOffset"], colBase), 5), p.shl32(tid, 7))))
    pfPtr = p.reg("b64"); p.emit(f"mad.wide.u32 {pfPtr}, {wStep}, {p.add32(kStep0, p.imm32(lead + stages - 1))}, {pfBase};")
    bIssued = [0]; pfIssued = [lead + stages - 1]                      # prefetching starts beyond the staged prologue tiles

    def issue_A(stepLocal):
        slot = stepLocal % stages
        for i in range(len(aDst)):
            guard = f"@{aPred[i]} " if aPred[i] is not None else ""
            p.emit(f"{guard}cp.async.cg.shared.global [{aDst[i]}+{A_RING + slot * A_BYTES}], [{aSrc[i]}+{32 * stepLocal}], 16;")

    def issue_B(stepLocal):
        assert stepLocal == bIssued[0]
        bIssued[0] += 1
        slot = stepLocal % bstages
        for i in range(len(bDst)):
            p.emit(f"cp.async.cg.shared.global [{bDst[i]}+{slot * B_BYTES}], [{bPtr[i]}], 16;")
            p.emit(f"add.u64 {bPtr[i]}, {bPtr[i]}, {wStep64};")

    def prefetch_B(stepLocal):
        if stepLocal >= steps: return
        assert stepLocal == pfIssued[0]
        p.emit(f"@{pPf} prefetch.global.L2 [{pfPtr}];")
        p.emit(f"add.u64 {pfPtr}, {pfPtr}, {wStep64};")
        pfIssued[0] += 1

    # ---- ldmatrix lane addresses (A: 16 rows x 32 B per m-tile of this warp's quarter; B: 16 columns per x4 of
    # this warp's half)
    l8 = p.and32(lane, 7); lm = p.shr32(lane, 3)
    aRowL = p.add32(l8, p.shl32(p.and32(lm, 1), 3)); aChunkL = p.shr32(lane, 4)
    aPchL = p.xor32(aChunkL, p.and32(p.shr32(aRowL, 2), 1))
    aLane = p.add32(smem, p.add32(p.shl32(p.add32(aRowL, p.mul32(rowQ, p.imm32(48))), 5), p.shl32(aPchL, 4)))
    bNL = p.add32(l8, p.shl32(p.shr32(lane, 4), 3)); bChunkL = p.and32(lm, 1)
    bPchL = p.xor32(bChunkL, p.and32(p.shr32(bNL, 2), 1))
    bLane = p.add32(smem, p.add32(p.imm32(B_RING), p.add32(p.shl32(p.add32(bNL, p.shl32(colH, 6)), 5), p.shl32(bPchL, 4))))
    def load_A(stepLocal):
        slot = stepLocal % stages
        return [ldm4(p, f"{aLane}+{A_RING + slot * A_BYTES + 512 * m}") for m in range(3)]

    def load_B(stepLocal, i2):
        slot = stepLocal % bstages
        return ldm4(p, f"{bLane}+{slot * B_BYTES + 512 * i2}")

    # output rows of this warp's tiles: 48 rowQ + 16 m + g + 8 h (row & 7 = g)
    outRow = [[p.add32(blockRow, p.add32(p.mul32(rowQ, p.imm32(48)), p.add32(g, p.imm32(16 * m + 8 * h)))) for h in range(2)] for m in range(3)]
    rowOk = [[p.setp("lt.u32", outRow[m][h], P["rows"]) for h in range(2)] for m in range(3)]

    # ---- cp.async groups: every step commits the weight group B(s + stages - 1 + lead) and then the A group
    # A(s + stages - 1); wait_group 2 stages - 6 at the top of step s completes A(s + 1) and everything older, which
    # includes B(s + 1) (committed `lead` steps earlier). The weights thus lead by stages - 2 + lead steps.
    # Weight staging and L2 prefetch start before the dependency wait (the weights do not depend on the producer).
    for s in range(lead + stages - 1, lead + stages - 1 + prefetch0 + prefetch):   # the staged tiles need no prefetch
        prefetch_B(s)
    for s in range(min(lead, steps)):
        issue_B(s)
    p.emit("cp.async.commit_group;")
    pWaitOn = p.setp("ne.u64", P["pWait"], "0")
    sync_wait(p, P["pWait"], zero32, zero32, P["waitExpected"], lane, pWaitOn, warp, error64=P["pError"])
    if res:
        # split 0 stages the residual tile [192][128 B] (16-byte chunk ^= row & 7) at RES_OFF, within group 0
        pSeed = p.setp("eq.u32", split, zero32)
        for i in range(BM * 8 // THREADS):   # BM x 8 chunks
            c = p.add32(tid, p.imm32(THREADS * i))
            row = p.shr32(c, 3); chunk = p.and32(c, 7)
            gRow = p.add32(blockRow, row)
            ok = p.reg("pred"); p.emit(f"setp.lt.and.u32 {ok}, {gRow}, {P['rows']}, {pSeed};")
            src = p.add64(P["pRes"], p.widen(p.add32(p.mul32(gRow, P["outputStride"]), p.add32(P["outputColumnOffset"], p.add32(colBase, p.shl32(chunk, 4))))))
            dst = p.add32(smem, p.add32(p.imm32(RES_OFF), p.add32(p.shl32(row, 7), p.shl32(p.xor32(chunk, p.and32(row, 7)), 4))))
            p.emit(f"@{ok} cp.async.cg.shared.global [{dst}], [{src}], 16;")
        auxAddr = p.add64(P["pAux"], p.widen(p.shl32(p.add32(P["auxHalfOffset"], p.add32(warpCol, p.shl32(t, 1))), 1)))
        scales = []
        for j in range(8):
            r = p.reg("b32"); p.emit(f"ld.global.nc.b32 {r}, [{auxAddr}+{16 * j}];"); scales.append(r)
    for s in range(stages - 1):
        if s + lead < steps: issue_B(s + lead)
        p.emit("cp.async.commit_group;")
        issue_A(s)
        p.emit("cp.async.commit_group;")
    acc = [[zero_tile(p) for _ in range(8)] for _ in range(3)]
    # ---- main loop: step s consumes stage s (fragments loaded during step s - 1), stages s and s + 1 are complete
    # after the barrier; the issue refills the slots read at step s - 1
    A = B = None
    for s in range(steps):
        p.emit(f"cp.async.wait_group {2 * stages - 6};")
        p.emit("bar.sync 0;")
        if s == 0:
            if res:
                # the chain seed round_f16(residual * scale) from the staged tile (split 0 only)
                resLane = p.add32(smem, p.add32(p.imm32(RES_OFF), p.shl32(t, 1)))
                codes = {}
                for m in range(3):
                    for h in range(2):
                        rowL = p.add32(p.mul32(rowQ, p.imm32(48)), p.add32(g, p.imm32(16 * m + 8 * h)))
                        rowOff = p.add32(resLane, p.shl32(rowL, 7))
                        for j in range(8):
                            chunk = p.add32(p.shl32(p.xor32(p.add32(p.shl32(colH, 2), p.imm32(j >> 1)), g), 4), p.imm32(8 * (j & 1)))
                            hc = p.reg("b16"); p.emit(f"ld.shared.b16 {hc}, [{p.add32(rowOff, chunk)}];")
                            codes[m, h, j] = hc
                for m in range(3):
                    for h in range(2):
                        for j in range(8):
                            v = hmul2(p, e4x2_to_f16x2(p, codes[m, h, j]), scales[j])
                            p.emit(f"@{pSeed} mov.b32 {acc[m][j][h]}, {v};")
            A = load_A(0); B = load_B(0, 0)
        j = s + stages - 1
        def issue_step():
            if j + lead < steps: issue_B(j + lead)
            if prefetch: prefetch_B(j + lead + prefetch0 + prefetch)
            p.emit("cp.async.commit_group;")
            if j < steps: issue_A(j)
            p.emit("cp.async.commit_group;")
        issue_step()   # the step's copies right after the barrier
        for i2 in range(4):
            if i2 < 3:
                Bn = load_B(s, i2 + 1)
            elif s + 1 < steps:
                An = load_A(s + 1); Bn = load_B(s + 1, 0)
            for m in range(3):
                mma_e4(p, acc[m][2 * i2], A[m], (B[0], B[1]), acc[m][2 * i2])
                mma_e4(p, acc[m][2 * i2 + 1], A[m], (B[2], B[3]), acc[m][2 * i2 + 1])
            B = Bn
            if i2 == 3 and s + 1 < steps: A = An
    p.emit("bar.sync 0;")                                     # the ring is dead: the staging tiles alias it
    endL = p.label("END")

    if splits > 1:
        # the partial in fragment order: [split][rowGroup][colGroup][warp][m, j, h][lane] f16x2 (a line per store)
        tileIndex = p.add32(colGroup, p.mul32(rowGroup, colGroups))
        tiles = p.mul32(rowGroups, colGroups)
        sliceBytes = p.mul32(tiles, p.imm32(PARTIAL_WG_BYTES))
        tileBase = p.mad_wide(p.add32(p.mul32(tileIndex, p.imm32(WARPS)), warp), p.imm32(48 * 128), P["pPartial"])
        laneBase = p.add64(tileBase, p.widen(p.shl32(lane, 2)))          # slice 0 of this tile / warp / lane
        ownBase = p.mad_wide(sliceBytes, split, laneBase)
        for m in range(3):
            for j in range(8):
                for h in range(2):
                    p.emit(f"st.global.b32 [{ownBase}+{(m * 16 + j * 2 + h) * 128}], {acc[m][j][h]};")
        p.emit("fence.acq_rel.gpu;")
        p.emit("bar.sync 0;")
        pT0 = p.setp("eq.u32", tid, zero32)
        countAddr = p.add64(P["pCount"], p.widen(p.shl32(tileIndex, 2)))
    if splits > 1:
        # last arrival of the column group reduces (threadfence reduction)
        old = p.reg("b32"); p.emit(f"mov.u32 {old}, 0;")
        p.emit(f"@{pT0} atom.acq_rel.gpu.global.add.u32 {old}, [{countAddr}], 1;")
        p.emit(f"@{pT0} st.shared.u32 [{smem}], {old};")
        p.emit("bar.sync 0;")
        p.emit(f"ld.shared.u32 {old}, [{smem}];")
        pLast = p.setp("eq.u32", old, p.imm32(splits - 1))
        p.emit(f"@!{pLast} bra {endL};")
        p.emit("fence.acq_rel.gpu;")
        total = [[[None, None] for _ in range(8)] for _ in range(3)]
        for z in range(splits):
            pIs = p.setp("eq.u32", split, p.imm32(z))
            pNot = p.reg("pred"); p.emit(f"not.pred {pNot}, {pIs};")
            zBase = p.mad_wide(sliceBytes, p.imm32(z), laneBase)
            loaded = {}
            for m in range(3):   # the slice's 48 loads first (independent), then the in-order sum
                for j in range(8):
                    for h in range(2):
                        v = p.reg("b32"); p.emit(f"mov.b32 {v}, 0;")
                        p.emit(f"@{pNot} ld.global.cg.b32 {v}, [{zBase}+{(m * 16 + j * 2 + h) * 128}];")
                        loaded[m, j, h] = v
            for m in range(3):
                for j in range(8):
                    for h in range(2):
                        v = p.selp32(pIs, acc[m][j][h], loaded[m, j, h])
                        total[m][j][h] = v if z == 0 else hadd2(p, total[m][j][h], v)
        acc = total
    # ---- publish
    if siluF:
        k = SiluConsts(p)
        for m in range(3):
            for j in range(8):
                for h in range(2):
                    acc[m][j][h] = silu(p, acc[m][j][h], k)
    if outF16:
        # f16 tile [192][256 B], 16-byte chunk ^= row & 7 (= g for this warp's rows): conflict-free C stores
        for m in range(3):
            for h in range(2):
                rowL = p.add32(p.mul32(rowQ, p.imm32(48)), p.add32(g, p.imm32(16 * m + 8 * h)))
                rowOff = p.add32(smem, p.add32(p.shl32(rowL, 8), p.shl32(t, 2)))
                for j in range(8):
                    chunk = p.shl32(p.xor32(p.add32(p.shl32(colH, 3), p.imm32(j)), g), 4)
                    p.emit(f"st.shared.b32 [{p.add32(rowOff, chunk)}], {acc[m][j][h]};")
    if outE4:
        # E4 tile [192][128 B], 16-byte chunk ^= row & 7
        for m in range(3):
            for h in range(2):
                rowL = p.add32(p.mul32(rowQ, p.imm32(48)), p.add32(g, p.imm32(16 * m + 8 * h)))
                rowOff = p.add32(smem, p.add32(p.imm32(E4OFF), p.add32(p.shl32(rowL, 7), p.shl32(t, 1))))
                for j in range(8):
                    chunk = p.add32(p.shl32(p.xor32(p.add32(p.shl32(colH, 2), p.imm32(j >> 1)), g), 4), p.imm32(8 * (j & 1)))
                    hc = cvt_e4x2(p, acc[m][j][h])
                    p.emit(f"st.shared.b16 [{p.add32(rowOff, chunk)}], {hc};")
    p.emit("bar.sync 0;")
    if outF16:
        for i in range(BM * 16 // THREADS):   # BM x 16 chunks of 16 B
            c = p.add32(tid, p.imm32(THREADS * i))
            row = p.shr32(c, 4); chunk = p.and32(c, 15)
            phys = p.add32(smem, p.add32(p.shl32(row, 8), p.shl32(p.xor32(chunk, p.and32(row, 7)), 4)))
            r = p.regs("b32", 4)
            p.emit(f"ld.shared.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{phys}];")
            gRowO = p.add32(blockRow, row)
            ok = p.setp("lt.u32", gRowO, P["rows"])
            off = p.add32(p.mul32(gRowO, P["outputStride"]), p.add32(P["outputColumnOffset"], p.add32(colBase, p.shl32(chunk, 3))))
            p.emit(f"@{ok} st.global.v4.b32 [{p.add64(P['pOut16'], p.widen(p.shl32(off, 1)))}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")
    if outE4:
        for i in range(BM * 8 // THREADS):   # BM x 8 chunks of 16 B
            c = p.add32(tid, p.imm32(THREADS * i))
            row = p.shr32(c, 3); chunk = p.and32(c, 7)
            phys = p.add32(smem, p.add32(p.imm32(E4OFF), p.add32(p.shl32(row, 7), p.shl32(p.xor32(chunk, p.and32(row, 7)), 4))))
            r = p.regs("b32", 4)
            p.emit(f"ld.shared.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{phys}];")
            gRowO = p.add32(blockRow, row)
            ok = p.setp("lt.u32", gRowO, P["rows"])
            off = p.add32(p.mul32(gRowO, P["outputStride"]), p.add32(P["outputColumnOffset"], p.add32(colBase, p.shl32(chunk, 4))))
            p.emit(f"@{ok} st.global.v4.b32 [{p.add64(P['pOut'], p.widen(off))}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")
    pSignalOn = p.setp("ne.u64", P["pSignal"], "0")
    sync_signal(p, [P["pSignal"]], tid, pSignalOn)
    p.emit(f"{endL}:")
    p.emit("ret;")
    return name, p.finish(), shared_bytes


if __name__ == "__main__":
    K, flags, splits, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    stages = int(sys.argv[5]) if len(sys.argv) > 5 else STAGES
    bstages = int(sys.argv[6]) if len(sys.argv) > 6 else BSTAGES
    prefetch = int(sys.argv[7]) if len(sys.argv) > 7 else PREFETCH
    max_regs = int(sys.argv[8]) if len(sys.argv) > 8 and int(sys.argv[8]) else None
    prefetch0 = int(sys.argv[9]) if len(sys.argv) > 9 else PREFETCH0
    bm = int(sys.argv[10]) if len(sys.argv) > 10 else 192
    name, text, shared = generate(K, flags, splits, stages, bstages, prefetch, max_regs, prefetch0, bm)
    open(out, "w").write(text)
    print(name, shared)
