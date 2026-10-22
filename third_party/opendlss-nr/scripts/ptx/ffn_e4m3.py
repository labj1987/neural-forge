"""Expert FFN + W3 of one block in a single PTX kernel: E4( E4(SiLU(x W1_e)) W2_e ) for every expert e of a row
group, then E4( skip * scale + narrow W3 ) with the narrow activations staged in shared memory.

Workgroup: E experts x ROWT row tiles of 16 = 4E * ROWT / 4 ... warps = E * ROWT (warp w: expert / column group
e = w / ROWT, row tile q = w % ROWT), rows per workgroup = 16 ROWT, grid = ceil(rows / (16 ROWT)).
  phase 1: hidden_e = x W1_e over K = C in k32 steps; every expert's W1 tile of the step sits in one ring stage
           (E x 4 KB, chunk-swizzled rows), A fragments straight from the state rows (2 steps in flight).
  phase 2: SiLU, E4, W2_e (the W2 tiles land in the free ring stage during the last W1 step; the hidden C fragment
           packs into the W2 A fragment through the permuted W1 columns), narrow tile [16 ROWT][C + 16] E4 in shared.
  phase 3: out[rows][32 e ..] = E4( skip * aux + narrow W3[:, 32 e ..] ), W3 tiles ring-staged per k32 step, the
           narrow rows read with ldmatrix (padded stride: conflict-free), output through the narrow region.
Arithmetic identical to mlp_e4m3 + gemm2_e4m3 (verified bit-exact). Dynamic shared memory (up to ~83 KB).

python ffn_e4m3.py <C> <ROWT> out.ptx [maxregs] [proj]
"""
import sys
from ptxgen import Ptx
from swin import *

HIDDEN = 128


def generate(C, rowt, max_regs=None, proj=False):
    """proj: the previous block's projection runs first (phase 0): state = E4(ffnPrev * auxAttn + attended Wproj),
    kept in shared memory as the FFN input and skip (the state never touches global memory)."""
    E = C // 32
    steps = C // 32
    warps = E * rowt
    threads = 32 * warps
    rowsPerWg = 16 * rowt
    stageBytes = E * 4096 + 2048          # W1 tiles of one k32 step for all experts ([e][128 n][32 B]) + the A tile [64][32 B]
    A_OFF = E * 4096                      # A tile offset within a W1 stage (chunk-swizzled rows, L2 copies)
    RING = 0
    NARROW = 2 * stageBytes
    narrowStride = C + 16
    sharedBytes = NARROW + rowsPerWg * narrowStride
    w3StageBytes = C * 32                 # W3 tiles of one k32 step ([C n][32 B])
    wpStageBytes = C * 32 + 2048          # Wproj tiles + the attended A tile (phase 0 ring inside W1 stage 0)
    assert 2 * wpStageBytes <= stageBytes
    name = f"ffn_e4m3_C{C}_R{rowt}" + ("_proj" if proj else "")
    p = Ptx()
    params = [("u64", "pA"), ("u64", "pW1"), ("u64", "pW2"), ("u64", "pW3"), ("u64", "pAux"), ("u64", "pOut"),
              ("u32", "rows"), ("u32", "auxHalfOffset"),
              ("u64", "pAtt"), ("u64", "pFfnPrev"), ("u64", "pWproj"), ("u32", "auxAttnHalf"), ("u64", "pAuxPrev"),
              ("u64", "pStateOut"), ("u32", "storeState"),
              ("u64", "pWaitRows"), ("u32", "waitExpected"), ("u32", "waitShiftY"), ("u64", "pWaitBands"), ("u64", "pSignal"),
              ("u32", "width"), ("u32", "waitMul"), ("u32", "waitGroupRows"), ("u64", "pError")]   # chaining (0 = off): wait on the previous block's window rows / state row bands (waitMul signals per row group of waitGroupRows rows), signal this block's FFN row bands
    p.entry(name, params, sharedBytes, threads, max_regs, dynamic_shared=True)
    P = {n: (p.load_param_u64(n) if t == "u64" else p.load_param_u32(n)) for t, n in params}
    tid = p.special("tid.x"); ctaid = p.special("ctaid.x")
    lane = p.and32(tid, 31); warp = p.shr32(tid, 5)
    li = LaneInfo(p, lane)
    g, t = li.g, li.t
    smem = p.shared_addr(p.imm32(0))
    zero = imm(p, 0)
    e = p.reg("b32"); q = p.reg("b32")
    if rowt & (rowt - 1) == 0:
        p.emit(f"shr.u32 {e}, {warp}, {rowt.bit_length() - 1};"); p.emit(f"and.b32 {q}, {warp}, {rowt - 1};")
    else:
        p.emit(f"div.u32 {e}, {warp}, {rowt};"); p.emit(f"rem.u32 {q}, {warp}, {rowt};")
    blockRow = p.mul32(ctaid, rowsPerWg)
    rowsMinus1 = p.sub32(P["rows"], 1)
    # this warp's rows g, g + 8 (clamped for the loads, never stored past the end)
    rowOk = []; aRow = []; rowIdx = []
    for h in range(2):
        r = p.add32(blockRow, p.add32(p.shl32(q, 4), p.add32(g, p.imm32(8 * h))))
        rowIdx.append(r)
        rowOk.append(p.setp("lt.u32", r, P["rows"]))
        rc = p.selp32(rowOk[h], r, rowsMinus1)
        aRow.append(p.add64(P["pA"], p.widen(p.add32(p.mul32(rc, C), p.shl32(t, 2)))))
    stgAddr = [p.add32(smem, RING + s * stageBytes) for s in range(2)]
    # ---- chaining: wait for the producers of this workgroup's rows
    rowLast = p.add32(blockRow, p.imm32(rowsPerWg - 1))
    pLastBig = p.setp("gt.u32", rowLast, rowsMinus1)
    rowLast = p.selp32(pLastBig, rowsMinus1, rowLast)
    yFirst = p.reg("b32"); p.emit(f"div.u32 {yFirst}, {blockRow}, {P['width']};")
    yLast = p.reg("b32"); p.emit(f"div.u32 {yLast}, {rowLast}, {P['width']};")
    pWaitRowsOn = p.setp("ne.u64", P["pWaitRows"], 0)
    wy0 = p.shr32(p.add32(yFirst, P["waitShiftY"]), 3); wy1 = p.shr32(p.add32(yLast, P["waitShiftY"]), 3)
    sync_wait(p, P["pWaitRows"], wy0, wy1, P["waitExpected"], lane, pWaitRowsOn, warp, error64=P["pError"])
    pWaitBandsOn = p.setp("ne.u64", P["pWaitBands"], 0)
    b0 = p.shr32(yFirst, 3); b1 = p.shr32(yLast, 3)
    sync_wait(p, P["pWaitBands"], b0, b1, lambda b: band_expected(p, b, P["width"], P["rows"], P["waitMul"], P["waitGroupRows"]), lane, pWaitBandsOn, warp, error64=P["pError"])

    # ---- W1 ring copies: E x 256 chunks per stage, 2 per thread: c = tid + threads i -> expert c >> 8, row (c & 255) >> 1, half c & 1
    w1Copies = []
    copiesPerThread = (256 * E + threads - 1) // threads
    copyExact = copiesPerThread * threads == 256 * E
    w1Preds = []
    for i in range(copiesPerThread):
        c = p.add32(tid, i * threads)
        w1Preds.append(None if copyExact else p.setp("lt.u32", c, 256 * E))
        ce = p.shr32(c, 8); cc = p.and32(c, 255)
        n = p.shr32(cc, 1); half = p.and32(cc, 1)
        pch = p.xor32(half, p.and32(p.shr32(n, 2), 1))
        dst = p.add32(p.shl32(ce, 12), p.add32(p.shl32(n, 5), p.shl32(pch, 4)))
        # W1 permuted tile-major [e][steps][128][32]: expert e step s row n at ((e steps + s) 128 + n) 32
        src = p.add64(P["pW1"], p.widen(p.add32(p.mul32(ce, steps * 4096), p.add32(p.shl32(n, 5), p.shl32(half, 4)))))
        w1Copies.append((dst, src))
    def issue_w1(stage, step):
        for (dst, src), pv in zip(w1Copies, w1Preds):
            saddr = p.add32(stgAddr[stage], dst)
            gaddr = p.add64(src, p.widen(p.imm32(step * 4096)))
            pred = f"@{pv} " if pv else ""
            p.emit(f"{pred}cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;")
        if not proj: issue_a(stgAddr[stage], A_OFF, P["pA"], step)
        p.emit("cp.async.commit_group;")
    # W2 tiles [e][4 kt][32 n][32 B] (E x 4 KB): same chunk mapping (row = kt * 32 + n within the expert)
    def issue_w2(stage):
        for i in range(copiesPerThread):
            c = p.add32(tid, i * threads)
            pv = None if copyExact else p.setp("lt.u32", c, 256 * E)
            ce = p.shr32(c, 8); cc = p.and32(c, 255)
            row = p.shr32(cc, 1); half = p.and32(cc, 1)
            pch = p.xor32(half, p.and32(p.shr32(row, 2), 1))
            dst = p.add32(stgAddr[stage], p.add32(p.shl32(ce, 12), p.add32(p.shl32(row, 5), p.shl32(pch, 4))))
            src = p.add64(P["pW2"], p.widen(p.add32(p.shl32(ce, 12), p.add32(p.shl32(row, 5), p.shl32(half, 4)))))
            pred = f"@{pv} " if pv else ""
            p.emit(f"{pred}cp.async.cg.shared.global [{dst}], [{src}], 16;")
        p.emit("cp.async.commit_group;")
    # W3 ring copies: C x 2 chunks per stage (in the ring region, stage s at RING + s * w3StageBytes)
    w3Chunks = C * 2
    w3Copies = []
    for i in range((w3Chunks + threads - 1) // threads):
        c = p.add32(tid, i * threads)
        pv = p.setp("lt.u32", c, w3Chunks) if w3Chunks % threads else None
        n = p.shr32(c, 1); half = p.and32(c, 1)
        pch = p.xor32(half, p.and32(p.shr32(n, 2), 1))
        dst = p.add32(p.shl32(n, 5), p.shl32(pch, 4))
        src = p.add64(P["pW3"], p.widen(p.add32(p.shl32(n, 5), p.shl32(half, 4))))
        w3Copies.append((dst, src, pv))
    def issue_w3(stage, step):
        for dst, src, pv in w3Copies:
            saddr = p.add32(p.add32(smem, RING + stage * w3StageBytes), dst)
            gaddr = p.add64(src, p.widen(p.imm32(step * C * 32)))
            pred = f"@{pv} " if pv else ""
            p.emit(f"{pred}cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;")
        p.emit("cp.async.commit_group;")

    # ---- fragment addressing
    l8 = p.and32(lane, 7); lm = p.shr32(lane, 3)
    bNL = p.add32(l8, p.shl32(p.shr32(lane, 4), 3)); bChunkL = p.and32(lm, 1)
    bPch = p.xor32(bChunkL, p.and32(p.shr32(bNL, 2), 1))
    bLaneOff = p.add32(p.shl32(bNL, 5), p.shl32(bPch, 4))
    w1B = [p.add32(stgAddr[s], p.add32(p.shl32(e, 12), bLaneOff)) for s in range(2)]   # this expert's tiles
    # A tile copies: 128 chunks (64 rows x 2 halves) per step, threads < 128: row = tid >> 1 (clamped to the last row)
    aCopyOn = p.setp("lt.u32", tid, 128)
    aRowC = p.shr32(tid, 1); aHalfC = p.and32(tid, 1)
    aRowG = p.add32(blockRow, aRowC)
    aRowG = p.selp32(p.setp("lt.u32", aRowG, P["rows"]), aRowG, rowsMinus1)
    aDstOff = p.add32(p.shl32(aRowC, 5), p.shl32(p.xor32(aHalfC, p.and32(p.shr32(aRowC, 2), 1)), 4))
    aSrcOff = p.widen(p.add32(p.mul32(aRowG, C), p.shl32(aHalfC, 4)))
    def issue_a(stageAddr, aOff, base64, step):
        saddr = p.add32(stageAddr, p.add32(aDstOff, p.imm32(aOff)))
        gaddr = p.add64(base64, p.add64(aSrcOff, p.widen(p.imm32(32 * step))))
        p.emit(f"@{aCopyOn} cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;")
    aRowL = p.add32(l8, p.shl32(p.and32(lm, 1), 3)); aChunkL = p.shr32(lane, 4)
    aFragOff = p.add32(p.shl32(p.add32(aRowL, p.shl32(q, 4)), 5), p.shl32(p.xor32(aChunkL, p.and32(p.shr32(aRowL, 2), 1)), 4))
    narrowBase = p.add32(smem, NARROW)
    narrowA = p.add32(narrowBase, p.add32(p.mul32(p.add32(p.shl32(q, 4), aRowL), narrowStride), p.shl32(aChunkL, 4)))
    w3B = [p.add32(p.add32(smem, RING + s * w3StageBytes), p.add32(p.shl32(e, 10), bLaneOff)) for s in range(2)]   # columns 32 e ..
    # residual seed for phase 3: E4 state (row, 32 e + 8 j + 2 t) pairs and the aux scales
    seedRegs = []
    auxAddr = p.add64(P["pAux"], p.widen(p.shl32(p.add32(P["auxHalfOffset"], p.add32(p.shl32(e, 5), p.shl32(t, 1))), 1)))
    scales = []
    for j in range(4):
        r = p.reg("b32"); p.emit(f"ld.global.nc.b32 {r}, [{auxAddr}+{16 * j}];"); scales.append(r)
    # the state tile (proj): [rowsPerWg][C + 16] E4 aliasing the narrow region (dead before phase 2 writes it)
    stateOut = p.add32(narrowBase, p.add32(p.mul32(p.add32(p.shl32(q, 4), g), narrowStride), p.add32(p.shl32(e, 5), p.shl32(t, 1))))
    if not proj:
        for h in range(2):
            resAddr = p.add64(P["pA"], p.widen(p.add32(p.mul32(p.selp32(rowOk[h], rowIdx[h], rowsMinus1), C), p.add32(p.shl32(e, 5), p.shl32(t, 1)))))
            for j in range(4):
                hc = p.reg("b16"); p.emit(f"ld.global.cg.b16 {hc}, [{resAddr}+{8 * j}];")
                seedRegs.append((j, h, hc))


    chunksPerRow = C // 16
    def store_tile(dstBase):
        """The [rowsPerWg][C] E4 tile in the narrow region -> global [rows][C] rows (16-byte chunks, one per thread)."""
        for i in range((rowsPerWg * chunksPerRow + threads - 1) // threads):
            c = p.add32(tid, i * threads)
            pv = p.setp("lt.u32", c, rowsPerWg * chunksPerRow) if (rowsPerWg * chunksPerRow) % threads else None
            row = p.reg("b32"); chunk = p.reg("b32")
            p.emit(f"div.u32 {row}, {c}, {chunksPerRow};"); p.emit(f"rem.u32 {chunk}, {c}, {chunksPerRow};")
            gRow = p.add32(blockRow, row)
            ok = p.setp("lt.u32", gRow, P["rows"])
            if pv is not None: p.emit(f"and.pred {ok}, {ok}, {pv};")
            r = p.regs("b32", 4)
            src = p.add32(narrowBase, p.add32(p.mul32(row, narrowStride), p.shl32(chunk, 4)))
            p.emit(f"ld.shared.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{src}];")
            addr = p.add64(dstBase, p.widen(p.add32(p.mul32(gRow, C), p.shl32(chunk, 4))))
            p.emit(f"@{ok} st.global.v4.b32 [{addr}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")

    w1First = 0
    if proj:
        # ================= phase 0: state = E4(ffnPrev * auxAttn + attended Wproj[:, 32 e ..]); Wproj ring in stage 0, W1 step 0 -> stage 1
        w1First = 1
        auxAttnAddr = p.add64(P["pAuxPrev"], p.widen(p.shl32(p.add32(P["auxAttnHalf"], p.add32(p.shl32(e, 5), p.shl32(t, 1))), 1)))
        scalesAttn = []
        for j in range(4):
            r = p.reg("b32"); p.emit(f"ld.global.nc.b32 {r}, [{auxAttnAddr}+{16 * j}];")
            scalesAttn.append(r)
        pacc = [[None, None] for _ in range(4)]
        for h in range(2):
            prevAddr = p.add64(P["pFfnPrev"], p.widen(p.add32(p.mul32(p.selp32(rowOk[h], rowIdx[h], rowsMinus1), C), p.add32(p.shl32(e, 5), p.shl32(t, 1)))))
            for j in range(4):
                hc = p.reg("b16"); p.emit(f"ld.global.cg.b16 {hc}, [{prevAddr}+{8 * j}];")
                pacc[j][h] = hmul2(p, e4x2_to_f16x2(p, hc), scalesAttn[j])
        # Wproj tiles share the W3 copy geometry: chunk offsets relative to the matrix base
        wpCopies = []
        for i in range((w3Chunks + threads - 1) // threads):
            c = p.add32(tid, i * threads)
            pv = p.setp("lt.u32", c, w3Chunks) if w3Chunks % threads else None
            n = p.shr32(c, 1); half = p.and32(c, 1)
            pch = p.xor32(half, p.and32(p.shr32(n, 2), 1))
            dst = p.add32(p.shl32(n, 5), p.shl32(pch, 4))
            src = p.add64(P["pWproj"], p.widen(p.add32(p.shl32(n, 5), p.shl32(half, 4))))
            wpCopies.append((dst, src, pv))
        wpB = [p.add32(p.add32(smem, RING + st * wpStageBytes), p.add32(p.shl32(e, 10), bLaneOff)) for st in range(2)]
        def issue_wproj(stage, step):
            for dst, src, pv in wpCopies:
                saddr = p.add32(p.add32(smem, RING + stage * wpStageBytes), dst)
                gaddr = p.add64(src, p.widen(p.imm32(step * C * 32)))
                pred = f"@{pv} " if pv else ""
                p.emit(f"{pred}cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;")
            issue_a(p.add32(smem, RING + stage * wpStageBytes), C * 32, P["pAtt"], step)
            p.emit("cp.async.commit_group;")
        issue_wproj(0, 0)
        issue_w1(1, 0)   # W1 step 0 lands in stage 1 during phase 0
        for step in range(steps):
            stage = step & 1
            p.emit("cp.async.wait_group 0;")
            p.emit("bar.sync 0;")
            if step + 1 < steps: issue_wproj(stage ^ 1, step + 1)
            else: p.emit("cp.async.commit_group;")
            a = ldm4(p, p.add32(p.add32(smem, RING + stage * wpStageBytes + C * 32), aFragOff))
            for i2 in range(2):
                r = ldm4(p, p.add32(wpB[stage], i2 * 512))
                mma_e4(p, pacc[2 * i2], a, (r[0], r[1]), pacc[2 * i2])
                mma_e4(p, pacc[2 * i2 + 1], a, (r[2], r[3]), pacc[2 * i2 + 1])
        # publish the state tile: E4 codes (the FFN input and skip)
        for j in range(4):
            for h in range(2):
                hc = cvt_e4x2(p, pacc[j][h])
                p.emit(f"st.shared.b16 [{stateOut}+{8 * j + 8 * narrowStride * h}], {hc};")
                seedRegs.append((j, h, hc))
        p.emit("bar.sync 0;")
        # W1 step 1 into stage 0 (the Wproj ring is dead)
        if steps > 1: issue_w1(0, 1)
        # optional materialization of the block state (parity captures)
        pStore = p.setp("ne.u32", P["storeState"], 0)
        skipStore = p.label("NOSTATE")
        p.emit(f"@!{pStore} bra {skipStore};")
        store_tile(P["pStateOut"])
        p.emit(f"{skipStore}:")

    # ================= phase 1: hidden = x W1_e
    if not proj:
        issue_w1(0, 0)
    hid = [zero_tile(p) for _ in range(HIDDEN // 8)]
    for step in range(steps):
        stage = (step + w1First) & 1
        p.emit("cp.async.wait_group 0;")
        p.emit("bar.sync 0;")
        if step + 1 < steps:
            if not proj or step >= 1: issue_w1(stage ^ 1, step + 1)
        else: issue_w2(stage ^ 1)
        if proj:
            a = ldm4(p, p.add32(narrowA, 32 * step))   # the state tile in shared
        else:
            a = ldm4(p, p.add32(stgAddr[stage], p.add32(aFragOff, p.imm32(A_OFF))))
        for i2 in range(8):
            r = ldm4(p, p.add32(w1B[stage], i2 * 512))
            mma_e4(p, hid[2 * i2], a, (r[0], r[1]), hid[2 * i2])
            mma_e4(p, hid[2 * i2 + 1], a, (r[2], r[3]), hid[2 * i2 + 1])
    # ================= phase 2: SiLU, E4, W2_e -> narrow tile
    p.emit("cp.async.wait_group 0;")
    p.emit("bar.sync 0;")
    w2Stage = (steps + w1First) & 1
    k = SiluConsts(p)
    for d in hid:
        d[0] = silu(p, d[0], k); d[1] = silu(p, d[1], k)
    narrow = [zero_tile(p) for _ in range(4)]
    for kt in range(4):
        qcodes = [[b16_to_b32(p, cvt_e4x2(p, hid[4 * kt + i][h])) for h in range(2)] for i in range(4)]
        a2 = [prmt(p, qcodes[0][0], qcodes[1][0], "0x5410"), prmt(p, qcodes[0][1], qcodes[1][1], "0x5410"),
              prmt(p, qcodes[2][0], qcodes[3][0], "0x5410"), prmt(p, qcodes[2][1], qcodes[3][1], "0x5410")]
        for i2 in range(2):
            r = ldm4(p, p.add32(w1B[w2Stage], kt * 1024 + i2 * 512))
            mma_e4(p, narrow[2 * i2], a2, (r[0], r[1]), narrow[2 * i2])
            mma_e4(p, narrow[2 * i2 + 1], a2, (r[2], r[3]), narrow[2 * i2 + 1])
    narrowOut = p.add32(narrowBase, p.add32(p.mul32(p.add32(p.shl32(q, 4), g), narrowStride), p.add32(p.shl32(e, 5), p.shl32(t, 1))))
    for j in range(4):
        for h in range(2):
            hc = cvt_e4x2(p, narrow[j][h])
            p.emit(f"st.shared.b16 [{narrowOut}+{8 * j + 8 * narrowStride * h}], {hc};")
    # W3 stage 0 into the ring (both W1/W2 stages are dead after this barrier)
    p.emit("bar.sync 0;")
    issue_w3(0, 0)
    # ================= phase 3: out = skip * aux + narrow W3[:, 32 e ..]
    acc = [[None, None] for _ in range(4)]
    for j, h, hc in seedRegs:
        acc[j][h] = hmul2(p, e4x2_to_f16x2(p, hc), scales[j])
    for step in range(steps):
        stage = step & 1
        p.emit("cp.async.wait_group 0;")
        p.emit("bar.sync 0;")
        if step + 1 < steps: issue_w3(stage ^ 1, step + 1)
        a = ldm4(p, p.add32(narrowA, 32 * step))
        for i2 in range(2):
            r = ldm4(p, p.add32(w3B[stage], i2 * 512))
            mma_e4(p, acc[2 * i2], a, (r[0], r[1]), acc[2 * i2])
            mma_e4(p, acc[2 * i2 + 1], a, (r[2], r[3]), acc[2 * i2 + 1])
    # ---- output through the narrow region (every warp finished reading it), then 16-byte stores: one chunk per thread
    p.emit("bar.sync 0;")
    for j in range(4):
        for h in range(2):
            hc = cvt_e4x2(p, acc[j][h])
            p.emit(f"st.shared.b16 [{narrowOut}+{8 * j + 8 * narrowStride * h}], {hc};")
    p.emit("bar.sync 0;")
    store_tile(P["pOut"])
    # ---- chaining: release the FFN rows (bands b0 .. b1 of this workgroup)
    pSignalOn = p.setp("ne.u64", P["pSignal"], 0)
    p.emit("fence.acq_rel.gpu;")
    p.emit("bar.sync 0;")
    pT0 = p.reg("pred"); p.emit(f"and.pred {pT0}, {pSignalOn}, {p.setp('eq.u32', tid, 0)};")
    bIter = p.reg("b32"); p.emit(f"mov.u32 {bIter}, {b0};")
    sigLoop = p.label("SIG"); sigDone = p.label("SIGDONE")
    p.emit(f"@!{pT0} bra {sigDone};")
    p.emit(f"{sigLoop}:")
    sigAddr = p.add64(P["pSignal"], p.widen(p.shl32(bIter, 2)))
    p.emit(f"red.release.gpu.global.add.u32 [{sigAddr}], 1;")
    p.emit(f"add.u32 {bIter}, {bIter}, 1;")
    pMoreB = p.setp("le.u32", bIter, b1)
    p.emit(f"@{pMoreB} bra {sigLoop};")
    p.emit(f"{sigDone}:")
    p.emit("ret;")
    return name, p.finish(), sharedBytes


if __name__ == "__main__":
    C, rowt, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    max_regs = int(sys.argv[4]) if len(sys.argv) > 4 else None
    proj = len(sys.argv) > 5 and int(sys.argv[5]) != 0
    name, text, shared = generate(C, rowt, max_regs, proj)
    open(out, "w").write(text)
    print(name, shared)
