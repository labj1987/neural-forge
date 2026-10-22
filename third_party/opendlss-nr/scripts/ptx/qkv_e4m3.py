"""QKV projection + cosine normalization + 8x8 window attention for (window, head) items in PTX
(the qkv_attention.comp dataflow at the fragment level; see block32_e4m3.py for the block-level version).

Workgroup = 4 warps, persistent over items (item = head * windowCount + window, stride = grid); warp q owns the
window's tokens 16q .. 16q+15.
  q/k/v = A(window rows of the E4 block state, K = C) x Wqkv[head columns] over K in k32 steps: the A fragments come
  straight from global memory (ADEPTH k32 steps in flight in registers, zero outside the image); the head's 96
  weight columns per k32 step are staged in a cp.async ring (3 KB per step).
  normalize q (learned scale) and k; K rows / transposed V of this head into shared (physical token order);
  S = prior[head] + Q K^T; the specified softmax; P V; E4 output at columns head * 32 of the attended tensor.
NaN handling: the normalization's NaN -> 0 (zero rows at the image border) runs under a warp vote; V and the
attended tile are converted directly (finite data: every state is a saturated publication).

python qkv_e4m3.py <C> out.ptx [maxregs]
"""
import sys
from ptxgen import Ptx
from swin import *

THREADS = 128
STAGES = 2        # cp.async ring depth (steps in flight = STAGES - 1)
ALD = "cg"        # A operand load cache hint (produced by the chained FFN launch: no L1 / texture path)
ADEPTH = 2        # k32 steps of A fragments in flight
B_BYTES = 96 * 32             # one k32 step of the head's B tiles: [96 n][32 B] (chunk-swizzled rows)
STEP_BYTES = B_BYTES + 64 * 32 # + the window's A tile [64 tokens][32 B] (chunk-swizzled rows, zero outside the image)
KROWS = STAGES * STEP_BYTES    # K [64 physical][32 B] swizzled (2 KB)
VTS = KROWS + 2048             # Vt [32 dim][64 B] swizzled (chunk ^= (dim >> 1) & 3)
STG = VTS + 2048               # per warp 512 B output staging
SHARED_BYTES = STG + 4 * 512


def generate(C, max_regs=None):
    assert C % 32 == 0
    steps = C // 32
    stages = min(STAGES, steps)
    name = f"qkv_e4m3_K{C}"
    p = Ptx()
    params = [("u64", "pState"), ("u64", "pWqkv"), ("u64", "pPrior"), ("u64", "pAux"), ("u64", "pOut"),
              ("u32", "width"), ("u32", "height"), ("u32", "shiftX"), ("u32", "shiftY"), ("u32", "windowsX"),
              ("u32", "scaleWord"), ("u32", "windowCount"), ("u32", "itemCount"),
              ("u64", "pWait"), ("u64", "pSignal"), ("u32", "waitMul"), ("u32", "waitGroupRows"), ("u64", "pError")]   # chaining: this block's FFN row bands (wait, waitMul signals per row group of waitGroupRows rows), window rows (signal); 0 = off
    p.entry(name, params, SHARED_BYTES, THREADS, max_regs)
    P = {n: (p.load_param_u64(n) if t == "u64" else p.load_param_u32(n)) for t, n in params}
    tid = p.special("tid.x"); ctaid = p.special("ctaid.x"); nctaid = p.special("nctaid.x")
    lane = p.and32(tid, 31); warp = p.shr32(tid, 5)
    li = LaneInfo(p, lane)
    g, t = li.g, li.t
    smem = p.shared_addr(p.imm32(0))
    zero = imm(p, 0)
    exp_k = ExpConsts(p)

    # ---- lane constants
    l8 = p.and32(lane, 7); lm = p.shr32(lane, 3)
    bNL = p.add32(l8, p.shl32(p.shr32(lane, 4), 3)); bChunkL = p.and32(lm, 1)
    bPch = p.xor32(bChunkL, p.and32(p.shr32(bNL, 2), 1))
    bLaneOff = p.add32(p.shl32(bNL, 5), p.shl32(bPch, 4))
    ringB = p.add32(smem, bLaneOff)
    kLane = p.add32(p.add32(smem, KROWS), bLaneOff)
    vSwz = p.and32(p.shr32(bNL, 1), 3)
    vLane = [p.add32(p.add32(smem, VTS), p.add32(p.shl32(bNL, 6), p.shl32(p.xor32(p.xor32(bChunkL, p.imm32(2 * st)), vSwz), 4)))
             for st in range(2)]
    # K rows / Vt publication of this warp's tokens g, g + 8 (physical order)
    phys = [tiled_token_regs(p, p.add32(p.shl32(warp, 4), p.add32(g, p.imm32(8 * h)))) for h in range(2)]
    kBase = []; vBase = []
    for h in range(2):
        swz = p.and32(p.shr32(phys[h], 2), 1)
        kBase.append([p.add32(p.add32(smem, KROWS), p.add32(p.shl32(phys[h], 5), p.add32(p.shl32(p.xor32(p.imm32(c), swz), 4), p.shl32(t, 1)))) for c in range(2)])
        vBase.append(p.add32(p.add32(smem, VTS), p.add32(p.shl32(t, 7), p.add32(p.shl32(p.xor32(p.shr32(phys[h], 4), t), 4), p.and32(phys[h], 15)))))
    # A tile chunk of this thread (one 16-byte copy per step): token = tid >> 1, half = tid & 1
    aTok = p.shr32(tid, 1); aHalf = p.and32(tid, 1)
    aTokX = p.and32(aTok, 7); aTokY = p.shr32(aTok, 3)
    aDst = p.add32(smem, p.add32(p.imm32(B_BYTES), p.add32(p.shl32(aTok, 5), p.shl32(p.xor32(aHalf, p.and32(p.shr32(aTok, 2), 1)), 4))))
    aRowL = p.add32(l8, p.shl32(p.and32(lm, 1), 3)); aChunkL = p.shr32(lane, 4)
    aPch = p.xor32(aChunkL, p.and32(p.shr32(aRowL, 2), 1))
    aLane = p.add32(smem, p.add32(p.imm32(B_BYTES), p.add32(p.shl32(p.add32(aRowL, p.shl32(warp, 4)), 5), p.shl32(aPch, 4))))
    # B ring copies: 192 chunks per step, 2 per thread; chunk c: row n = c >> 1, half = c & 1, chunk ^= (n >> 2) & 1
    copies = []
    for i in range(2):
        c = p.add32(tid, i * THREADS)
        pv = p.setp("lt.u32", c, 192)
        n = p.shr32(c, 1); half = p.and32(c, 1)
        pch = p.xor32(half, p.and32(p.shr32(n, 2), 1))
        dst = p.add32(smem, p.add32(p.shl32(n, 5), p.shl32(pch, 4)))
        copies.append((dst, p.shl32(c, 4), pv))
    # output staging / store lane
    stgWarp = p.add32(p.add32(smem, STG), p.shl32(warp, 9))
    stgOut = p.add32(stgWarp, p.add32(p.shl32(g, 5), p.shl32(t, 1)))
    ldRow = p.shr32(lane, 1); ldPart = p.and32(lane, 1)
    stgRead = p.add32(stgWarp, p.add32(p.shl32(ldRow, 5), p.shl32(ldPart, 4)))
    priorRowOff = p.add32(p.shl32(p.add32(p.shl32(warp, 4), g), 7), p.shl32(t, 2))   # + head * 8192

    # ================= persistent item loop (the next item's first weight stage and A fragments are prefetched
    # during the current item's attention phase)
    heads = C // 32
    headShift = heads.bit_length() - 1
    assert 1 << headShift == heads
    windowsXRcp = rcp_f32(p, P["windowsX"])
    def geometry(itemReg):
        # item = window * heads + head (heads is a power of two); window -> (wy, wx) by the f32 reciprocal
        gm = {}
        gm["head"] = p.and32(itemReg, heads - 1)
        window = p.shr32(itemReg, headShift)
        wy, wx = fast_divmod(p, window, P["windowsX"], windowsXRcp)
        gm["windowX"] = p.sub32(p.shl32(wx, 3), P["shiftX"]); gm["windowY"] = p.sub32(p.shl32(wy, 3), P["shiftY"])
        # this thread's A chunk: token tid >> 1 (pixel windowX + (token & 7), windowY + (token >> 3)), half tid & 1
        xa = p.add32(gm["windowX"], aTokX); ya = p.add32(gm["windowY"], aTokY)
        pvx = p.setp("lt.u32", xa, P["width"]); pvy = p.setp("lt.u32", ya, P["height"])
        pv = p.reg("pred"); p.emit(f"and.pred {pv}, {pvx}, {pvy};")
        pixel = p.mad32(ya, P["width"], xa)
        off = p.selp32(pv, p.add32(p.mul32(pixel, C), p.shl32(aHalf, 4)), p.imm32(0))
        gm["aSrc"] = p.add64(P["pState"], p.widen(off))
        gm["aSize"] = p.selp32(pv, p.imm32(16), p.imm32(0))
        gm["wHead"] = p.add64(P["pWqkv"], p.widen(p.mul32(gm["head"], 96 * 32)))
        gm["wy"] = wy
        # pixel-row bands of the window: rows [max(8 wy - shiftY, 0), min(+7, height - 1)]
        yTop = gm["windowY"]
        pNeg = p.setp("gt.s32", p.imm32(0), yTop)
        y0 = p.selp32(pNeg, p.imm32(0), yTop)
        y1 = p.add32(yTop, 7)
        hm1 = p.sub32(P["height"], 1)
        pBig = p.setp("gt.u32", y1, hm1)
        y1 = p.selp32(pBig, hm1, y1)
        gm["band0"] = p.shr32(y0, 3); gm["band1"] = p.shr32(y1, 3)
        return gm

    rowsTotal = p.mul32(P["width"], P["height"])
    pWaitOn = p.setp("ne.u64", P["pWait"], 0)
    pSignalOn = p.setp("ne.u64", P["pSignal"], 0)
    def wait_item(gm, guard=None):
        g2 = pWaitOn
        if guard is not None:
            g2 = p.reg("pred"); p.emit(f"and.pred {g2}, {pWaitOn}, {guard};")
        sync_wait(p, P["pWait"], gm["band0"], gm["band1"], lambda b: band_expected(p, b, P["width"], rowsTotal, P["waitMul"], P["waitGroupRows"]), lane, g2, warp, error64=P["pError"])

    def issue_stage(gm, stageIndex, step, guard=None):
        # the head's weight tiles (L1-cached, constant) and the window's A tile (L2 only: produced by the chained FFN)
        for dst, srcOff, pv in copies:
            saddr = p.add32(dst, p.imm32(stageIndex * STEP_BYTES))
            gaddr = p.add64(gm["wHead"], p.widen(p.add32(srcOff, p.imm32(step * 3 * C * 32))))
            pred = pv
            if guard is not None:
                pred = p.reg("pred"); p.emit(f"and.pred {pred}, {pv}, {guard};")
            p.emit(f"@{pred} cp.async.ca.shared.global [{saddr}], [{gaddr}], 16;")
        saddr = p.add32(aDst, p.imm32(stageIndex * STEP_BYTES))
        gaddr = p.add64(gm["aSrc"], p.widen(p.imm32(32 * step)))
        pred = f"@{guard} " if guard is not None else ""
        p.emit(f"{pred}cp.async.cg.shared.global [{saddr}], [{gaddr}], 16, {gm['aSize']};")
        p.emit("cp.async.commit_group;")

    def carry_regs():
        return {"head": p.reg("b32"), "windowX": p.reg("b32"), "windowY": p.reg("b32"), "wy": p.reg("b32"),
                "aSrc": p.reg("b64"), "aSize": p.reg("b32"), "wHead": p.reg("b64")}

    def carry_assign(dst, gm, aList=None):
        for k in ("head", "windowX", "windowY", "wy", "aSize"): p.emit(f"mov.b32 {dst[k]}, {gm[k]};")
        p.emit(f"mov.b64 {dst['aSrc']}, {gm['aSrc']};")
        p.emit(f"mov.b64 {dst['wHead']}, {gm['wHead']};")

    item = p.reg("b32"); p.emit(f"mov.u32 {item}, {ctaid};")
    pItem = p.setp("lt.u32", item, P["itemCount"])
    endL = p.label("END"); loopL = p.label("ITEM")
    p.emit(f"@!{pItem} bra {endL};")
    gm0 = geometry(item)
    wait_item(gm0)
    for j in range(stages - 1): issue_stage(gm0, j, j)
    carried = carry_regs()
    carry_assign(carried, gm0)
    sKeep = [[p.reg("b32"), p.reg("b32")] for _ in range(8)]   # the item's scores for the attention tail
    for m in range(8):
        p.emit(f"mov.b32 {sKeep[m][0]}, 0;"); p.emit(f"mov.b32 {sKeep[m][1]}, 0;")

    def emit_tail(gmT):
        """softmax, P V, E4 output of the item whose scores are in sKeep and geometry in gmT (straight-line code)."""
        w = softmax(p, sKeep, li, exp_k)
        pa = [acc_to_a(p, w[0:4], li), acc_to_a(p, w[4:8], li)]
        o = [zero_tile(p) for _ in range(4)]
        for step in range(2):
            for d2 in range(2):
                r = ldm4(p, p.add32(vLane[step], d2 * 1024))
                mma_e4(p, o[2 * d2], pa[step], (r[0], r[1]), o[2 * d2])
                mma_e4(p, o[2 * d2 + 1], pa[step], (r[2], r[3]), o[2 * d2 + 1])
        for j in range(4):
            for h in range(2):
                hc = cvt_e4x2(p, o[j][h])
                p.emit(f"st.shared.b16 [{stgOut}+{8 * j + 256 * h}], {hc};")
        p.emit("bar.warp.sync 0xffffffff;")
        r = p.regs("b32", 4)
        p.emit(f"ld.shared.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{stgRead}];")
        tokX = p.add32(gmT["windowX"], p.and32(ldRow, 7)); tokY = p.add32(gmT["windowY"], p.add32(p.shl32(warp, 1), p.shr32(lane, 4)))
        pvx2 = p.setp("lt.u32", tokX, P["width"]); pvy2 = p.setp("lt.u32", tokY, P["height"])
        pv = p.reg("pred"); p.emit(f"and.pred {pv}, {pvx2}, {pvy2};")
        pixel = p.mad32(tokY, P["width"], tokX)
        addr = p.add64(P["pOut"], p.widen(p.add32(p.mul32(pixel, C), p.add32(p.shl32(gmT["head"], 5), p.shl32(ldPart, 4)))))
        p.emit(f"@{pv} st.global.v4.b32 [{addr}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")

    def signal_item(gmT):
        # release the attended rows of the item: window row counter (fence + barrier + one red)
        p.emit("fence.acq_rel.gpu;")
        p.emit("bar.sync 0;")
        pSig = p.reg("pred"); p.emit(f"and.pred {pSig}, {pSignalOn}, {p.setp('eq.u32', tid, 0)};")
        sigAddr = p.add64(P["pSignal"], p.widen(p.shl32(gmT["wy"], 2)))
        p.emit(f"@{pSig} red.release.gpu.global.add.u32 [{sigAddr}], 1;")

    p.emit(f"{loopL}:")
    gm = carried
    head, windowX, windowY = gm["head"], gm["windowX"], gm["windowY"]
    # prior rows of this head and the head scale: issued first
    priorLane = p.add64(P["pPrior"], p.widen(p.add32(p.shl32(head, 13), priorRowOff)))
    priorRegs = []
    for m in range(8):
        c0 = p.reg("b32"); p.emit(f"ld.global.nc.b32 {c0}, [{priorLane}+{16 * m}];")
        c1 = p.reg("b32"); p.emit(f"ld.global.nc.b32 {c1}, [{priorLane}+{1024 + 16 * m}];")
        priorRegs.append([c0, c1])
    scaleAddr = p.add64(P["pAux"], p.widen(p.shl32(p.add32(P["scaleWord"], head), 2)))
    scaleF = p.reg("f32"); p.emit(f"ld.global.nc.f32 {scaleF}, [{scaleAddr}];")
    acc = [zero_tile(p) for _ in range(12)]   # q tiles 0..3, k 4..7, v 8..11
    for step in range(steps):
        stageIndex = step % stages
        if step + stages - 1 < steps:
            issue_stage(gm, (step + stages - 1) % stages, step + stages - 1)
        else:
            p.emit("cp.async.commit_group;")
        p.emit(f"cp.async.wait_group {stages - 1};")
        p.emit("bar.sync 0;")
        a = ldm4(p, p.add32(aLane, stageIndex * STEP_BYTES))
        base = p.add32(ringB, stageIndex * STEP_BYTES)
        for i2 in range(6):
            r = ldm4(p, p.add32(base, i2 * 512))
            mma_e4(p, acc[2 * i2], a, (r[0], r[1]), acc[2 * i2])
            mma_e4(p, acc[2 * i2 + 1], a, (r[2], r[3]), acc[2 * i2 + 1])
        if step + 1 < steps: p.emit("bar.sync 0;")   # the stage is free for the next copy
    scaleH = p.reg("f16"); p.emit(f"cvt.rn.f16.f32 {scaleH}, {scaleF};")
    scale2 = pack16(p, scaleH, scaleH)
    qn = normalize(p, acc[0:4], li, zero, scale2)
    kn = normalize(p, acc[4:8], li, zero)
    mq = acc_to_a(p, qn, li)
    for h in range(2):
        for j in range(4):
            hc = cvt_e4x2(p, kn[j][h])
            p.emit(f"st.shared.b16 [{kBase[h][j >> 1]}+{8 * (j & 1)}], {hc};")
            vc = cvt_e4x2(p, acc[8 + j][h])
            vhi = p.reg("b16"); p.emit(f"shr.b16 {vhi}, {vc}, 8;")
            p.emit(f"st.shared.u8 [{vBase[h]}+{512 * j}], {vc};")
            p.emit(f"st.shared.u8 [{vBase[h]}+{512 * j + 64}], {vhi};")
    p.emit("bar.sync 0;")
    # prefetch the next item: its first weight stage (the ring is free) and A fragments
    nextItem = p.add32(item, nctaid)
    pNext = p.setp("lt.u32", nextItem, P["itemCount"])
    gmN = geometry(nextItem)
    wait_item(gmN, pNext)
    for j in range(stages - 1): issue_stage(gmN, j, j, pNext)
    s = []
    for m in range(8):
        s.append([priorRegs[m][0], priorRegs[m][1]])
    for m2 in range(4):
        r = ldm4(p, p.add32(kLane, m2 * 512))
        mma_e4(p, s[2 * m2], mq, (r[0], r[1]), s[2 * m2])
        mma_e4(p, s[2 * m2 + 1], mq, (r[2], r[3]), s[2 * m2 + 1])
    for m in range(8):
        p.emit(f"mov.b32 {sKeep[m][0]}, {s[m][0]};"); p.emit(f"mov.b32 {sKeep[m][1]}, {s[m][1]};")
    emit_tail(gm)
    signal_item(gm)
    carry_assign(carried, gmN)
    p.emit(f"mov.u32 {item}, {nextItem};")
    p.emit(f"@{pNext} bra {loopL};")
    p.emit(f"{endL}:")
    p.emit("ret;")
    return name, p.finish()


if __name__ == "__main__":
    C, out = int(sys.argv[1]), sys.argv[2]
    max_regs = int(sys.argv[3]) if len(sys.argv) > 3 else None
    name, text = generate(C, max_regs)
    open(out, "w").write(text)
    print(name)
