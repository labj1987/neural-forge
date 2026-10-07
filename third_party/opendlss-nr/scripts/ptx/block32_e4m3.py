"""Fused 32-channel Swin block (blocks 0-4, 66-70) in PTX: the fused_block32.comp dataflow at the fragment level.

One workgroup = 4 warps = one 8x8 window (warp w: tokens 16w .. 16w+15 = window rows 2w, 2w+1), persistent over
windows. Per warp everything is a 16-row fragment: FFN (x W1 -> SiLU -> E4 -> W2 + skip * ffnScale, in four k32
quarters with W1's columns permuted so the hidden C fragment packs straight into the W2 A fragment), q/k/v
projections, cosine normalization, K rows / transposed V into shared memory (physical token order), S = Q K^T +
prior, the specified softmax, P V, projection + ffn * attnScale, E4 output. Every arithmetic step is that of the
GLSL kernel (same instructions on the same fragment elements); only the data movement differs.

Variants (flags): PRE (block 0: f32 features x f16 adapter in the input path), POST (block 70: learned post
blend of block-0 E4 and the 2x upsampled half-res E4), UPRES (block 66: f16 low-res projection + E4 skip *
scale), HEAD (RGBA head from the f16 result), POOL (2x2 box pool of the f16 result to E4), OUT_E4.

python block32_e4m3.py <flags> out.ptx      flags: sum of E4=2 PRE=8 POST=16 HEAD=32 POOL=64 UPRES=128
"""
import sys
from ptxgen import Ptx
from swin import *

F_OUT_E4, F_PRE, F_POST, F_HEAD, F_POOL, F_UPRES = 2, 8, 16, 32, 64, 128

# shared memory map (bytes)
W1S, W2S, WQKVS, WPROJS = 0, 4096, 8192, 11264
KROWS, VTS = 12288, 15360          # K [64 physical][48 B]; Vt [32 dim][80 B]
STG = 17920                        # per warp 1280 B: f16 tile [16][80 B]; the E4 tile [16][32 B] aliases its start
STG_WARP = 1280
WF16S = STG + 4 * STG_WARP         # adapter [32 n][16 k] f16 / head [16 n][32 k] f16 (1 KB)
SHARED_BYTES = WF16S + 1024
THREADS = 128
A2A_SMEM = 0      # accumulator -> A fragment through the per-warp staging tile (else shuffles)


def generate(flags, max_regs=None):
    pre, post, upres = bool(flags & F_PRE), bool(flags & F_POST), bool(flags & F_UPRES)
    head, pool, outE4 = bool(flags & F_HEAD), bool(flags & F_POOL), bool(flags & F_OUT_E4)
    assert pre + post + upres <= 1
    name = f"block32_e4m3_f{flags}"
    p = Ptx()
    params = [("u64", "pState"), ("u64", "pLow"), ("u64", "pW1"), ("u64", "pW2"), ("u64", "pWqkv"), ("u64", "pWproj"),
              ("u64", "pAux"), ("u64", "pPrior"), ("u64", "pOutE4"), ("u64", "pOut2"), ("u64", "pWf16"),
              ("u32", "width"), ("u32", "height"), ("u32", "shiftX"), ("u32", "shiftY"), ("u32", "windowsX"),
              ("u32", "windowCount"), ("u32", "auxFfnHalf"), ("u32", "auxAttnHalf"), ("u32", "scaleWord"),
              ("u32", "auxInputHalf"), ("u32", "auxAdapterHalf"), ("u32", "lowWidth"),
              ("u64", "pWait"), ("u32", "waitExpected"), ("u32", "waitShiftY"), ("u32", "waitScale"), ("u64", "pSignal"), ("u64", "pError")]
    # chaining (pWait / pSignal = 0: off): wait on the producer block's window rows covering this window's pixel rows
    # (waitScale 0: same resolution, 1: producer at 2x (pool), 2: producer at half (upres / post)), signal this block's
    p.entry(name, params, SHARED_BYTES, THREADS, max_regs)
    P = {n: (p.load_param_u64(n) if t == "u64" else p.load_param_u32(n)) for t, n in params}
    tid = p.special("tid.x"); ctaid = p.special("ctaid.x"); nctaid = p.special("nctaid.x")
    lane = p.and32(tid, 31); warp = p.shr32(tid, 5)
    li = LaneInfo(p, lane)
    g, t = li.g, li.t
    smem = p.shared_addr(p.imm32(0))
    zero = imm(p, 0)
    silu_k = SiluConsts(p)
    exp_k = ExpConsts(p)

    # ---- stage the block weights once: [rows][32 B] tiles, chunk ^= (row >> 2) & 1
    def stage_e4_rows(src, dstBase, chunks):
        for i in range((chunks + THREADS - 1) // THREADS):
            c = p.add32(tid, i * THREADS)
            pv = p.setp("lt.u32", c, chunks) if chunks % THREADS else None
            row = p.shr32(c, 1); chunk = p.and32(c, 1)
            pch = p.xor32(chunk, p.and32(p.shr32(row, 2), 1))
            dst = p.add32(p.add32(smem, dstBase), p.add32(p.shl32(row, 5), p.shl32(pch, 4)))
            gaddr = p.add64(src, p.widen(p.shl32(c, 4)))
            r = p.regs("b32", 4)
            pred = f"@{pv} " if pv else ""
            p.emit(f"{pred}ld.global.nc.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{gaddr}];")
            p.emit(f"{pred}st.shared.v4.b32 [{dst}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")
    stage_e4_rows(P["pW1"], W1S, 256)
    stage_e4_rows(P["pW2"], W2S, 256)
    stage_e4_rows(P["pWqkv"], WQKVS, 192)
    stage_e4_rows(P["pWproj"], WPROJS, 64)
    if pre:
        # adapter f16 [16 k][32 n] (global) -> shared [32 n][16 k] with the 32-byte-row chunk swizzle
        n = p.shr32(tid, 2); kq = p.and32(tid, 3)
        hs = []
        for i in range(4):
            k = p.add32(p.shl32(kq, 2), i)
            addr = p.add64(P["pWf16"], p.widen(p.shl32(p.add32(p.shl32(k, 5), n), 1)))
            h = p.reg("b16"); p.emit(f"ld.global.nc.b16 {h}, [{addr}];"); hs.append(h)
        w0 = p.reg("b32"); p.emit(f"mov.b32 {w0}, {{{hs[0]}, {hs[1]}}};")
        w1 = p.reg("b32"); p.emit(f"mov.b32 {w1}, {{{hs[2]}, {hs[3]}}};")
        pch = p.xor32(p.shr32(kq, 1), p.and32(p.shr32(n, 2), 1))
        dst = p.add32(p.add32(smem, WF16S), p.add32(p.shl32(n, 5), p.add32(p.shl32(pch, 4), p.shl32(p.and32(kq, 1), 3))))
        p.emit(f"st.shared.v2.b32 [{dst}], {{{w0}, {w1}}};")
    if head:
        # head f16 [32 k][16 n] (global, padded N = 16) -> shared [16 n][32 k] (64-byte rows)
        n = p.shr32(tid, 3); kq = p.and32(tid, 7)
        hs = []
        for i in range(4):
            k = p.add32(p.shl32(kq, 2), i)
            addr = p.add64(P["pWf16"], p.widen(p.shl32(p.add32(p.shl32(k, 4), n), 1)))
            h = p.reg("b16"); p.emit(f"ld.global.nc.b16 {h}, [{addr}];"); hs.append(h)
        w0 = p.reg("b32"); p.emit(f"mov.b32 {w0}, {{{hs[0]}, {hs[1]}}};")
        w1 = p.reg("b32"); p.emit(f"mov.b32 {w1}, {{{hs[2]}, {hs[3]}}};")
        dst = p.add32(p.add32(smem, WF16S), p.add32(p.shl32(n, 6), p.shl32(kq, 3)))
        p.emit(f"st.shared.v2.b32 [{dst}], {{{w0}, {w1}}};")

    # ---- per-lane constants
    # skip scale pairs: accumulator tile j of either row half holds columns 8j + 2t, +1
    auxBase = P["pAux"]
    def aux_pairs(halfOffsetReg):
        regs = []
        for j in range(4):
            off = p.add32(p.shl32(halfOffsetReg, 1), p.add32(p.shl32(t, 2), p.imm32(16 * j)))
            addr = p.add64(auxBase, p.widen(off))
            r = p.reg("b32"); p.emit(f"ld.global.nc.b32 {r}, [{addr}];"); regs.append(r)
        return regs
    auxFfn = aux_pairs(P["auxFfnHalf"])
    auxAttn = aux_pairs(P["auxAttnHalf"])
    scaleAddr = p.add64(auxBase, p.widen(p.shl32(P["scaleWord"], 2)))
    scaleF = p.reg("f32"); p.emit(f"ld.global.nc.f32 {scaleF}, [{scaleAddr}];")
    scaleH = p.reg("f16"); p.emit(f"cvt.rn.f16.f32 {scaleH}, {scaleF};")
    scale2 = pack16(p, scaleH, scaleH)
    # load lane: row = lane >> 1 of this warp's 16 tokens, part = lane & 1 (16 bytes of the E4 row)
    ldRow = p.shr32(lane, 1); ldPart = p.and32(lane, 1)
    ldTokX = p.and32(ldRow, 7); ldTokY = p.add32(p.shl32(warp, 1), p.shr32(lane, 4))
    ldSw = p.and32(p.shr32(ldRow, 2), 1)
    stgWarp = p.add32(p.add32(smem, STG), p.mul32(warp, STG_WARP))
    stgLdE4 = p.add32(stgWarp, p.add32(p.shl32(ldRow, 5), p.shl32(p.xor32(ldPart, ldSw), 4)))   # this lane's 16-byte E4 chunk
    stgLdF16 = p.add32(stgWarp, p.add32(p.mul32(ldRow, 80), p.shl32(ldPart, 5)))                # this lane's 32-byte f16 chunk
    # ldmatrix lane addressing for [16 rows][32 B] tiles (A) and [n rows][32 B] tiles (B), both swizzled
    l8 = p.and32(lane, 7); lm = p.shr32(lane, 3)
    aRowL = p.add32(l8, p.shl32(p.and32(lm, 1), 3)); aChunkL = p.shr32(lane, 4)
    aPch = p.xor32(aChunkL, p.and32(p.shr32(aRowL, 2), 1))
    aLaneOff = p.add32(p.shl32(aRowL, 5), p.shl32(aPch, 4))
    stgA = p.add32(stgWarp, aLaneOff)
    bNL = p.add32(l8, p.shl32(p.shr32(lane, 4), 3)); bChunkL = p.and32(lm, 1)
    bPch = p.xor32(bChunkL, p.and32(p.shr32(bNL, 2), 1))
    bLaneOff = p.add32(p.shl32(bNL, 5), p.shl32(bPch, 4))
    w1B = p.add32(p.add32(smem, W1S), bLaneOff)
    w2B = p.add32(p.add32(smem, W2S), bLaneOff)
    wqkvB = p.add32(p.add32(smem, WQKVS), bLaneOff)
    wprojB = p.add32(p.add32(smem, WPROJS), bLaneOff)
    wf16B = p.add32(p.add32(smem, WF16S), bLaneOff)
    # C-layout b16 loads from the E4 staging tile: row g (+8), byte column 8j + 2t -> chunk j >> 1 (swizzled), +8 (j & 1)
    sw_g = p.and32(p.shr32(g, 2), 1)
    cBase = [p.add32(stgWarp, p.add32(p.shl32(g, 5), p.add32(p.shl32(p.xor32(p.imm32(c), sw_g), 4), p.shl32(t, 1)))) for c in range(2)]
    # C-layout b32 loads from the f16 staging tile: row g (+8) at 80 B, word 4j + t
    cBaseF16 = p.add32(stgWarp, p.add32(p.mul32(g, 80), p.shl32(t, 2)))
    # K rows / Vt publication: this lane's tokens g and g + 8 -> physical indices
    phys = [tiled_token_regs(p, p.add32(p.shl32(warp, 4), p.add32(g, p.imm32(8 * h)))) for h in range(2)]
    kPub = [p.add32(p.add32(smem, KROWS), p.add32(p.mul32(phys[h], 48), p.shl32(t, 1))) for h in range(2)]
    vPub = [p.add32(p.add32(smem, VTS), p.add32(p.mul32(t, 160), phys[h])) for h in range(2)]
    # ldmatrix addressing for K [64][48 B] (n = key rows) and Vt [32][80 B] (n = dim rows)
    kLane = p.add32(p.add32(smem, KROWS), p.add32(p.mul32(bNL, 48), p.shl32(bChunkL, 4)))
    vLane = p.add32(p.add32(smem, VTS), p.add32(p.mul32(bNL, 80), p.shl32(bChunkL, 4)))
    # prior rows for queries 16 warp + g (+8): [64][64] f16, keys 8m + 2t
    priorLane = p.add64(P["pPrior"], p.widen(p.add32(p.shl32(p.add32(p.shl32(warp, 4), g), 7), p.shl32(t, 2))))
    quarter2 = imm(p, "0x34003400")   # 0.25, 0.25
    if post or upres:
        blendA = []; blendB = []
        for i in range(8):
            c = p.add32(p.shl32(ldPart, 4), p.imm32(2 * i))
            a = p.add64(auxBase, p.widen(p.shl32(p.add32(P["auxInputHalf"], c), 1)))
            r = p.reg("b32"); p.emit(f"ld.global.nc.b32 {r}, [{a}];"); blendA.append(r)
            if post:
                b = p.add64(auxBase, p.widen(p.shl32(p.add32(P["auxAdapterHalf"], c), 1)))
                r = p.reg("b32"); p.emit(f"ld.global.nc.b32 {r}, [{b}];"); blendB.append(r)
    # the attention prior rows of this warp's 16 queries never change: 16 registers for the whole kernel
    priorRegs = []
    for m in range(8):
        c0 = p.reg("b32"); p.emit(f"ld.global.nc.b32 {c0}, [{priorLane}+{16 * m}];")
        c1 = p.reg("b32"); p.emit(f"ld.global.nc.b32 {c1}, [{priorLane}+{1024 + 16 * m}];")
        priorRegs.append([c0, c1])
    p.emit("bar.sync 0;")   # weights staged

    # ================= persistent window loop (the next window's input words are prefetched into registers)
    def geometry(windowReg):
        wx = p.reg("b32"); p.emit(f"rem.u32 {wx}, {windowReg}, {P['windowsX']};")
        wy = p.reg("b32"); p.emit(f"div.u32 {wy}, {windowReg}, {P['windowsX']};")
        gm = {}
        gm["windowX"] = p.sub32(p.shl32(wx, 3), P["shiftX"]); gm["windowY"] = p.sub32(p.shl32(wy, 3), P["shiftY"])
        gm["x"] = p.add32(gm["windowX"], ldTokX); gm["y"] = p.add32(gm["windowY"], ldTokY)
        pvx = p.setp("lt.u32", gm["x"], P["width"]); pvy = p.setp("lt.u32", gm["y"], P["height"])
        gm["valid"] = p.reg("pred"); p.emit(f"and.pred {gm['valid']}, {pvx}, {pvy};")
        gm["pixel"] = p.mad32(gm["y"], P["width"], gm["x"])
        if post or upres: gm["lowPixel"] = p.mad32(p.shr32(gm["y"], 1), P["lowWidth"], p.shr32(gm["x"], 1))
        gm["wy"] = wy
        # producer window rows: this window's pixel rows [max(8 wy - shiftY, 0), min(+7, height - 1)] mapped to the
        # producer's resolution (x2 / x0.5), then (y + producer shiftY) >> 3
        yTop = gm["windowY"]
        pNeg = p.setp("gt.s32", p.imm32(0), yTop)
        y0 = p.selp32(pNeg, p.imm32(0), yTop)
        y1 = p.add32(yTop, 7)
        hm1 = p.sub32(P["height"], 1)
        pBig = p.setp("gt.u32", y1, hm1)
        y1 = p.selp32(pBig, hm1, y1)
        pS1 = p.setp("eq.u32", P["waitScale"], 1); pS2 = p.setp("eq.u32", P["waitScale"], 2)
        y0 = p.selp32(pS1, p.shl32(y0, 1), p.selp32(pS2, p.shr32(y0, 1), y0))
        y1 = p.selp32(pS1, p.add32(p.shl32(y1, 1), p.imm32(1)), p.selp32(pS2, p.shr32(y1, 1), y1))
        gm["wr0"] = p.shr32(p.add32(y0, P["waitShiftY"]), 3); gm["wr1"] = p.shr32(p.add32(y1, P["waitShiftY"]), 3)
        return gm

    pWaitOn = p.setp("ne.u64", P["pWait"], 0)
    pSignalOn = p.setp("ne.u64", P["pSignal"], 0)
    def wait_window(gm, guard=None):
        g2 = pWaitOn
        if guard is not None:
            g2 = p.reg("pred"); p.emit(f"and.pred {g2}, {pWaitOn}, {guard};")
        sync_wait(p, P["pWait"], gm["wr0"], gm["wr1"], P["waitExpected"], lane, g2, warp, error64=P["pError"])

    def load_v4(base, byteOff, pred, produced):
        """produced: written by a chained (barrier-free) launch -> L2 path; else the non-coherent (texture) path."""
        r = p.regs("b32", 4)
        for q in r: p.emit(f"mov.b32 {q}, 0;")
        addr = p.add64(base, p.widen(byteOff))
        hint = "cg" if produced else "nc"
        p.emit(f"@{pred} ld.global.{hint}.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{addr}];")
        return r

    def issue_loads(gm, pred):
        pixel = gm["pixel"]
        if pre:
            # features [tokens][16] f32: 64 B per token, this lane's 8 floats at part * 32
            return (load_v4(P["pState"], p.add32(p.shl32(pixel, 6), p.shl32(ldPart, 5)), pred, False) +
                    load_v4(P["pState"], p.add32(p.add32(p.shl32(pixel, 6), p.shl32(ldPart, 5)), p.imm32(16)), pred, False))
        # the state is produced by the previous chained block except for the post block (block 0's output) and the
        # upres block (the skip from the encoder); the low-resolution input of the post block is chained
        regs = load_v4(P["pState"], p.add32(p.shl32(pixel, 5), p.shl32(ldPart, 4)), pred, not (post or upres))
        if post:
            regs += load_v4(P["pLow"], p.add32(p.shl32(gm["lowPixel"], 5), p.shl32(ldPart, 4)), pred, True)
        elif upres:
            regs += load_v4(P["pLow"], p.add32(p.shl32(gm["lowPixel"], 6), p.shl32(ldPart, 5)), pred, False)
            regs += load_v4(P["pLow"], p.add32(p.add32(p.shl32(gm["lowPixel"], 6), p.shl32(ldPart, 5)), p.imm32(16)), pred, False)
        return regs

    window = p.reg("b32"); p.emit(f"mov.u32 {window}, {ctaid};")
    pW = p.setp("lt.u32", window, P["windowCount"])
    endL = p.label("END"); loopL = p.label("WINDOW")
    p.emit(f"@!{pW} bra {endL};")
    gm0 = geometry(window)
    wait_window(gm0)
    first = issue_loads(gm0, gm0["valid"])
    carried = p.regs("b32", len(first))
    for c, f in zip(carried, first): p.emit(f"mov.b32 {c}, {f};")
    p.emit(f"{loopL}:")
    gm = geometry(window)
    windowX, windowY, x, y, valid, pixel = gm["windowX"], gm["windowY"], gm["x"], gm["y"], gm["valid"], gm["pixel"]
    inputs = carried
    # prefetch the next window
    nextWindow = p.add32(window, nctaid)
    pNext = p.setp("lt.u32", nextWindow, P["windowCount"])
    gmN = geometry(nextWindow)
    wait_window(gmN, pNext)
    pLoadN = p.reg("pred"); p.emit(f"and.pred {pLoadN}, {gmN['valid']}, {pNext};")
    nxt = issue_loads(gmN, pLoadN)

    def st_v4(addr, r):
        p.emit(f"st.shared.v4.b32 [{addr}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")

    def warp_sync():
        p.emit("bar.warp.sync 0xffffffff;")

    def e4_word_pairs(words):
        """four b32 words (16 E4 codes) -> eight .b16 code pairs in order."""
        out = []
        for w in words:
            lo = p.reg("b16"); hi = p.reg("b16")
            p.emit(f"mov.b32 {{{lo}, {hi}}}, {w};")
            out += [lo, hi]
        return out

    # ---- Phase F inputs: x (E4 A fragment) and cInit (f16 C tiles = skip)
    cInit = [[None, None] for _ in range(4)]
    if pre:
        f0, f1 = inputs[0:4], inputs[4:8]
        hw = []
        for a, b in ((f0[0], f0[1]), (f0[2], f0[3]), (f1[0], f1[1]), (f1[2], f1[3])):
            r = p.reg("b32"); p.emit(f"cvt.rn.f16x2.f32 {r}, {b}, {a};"); hw.append(r)
        st_v4(stgLdE4, hw)
        warp_sync()
        fa = ldm4(p, stgA)
        adapted = []
        for i2 in range(2):
            r = ldm4(p, p.add32(wf16B, i2 * 512))
            for jj in range(2):
                d = zero_tile(p)
                mma_f16(p, d, fa, (r[0], r[1]) if jj == 0 else (r[2], r[3]), d)
                adapted.append(d)
        for j in range(4):
            cInit[j] = [adapted[j][0], adapted[j][1]]
        xa = acc_to_a(p, adapted, li, zero)   # adapter output: keep the exact NaN -> 0 (f32 features)
        warp_sync()
    elif post or upres:
        xw = inputs[0:4]
        if post: sk = inputs[4:8]
        else: sk0, sk1 = inputs[4:8], inputs[8:12]
        adPairs = e4_word_pairs(xw)
        if post: upPairs = e4_word_pairs(sk)
        raw = []
        for i in range(8):
            ad = nanzero(p, e4x2_to_f16x2(p, adPairs[i]), zero)
            if post:
                up = nanzero(p, e4x2_to_f16x2(p, upPairs[i]), zero)
                inputValue = hmul2(p, up, blendA[i])
                raw.append(hfma2(p, ad, blendB[i], inputValue))
            else:
                projected = (sk0 + sk1)[i]
                raw.append(hfma2(p, ad, blendA[i], projected))
        st_v4(stgLdF16, raw[0:4]); st_v4(p.add32(stgLdF16, 16), raw[4:8])
        warp_sync()
        for j in range(4):
            for h in range(2):
                r = p.reg("b32"); p.emit(f"ld.shared.b32 {r}, [{cBaseF16}+{16 * j + 640 * h}];"); cInit[j][h] = r
        warp_sync()   # the E4 tile aliases the f16 tile
        codes = [cvt_e4x2(p, nanzero(p, raw[i], zero)) for i in range(8)]
        cw = [pack16(p, codes[2 * i], codes[2 * i + 1]) for i in range(4)]
        st_v4(stgLdE4, cw)
        warp_sync()
        xa = ldm4(p, stgA)
        warp_sync()
    else:
        xw = inputs[0:4]
        st_v4(stgLdE4, xw)
        warp_sync()
        xa = ldm4(p, stgA)
        for j in range(4):
            for h in range(2):
                hcode = p.reg("b16"); p.emit(f"ld.shared.b16 {hcode}, [{cBase[j >> 1]}+{8 * (j & 1) + 256 * h}];")
                cInit[j][h] = e4x2_to_f16x2(p, hcode)
        warp_sync()
    # scaled skip seeds the W2 accumulators
    ffn = [[hmul2(p, cInit[j][h], auxFfn[j]) for h in range(2)] for j in range(4)]

    # ---- FFN: four quarters of 32 hidden units (W1 columns permuted), W2 chained in K order
    for qq in range(4):
        hid = []
        for i2 in range(2):
            r = ldm4(p, p.add32(w1B, (2 * qq + i2) * 512))
            for jj in range(2):
                d = zero_tile(p)
                mma_e4(p, d, xa, (r[0], r[1]) if jj == 0 else (r[2], r[3]), d)
                hid.append(d)
        for d in hid:
            d[0] = silu(p, d[0], silu_k); d[1] = silu(p, d[1], silu_k)
        q = [[b16_to_b32(p, cvt_e4x2(p, d[h])) for h in range(2)] for d in hid]
        a2 = [prmt(p, q[0][0], q[1][0], "0x5410"), prmt(p, q[0][1], q[1][1], "0x5410"),
              prmt(p, q[2][0], q[3][0], "0x5410"), prmt(p, q[2][1], q[3][1], "0x5410")]
        for i2 in range(2):
            r = ldm4(p, p.add32(w2B, qq * 1024 + i2 * 512))
            for jj in range(2):
                j = 2 * i2 + jj
                mma_e4(p, ffn[j], a2, (r[0], r[1]) if jj == 0 else (r[2], r[3]), ffn[j])
    ffnQ = acc_to_a_smem(p, ffn, cBase, stgA) if A2A_SMEM else acc_to_a(p, ffn, li)

    # ---- q, k, v
    qkv = []
    for m in range(3):
        acc = []
        for i2 in range(2):
            r = ldm4(p, p.add32(wqkvB, (2 * m + i2) * 512))
            for jj in range(2):
                d = zero_tile(p)
                mma_e4(p, d, ffnQ, (r[0], r[1]) if jj == 0 else (r[2], r[3]), d)
                acc.append(d)
        qkv.append(acc)
    qn = normalize(p, qkv[0], li, zero, scale2)
    kn = normalize(p, qkv[1], li, zero)
    mq = acc_to_a_smem(p, qn, cBase, stgA) if A2A_SMEM else acc_to_a(p, qn, li)
    for j in range(4):
        for h in range(2):
            hc = cvt_e4x2(p, kn[j][h])
            p.emit(f"st.shared.b16 [{kPub[h]}+{8 * j}], {hc};")
            vc = cvt_e4x2(p, qkv[2][j][h])
            vhi = p.reg("b16"); p.emit(f"shr.b16 {vhi}, {vc}, 8;")
            p.emit(f"st.shared.u8 [{vPub[h]}+{640 * j}], {vc};")
            p.emit(f"st.shared.u8 [{vPub[h]}+{640 * j + 80}], {vhi};")
    p.emit("bar.sync 0;")   # K / V rows of all four warps visible

    # ---- attention: S = prior + Q K^T (physical key order), softmax, P V
    s = []
    for m in range(8):
        c0 = p.reg("b32"); p.emit(f"mov.b32 {c0}, {priorRegs[m][0]};")
        c1 = p.reg("b32"); p.emit(f"mov.b32 {c1}, {priorRegs[m][1]};")
        s.append([c0, c1])
    for m2 in range(4):
        r = ldm4(p, p.add32(kLane, m2 * 768))
        mma_e4(p, s[2 * m2], mq, (r[0], r[1]), s[2 * m2])
        mma_e4(p, s[2 * m2 + 1], mq, (r[2], r[3]), s[2 * m2 + 1])
    w = softmax(p, s, li, exp_k)
    pa = [acc_to_a_smem(p, w[0:4], cBase, stgA), acc_to_a_smem(p, w[4:8], cBase, stgA)] if A2A_SMEM else [acc_to_a(p, w[0:4], li), acc_to_a(p, w[4:8], li)]
    o = [zero_tile(p) for _ in range(4)]
    for step in range(2):
        for d2 in range(2):
            r = ldm4(p, p.add32(vLane, d2 * 1280 + step * 32))
            mma_e4(p, o[2 * d2], pa[step], (r[0], r[1]), o[2 * d2])
            mma_e4(p, o[2 * d2 + 1], pa[step], (r[2], r[3]), o[2 * d2 + 1])
    attended = acc_to_a_smem(p, o, cBase, stgA) if A2A_SMEM else acc_to_a(p, o, li)
    result = [[hmul2(p, ffn[j][h], auxAttn[j]) for h in range(2)] for j in range(4)]
    for i2 in range(2):
        r = ldm4(p, p.add32(wprojB, i2 * 512))
        for jj in range(2):
            j = 2 * i2 + jj
            mma_e4(p, result[j], attended, (r[0], r[1]) if jj == 0 else (r[2], r[3]), result[j])

    # ---- outputs
    stgOut = p.add32(stgWarp, p.add32(p.shl32(g, 5), p.shl32(t, 1)))   # E4 output tile [16][32 B] (plain rows)
    stgRead = p.add32(stgWarp, p.add32(p.shl32(ldRow, 5), p.shl32(ldPart, 4)))
    if outE4:
        for j in range(4):
            for h in range(2):
                hc = cvt_e4x2(p, result[j][h])
                p.emit(f"st.shared.b16 [{stgOut}+{8 * j + 256 * h}], {hc};")
        warp_sync()
        r = p.regs("b32", 4)
        p.emit(f"ld.shared.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{stgRead}];")
        addr = p.add64(P["pOutE4"], p.widen(p.add32(p.shl32(pixel, 5), p.shl32(ldPart, 4))))
        p.emit(f"@{valid} st.global.v4.b32 [{addr}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")
        warp_sync()
    if pool:
        # ds_fp8: window rows 2w, 2w+1 (rows g, g+8 of the tile) pool to half-resolution pixels (windowX + 2i, windowY + 2w):
        # ((a + b) + (c + d)) * 0.25 in halves; lanes with even g hold pixel i = g >> 1.
        poolStg = p.add32(stgWarp, 512)
        gEven = p.setp("eq.u32", p.and32(g, 1), 0)
        for j in range(4):
            a, c = result[j][0], result[j][1]
            b = shfl_xor(p, a, 4); d = shfl_xor(p, c, 4)
            top = hadd2(p, a, b); bottom = hadd2(p, c, d)
            sm = hadd2(p, top, bottom)
            v = hmul2(p, sm, quarter2)
            hc = cvt_e4x2(p, v)
            addr = p.add32(poolStg, p.add32(p.shl32(p.shr32(g, 1), 5), p.add32(p.shl32(t, 1), p.imm32(8 * j))))
            p.emit(f"@{gEven} st.shared.b16 [{addr}], {hc};")
        warp_sync()
        pLane8 = p.setp("lt.u32", lane, 8)
        i = p.shr32(lane, 1); part = p.and32(lane, 1)
        r = p.regs("b32", 4)
        raddr = p.add32(poolStg, p.add32(p.shl32(i, 5), p.shl32(part, 4)))
        p.emit(f"@{pLane8} ld.shared.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{raddr}];")
        sx = p.add32(windowX, p.shl32(i, 1)); sy = p.add32(windowY, p.shl32(warp, 1))
        sx1 = p.add32(sx, 1); sy1 = p.add32(sy, 1)
        pvx = p.setp("lt.u32", sx1, P["width"]); pvy = p.setp("lt.u32", sy1, P["height"])
        pv = p.reg("pred"); p.emit(f"and.pred {pv}, {pvx}, {pvy};")
        p.emit(f"and.pred {pv}, {pv}, {pLane8};")
        pnx = p.setp("ge.s32", sx, 0); pny = p.setp("ge.s32", sy, 0)
        p.emit(f"and.pred {pv}, {pv}, {pnx};"); p.emit(f"and.pred {pv}, {pv}, {pny};")
        low = p.mad32(p.shr32(sy, 1), P["lowWidth"], p.shr32(sx, 1))
        addr = p.add64(P["pOut2"], p.widen(p.add32(p.shl32(low, 5), p.shl32(part, 4))))
        p.emit(f"@{pv} st.global.v4.b32 [{addr}], {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}};")
        warp_sync()
    if head:
        # RGBA head: two k16 steps of the f16 result x head weights [16 n][32 k] (n8 tile 0 holds columns 0..3)
        hd = zero_tile(p)
        hLane = p.add32(p.add32(smem, WF16S), p.add32(p.shl32(l8, 6), p.shl32(p.and32(lm, 1), 4)))
        for step in range(2):
            b = ldm2(p, p.add32(hLane, step * 32))
            mma_f16(p, hd, acc_f16_to_a(result, step), b, hd)
        pT2 = p.setp("lt.u32", t, 2)
        for h in range(2):
            xo = p.add32(windowX, g); yo = p.add32(windowY, p.add32(p.shl32(warp, 1), p.imm32(h)))
            pvx = p.setp("lt.u32", xo, P["width"]); pvy = p.setp("lt.u32", yo, P["height"])
            pv = p.reg("pred"); p.emit(f"and.pred {pv}, {pvx}, {pvy};"); p.emit(f"and.pred {pv}, {pv}, {pT2};")
            lo, hi = unpack16(p, hd[h])
            f0 = p.reg("f32"); p.emit(f"cvt.f32.f16 {f0}, {lo};")
            f1 = p.reg("f32"); p.emit(f"cvt.f32.f16 {f1}, {hi};")
            px = p.mad32(yo, P["width"], xo)
            addr = p.add64(P["pOut2"], p.widen(p.add32(p.shl32(px, 4), p.shl32(t, 3))))
            p.emit(f"@{pv} st.global.v2.f32 [{addr}], {{{f0}, {f1}}};")
    p.emit("fence.acq_rel.gpu;")
    p.emit("bar.sync 0;")   # K / V and staging are reused by the next window; the window's stores are released
    pSig = p.reg("pred"); p.emit(f"and.pred {pSig}, {pSignalOn}, {p.setp('eq.u32', tid, 0)};")
    sigAddr = p.add64(P["pSignal"], p.widen(p.shl32(gm["wy"], 2)))
    p.emit(f"@{pSig} red.release.gpu.global.add.u32 [{sigAddr}], 1;")
    for c, n in zip(carried, nxt): p.emit(f"mov.b32 {c}, {n};")
    p.emit(f"mov.u32 {window}, {nextWindow};")
    p.emit(f"@{pNext} bra {loopL};")
    p.emit(f"{endL}:")
    p.emit("ret;")
    return name, p.finish()


if __name__ == "__main__":
    flags, out = int(sys.argv[1]), sys.argv[2]
    max_regs = int(sys.argv[3]) if len(sys.argv) > 3 else None
    name, text = generate(flags, max_regs)
    open(out, "w").write(text)
    print(name)
