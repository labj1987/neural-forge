"""Global ViT cosine attention (global_attention.comp) for one (head, 64-query block): the head's K/V rows of all
padded tokens are normalized / quantized into shared memory (K rows [token][32 B], V transposed [dim][token] with the
keys of every 32-group permuted so that the exp'd score fragments pack straight into the P·V A operand), the four
warps normalize their 16 query rows, then per 64-key block: S = Q K^T, the ViT exponential (f16 fma, clamp, bit
shift), the specified block sum ((b0 + b1) + b2) + b3 / ((t0 + t1) + t2) + t3 / even + odd, running total per row,
unnormalized E4 weights times V accumulated over the blocks in key order; the padding correction and the f32
reciprocal (rounded to half) are applied afterwards. Arithmetic identical to the GLSL kernel.

Chaining: waits for `waitExpected` signals at pWait (the QKV GEMM), signals pSignal once per workgroup.

python global_attention_e4m3.py <paddedTokens> out.ptx
"""
import sys
from ptxgen import Ptx
from swin import *

THREADS = 128
QROWS = 0                     # [64][32 B] swizzled (chunk ^= (row >> 2) & 1)
KROWS = 2048                  # [padded][32 B] swizzled



def generate(padded):
    assert padded % 64 == 0 and padded <= 256
    blocks = padded // 64
    VT = KROWS + padded * 32                        # [32 dims][VT_STRIDE B]
    VT_STRIDE = padded + 16                          # (stride / 4) mod 32 = 4 or 20: conflict-free ldmatrix rows
    shared_bytes = VT + 32 * VT_STRIDE
    name = f"global_attention_e4m3_p{padded}"
    p = Ptx()
    params = [("u64", "pQkv"), ("u64", "pAux"), ("u64", "pOut"), ("u32", "tokens"), ("u32", "heads"), ("u32", "scaleWordOffset"),
              ("u64", "pWait"), ("u32", "waitExpected"), ("u64", "pSignal"), ("u64", "pError")]
    p.entry(name, params, shared_bytes, THREADS, None, dynamic_shared=True)
    P = {n: (p.load_param_u64(n) if t == "u64" else p.load_param_u32(n)) for t, n in params}
    tid = p.special("tid.x"); head = p.special("ctaid.x"); qb = p.special("ctaid.y")
    lane = p.and32(tid, 31); warp = p.shr32(tid, 5)
    li = LaneInfo(p, lane)
    g, t = li.g, li.t
    smem = p.shared_addr(p.imm32(0))
    zero32 = p.imm32(0)
    channels = p.mul32(P["heads"], p.imm32(32))
    rowBytes = p.mul32(P["heads"], p.imm32(192))              # f16 [heads * 96] per token
    headByte = p.mul32(head, p.imm32(192))
    # per-head learned scale (f32 -> f16), f16(sqrt 32) = 0x45A8
    learned = p.reg("f32"); p.emit(f"ld.global.nc.f32 {learned}, [{p.add64(P['pAux'], p.widen(p.shl32(p.add32(P['scaleWordOffset'], head), 2)))}];")
    learnedH = p.reg("f16"); p.emit(f"cvt.rn.f16.f32 {learnedH}, {learned};")
    learned2 = pack16(p, learnedH, learnedH)
    hs2 = imm(p, "0x45A845A8")
    # ---- chain wait (the QKV rows are produced by the chained GEMM)
    pWaitOn = p.setp("ne.u64", P["pWait"], "0")
    sync_wait(p, P["pWait"], zero32, zero32, P["waitExpected"], lane, pWaitOn, warp, error64=P["pError"])

    def load_row(token, valid, part):
        """16 words of f16 (part 0 = Q, 1 = K, 2 = V of the head) for `token` (zeros when not valid)."""
        addr = p.add64(P["pQkv"], p.widen(p.add32(p.mul32(token, rowBytes), p.add32(headByte, p.imm32(64 * part)))))
        w = []
        for c in range(4):
            r = p.regs("b32", 4)
            for x in r: p.emit(f"mov.b32 {x}, 0;")
            p.emit(f"@{valid} ld.global.cg.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{addr}+{16 * c}];")
            w += r
        return w

    def store_row16(base, row, codes):
        """8 words of E4 codes -> a [row][32 B] swizzled tile at `base`."""
        swz = p.and32(p.shr32(row, 2), 1)
        for c in range(2):
            addr = p.add32(base, p.add32(p.shl32(row, 5), p.shl32(p.xor32(p.imm32(c), swz), 4)))
            p.emit(f"st.shared.v4.b32 [{addr}], {{{codes[4 * c]}, {codes[4 * c + 1]}, {codes[4 * c + 2]}, {codes[4 * c + 3]}}};")

    # ---- Q of this workgroup's 64 tokens (threads 0..63)
    pQ = p.setp("lt.u32", tid, p.imm32(64))
    qToken = p.add32(p.shl32(qb, 6), tid)
    qValid = p.reg("pred"); p.emit(f"setp.lt.and.u32 {qValid}, {qToken}, {P['tokens']}, {pQ};")
    qL = p.label("QDONE")
    p.emit(f"@!{pQ} bra {qL};")
    wq = load_row(qToken, qValid, 0)
    qc = vit_quantize(p, wq, vit_norm(p, wq), True, hs2, learned2, zero32)
    qc = [p.selp32(qValid, c, zero32) for c in qc]
    store_row16(p.add32(smem, p.imm32(QROWS)), tid, qc)
    p.emit(f"{qL}:")
    # ---- K / V of all padded tokens: thread t handles tokens t, t + 128; V transposed with the 32-group permutation
    # physical position of key k: (k & ~31) | (k & 16) | (((k & 7) >> 1) << 2) | (k & 1) | (((k & 15) >> 3) << 1)
    for local0 in range(0, padded, THREADS):
        token = p.add32(tid, p.imm32(local0))
        pIn = p.setp("lt.u32", token, p.imm32(padded)) if local0 + THREADS > padded else None
        valid = p.reg("pred")
        if pIn is not None: p.emit(f"setp.lt.and.u32 {valid}, {token}, {P['tokens']}, {pIn};")
        else: p.emit(f"setp.lt.u32 {valid}, {token}, {P['tokens']};")
        skipL = p.label("KVSKIP")
        if pIn is not None: p.emit(f"@!{pIn} bra {skipL};")
        wk = load_row(token, valid, 1)
        wv = load_row(token, valid, 2)
        kc = vit_quantize(p, wk, vit_norm(p, wk), False, hs2, learned2, zero32)
        kc = [p.selp32(valid, c, zero32) for c in kc]
        store_row16(p.add32(smem, p.imm32(KROWS)), token, kc)
        vc = [pack16(p, cvt_e4x2(p, wv[2 * i]), cvt_e4x2(p, wv[2 * i + 1])) for i in range(8)]
        vc = [p.selp32(valid, c, zero32) for c in vc]
        ppos = p.or32(p.or32(p.and32(token, p.imm32(0xffffffe0 | 16)), p.shl32(p.shr32(p.and32(token, 7), 1), 2)),
                      p.or32(p.and32(token, 1), p.shl32(p.shr32(p.and32(token, 15), 3), 1)))
        vAddr = p.add32(smem, p.add32(p.imm32(VT), ppos))
        for i in range(8):     # word i holds dims 4 i .. 4 i + 3
            for b in range(4):
                byte = p.reg("b32")
                if b == 0: p.emit(f"mov.b32 {byte}, {vc[i]};")
                else: p.emit(f"shr.u32 {byte}, {vc[i]}, {8 * b};")
                p.emit(f"st.shared.u8 [{vAddr}+{(4 * i + b) * VT_STRIDE}], {byte};")
        p.emit(f"{skipL}:")
    p.emit("bar.sync 0;")
    # ---- fragments: Q A (this warp's 16 rows), K B rows, V^T B rows
    l8 = p.and32(lane, 7); lm = p.shr32(lane, 3)
    aRowL = p.add32(l8, p.shl32(p.and32(lm, 1), 3)); aChunkL = p.shr32(lane, 4)
    aPchL = p.xor32(aChunkL, p.and32(p.shr32(aRowL, 2), 1))
    aLane = p.add32(smem, p.add32(p.imm32(QROWS), p.add32(p.shl32(p.add32(aRowL, p.shl32(warp, 4)), 5), p.shl32(aPchL, 4))))
    mq = ldm4(p, aLane)
    bNL = p.add32(l8, p.shl32(p.shr32(lane, 4), 3)); bChunkL = p.and32(lm, 1)
    bPchL = p.xor32(bChunkL, p.and32(p.shr32(bNL, 2), 1))
    kLane = p.add32(smem, p.add32(p.imm32(KROWS), p.add32(p.shl32(bNL, 5), p.shl32(bPchL, 4))))
    vLane = p.add32(smem, p.add32(p.imm32(VT), p.add32(p.mul32(bNL, p.imm32(VT_STRIDE)), p.shl32(bChunkL, 4))))
    k = VitExpConsts(p)
    o = [zero_tile(p) for _ in range(4)]                      # 16 x 32 output: 4 n-tiles of 8 dims
    total = [None, None]                                      # f16 per row half (g, g + 8)
    for b in range(blocks):
        s = [zero_tile(p) for _ in range(8)]
        for m2 in range(4):
            r = ldm4(p, f"{kLane}+{b * 2048 + m2 * 512}")
            mma_e4(p, s[2 * m2], mq, (r[0], r[1]), s[2 * m2])
            mma_e4(p, s[2 * m2 + 1], mq, (r[2], r[3]), s[2 * m2 + 1])
        e = [[vit_exp(p, s[j][h], k) for h in range(2)] for j in range(8)]
        for h in range(2):
            bp = [hadd2(p, e[2 * j][h], e[2 * j + 1][h]) for j in range(4)]
            tp = hadd2(p, hadd2(p, hadd2(p, bp[0], bp[1]), bp[2]), bp[3])
            tq = [shfl_idx(p, tp, src) for src in li.quadLanes]
            parts = [unpack16(p, x) for x in tq]
            even = hadd(p, hadd(p, hadd(p, parts[0][0], parts[1][0]), parts[2][0]), parts[3][0])
            odd = hadd(p, hadd(p, hadd(p, parts[0][1], parts[1][1]), parts[2][1]), parts[3][1])
            blockSum = hadd(p, even, odd)
            total[h] = blockSum if b == 0 else hadd(p, total[h], blockSum)   # total = 0 + blockSum is exact
        # P (E4 weights) as the A operand: k32 group kb = n-tiles 4 kb .. 4 kb + 3 (the V tile's key permutation)
        for kb in range(2):
            c = [[cvt_e4x2(p, e[4 * kb + j][h]) for h in range(2)] for j in range(4)]
            pa = [pack16(p, c[0][0], c[1][0]), pack16(p, c[0][1], c[1][1]), pack16(p, c[2][0], c[3][0]), pack16(p, c[2][1], c[3][1])]
            for d2 in range(2):
                r = ldm4(p, f"{vLane}+{b * 64 + kb * 32 + d2 * 16 * VT_STRIDE}")
                mma_e4(p, o[2 * d2], pa, (r[0], r[1]), o[2 * d2])
                mma_e4(p, o[2 * d2 + 1], pa, (r[2], r[3]), o[2 * d2 + 1])
    # ---- padding correction, reciprocal, output
    padding = p.sub32(p.imm32(padded), P["tokens"])
    pPad = p.setp("ne.u32", padding, zero32)
    expZero = vit_exp(p, zero32, k)
    ez, _ = unpack16(p, expZero)
    ezf = p.reg("f32"); p.emit(f"cvt.f32.f16 {ezf}, {ez};")
    padf = p.reg("f32"); p.emit(f"cvt.rn.f32.u32 {padf}, {padding};")
    corrF = p.reg("f32"); p.emit(f"mul.rn.f32 {corrF}, {ezf}, {padf};")
    corr = p.reg("f16"); p.emit(f"cvt.rn.f16.f32 {corr}, {corrF};")
    for h in range(2):
        corrected = p.reg("f16"); p.emit(f"sub.rn.f16 {corrected}, {total[h]}, {corr};")
        p.emit(f"@{pPad} mov.b16 {total[h]}, {corrected};")
    for h in range(2):
        f = p.reg("f32"); p.emit(f"cvt.f32.f16 {f}, {total[h]};")
        rc = p.reg("f32"); p.emit(f"rcp.rn.f32 {rc}, {f};")
        rh = p.reg("f16"); p.emit(f"cvt.rn.f16.f32 {rh}, {rc};")
        rf = p.reg("f32"); p.emit(f"cvt.f32.f16 {rf}, {rh};")
        token = p.add32(p.shl32(qb, 6), p.add32(p.shl32(warp, 4), p.add32(g, p.imm32(8 * h))))
        ok = p.setp("lt.u32", token, P["tokens"])
        outAddr = p.add64(P["pOut"], p.widen(p.add32(p.mul32(token, channels), p.add32(p.shl32(head, 5), p.shl32(t, 1)))))
        for j in range(4):
            lo, hi = unpack16(p, o[j][h])
            flo = p.reg("f32"); p.emit(f"cvt.f32.f16 {flo}, {lo};"); p.emit(f"mul.rn.f32 {flo}, {flo}, {rf};")
            fhi = p.reg("f32"); p.emit(f"cvt.f32.f16 {fhi}, {hi};"); p.emit(f"mul.rn.f32 {fhi}, {fhi}, {rf};")
            v = p.reg("b32"); p.emit(f"cvt.rn.f16x2.f32 {v}, {fhi}, {flo};")
            hc = cvt_e4x2(p, v)
            p.emit(f"@{ok} st.global.b16 [{outAddr}+{8 * j}], {hc};")
    pSignalOn = p.setp("ne.u64", P["pSignal"], "0")
    sync_signal(p, [P["pSignal"]], tid, pSignalOn)
    p.emit("ret;")
    return name, p.finish(), shared_bytes


if __name__ == "__main__":
    padded, out = int(sys.argv[1]), sys.argv[2]
    name, text, shared = generate(padded)
    open(out, "w").write(text)
    print(name, shared)
