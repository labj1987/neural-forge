"""Global ViT cosine attention for any token count (high resolutions), in two PTX passes with the arithmetic of
global_attention_e4m3.py (bit-identical per key block, blocks accumulated in key order):

1. global_normalize_e4m3: one thread per (token, head): Q normalized (times f16(sqrt 32) and the learned scale),
   K normalized, V as E4 codes, written per head so the attention streams contiguous tiles:
     Q  [head][padded][32 B]                          at 0
     K  [head][padded][32 B]                          at heads * padded * 32
     V^T[head][32 dims][padded] (32-key permutation)  at 2 * heads * padded * 32
   Tokens >= `tokens` (padding) are zero codes. Grid (padded / 64, heads), 64 threads.

2. global_attention_stream_e4m3: one workgroup per (head, 64-query block), four warps of 16 query rows; the
   key blocks (K tile 2 KB + V^T tile 32 x 80 B) stream through a STAGES-deep cp.async ring; per block the same
   S = Q K^T, ViT exponential, the specified block sums, unnormalized P V as the resident kernel; padding correction,
   f32 reciprocal and the E4 output afterwards. Grid (heads, padded / 64), 128 threads, static shared memory.

Chaining: the normalize waits for the QKV GEMM's signals and signals once per workgroup; the attention waits for
the normalize workgroups and signals once per workgroup (the projection GEMM waits for heads * padded / 64).

python global_attention_stream_e4m3.py normalize|attention out.ptx [stages]
"""
import sys
from ptxgen import Ptx
from swin import *

QTILE = 0                      # [64][32 B] swizzled
STAGE0 = 2048
K_BYTES = 2048                 # [64 keys][32 B] swizzled
VT_STRIDE = 80                 # 64 keys + 16 pad: (80 / 4) mod 32 = 20, conflict-free ldmatrix rows
VT_BYTES = 32 * VT_STRIDE
STAGE_BYTES = K_BYTES + VT_BYTES


def key_position(p, token):
    """Physical key position within the V^T row: (k & ~31) | (k & 16) | (((k & 7) >> 1) << 2) | (k & 1) | (((k & 15) >> 3) << 1)."""
    return p.or32(p.or32(p.and32(token, p.imm32(0xffffffe0 | 16)), p.shl32(p.shr32(p.and32(token, 7), 1), 2)),
                  p.or32(p.and32(token, 1), p.shl32(p.shr32(p.and32(token, 15), 3), 1)))


def generate_normalize():
    name = "global_normalize_e4m3"
    p = Ptx()
    params = [("u64", "pQkv"), ("u64", "pAux"), ("u64", "pOut"), ("u32", "tokens"), ("u32", "padded"), ("u32", "heads"),
              ("u32", "scaleWordOffset"), ("u64", "pWait"), ("u32", "waitExpected"), ("u64", "pSignal"), ("u64", "pError")]
    p.entry(name, params, 0, 64)
    P = {n: (p.load_param_u64(n) if t == "u64" else p.load_param_u32(n)) for t, n in params}
    tid = p.special("tid.x"); tb = p.special("ctaid.x"); head = p.special("ctaid.y")
    lane = p.and32(tid, 31); warp = p.shr32(tid, 5)
    zero32 = p.imm32(0)
    token = p.add32(p.shl32(tb, 6), tid)
    valid = p.setp("lt.u32", token, P["tokens"])
    pWaitOn = p.setp("ne.u64", P["pWait"], "0")
    sync_wait(p, P["pWait"], zero32, zero32, P["waitExpected"], lane, pWaitOn, warp, error64=P["pError"])
    rowBytes = p.mul32(P["heads"], p.imm32(192))
    headByte = p.mul32(head, p.imm32(192))
    learned = p.reg("f32"); p.emit(f"ld.global.nc.f32 {learned}, [{p.add64(P['pAux'], p.widen(p.shl32(p.add32(P['scaleWordOffset'], head), 2)))}];")
    learnedH = p.reg("f16"); p.emit(f"cvt.rn.f16.f32 {learnedH}, {learned};")
    learned2 = pack16(p, learnedH, learnedH)
    hs2 = imm(p, "0x45A845A8")
    src = p.add64(P["pQkv"], p.widen(p.add32(p.mul32(token, rowBytes), headByte)))

    def load_row(part):
        w = []
        for c in range(4):
            r = p.regs("b32", 4)
            for x in r: p.emit(f"mov.b32 {x}, 0;")
            p.emit(f"@{valid} ld.global.cg.v4.b32 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{src}+{64 * part + 16 * c}];")
            w += r
        return w
    wq = load_row(0); wk = load_row(1); wv = load_row(2)
    qc = vit_quantize(p, wq, vit_norm(p, wq), True, hs2, learned2, zero32)
    kc = vit_quantize(p, wk, vit_norm(p, wk), False, hs2, learned2, zero32)
    vc = [pack16(p, cvt_e4x2(p, wv[2 * i]), cvt_e4x2(p, wv[2 * i + 1])) for i in range(8)]
    qc = [p.selp32(valid, c, zero32) for c in qc]
    kc = [p.selp32(valid, c, zero32) for c in kc]
    vc = [p.selp32(valid, c, zero32) for c in vc]
    headRows = p.mul32(head, P["padded"])                           # head * padded
    rowIndex = p.add32(headRows, token)
    planeBytes = p.mul32(p.mul32(P["heads"], P["padded"]), p.imm32(32))
    qAddr = p.add64(P["pOut"], p.widen(p.shl32(rowIndex, 5)))
    kAddr = p.add64(qAddr, p.widen(planeBytes))
    for c in range(2):
        p.emit(f"st.global.v4.b32 [{qAddr}+{16 * c}], {{{qc[4 * c]}, {qc[4 * c + 1]}, {qc[4 * c + 2]}, {qc[4 * c + 3]}}};")
        p.emit(f"st.global.v4.b32 [{kAddr}+{16 * c}], {{{kc[4 * c]}, {kc[4 * c + 1]}, {kc[4 * c + 2]}, {kc[4 * c + 3]}}};")
    # V^T: byte (dim) of the token at [head][dim][position]
    vBase = p.add64(P["pOut"], p.widen(p.shl32(planeBytes, 1)))
    vAddr = p.add64(vBase, p.widen(p.add32(p.mul32(headRows, p.imm32(32)), key_position(p, token))))
    for i in range(8):
        for b in range(4):
            byte = p.reg("b32")
            if b == 0: p.emit(f"mov.b32 {byte}, {vc[i]};")
            else: p.emit(f"shr.u32 {byte}, {vc[i]}, {8 * b};")
            p.emit(f"st.global.u8 [{p.add64(vAddr, p.widen(p.mul32(P['padded'], p.imm32(4 * i + b))))}], {byte};")
    pSignalOn = p.setp("ne.u64", P["pSignal"], "0")
    sync_signal(p, [P["pSignal"]], tid, pSignalOn)
    p.emit("ret;")
    return name, p.finish(), 0


def generate_attention(stages=4):
    name = "global_attention_stream_e4m3"
    shared_bytes = STAGE0 + stages * STAGE_BYTES
    p = Ptx()
    params = [("u64", "pNorm"), ("u64", "pOut"), ("u32", "tokens"), ("u32", "padded"), ("u32", "heads"),
              ("u64", "pWait"), ("u32", "waitExpected"), ("u64", "pSignal"), ("u64", "pError")]
    p.entry(name, params, shared_bytes, 128)
    P = {n: (p.load_param_u64(n) if t == "u64" else p.load_param_u32(n)) for t, n in params}
    tid = p.special("tid.x"); head = p.special("ctaid.x"); qb = p.special("ctaid.y")
    lane = p.and32(tid, 31); warp = p.shr32(tid, 5)
    li = LaneInfo(p, lane)
    g, t = li.g, li.t
    smem = p.shared_addr(p.imm32(0))
    zero32 = p.imm32(0)
    channels = p.mul32(P["heads"], p.imm32(32))
    blocks = p.shr32(P["padded"], 6)
    pWaitOn = p.setp("ne.u64", P["pWait"], "0")
    sync_wait(p, P["pWait"], zero32, zero32, P["waitExpected"], lane, pWaitOn, warp, error64=P["pError"])
    # ---- source addresses (this head)
    headRows = p.mul32(head, P["padded"])
    planeBytes = p.mul32(p.mul32(P["heads"], P["padded"]), p.imm32(32))
    qRowsAddr = p.add64(P["pNorm"], p.widen(p.shl32(p.add32(headRows, p.shl32(qb, 6)), 5)))     # Q rows of the query block
    kPlane = p.add64(P["pNorm"], p.widen(planeBytes))
    kHead = p.add64(kPlane, p.widen(p.shl32(headRows, 5)))                                        # K rows of the head
    vHead = p.add64(p.add64(P["pNorm"], p.widen(p.shl32(planeBytes, 1))), p.widen(p.mul32(headRows, p.imm32(32))))  # V^T rows
    # per-thread copy assignments: K / Q chunk (row t >> 1, chunk t & 1, swizzled), V^T chunk (dim t >> 2, chunk t & 3)
    cRow = p.shr32(tid, 1); cChunk = p.and32(tid, 1)
    cSwz = p.and32(p.shr32(cRow, 2), 1)
    kDstOff = p.add32(p.shl32(cRow, 5), p.shl32(p.xor32(cChunk, cSwz), 4))                       # within a K / Q tile
    kSrcOff = p.widen(p.add32(p.shl32(cRow, 5), p.shl32(cChunk, 4)))                             # within a 64-row block
    vDim = p.shr32(tid, 2); vChunk = p.and32(tid, 3)
    vDstOff = p.add32(p.mul32(vDim, p.imm32(VT_STRIDE)), p.shl32(vChunk, 4))
    vSrc = p.add64(vHead, p.widen(p.add32(p.mul32(vDim, P["padded"]), p.shl32(vChunk, 4))))     # + 64 * block
    kSrc = p.add64(kHead, kSrcOff)                                                                 # + 2048 * block
    # Q tile
    p.emit(f"cp.async.cg.shared.global [{p.add32(smem, p.add32(p.imm32(QTILE), kDstOff))}], [{p.add64(qRowsAddr, kSrcOff)}], 16;")

    def issue(block, stage):
        """cp.async of key block `block` (register) into stage `stage` (literal), guarded by block < blocks; one group."""
        ok = p.setp("lt.u32", block, blocks)
        kS = p.add64(kSrc, p.widen(p.shl32(block, 11)))
        vS = p.add64(vSrc, p.widen(p.shl32(block, 6)))
        kD = p.add32(smem, p.add32(p.imm32(STAGE0 + stage * STAGE_BYTES), kDstOff))
        vD = p.add32(smem, p.add32(p.imm32(STAGE0 + stage * STAGE_BYTES + K_BYTES), vDstOff))
        p.emit(f"@{ok} cp.async.cg.shared.global [{kD}], [{kS}], 16;")
        p.emit(f"@{ok} cp.async.cg.shared.global [{vD}], [{vS}], 16;")
        p.emit("cp.async.commit_group;")
    for s in range(stages - 1):
        issue(p.imm32(s), s)
    # ---- fragments
    l8 = p.and32(lane, 7); lm = p.shr32(lane, 3)
    aRowL = p.add32(l8, p.shl32(p.and32(lm, 1), 3)); aChunkL = p.shr32(lane, 4)
    aPchL = p.xor32(aChunkL, p.and32(p.shr32(aRowL, 2), 1))
    aLane = p.add32(smem, p.add32(p.imm32(QTILE), p.add32(p.shl32(p.add32(aRowL, p.shl32(warp, 4)), 5), p.shl32(aPchL, 4))))
    bNL = p.add32(l8, p.shl32(p.shr32(lane, 4), 3)); bChunkL = p.and32(lm, 1)
    bPchL = p.xor32(bChunkL, p.and32(p.shr32(bNL, 2), 1))
    kLane = p.add32(smem, p.add32(p.imm32(STAGE0), p.add32(p.shl32(bNL, 5), p.shl32(bPchL, 4))))
    vLane = p.add32(smem, p.add32(p.imm32(STAGE0 + K_BYTES), p.add32(p.mul32(bNL, p.imm32(VT_STRIDE)), p.shl32(bChunkL, 4))))
    k = VitExpConsts(p)
    o = [zero_tile(p) for _ in range(4)]
    total = [p.reg("f16"), p.reg("f16")]
    for h in range(2): p.emit(f"mov.b16 {total[h]}, 0;")
    mq = None
    # ---- the key-block loop, unrolled `stages` times so every stage address is a literal
    b = p.reg("b32"); p.emit(f"mov.u32 {b}, 0;")
    loopL = p.label("BLOCKS"); doneL = p.label("BLOCKSDONE")
    p.emit(f"{loopL}:")
    for s in range(stages):
        pB = p.setp("lt.u32", b, blocks)
        p.emit(f"@!{pB} bra {doneL};")
        p.emit(f"cp.async.wait_group {stages - 2};")
        p.emit("bar.sync 0;")
        # refill the stage consumed last iteration with block b + stages - 1
        issue(p.add32(b, p.imm32(stages - 1)), (s + stages - 1) % stages)
        if mq is None:
            mq = ldm4(p, aLane)   # Q fragment (the tile arrived with the first group)
        st = s * STAGE_BYTES
        sTiles = [zero_tile(p) for _ in range(8)]
        for m2 in range(4):
            r = ldm4(p, f"{kLane}+{st + m2 * 512}")
            mma_e4(p, sTiles[2 * m2], mq, (r[0], r[1]), sTiles[2 * m2])
            mma_e4(p, sTiles[2 * m2 + 1], mq, (r[2], r[3]), sTiles[2 * m2 + 1])
        e = [[vit_exp(p, sTiles[j][h], k) for h in range(2)] for j in range(8)]
        for h in range(2):
            bp = [hadd2(p, e[2 * j][h], e[2 * j + 1][h]) for j in range(4)]
            tp = hadd2(p, hadd2(p, hadd2(p, bp[0], bp[1]), bp[2]), bp[3])
            tq = [shfl_idx(p, tp, src) for src in li.quadLanes]
            parts = [unpack16(p, x) for x in tq]
            even = hadd(p, hadd(p, hadd(p, parts[0][0], parts[1][0]), parts[2][0]), parts[3][0])
            odd = hadd(p, hadd(p, hadd(p, parts[0][1], parts[1][1]), parts[2][1]), parts[3][1])
            blockSum = hadd(p, even, odd)
            p.emit(f"add.rn.f16 {total[h]}, {total[h]}, {blockSum};")
        for kb in range(2):
            c = [[cvt_e4x2(p, e[4 * kb + j][h]) for h in range(2)] for j in range(4)]
            pa = [pack16(p, c[0][0], c[1][0]), pack16(p, c[0][1], c[1][1]), pack16(p, c[2][0], c[3][0]), pack16(p, c[2][1], c[3][1])]
            for d2 in range(2):
                r = ldm4(p, f"{vLane}+{st + kb * 32 + d2 * 16 * VT_STRIDE}")
                mma_e4(p, o[2 * d2], pa, (r[0], r[1]), o[2 * d2])
                mma_e4(p, o[2 * d2 + 1], pa, (r[2], r[3]), o[2 * d2 + 1])
        p.emit(f"add.u32 {b}, {b}, 1;")
    p.emit(f"bra {loopL};")
    p.emit(f"{doneL}:")
    p.emit("cp.async.wait_group 0;")
    # ---- padding correction, reciprocal, output (as the resident kernel)
    padding = p.sub32(P["padded"], P["tokens"])
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
    which, out = sys.argv[1], sys.argv[2]
    stages = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    name, text, shared = generate_normalize() if which == "normalize" else generate_attention(stages)
    open(out, "w").write(text)
    print(name, shared)
