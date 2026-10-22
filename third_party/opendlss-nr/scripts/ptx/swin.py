"""Shared PTX building blocks for the DLSS-NR Swin kernels (fragment-level, exact arithmetic).

Fragment conventions (mma.sync m16n8k32 e4m3 -> f16, and m16n8k16 f16):
  lane l: g = l >> 2 (row), t = l & 3.
  C/D tile (16 x 8): [c0, c1] f16x2 regs; c0 = (row g, cols 2t, 2t+1), c1 = (row g+8, same cols).
  A (16 x 32 e4m3): [a0..a3] b32; a0 = (row g, k 4t..4t+3), a1 = (row g+8, k 4t..), a2 = (row g, 16+4t..), a3 = (row g+8, 16+4t..).
  B (32 x 8 e4m3): [b0, b1]; b0 = (k 4t..4t+3, col g), b1 = (k 16+4t.., col g).
  A f16 (16 x 16): a0 = (row g, k 2t..2t+1), a1 = (row g+8, ..), a2 = (row g, 8+2t..), a3 = (row g+8, 8+2t..).
  B f16 (16 x 8): b0 = (k 2t..2t+1, col g), b1 = (k 8+2t.., col g).
A 16 x 32 f16 accumulator (four C tiles j = 0..3) holds in lane (g, t) the columns 8j + 2t, 8j + 2t + 1
of rows g and g + 8: the coopmat "element 8h + 2j + i" layout of the GLSL kernels.
"""


def mma_e4(p, d, a, b, c):
    p.emit(f"mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {{{d[0]}, {d[1]}}}, {{{a[0]}, {a[1]}, {a[2]}, {a[3]}}}, "
           f"{{{b[0]}, {b[1]}}}, {{{c[0]}, {c[1]}}};")


def mma_f16(p, d, a, b, c):
    p.emit(f"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {{{d[0]}, {d[1]}}}, {{{a[0]}, {a[1]}, {a[2]}, {a[3]}}}, "
           f"{{{b[0]}, {b[1]}}}, {{{c[0]}, {c[1]}}};")


def ldm4(p, addr):
    r = p.regs("b32", 4)
    p.emit(f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, [{addr}];")
    return r


def ldm2(p, addr):
    r = p.regs("b32", 2)
    p.emit(f"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{r[0]}, {r[1]}}}, [{addr}];")
    return r


def zero_tile(p):
    d = p.regs("b32", 2)
    p.emit(f"mov.b32 {d[0]}, 0;"); p.emit(f"mov.b32 {d[1]}, 0;")
    return d


def imm(p, value):
    r = p.reg("b32"); p.emit(f"mov.b32 {r}, {value};"); return r


def f16x2_op(p, op, a, b):
    r = p.reg("b32"); p.emit(f"{op}.f16x2 {r}, {a}, {b};"); return r


def hadd2(p, a, b): return f16x2_op(p, "add.rn", a, b)
def hmul2(p, a, b): return f16x2_op(p, "mul.rn", a, b)
def hmax2(p, a, b): return f16x2_op(p, "max", a, b)
def hmin2(p, a, b): return f16x2_op(p, "min", a, b)


def hfma2(p, a, b, c):
    r = p.reg("b32"); p.emit(f"fma.rn.f16x2 {r}, {a}, {b}, {c};"); return r


def hadd(p, a, b):
    r = p.reg("f16"); p.emit(f"add.rn.f16 {r}, {a}, {b};"); return r


def unpack16(p, r):
    lo = p.reg("f16"); hi = p.reg("f16")
    p.emit(f"mov.b32 {{{lo}, {hi}}}, {r};")
    return lo, hi


def pack16(p, lo, hi):
    r = p.reg("b32"); p.emit(f"mov.b32 {r}, {{{lo}, {hi}}};"); return r


def cvt_e4x2(p, v):
    """f16x2 -> two E4M3 codes (saturating, NaN -> 0x7f) in a .b16 register."""
    h = p.reg("b16"); p.emit(f"cvt.rn.satfinite.e4m3x2.f16x2 {h}, {v};"); return h


def e4x2_to_f16x2(p, h):
    """Two E4M3 codes (.b16) -> f16x2 (exact widening; NaN codes -> NaN)."""
    r = p.reg("b32"); p.emit(f"cvt.rn.f16x2.e4m3x2 {r}, {h};"); return r


def b16_to_b32(p, h):
    r = p.reg("b32"); p.emit(f"cvt.u32.u16 {r}, {h};"); return r


def b32_to_b16(p, r):
    h = p.reg("b16"); p.emit(f"cvt.u16.u32 {h}, {r};"); return h


def nanzero(p, v, zero):
    """f16x2: NaN lanes -> 0, everything else unchanged (max/min return the non-NaN operand)."""
    pos = hmax2(p, v, zero)
    neg = hmin2(p, v, zero)
    return hadd2(p, pos, neg)


def shfl_idx(p, v, src):
    r = p.reg("b32"); p.emit(f"shfl.sync.idx.b32 {r}, {v}, {src}, 0x1f, 0xffffffff;"); return r


def shfl_xor(p, v, mask):
    r = p.reg("b32"); p.emit(f"shfl.sync.bfly.b32 {r}, {v}, {mask}, 0x1f, 0xffffffff;"); return r


def prmt(p, a, b, sel):
    r = p.reg("b32"); p.emit(f"prmt.b32 {r}, {a}, {b}, {sel};"); return r


class SiluConsts:
    def __init__(self, p):
        self.c4 = imm(p, "0x44004400")      # 4.0
        self.cm4 = imm(p, "0xC400C400")     # -4.0
        self.cA = imm(p, "0xAB28AB28")      # -0.055908203125
        self.cB = imm(p, "0x37283728")      # 0.447265625
        self.cC = imm(p, "0x3B283B28")      # 0.89453125


def silu(p, v, k):
    """siluPair: clamp to [-4, 4], inner = fma(-0.0559, |bounded|, 0.4473), poly = fma(bounded, inner, 0.8945), v * poly."""
    bounded = hmax2(p, v, k.cm4)
    bounded = hmin2(p, bounded, k.c4)
    ab = p.reg("b32"); p.emit(f"abs.f16x2 {ab}, {bounded};")
    inner = hfma2(p, k.cA, ab, k.cB)
    poly = hfma2(p, bounded, inner, k.cC)
    return hmul2(p, v, poly)


class LaneInfo:
    """Per-lane constants for the fragment shuffles."""
    def __init__(self, p, lane):
        self.lane = lane
        self.g = p.shr32(lane, 2)
        self.t = p.and32(lane, 3)
        quad = p.and32(lane, 28)
        self.quad = quad
        # acc_to_a sources: lane (g, 2 (t & 1)) and (g, 2 (t & 1) + 1); selector picks the low pairs (t >> 1 == 0) or the high pairs
        tl = p.and32(lane, 1)
        self.srcA = p.add32(quad, p.shl32(tl, 1))
        self.srcB = p.add32(self.srcA, 1)
        th = p.and32(p.shr32(lane, 1), 1)
        pHi = p.setp("ne.u32", th, 0)
        self.sel = p.selp32(pHi, imm(p, "0x7632"), imm(p, "0x5410"))
        self.quadLanes = [quad, p.add32(quad, 1), p.add32(quad, 2), p.add32(quad, 3)]


def acc_to_a(p, tiles, li, zero=None):
    """Four C tiles (16 x 32 f16) -> the E4M3 A fragment of the same 16 x 32 (NaN -> 0 when `zero` is given)."""
    q = []
    for j in range(4):
        row = []
        for h in range(2):
            v = tiles[j][h]
            if zero is not None: v = nanzero(p, v, zero)
            row.append(cvt_e4x2(p, v))
        q.append(row)
    a = []
    for pair in range(2):                # k 0..15 (tiles 0, 1) then k 16..31 (tiles 2, 3)
        for h in range(2):
            w = pack16(p, q[2 * pair][h], q[2 * pair + 1][h])
            s0 = shfl_idx(p, w, li.srcA); s1 = shfl_idx(p, w, li.srcB)
            a.append(prmt(p, s0, s1, li.sel))
    # order: a0 (row g, k 0..15), a1 (row g+8, k 0..15), a2 (row g, k 16..31), a3 (row g+8, k 16..31)
    return [a[0], a[1], a[2], a[3]]


def acc_f16_to_a(tiles, step):
    """16 x 32 f16 accumulator -> f16 A fragment of k16 step `step` (cols 16 step ..): a register renaming."""
    return [tiles[2 * step][0], tiles[2 * step][1], tiles[2 * step + 1][0], tiles[2 * step + 1][1]]


def normalize(p, tiles, li, zero, scale2=None, guarded=True):
    """Cosine normalization of a 16 x 32 accumulator (per row half), reference association order:
    r[c] = fma(v[c], v[c], v[c+16]^2) (c < 16), s8 = r[c] + r[c+8], s4 (lane ^ 2), s2 (lane ^ 1), s1 = s2.x + s2.y,
    norm = f16(rsqrt(f32(s1))); v * norm (* scale); NaN -> 0.
    guarded: NaN can only come from 0 * inf (a zero row: s1 == 0), so the NaN -> 0 pass runs under a warp-uniform
    branch taken only when some lane's row sum is zero (image borders)."""
    out = [[None, None] for _ in range(4)]
    anyZero = p.reg("pred"); p.emit(f"setp.ne.u32 {anyZero}, 0, 0;")   # false
    for h in range(2):
        p0, p1, p2, p3 = (tiles[j][h] for j in range(4))
        hs0 = hmul2(p, p2, p2)
        r0 = hfma2(p, p0, p0, hs0)
        hs1 = hmul2(p, p3, p3)
        r1 = hfma2(p, p1, p1, hs1)
        s8 = hadd2(p, r0, r1)
        s8o = shfl_xor(p, s8, 2)
        s4 = hadd2(p, s8, s8o)
        s4o = shfl_xor(p, s4, 1)
        s2 = hadd2(p, s4, s4o)
        lo, hi = unpack16(p, s2)
        s1 = hadd(p, lo, hi)
        f = p.reg("f32"); p.emit(f"cvt.f32.f16 {f}, {s1};")
        if guarded:
            pz = p.reg("pred"); p.emit(f"setp.eq.f32 {pz}, {f}, 0f00000000;")
            p.emit(f"or.pred {anyZero}, {anyZero}, {pz};")
        rs = p.reg("f32"); p.emit(f"rsqrt.approx.f32 {rs}, {f};")
        norm = p.reg("f16"); p.emit(f"cvt.rn.f16.f32 {norm}, {rs};")
        norm2 = pack16(p, norm, norm)
        for j in range(4):
            n = hmul2(p, tiles[j][h], norm2)
            if scale2 is not None: n = hmul2(p, n, scale2)
            out[j][h] = n if guarded else nanzero(p, n, zero)
    if guarded:
        vote = p.reg("pred"); p.emit(f"vote.sync.any.pred {vote}, {anyZero}, 0xffffffff;")
        skip = p.label("NZ")
        p.emit(f"@!{vote} bra {skip};")
        for j in range(4):
            for h in range(2):
                clean = nanzero(p, out[j][h], zero)
                p.emit(f"mov.b32 {out[j][h]}, {clean};")
        p.emit(f"{skip}:")
    return out


class ExpConsts:
    def __init__(self, p):
        self.a = p.reg("f32"); p.emit(f"mov.f32 {self.a}, 0f3D380000;")    # 0.044921875
        self.b = p.reg("f32"); p.emit(f"mov.f32 {self.b}, 0f3FA68000;")    # 1.30078125
        self.lo = imm(p, "0x3C203C20")                                     # 1.03125
        self.hi = imm(p, "0x3E473E47")                                     # 1.5693359375
        self.m = imm(p, "0xffe0ffe0")
        self.x = imm(p, "0x80008000")


def exp_weight(p, s, k):
    """the exponential approximation on a packed pair: f32 affine, f16 clamp, (bits << 5 & 0xffe0) ^ 0x8000 per half."""
    lo, hi = unpack16(p, s)
    flo = p.reg("f32"); p.emit(f"cvt.f32.f16 {flo}, {lo};")
    fhi = p.reg("f32"); p.emit(f"cvt.f32.f16 {fhi}, {hi};")
    p.emit(f"fma.rn.f32 {flo}, {flo}, {k.a}, {k.b};")
    p.emit(f"fma.rn.f32 {fhi}, {fhi}, {k.a}, {k.b};")
    h = p.reg("b32"); p.emit(f"cvt.rn.f16x2.f32 {h}, {fhi}, {flo};")
    h = hmax2(p, h, k.lo)
    h = hmin2(p, h, k.hi)
    r = p.reg("b32"); p.emit(f"shl.b32 {r}, {h}, 5;")
    d = p.reg("b32"); p.emit(f"lop3.b32 {d}, {r}, {k.m}, {k.x}, 0x6A;")   # (a & b) ^ c
    return d


def softmax(p, s_tiles, li, k):
    """16 x 64 score accumulator (8 C tiles) -> attention weights in the same layout (the specified softmax reduction order)."""
    w = [[None, None] for _ in range(8)]
    for h in range(2):
        e = [exp_weight(p, s_tiles[m][h], k) for m in range(8)]
        bp = [hadd2(p, e[2 * j], e[2 * j + 1]) for j in range(4)]
        t01 = hadd2(p, bp[0], bp[1])
        t012 = hadd2(p, t01, bp[2])
        tp = hadd2(p, t012, bp[3])
        tq = [shfl_idx(p, tp, src) for src in li.quadLanes]
        parts = [unpack16(p, t) for t in tq]
        e01 = hadd(p, parts[0][0], parts[1][0]); e012 = hadd(p, e01, parts[2][0]); even = hadd(p, e012, parts[3][0])
        o01 = hadd(p, parts[0][1], parts[1][1]); o012 = hadd(p, o01, parts[2][1]); odd = hadd(p, o012, parts[3][1])
        total = hadd(p, even, odd)
        f = p.reg("f32"); p.emit(f"cvt.f32.f16 {f}, {total};")
        rc = p.reg("f32"); p.emit(f"rcp.rn.f32 {rc}, {f};")
        rec = p.reg("f16"); p.emit(f"cvt.rn.f16.f32 {rec}, {rc};")
        rec2 = pack16(p, rec, rec)
        for m in range(8):
            w[m][h] = hmul2(p, e[m], rec2)
    return w


def tiled_token_regs(p, token):
    """natural token (0..63) -> physical (tiled) index: (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3)."""
    x = p.and32(token, 7); y = p.shr32(token, 3)
    a = p.shl32(p.shr32(y, 2), 5)
    b = p.shl32(p.shr32(x, 2), 4)
    c = p.shl32(p.and32(y, 3), 2)
    d = p.and32(x, 3)
    return p.add32(p.add32(a, b), p.add32(c, d))


def fast_divmod(p, n, d, dRcp):
    """Exact unsigned division of n (< 2^24) by d (> 0) via the f32 reciprocal dRcp = 1 / d (rcp.rn), corrected by
    one remainder check each way: returns (q, r)."""
    f = p.reg("f32"); p.emit(f"cvt.rn.f32.u32 {f}, {n};")
    qf = p.reg("f32"); p.emit(f"mul.rn.f32 {qf}, {f}, {dRcp};")
    q = p.reg("b32"); p.emit(f"cvt.rzi.u32.f32 {q}, {qf};")
    r = p.reg("b32"); p.emit(f"mul.lo.u32 {r}, {q}, {d};"); p.emit(f"sub.u32 {r}, {n}, {r};")
    pNeg = p.setp("lt.s32", r, 0)            # r < 0 as signed: q one too large
    pBig = p.setp("ge.u32", r, d)            # r >= d: q one too small (only when r >= 0)
    p.emit(f"@{pNeg} sub.u32 {q}, {q}, 1;"); p.emit(f"@{pNeg} add.u32 {r}, {r}, {d};")
    pBig2 = p.reg("pred"); p.emit(f"and.pred {pBig2}, {pBig}, !{pNeg};")
    p.emit(f"@{pBig2} add.u32 {q}, {q}, 1;"); p.emit(f"@{pBig2} sub.u32 {r}, {r}, {d};")
    return q, r


def rcp_f32(p, d):
    f = p.reg("f32"); p.emit(f"cvt.rn.f32.u32 {f}, {d};")
    r = p.reg("f32"); p.emit(f"rcp.rn.f32 {r}, {f};")
    return r


def acc_to_a_smem(p, tiles, cBase, aLane, zero=None):
    """Four C tiles (16 x 32 f16) -> the E4M3 A fragment through a per-warp [16][32 B] chunk-swizzled staging tile:
    cBase[c] = tile + g * 32 + ((c ^ ((g >> 2) & 1)) * 16) + 2 t for chunk c (the C-layout store addresses),
    aLane = tile + aRowL * 32 + (aChunkL ^ ((aRowL >> 2) & 1)) * 16 (the ldmatrix address). NaN -> 0 when `zero`."""
    p.emit("bar.warp.sync 0xffffffff;")   # the previous fragment load from the tile has completed
    for j in range(4):
        for h in range(2):
            v = tiles[j][h]
            if zero is not None: v = nanzero(p, v, zero)
            hc = cvt_e4x2(p, v)
            p.emit(f"st.shared.b16 [{cBase[j >> 1]}+{8 * (j & 1) + 256 * h}], {hc};")
    p.emit("bar.warp.sync 0xffffffff;")
    return ldm4(p, aLane)


# ---- cross-kernel dependency counters (barrier-free chaining of consecutive launches)
# Counters are u32 in a per-frame zeroed buffer. Producers: fence.acq_rel.gpu + bar.sync + thread 0 red.release.gpu.add.
# Consumers: every warp polls its counters (lanes in parallel) with ld.acquire.gpu until all reach the expected count.

SLEEP_MAX = 256   # ns: polling backoff ceiling
WAIT_LIMIT_NS = 1_000_000_000   # a wait this long is a failed scheduling assumption (docs/execution.md), not a slow frame


def sync_wait(p, base64, first, last, expected, lane, guard=None, warp=None, *, error64):
    """Spin until counters [first .. last] (at most 32) at base64 (u64 byte address of counter 0) are all >= expected
    (a register, or a callable emitting the expected count for the counter index register). `guard` (pred) skips the
    wait when false. With `warp` given, only warp 0 polls (exponential backoff 128 .. 256 ns) and a bar.sync
    releases the workgroup; otherwise every warp polls for itself.
    A wait that lasts WAIT_LIMIT_NS gives up instead of hanging: it sets bit 31 of the counters it is stuck on (which
    releases every other waiter on them), adds one to the u32 at error64 and, if it is the first, records the low
    32 bits of a stuck counter's address at error64 + 4. Every other wait still spinning then gives up too, so the
    frame completes one limit later and the host reports it."""
    skip = p.label("NOWAIT")
    if guard is not None: p.emit(f"@!{guard} bra {skip};")
    if warp is not None:
        pW0 = p.setp("ne.u32", warp, 0)
        joinL = p.label("WJOIN")
        p.emit(f"@{pW0} bra {joinL};")
    idx = p.add32(first, lane)
    pMine = p.setp("le.u32", idx, last)
    if callable(expected): expected = expected(idx)
    addr = p.add64(base64, p.widen(p.shl32(idx, 2)))
    sleep = p.reg("b32"); p.emit(f"mov.u32 {sleep}, 128;")
    start = p.reg("b64"); p.emit(f"mov.u64 {start}, %globaltimer;")
    loop = p.label("WAIT")
    p.emit(f"{loop}:")
    v = p.reg("b32"); p.emit(f"mov.u32 {v}, 0xffffffff;")
    p.emit(f"@{pMine} ld.acquire.gpu.global.u32 {v}, [{addr}];")
    pOk = p.setp("ge.u32", v, expected)
    pAll = p.reg("pred"); p.emit(f"vote.sync.all.pred {pAll}, {pOk}, 0xffffffff;")
    done = p.label("WAITED")
    p.emit(f"@{pAll} bra {done};")
    p.emit(f"nanosleep.u32 {sleep};")
    p.emit(f"shl.b32 {sleep}, {sleep}, 1;")
    p.emit(f"min.u32 {sleep}, {sleep}, {SLEEP_MAX};")
    elapsed = p.reg("b64"); p.emit(f"mov.u64 {elapsed}, %globaltimer;"); p.emit(f"sub.u64 {elapsed}, {elapsed}, {start};")
    pLate = p.setp("gt.u64", elapsed, WAIT_LIMIT_NS)
    gaveUp = p.reg("b32"); p.emit(f"ld.relaxed.gpu.global.u32 {gaveUp}, [{error64}];")   # once one wait gave up, all do
    pAbandon = p.setp("ne.u32", gaveUp, 0)
    pGiveUp = p.reg("pred"); p.emit(f"or.pred {pGiveUp}, {pLate}, {pAbandon};")
    p.emit(f"@!{pGiveUp} bra {loop};")
    pStuck = p.reg("pred"); p.emit(f"and.pred {pStuck}, {pMine}, !{pOk};")
    p.emit(f"@{pStuck} red.relaxed.gpu.global.or.b32 [{addr}], 0x80000000;")
    pLane0 = p.setp("eq.u32", lane, 0)
    p.emit(f"@{pLane0} red.relaxed.gpu.global.add.u32 [{error64}], 1;")
    addrLow = p.reg("b32"); p.emit(f"cvt.u32.u64 {addrLow}, {addr};")
    first_ = p.reg("b32"); p.emit(f"@{pStuck} atom.relaxed.gpu.global.cas.b32 {first_}, [{error64}+4], 0, {addrLow};")
    p.emit(f"{done}:")
    if warp is not None:
        p.emit(f"{joinL}:")
        p.emit("bar.sync 0;")
    p.emit(f"{skip}:")


def sync_signal(p, addrs, tid, guard=None):
    """After this workgroup's global stores: release them and increment the counters at addrs (u64 regs) by one.
    All threads must reach this point (bar.sync inside)."""
    p.emit("fence.acq_rel.gpu;")
    p.emit("bar.sync 0;")
    pT0 = p.setp("eq.u32", tid, 0)
    if guard is not None:
        pG = p.reg("pred"); p.emit(f"and.pred {pG}, {pT0}, {guard};"); pT0 = pG
    for a in addrs:
        p.emit(f"@{pT0} red.release.gpu.global.add.u32 [{a}], 1;")


def band_range(p, rowFirst, rowLast, width):
    """Pixel-row bands (8 rows) touched by token rows [rowFirst, rowLast]: (b0, b1)."""
    y0 = p.reg("b32"); p.emit(f"div.u32 {y0}, {rowFirst}, {width};")
    y1 = p.reg("b32"); p.emit(f"div.u32 {y1}, {rowLast}, {width};")
    return p.shr32(y0, 3), p.shr32(y1, 3)


def band_expected(p, band, width, rows, mul=None, groupRows=None):
    """Number of producer row groups (of `groupRows` token rows each; a register, 64 when None) intersecting pixel-row
    band `band` (rows [8 band, 8 band + 8) of `width` tokens), times `mul` (the producer's signals per row group,
    e.g. its column-group count). The producer signals every band its rows touch once per workgroup, so this count
    must be exactly its workgroup size: a smaller group size here would release the consumer early (the C=256
    FFN runs 48-row workgroups)."""
    t0 = p.mul32(p.shl32(band, 3), width)
    t1 = p.sub32(p.add32(t0, p.shl32(width, 3)), 1)
    rowsMinus1 = p.sub32(rows, 1)
    pClamp = p.setp("gt.u32", t1, rowsMinus1)
    t1c = p.selp32(pClamp, rowsMinus1, t1)
    if groupRows is None:
        g0, g1 = p.shr32(t0, 6), p.shr32(t1c, 6)
    else:
        g0 = p.reg("b32"); p.emit(f"div.u32 {g0}, {t0}, {groupRows};")
        g1 = p.reg("b32"); p.emit(f"div.u32 {g1}, {t1c}, {groupRows};")
    n = p.add32(p.sub32(g1, g0), 1)
    return n if mul is None else p.mul32(n, mul)


# ---- ViT variants of the same publication rules (different constants, different orderings)
def vit_exp(p, s, k):
    """expWeightPairVit on a packed pair: fma.rn.f16x2 (0.0895, 1.709), clamp [1.4395, 1.9775], ((bits << 4) + 0x4000) & 0xffff per half."""
    a = hfma2(p, s, k.c1, k.c2)
    a = hmax2(p, a, k.lo)
    a = hmin2(p, a, k.hi)
    lo = p.reg("b32"); p.emit(f"shl.b32 {lo}, {a}, 4;"); p.emit(f"add.u32 {lo}, {lo}, 0x4000;"); p.emit(f"and.b32 {lo}, {lo}, 0xffff;")
    hi = p.reg("b32"); p.emit(f"shr.u32 {hi}, {a}, 12;"); p.emit(f"and.b32 {hi}, {hi}, 0xfff0;"); p.emit(f"add.u32 {hi}, {hi}, 0x4000;")
    p.emit(f"and.b32 {hi}, {hi}, 0xffff;"); p.emit(f"shl.b32 {hi}, {hi}, 16;")
    r = p.reg("b32"); p.emit(f"or.b32 {r}, {lo}, {hi};")
    return r


class VitExpConsts:
    def __init__(self, p):
        self.c1 = imm(p, "0x2DBB2DBB")   # f16(0.08953946828842163)
        self.c2 = imm(p, "0x3ED63ED6")   # f16(1.7093614339828491)
        self.lo = imm(p, "0x3DC23DC2")   # 1.439453125
        self.hi = imm(p, "0x3FE93FE9")   # 1.9775390625


def vit_norm(p, w):
    """inverseNormVit on 32 halves (16 packed words): r[m] = f16(f32(v[m])^2 + f32(f16(v[m+8]^2))) per component,
    s8/s4/s2 in f16x2, s1 = s2.x + s2.y, f16(rsqrt(f32(s1))) broadcast to a pair."""
    r = []
    for m in range(8):
        hs = hmul2(p, w[m + 8], w[m + 8])
        lo, hi = unpack16(p, w[m])
        flo = p.reg("f32"); p.emit(f"cvt.f32.f16 {flo}, {lo};")
        fhi = p.reg("f32"); p.emit(f"cvt.f32.f16 {fhi}, {hi};")
        p.emit(f"mul.rn.f32 {flo}, {flo}, {flo};"); p.emit(f"mul.rn.f32 {fhi}, {fhi}, {fhi};")
        hlo, hhi = unpack16(p, hs)
        glo = p.reg("f32"); p.emit(f"cvt.f32.f16 {glo}, {hlo};")
        ghi = p.reg("f32"); p.emit(f"cvt.f32.f16 {ghi}, {hhi};")
        p.emit(f"add.rn.f32 {flo}, {flo}, {glo};"); p.emit(f"add.rn.f32 {fhi}, {fhi}, {ghi};")
        d = p.reg("b32"); p.emit(f"cvt.rn.f16x2.f32 {d}, {fhi}, {flo};")
        r.append(d)
    s8 = [hadd2(p, r[m], r[m + 4]) for m in range(4)]
    s4a = hadd2(p, s8[0], s8[2]); s4b = hadd2(p, s8[1], s8[3])
    s2 = hadd2(p, s4a, s4b)
    lo, hi = unpack16(p, s2)
    s1 = hadd(p, lo, hi)
    f = p.reg("f32"); p.emit(f"cvt.f32.f16 {f}, {s1};")
    rs = p.reg("f32"); p.emit(f"rsqrt.approx.f32 {rs}, {f};")
    n = p.reg("f16"); p.emit(f"cvt.rn.f16.f32 {n}, {rs};")
    return pack16(p, n, n)


def vit_quantize(p, w, norm2, query, hs2, learned2, zero):
    """scaleQuantizeVit: 32 halves times the norm (queries: then f16(sqrt 32), then the learned scale), E4 codes (8 words).
    A zero row has norm = inf and 0 * inf = NaN: the GLSL conversion yields 0 for it (nanzero before the cvt)."""
    codes = []
    for i in range(8):
        a = hmul2(p, w[2 * i], norm2); b = hmul2(p, w[2 * i + 1], norm2)
        if query:
            a = hmul2(p, hmul2(p, a, hs2), learned2); b = hmul2(p, hmul2(p, b, hs2), learned2)
        a = nanzero(p, a, zero); b = nanzero(p, b, zero)
        codes.append(pack16(p, cvt_e4x2(p, a), cvt_e4x2(p, b)))
    return codes
