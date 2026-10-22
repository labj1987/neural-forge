"""Split-K reduction + publication (gemm_reduce.comp): f16 partials [S][rows][N] summed in slice order
(((p0 + p1) + p2) + p3 ...), then SiLU / E4M3 / f16 outputs. One thread per 8 columns.

python reduce_e4m3.py <splits> <flags> out.ptx        flags: SILU=2 E4=4 F16=8
"""
import sys
from ptxgen import Ptx
from swin import *

F_SILU, F_E4, F_F16 = 2, 4, 8


def generate(splits, flags):
    siluF, outE4, outF16 = bool(flags & F_SILU), bool(flags & F_E4), bool(flags & F_F16)
    name = f"reduce_e4m3_s{splits}_f{flags}"
    p = Ptx()
    params = [("u64", "pPartial"), ("u64", "pOut"), ("u64", "pOut16"), ("u32", "rows"), ("u32", "N"), ("u32", "splitStride"),
              ("u32", "outputStride"), ("u32", "outputColumnOffset")]
    p.entry(name, params, 0, 256)
    P = {n: (p.load_param_u64(n) if t == "u64" else p.load_param_u32(n)) for t, n in params}
    tid = p.special("tid.x"); ctaid = p.special("ctaid.x")
    group = p.add32(tid, p.shl32(ctaid, 8))          # 8-column group index
    groupsPerRow = p.shr32(P["N"], 3)
    row = p.reg("b32"); p.emit(f"div.u32 {row}, {group}, {groupsPerRow};")
    col = p.shl32(p.sub32(group, p.mul32(row, groupsPerRow)), 3)
    pOk = p.setp("lt.u32", row, P["rows"])
    endL = p.label("END")
    p.emit(f"@!{pOk} bra {endL};")
    base = p.add64(P["pPartial"], p.widen(p.shl32(p.add32(p.mul32(row, P["N"]), col), 1)))
    total = p.regs("b32", 4)
    p.emit(f"ld.global.cg.v4.b32 {{{total[0]}, {total[1]}, {total[2]}, {total[3]}}}, [{base}];")
    for s in range(1, splits):
        part = p.regs("b32", 4)
        addr = p.add64(base, p.widen(p.mul32(P["splitStride"], p.imm32(2 * s))))
        p.emit(f"ld.global.cg.v4.b32 {{{part[0]}, {part[1]}, {part[2]}, {part[3]}}}, [{addr}];")
        for i in range(4):
            total[i] = hadd2(p, total[i], part[i])
    if siluF:
        k = SiluConsts(p)
        for i in range(4): total[i] = silu(p, total[i], k)
    outIdx = p.add32(p.mul32(row, P["outputStride"]), p.add32(P["outputColumnOffset"], col))
    if outE4:
        codes = [cvt_e4x2(p, v) for v in total]
        w0 = pack16(p, codes[0], codes[1]); w1 = pack16(p, codes[2], codes[3])
        addr = p.add64(P["pOut"], p.widen(outIdx))
        p.emit(f"st.global.v2.b32 [{addr}], {{{w0}, {w1}}};")
    if outF16:
        addr = p.add64(P["pOut16"], p.widen(p.shl32(outIdx, 1)))
        p.emit(f"st.global.v4.b32 [{addr}], {{{total[0]}, {total[1]}, {total[2]}, {total[3]}}};")
    p.emit(f"{endL}:")
    p.emit("ret;")
    return name, p.finish()


if __name__ == "__main__":
    splits, flags, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    name, text = generate(splits, flags)
    open(out, "w").write(text)
    print(name)
