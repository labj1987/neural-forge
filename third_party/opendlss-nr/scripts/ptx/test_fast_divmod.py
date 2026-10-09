"""Regression test for swin.fast_divmod: interpret the PTX it emits, over every n < 2^24, and compare with exact
unsigned division. The emitted instructions are what is executed (so an impossible `setp` is caught), with the
IEEE semantics PTX gives them. Needs numpy.

  python scripts/ptx/test_fast_divmod.py [max divisor, default 32]
"""
import os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptxgen, swin  # noqa: E402

INSTR = re.compile(r"^(?:@(!?)(%\w+)\s+)?([\w.]+)\s+(.*);$")


def emit():
    p = ptxgen.Ptx()
    n, d = p.reg("b32"), p.reg("b32")
    q, r = swin.fast_divmod(p, n, d, swin.rcp_f32(p, d))
    return [line.strip() for line in p.body], n, d, q, r


def execute(lines, regs):
    def val(token):
        negate = token.startswith("!")
        token = token.lstrip("!")
        v = regs[token] if token.startswith("%") else np.uint32(int(token, 0) & 0xffffffff)
        return ~v if negate else v

    s32 = lambda v: np.asarray(v, dtype=np.uint32).view(np.int32)
    for line in lines:
        negate, guard, op, args = INSTR.match(line).groups()
        dst, *src = [a.strip() for a in args.split(",")]
        a = [val(s) for s in src]
        if op == "cvt.rn.f32.u32": out = a[0].astype(np.float32)
        elif op == "rcp.rn.f32": out = np.float32(1) / a[0]
        elif op == "mul.rn.f32": out = a[0] * a[1]
        elif op == "cvt.rzi.u32.f32": out = np.clip(np.trunc(a[0].astype(np.float64)), 0, 2**32 - 1).astype(np.uint32)
        elif op == "mul.lo.u32": out = (a[0].astype(np.uint64) * a[1] & 0xffffffff).astype(np.uint32)
        elif op == "sub.u32": out = a[0] - a[1]
        elif op == "add.u32": out = a[0] + a[1]
        elif op == "and.pred": out = a[0] & a[1]
        elif op.startswith("setp."):
            _, cmp, kind = op.split(".")
            x, y = (s32(a[0]), s32(a[1])) if kind == "s32" else (a[0], a[1])
            out = {"gt": x > y, "ge": x >= y, "lt": x < y, "le": x <= y, "eq": x == y, "ne": x != y}[cmp]
        else: raise SystemExit(f"test_fast_divmod: {op} is outside the interpreted subset")
        if guard: out = np.where(~regs[guard] if negate else regs[guard], out, regs[dst])
        regs[dst] = out
    return regs


def main():
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    lines, nr, dr, qr, rr = emit()
    n = np.arange(1 << 24, dtype=np.uint32)
    failures = 0
    for d in list(range(1, top + 1)):
        regs = execute(lines, {nr: n, dr: np.full(n.shape, d, dtype=np.uint32)})
        wrong = (regs[qr] != n // d) | (regs[rr] != n % d)
        if wrong.any():
            first = int(np.argmax(wrong))
            print(f"d = {d}: {int(wrong.sum())} wrong, first n = {first} -> ({regs[qr][first]}, {regs[rr][first]})")
            failures += 1
    print(f"fast_divmod: every n < 2^24, d = 1..{top}: " + ("FAIL" if failures else "exact"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
