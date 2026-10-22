"""Tiny PTX emitter used by the dlss5vk kernel generators.

Registers are virtual (PTX renames them); we only need unique names. Loops that
depend on compile-time parameters are unrolled here in Python so every shared
memory offset and fragment index is a literal.
"""


class Ptx:
    def __init__(self, target="sm_89", version="8.7"):
        self.lines = [f".version {version}", f".target {target}", ".address_size 64", ""]
        self.body = []
        self.counter = {}
        self.labels = 0

    # ---- registers
    def reg(self, kind):
        """Return a fresh register name of the given PTX type ('b32', 'b64', 'pred', 'f32', 'b16', 'f16x2')."""
        prefix = {"b32": "r", "b64": "rd", "pred": "p", "f32": "f", "b16": "h", "f16x2": "hx", "u32": "u", "f16": "hf"}[kind]
        n = self.counter.get(kind, 0)
        self.counter[kind] = n + 1
        return f"%{prefix}{n}"

    def regs(self, kind, count):
        return [self.reg(kind) for _ in range(count)]

    def label(self, base="L"):
        self.labels += 1
        return f"{base}_{self.labels}"

    # ---- emission
    def emit(self, text):
        self.body.append("  " + text)

    def comment(self, text):
        self.body.append("  // " + text)

    def entry(self, name, params, shared_bytes, max_threads=None, max_regs=None, dynamic_shared=False):
        """Begin an entry point. params: list of (ptxtype, name). dynamic_shared: declare an extern (launch-sized)
        shared array instead of a static one (static arrays are limited to 48 KB)."""
        decl = ", ".join(f".param .{t} {n}" for t, n in params)
        header = f".visible .entry {name}({decl})"
        if max_threads:
            header += "\n.maxntid " + str(max_threads) + ", 1, 1"
        if max_regs:
            header += "\n.maxnreg " + str(max_regs)
        self._entry = (header, 0 if dynamic_shared else shared_bytes)
        if dynamic_shared:
            self.lines.append(f"// dynamic_shared {shared_bytes}")   # the launch's shared size (Kernels::ptxKernel parses it)
            self.lines.append(".extern .shared .align 128 .b8 smem[];")

    def finish(self):
        header, shared_bytes = self._entry
        regdecl = []
        for kind, count in self.counter.items():
            ptxtype = {"b32": ".b32", "b64": ".b64", "pred": ".pred", "f32": ".f32", "b16": ".b16", "f16x2": ".b32", "u32": ".u32", "f16": ".f16"}[kind]
            prefix = {"b32": "r", "b64": "rd", "pred": "p", "f32": "f", "b16": "h", "f16x2": "hx", "u32": "u", "f16": "hf"}[kind]
            regdecl.append(f"  .reg {ptxtype} %{prefix}<{count}>;")
        shared = f"  .shared .align 128 .b8 smem[{shared_bytes}];" if shared_bytes else ""
        text = "\n".join(self.lines) + "\n" + header + "\n{\n" + "\n".join(regdecl) + "\n" + shared + "\n" + "\n".join(self.body) + "\n}\n"
        return text

    # ---- common helpers
    def load_param_u64(self, name):
        r = self.reg("b64")
        self.emit(f"ld.param.u64 {r}, [{name}];")
        return r

    def load_param_u32(self, name):
        r = self.reg("b32")
        self.emit(f"ld.param.u32 {r}, [{name}];")
        return r

    def special(self, sreg):
        r = self.reg("b32")
        self.emit(f"mov.u32 {r}, %{sreg};")
        return r

    def imm32(self, value):
        r = self.reg("b32")
        self.emit(f"mov.u32 {r}, {value};")
        return r

    def op32(self, opcode, a, b):
        r = self.reg("b32")
        self.emit(f"{opcode} {r}, {a}, {b};")
        return r

    def add32(self, a, b): return self.op32("add.u32", a, b)
    def sub32(self, a, b): return self.op32("sub.u32", a, b)
    def mul32(self, a, b): return self.op32("mul.lo.u32", a, b)
    def and32(self, a, b): return self.op32("and.b32", a, b)
    def or32(self, a, b): return self.op32("or.b32", a, b)
    def xor32(self, a, b): return self.op32("xor.b32", a, b)
    def shl32(self, a, b): return self.op32("shl.b32", a, b)
    def shr32(self, a, b): return self.op32("shr.u32", a, b)

    def mad32(self, a, b, c):
        r = self.reg("b32")
        self.emit(f"mad.lo.u32 {r}, {a}, {b}, {c};")
        return r

    def mad_wide(self, a, b, c64):
        """64-bit: a(u32) * b(u32) + c(u64)."""
        r = self.reg("b64")
        self.emit(f"mad.wide.u32 {r}, {a}, {b}, {c64};")
        return r

    def add64(self, a, b):
        r = self.reg("b64")
        self.emit(f"add.u64 {r}, {a}, {b};")
        return r

    def widen(self, a):
        r = self.reg("b64")
        self.emit(f"cvt.u64.u32 {r}, {a};")
        return r

    def setp(self, cmp, a, b):
        p = self.reg("pred")
        self.emit(f"setp.{cmp} {p}, {a}, {b};")
        return p

    def selp32(self, p, a, b):
        r = self.reg("b32")
        self.emit(f"selp.b32 {r}, {a}, {b}, {p};")
        return r

    def shared_addr(self, byte_offset_reg):
        """32-bit shared-space address of smem + offset."""
        r = self.reg("b32")
        self.emit(f"mov.u32 {r}, smem;")
        return self.add32(r, byte_offset_reg)
