//! Tiny PTX emitter used by the kernel generators.
//!
//! Registers are virtual (PTX renames them); we only need unique names. Loops that depend on compile-time
//! parameters are unrolled here so every shared memory offset and fragment index is a literal.
//!
//! The emitter is shared by reference (`&Ptx`, interior mutability), so nested emission reads like the
//! expressions it builds: `p.add32(&p.shl32(&a, 2), &b)`. Operands are anything `Display` (register names, integer
//! immediates, `"0x..."` literals, `"reg+offset"` addresses); arguments are evaluated left to right, inner calls
//! first, which is the emission order.

use std::cell::RefCell;
use std::fmt::Display;

/// A register kind's PTX declaration type and name prefix.
fn kind_info(kind: &str) -> (&'static str, &'static str) {
    match kind {
        "b32" => (".b32", "r"),
        "b64" => (".b64", "rd"),
        "pred" => (".pred", "p"),
        "f32" => (".f32", "f"),
        "b16" => (".b16", "h"),
        "f16x2" => (".b32", "hx"),
        "u32" => (".u32", "u"),
        "f16" => (".f16", "hf"),
        _ => panic!("unknown register kind {kind}"),
    }
}

#[derive(Default)]
struct State {
    lines: Vec<String>,
    body: Vec<String>,
    /// Registers allocated per kind, in the order each kind was first used (the declaration order).
    counter: Vec<(&'static str, usize)>,
    labels: usize,
    entry: Option<(String, usize)>,
}

/// One PTX module with a single entry point.
pub struct Ptx {
    s: RefCell<State>,
}

impl Default for Ptx {
    fn default() -> Self {
        Self::new()
    }
}

impl Ptx {
    /// A module for sm_89, PTX 8.7.
    pub fn new() -> Self {
        Self::with_target("sm_89", "8.7")
    }

    pub fn with_target(target: &str, version: &str) -> Self {
        let lines = vec![format!(".version {version}"), format!(".target {target}"), ".address_size 64".to_string(), String::new()];
        Self { s: RefCell::new(State { lines, ..State::default() }) }
    }

    // ---- registers

    /// A fresh register name of the given kind ("b32", "b64", "pred", "f32", "b16", "f16x2", "u32", "f16").
    pub fn reg(&self, kind: &'static str) -> String {
        let (_, prefix) = kind_info(kind);
        let mut s = self.s.borrow_mut();
        let n = match s.counter.iter_mut().find(|(k, _)| *k == kind) {
            Some((_, n)) => {
                *n += 1;
                *n - 1
            }
            None => {
                s.counter.push((kind, 1));
                0
            }
        };
        format!("%{prefix}{n}")
    }

    pub fn regs(&self, kind: &'static str, count: usize) -> Vec<String> {
        (0..count).map(|_| self.reg(kind)).collect()
    }

    pub fn label(&self, base: &str) -> String {
        let mut s = self.s.borrow_mut();
        s.labels += 1;
        format!("{base}_{}", s.labels)
    }

    // ---- emission

    pub fn emit(&self, text: impl Display) {
        self.s.borrow_mut().body.push(format!("  {text}"));
    }

    pub fn comment(&self, text: impl Display) {
        self.s.borrow_mut().body.push(format!("  // {text}"));
    }

    /// A line of the module header (before the entry point).
    pub fn header_line(&self, text: impl Display) {
        self.s.borrow_mut().lines.push(text.to_string());
    }

    /// The emitted body lines, without their indentation (the fast_divmod test interprets them).
    pub fn body_lines(&self) -> Vec<String> {
        self.s.borrow().body.iter().map(|l| l.trim().to_string()).collect()
    }

    /// Begin an entry point. params: (ptxtype, name). dynamic_shared: declare an extern (launch-sized) shared array
    /// instead of a static one (static arrays are limited to 48 KB).
    pub fn entry(&self, name: &str, params: &[(&str, &str)], shared_bytes: usize, max_threads: Option<u32>, max_regs: Option<u32>, dynamic_shared: bool) {
        let decl = params.iter().map(|(t, n)| format!(".param .{t} {n}")).collect::<Vec<_>>().join(", ");
        let mut header = format!(".visible .entry {name}({decl})");
        if let Some(t) = max_threads.filter(|&t| t != 0) {
            header += &format!("\n.maxntid {t}, 1, 1");
        }
        if let Some(r) = max_regs.filter(|&r| r != 0) {
            header += &format!("\n.maxnreg {r}");
        }
        let mut s = self.s.borrow_mut();
        s.entry = Some((header, if dynamic_shared { 0 } else { shared_bytes }));
        if dynamic_shared {
            // the launch's shared size (Kernels::ptxKernel parses it)
            s.lines.push(format!("// dynamic_shared {shared_bytes}"));
            s.lines.push(".extern .shared .align 128 .b8 smem[];".to_string());
        }
    }

    /// The module's text.
    pub fn finish(&self) -> String {
        let s = self.s.borrow();
        let (header, shared_bytes) = s.entry.as_ref().expect("finish() before entry()");
        let regdecl: Vec<String> = s
            .counter
            .iter()
            .map(|(kind, count)| {
                let (ptxtype, prefix) = kind_info(kind);
                format!("  .reg {ptxtype} %{prefix}<{count}>;")
            })
            .collect();
        let shared = if *shared_bytes != 0 { format!("  .shared .align 128 .b8 smem[{shared_bytes}];") } else { String::new() };
        format!("{}\n{header}\n{{\n{}\n{shared}\n{}\n}}\n", s.lines.join("\n"), regdecl.join("\n"), s.body.join("\n"))
    }

    // ---- common helpers

    pub fn load_param_u64(&self, name: &str) -> String {
        let r = self.reg("b64");
        self.emit(format!("ld.param.u64 {r}, [{name}];"));
        r
    }

    pub fn load_param_u32(&self, name: &str) -> String {
        let r = self.reg("b32");
        self.emit(format!("ld.param.u32 {r}, [{name}];"));
        r
    }

    pub fn special(&self, sreg: &str) -> String {
        let r = self.reg("b32");
        self.emit(format!("mov.u32 {r}, %{sreg};"));
        r
    }

    pub fn imm32(&self, value: impl Display) -> String {
        let r = self.reg("b32");
        self.emit(format!("mov.u32 {r}, {value};"));
        r
    }

    pub fn op32(&self, opcode: &str, a: impl Display, b: impl Display) -> String {
        let r = self.reg("b32");
        self.emit(format!("{opcode} {r}, {a}, {b};"));
        r
    }

    pub fn add32(&self, a: impl Display, b: impl Display) -> String {
        self.op32("add.u32", a, b)
    }
    pub fn sub32(&self, a: impl Display, b: impl Display) -> String {
        self.op32("sub.u32", a, b)
    }
    pub fn mul32(&self, a: impl Display, b: impl Display) -> String {
        self.op32("mul.lo.u32", a, b)
    }
    pub fn and32(&self, a: impl Display, b: impl Display) -> String {
        self.op32("and.b32", a, b)
    }
    pub fn or32(&self, a: impl Display, b: impl Display) -> String {
        self.op32("or.b32", a, b)
    }
    pub fn xor32(&self, a: impl Display, b: impl Display) -> String {
        self.op32("xor.b32", a, b)
    }
    pub fn shl32(&self, a: impl Display, b: impl Display) -> String {
        self.op32("shl.b32", a, b)
    }
    pub fn shr32(&self, a: impl Display, b: impl Display) -> String {
        self.op32("shr.u32", a, b)
    }

    pub fn mad32(&self, a: impl Display, b: impl Display, c: impl Display) -> String {
        let r = self.reg("b32");
        self.emit(format!("mad.lo.u32 {r}, {a}, {b}, {c};"));
        r
    }

    /// 64-bit: a(u32) * b(u32) + c(u64).
    pub fn mad_wide(&self, a: impl Display, b: impl Display, c64: impl Display) -> String {
        let r = self.reg("b64");
        self.emit(format!("mad.wide.u32 {r}, {a}, {b}, {c64};"));
        r
    }

    pub fn add64(&self, a: impl Display, b: impl Display) -> String {
        let r = self.reg("b64");
        self.emit(format!("add.u64 {r}, {a}, {b};"));
        r
    }

    pub fn widen(&self, a: impl Display) -> String {
        let r = self.reg("b64");
        self.emit(format!("cvt.u64.u32 {r}, {a};"));
        r
    }

    pub fn setp(&self, cmp: &str, a: impl Display, b: impl Display) -> String {
        let p = self.reg("pred");
        self.emit(format!("setp.{cmp} {p}, {a}, {b};"));
        p
    }

    pub fn selp32(&self, p: impl Display, a: impl Display, b: impl Display) -> String {
        let r = self.reg("b32");
        self.emit(format!("selp.b32 {r}, {a}, {b}, {p};"));
        r
    }

    /// 32-bit shared-space address of smem + offset.
    pub fn shared_addr(&self, byte_offset_reg: impl Display) -> String {
        let r = self.reg("b32");
        self.emit(format!("mov.u32 {r}, smem;"));
        self.add32(r, byte_offset_reg)
    }
}
