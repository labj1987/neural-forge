//! Regression test for swin::fast_divmod: interpret the PTX it emits, over every n < 2^24, and compare with exact
//! unsigned division, for d = 1..=32. The emitted instructions are what is executed (so an impossible `setp` is
//! caught), with the IEEE semantics PTX gives them.

use std::collections::HashMap;

use opendlss_nr_ptx::ptxgen::Ptx;
use opendlss_nr_ptx::swin;

/// An operand: a register slot (negated for `!%p`), or an immediate.
#[derive(Clone, Copy)]
enum Operand {
    Reg(usize, bool),
    Imm(u32),
}

/// The interpreted subset of PTX.
#[derive(Clone, Copy)]
enum Op {
    CvtF32U32,
    RcpF32,
    MulF32,
    CvtRziU32F32,
    MulLoU32,
    SubU32,
    AddU32,
    AndPred,
    Setp(std::cmp::Ordering, bool, bool),
}

impl Op {
    /// The op, and whether its result is negated (setp.ne is a negated setp.eq). `Setp`: the ordering that makes it
    /// true, whether equal does too, and whether the comparison is signed.
    fn parse(op: &str) -> (Op, bool) {
        let simple = match op {
            "cvt.rn.f32.u32" => Some(Op::CvtF32U32),
            "rcp.rn.f32" => Some(Op::RcpF32),
            "mul.rn.f32" => Some(Op::MulF32),
            "cvt.rzi.u32.f32" => Some(Op::CvtRziU32F32),
            "mul.lo.u32" => Some(Op::MulLoU32),
            "sub.u32" => Some(Op::SubU32),
            "add.u32" => Some(Op::AddU32),
            "and.pred" => Some(Op::AndPred),
            _ => None,
        };
        if let Some(op) = simple {
            return (op, false);
        }
        let parts: Vec<&str> = op.split('.').collect();
        let (cmp, signed) = match parts.as_slice() {
            ["setp", cmp, kind] => (*cmp, *kind == "s32"),
            _ => panic!("fast_divmod test: {op} is outside the interpreted subset"),
        };
        use std::cmp::Ordering::*;
        match cmp {
            "gt" => (Op::Setp(Greater, false, signed), false),
            "ge" => (Op::Setp(Greater, true, signed), false),
            "lt" => (Op::Setp(Less, false, signed), false),
            "le" => (Op::Setp(Less, true, signed), false),
            "eq" => (Op::Setp(Equal, true, signed), false),
            "ne" => (Op::Setp(Equal, true, signed), true),
            _ => panic!("fast_divmod test: {op} is outside the interpreted subset"),
        }
    }
}

/// One instruction: `[@[!]guard] op dst, src...;` (`invert`: the result is negated, for setp.ne).
struct Instr {
    guard: Option<(usize, bool)>,
    op: Op,
    invert: bool,
    dst: usize,
    src: Vec<Operand>,
}

struct Program {
    instrs: Vec<Instr>,
    slots: HashMap<String, usize>,
}

fn parse(lines: &[String]) -> Program {
    let mut slots: HashMap<String, usize> = HashMap::new();
    let mut slot = |name: &str| {
        let n = slots.len();
        *slots.entry(name.to_string()).or_insert(n)
    };
    let mut instrs = Vec::new();
    for line in lines {
        let line = line.strip_suffix(';').unwrap_or_else(|| panic!("not an instruction: {line}"));
        let (guard, rest) = match line.strip_prefix('@') {
            Some(g) => {
                let (g, rest) = g.split_once(char::is_whitespace).unwrap();
                let negate = g.starts_with('!');
                (Some((slot(g.trim_start_matches('!')), negate)), rest.trim())
            }
            None => (None, line),
        };
        let (op, args) = rest.split_once(char::is_whitespace).unwrap();
        let mut args = args.split(',').map(str::trim);
        let dst = slot(args.next().unwrap());
        let src = args
            .map(|a| {
                let negate = a.starts_with('!');
                let a = a.trim_start_matches('!');
                if a.starts_with('%') {
                    Operand::Reg(slot(a), negate)
                } else {
                    let v = match a.strip_prefix("0x") {
                        Some(hex) => u32::from_str_radix(hex, 16).unwrap(),
                        None => a.parse().unwrap(),
                    };
                    Operand::Imm(v)
                }
            })
            .collect();
        let (op, invert) = Op::parse(op);
        instrs.push(Instr { guard, op, invert, dst, src });
    }
    Program { instrs, slots }
}

/// Runs the program on one input; registers hold raw 32-bit values (f32 as bits, predicates as 0 / 1).
fn execute(program: &Program, regs: &mut [u32]) {
    let val = |regs: &[u32], o: Operand| match o {
        Operand::Reg(i, negate) => {
            if negate {
                !regs[i] & 1
            } else {
                regs[i]
            }
        }
        Operand::Imm(v) => v,
    };
    for ins in &program.instrs {
        if let Some((g, negate)) = ins.guard {
            if (regs[g] != 0) == negate {
                continue;
            }
        }
        let mut a = [0u32; 3];
        for (x, &o) in a.iter_mut().zip(&ins.src) {
            *x = val(regs, o);
        }
        let f = f32::from_bits;
        let out = match ins.op {
            Op::CvtF32U32 => (a[0] as f32).to_bits(),
            Op::RcpF32 => (1.0f32 / f(a[0])).to_bits(),
            Op::MulF32 => (f(a[0]) * f(a[1])).to_bits(),
            // truncation toward zero, saturating to [0, 2^32 - 1] (NaN -> 0)
            Op::CvtRziU32F32 => f(a[0]) as u32,
            Op::MulLoU32 => a[0].wrapping_mul(a[1]),
            Op::SubU32 => a[0].wrapping_sub(a[1]),
            Op::AddU32 => a[0].wrapping_add(a[1]),
            Op::AndPred => a[0] & a[1],
            Op::Setp(want, or_equal, signed) => {
                let ord = if signed { (a[0] as i32).cmp(&(a[1] as i32)) } else { a[0].cmp(&a[1]) };
                u32::from((ord == want || (or_equal && ord.is_eq())) != ins.invert)
            }
        };
        regs[ins.dst] = out;
    }
}

#[test]
fn fast_divmod_is_exact_below_2_pow_24() {
    let p = Ptx::new();
    let (n, d) = (p.reg("b32"), p.reg("b32"));
    let rcp = swin::rcp_f32(&p, &d);
    let (q, r) = swin::fast_divmod(&p, &n, &d, &rcp);
    let program = parse(&p.body_lines());
    let slot = |name: &str| program.slots[name];
    let (ns, ds, qs, rs) = (slot(&n), slot(&d), slot(&q), slot(&r));
    let top = 32u32;
    let failures: Vec<String> = std::thread::scope(|scope| {
        let workers: Vec<_> = (1..=top)
            .map(|dv| {
                let program = &program;
                scope.spawn(move || {
                    let mut regs = vec![0u32; program.slots.len()];
                    let (mut wrong, mut first) = (0u64, None);
                    for nv in 0..1u32 << 24 {
                        regs.fill(0);
                        regs[ns] = nv;
                        regs[ds] = dv;
                        execute(program, &mut regs);
                        if regs[qs] != nv / dv || regs[rs] != nv % dv {
                            wrong += 1;
                            first.get_or_insert((nv, regs[qs], regs[rs]));
                        }
                    }
                    first.map(|(nv, q, r)| format!("d = {dv}: {wrong} wrong, first n = {nv} -> ({q}, {r})"))
                })
            })
            .collect();
        workers.into_iter().filter_map(|w| w.join().unwrap()).collect()
    });
    assert!(failures.is_empty(), "fast_divmod: every n < 2^24, d = 1..{top}: FAIL\n{}", failures.join("\n"));
}
