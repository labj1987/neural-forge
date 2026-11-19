//! Where a function's return address goes: a static data-flow scan of x86-64 code, used by [`super`] to find the NGX
//! core's exports whose code reads its caller.
//!
//! An export reads its caller when the return address it was called with (`[rsp]` at its entry) reaches the address
//! argument (`rdx`) of `GetModuleHandleExA`/`W`, called through the import table: in its own code, in code it jumps
//! to (a tail call keeps the return address), or through at most [`MAX_CALL_DEPTH`] direct calls that pass it on as
//! an argument (a function that deep is still scanned for its frame, its own calls are not). The scan follows registers and the stack slots of each function's frame (pushes, `sub rsp`, moves
//! between registers and frame slots, `lea` of frame addresses, `__chkstk`'s `sub rsp, rax`), merges the states where
//! paths meet, and forgets a value at anything it does not model; it does not follow values through the heap or
//! through computed jumps (switch tables). Wherever the return address goes that the scan cannot see the end of, it
//! says so instead of staying silent ([`Flow::sinks`]): another import's argument (a new way of reading the caller),
//! a call deeper than [`MAX_CALL_DEPTH`] ([`DEPTH_LIMIT`]) and an indirect call it cannot resolve ([`INDIRECT_CALL`]).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register};

/// How many direct calls deep the return address is followed when it is passed on as an argument.
pub const MAX_CALL_DEPTH: u32 = 4;
/// The sink of a register argument passed to a direct call deeper than [`MAX_CALL_DEPTH`].
pub const DEPTH_LIMIT: &str = "<a call deeper than the scan follows>";
/// The sink of a register argument passed to an indirect call that is not to an import.
pub const INDIRECT_CALL: &str = "<an indirect call>";
/// The sink of the value at `[rsp]` (argument 0) at an indirect jump that is not to an import: a tail call the scan
/// cannot follow, which hands its target the return address there.
pub const INDIRECT_JUMP: &str = "<an indirect tail jump>";
/// Instructions one function's scan may visit before it gives up (each is visited again only when its state changes).
const MAX_STEPS: usize = 200_000;
/// Arguments followed into calls: four registers, then the stack.
const MAX_ARGS: u8 = 16;
/// The registers of the first four arguments (Windows x64): rcx, rdx, r8, r9.
const ARG_REGS: [usize; 4] = [1, 2, 8, 9];
/// The registers a call may change: rax, rcx, rdx, r8 to r11.
const VOLATILE: [usize; 7] = [0, 1, 2, 8, 9, 10, 11];
const RAX: usize = 0;
const RSP: usize = 4;

/// The code to scan: its executable bytes by address, and the import table's slots by address.
#[derive(Default)]
pub struct Image<'a> {
    regions: Vec<(u64, &'a [u8])>,
    imports: HashMap<u64, String>,
    cfg_dispatch: Option<u64>,
    cfg_check: Option<u64>,
}

impl<'a> Image<'a> {
    /// Code at virtual address `va` (an executable section's bytes).
    pub fn add_code(&mut self, va: u64, bytes: &'a [u8]) {
        self.regions.push((va, bytes));
    }

    /// The import table slot at `va` holds the address of `name`.
    pub fn add_import(&mut self, va: u64, name: impl Into<String>) {
        self.imports.insert(va, name.into());
    }

    /// The pointer Control Flow Guard's indirect calls go through (`GuardCFDispatchFunctionPointer`): a call through
    /// it calls `rax`.
    pub fn set_cfg_dispatch(&mut self, va: u64) {
        self.cfg_dispatch = Some(va);
    }

    /// The pointer Control Flow Guard's check before an indirect call goes through (`GuardCFCheckFunctionPointer`):
    /// it only checks `rcx`, and keeps the argument registers.
    pub fn set_cfg_check(&mut self, va: u64) {
        self.cfg_check = Some(va);
    }

    fn code_at(&self, va: u64) -> Option<&'a [u8]> {
        self.regions.iter().find_map(|&(start, bytes)| {
            let off = usize::try_from(va.checked_sub(start)?).ok()?;
            (off < bytes.len()).then(|| &bytes[off..])
        })
    }
}

/// Where values of one function went: for each import and argument position, whether the function's own return
/// address reached it and which of its parameters did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Flow {
    /// (import name, argument index) to what reached that argument; also [`DEPTH_LIMIT`] and [`INDIRECT_CALL`].
    pub sinks: BTreeMap<(String, u8), Reached>,
    /// The scan stopped before it had seen every path (its step limit, or code that could not be decoded).
    pub incomplete: bool,
    /// A bit per stack argument slot (`[rsp + 8 + 8k]` at entry, the register arguments' home space included) the
    /// function writes: a caller's value there does not survive the call.
    writes_args: u32,
    /// A bit per register unchanged on every return (a stack probe such as `__chkstk` keeps rax and the arguments).
    kept: u16,
}

impl Default for Flow {
    fn default() -> Self {
        Flow { sinks: BTreeMap::new(), incomplete: false, writes_args: 0, kept: u16::MAX }
    }
}

/// What reached one import argument.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reached {
    /// The function's own return address.
    pub ret: bool,
    /// A bit per parameter of the function.
    pub params: u32,
}

impl Flow {
    /// The imports (and argument index) the function's return address reached.
    pub fn return_address_sinks(&self) -> BTreeSet<(String, u8)> {
        self.sinks.iter().filter(|(_, r)| r.ret).map(|(k, _)| k.clone()).collect()
    }

    /// Whether the return address reached `GetModuleHandleExA` or `GetModuleHandleExW`'s address argument.
    pub fn reads_caller(&self) -> bool {
        self.return_address_sinks().iter().any(|(name, arg)| is_module_from_address(name) && *arg == 1)
    }

    fn record(&mut self, sink: (&str, u8), v: Val) {
        if v.may != 0 {
            let e = self.sinks.entry((sink.0.to_string(), sink.1)).or_default();
            e.ret |= v.may & RET != 0;
            e.params |= v.may & !RET;
        }
    }

    /// Records what reaches each register argument of a call whose callee the scan cannot follow.
    fn record_register_args(&mut self, sink: &str, st: &State) {
        for k in 0..ARG_REGS.len() as u8 {
            self.record((sink, k), st.arg(k));
        }
    }
}

/// Whether an import is one that turns an address into its module.
pub fn is_module_from_address(name: &str) -> bool {
    matches!(name, "GetModuleHandleExA" | "GetModuleHandleExW")
}

/// What a value is known to be on every path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Known {
    #[default]
    Unknown,
    /// An address in the function's frame: the entry `rsp` plus this.
    Stack(i64),
    Const(u64),
    /// The address of the import whose table slot is at this address.
    Import(u64),
}

/// The bit of [`Val::may`] for the function's own return address; bit `k` is its parameter `k` (0 = rcx).
const RET: u32 = 1 << 31;

/// A value: what it is known to be, and which of the function's inputs it is on at least one path (copied by moves,
/// pushes and pops, dropped by any other change, kept where paths meet).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Val {
    known: Known,
    may: u32,
}

impl Val {
    const UNKNOWN: Val = Val { known: Known::Unknown, may: 0 };

    fn known(known: Known) -> Val {
        Val { known, may: 0 }
    }

    fn input(bit: u32) -> Val {
        Val { known: Known::Unknown, may: bit }
    }

    fn join(self, other: Val) -> Val {
        Val { known: if self.known == other.known { self.known } else { Known::Unknown }, may: self.may | other.may }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct State {
    regs: [Val; 16],
    /// 8-byte frame slots by their offset from the entry `rsp`.
    slots: BTreeMap<i64, Val>,
    /// `rax` before the stack probe just called, for its `sub rsp, rax`.
    chkstk: Option<u64>,
    /// A bit per register some path to here wrote.
    written: u16,
    /// [`Flow::writes_args`] on the paths to here.
    arg_writes: u32,
}

impl State {
    fn entry() -> State {
        let mut regs = [Val::UNKNOWN; 16];
        regs[RSP] = Val::known(Known::Stack(0));
        for (k, &r) in ARG_REGS.iter().enumerate() {
            regs[r] = Val::input(1 << k);
        }
        let mut slots = BTreeMap::from([(0, Val::input(RET))]);
        for k in 4..MAX_ARGS {
            slots.insert(8 + 8 * i64::from(k), Val::input(1 << k));
        }
        State { regs, slots, chkstk: None, written: 0, arg_writes: 0 }
    }

    /// Merges `other` into `self` ([`Val::join`]). Whether `self` changed.
    fn join(&mut self, other: &State) -> bool {
        let before = self.clone();
        for (a, b) in self.regs.iter_mut().zip(other.regs) {
            *a = a.join(b);
        }
        for (k, v) in &other.slots {
            let mine = self.slots.get(k).copied().unwrap_or(Val::UNKNOWN);
            self.slots.insert(*k, mine.join(*v));
        }
        for (k, v) in self.slots.iter_mut() {
            if !other.slots.contains_key(k) {
                *v = v.join(Val::UNKNOWN);
            }
        }
        self.slots.retain(|_, v| *v != Val::UNKNOWN);
        if self.chkstk != other.chkstk {
            self.chkstk = None;
        }
        self.written |= other.written;
        self.arg_writes |= other.arg_writes;
        *self != before
    }

    fn arg(&self, k: u8) -> Val {
        match ARG_REGS.get(usize::from(k)) {
            Some(&r) => self.regs[r],
            None => match self.regs[RSP].known {
                Known::Stack(d) => self.slots.get(&(d + 8 * i64::from(k))).copied().unwrap_or(Val::UNKNOWN),
                _ => Val::UNKNOWN,
            },
        }
    }

    /// Forgets the slots `len` bytes from `off` overlap, noting a write to the function's own stack arguments.
    fn forget_slots(&mut self, off: i64, len: i64) {
        if len <= 0 {
            return;
        }
        let gone: Vec<i64> = self.slots.range(off - 7..off + len).map(|(k, _)| *k).collect();
        for k in gone {
            self.slots.remove(&k);
        }
        for k in 0..MAX_ARGS {
            let slot = 8 + 8 * i64::from(k);
            if off < slot + 8 && slot < off + len {
                self.arg_writes |= 1 << k;
            }
        }
    }

    fn set_reg(&mut self, g: usize, v: Val) {
        self.written |= 1 << g;
        self.regs[g] = v;
    }
}

fn gpr(r: Register) -> Option<usize> {
    let f = r.full_register();
    f.is_gpr64().then(|| f as usize - Register::RAX as usize)
}

/// Where a memory operand points.
#[derive(Clone, Copy)]
enum Addr {
    Stack(i64),
    Abs(u64),
}

/// Scans an image's functions, keeping each function's result.
pub struct Scanner<'i, 'a> {
    image: &'i Image<'a>,
    decoded: HashMap<u64, Option<Instruction>>,
    flows: HashMap<(u64, u32), Flow>,
    info: InstructionInfoFactory,
}

impl<'i, 'a> Scanner<'i, 'a> {
    pub fn new(image: &'i Image<'a>) -> Self {
        Scanner { image, decoded: HashMap::new(), flows: HashMap::new(), info: InstructionInfoFactory::new() }
    }

    /// The function at `va`, called from outside.
    pub fn function(&mut self, va: u64) -> Flow {
        self.flow(va, 0)
    }

    fn decode(&mut self, va: u64) -> Option<Instruction> {
        if let Some(i) = self.decoded.get(&va) {
            return *i;
        }
        let i = self.image.code_at(va).and_then(|bytes| {
            let i = Decoder::with_ip(64, bytes, va, DecoderOptions::NONE).decode();
            (!i.is_invalid()).then_some(i)
        });
        self.decoded.insert(va, i);
        i
    }

    fn flow(&mut self, entry: u64, depth: u32) -> Flow {
        if let Some(f) = self.flows.get(&(entry, depth)) {
            return f.clone();
        }
        // (A function is never asked for again while it is scanned: each call goes one level deeper, up to the limit.)
        let mut flow = Flow::default();
        let mut states: HashMap<u64, State> = HashMap::from([(entry, State::entry())]);
        let mut work = vec![entry];
        let mut steps = 0;
        while let Some(ip) = work.pop() {
            steps += 1;
            if steps > MAX_STEPS {
                flow.incomplete = true;
                break;
            }
            let Some(instr) = self.decode(ip) else {
                flow.incomplete = true;
                continue;
            };
            let mut st = states[&ip].clone();
            for next in self.step(&mut st, &instr, depth, &mut flow) {
                let changed = match states.get_mut(&next) {
                    Some(s) => s.join(&st),
                    None => {
                        states.insert(next, st.clone());
                        true
                    }
                };
                if changed {
                    work.push(next);
                }
            }
        }
        flow.writes_args = states.values().fold(0, |a, s| a | s.arg_writes);
        if flow.incomplete {
            flow.kept = 0;
        }
        self.flows.insert((entry, depth), flow.clone());
        flow
    }

    fn mem_addr(st: &State, i: &Instruction) -> Option<Addr> {
        if matches!(i.segment_prefix(), Register::FS | Register::GS) {
            return None;
        }
        if i.is_ip_rel_memory_operand() {
            return Some(Addr::Abs(i.ip_rel_memory_address()));
        }
        if i.memory_index() != Register::None {
            return None;
        }
        let disp = i.memory_displacement64();
        if i.memory_base() == Register::None {
            return Some(Addr::Abs(disp));
        }
        match st.regs[gpr(i.memory_base())?].known {
            Known::Stack(d) => Some(Addr::Stack(d.wrapping_add(disp as i64))),
            Known::Const(c) => Some(Addr::Abs(c.wrapping_add(disp))),
            _ => None,
        }
    }

    fn load(&self, st: &State, addr: Option<Addr>, size: usize) -> Val {
        match addr {
            Some(Addr::Stack(off)) if size == 8 => st.slots.get(&off).copied().unwrap_or(Val::UNKNOWN),
            Some(Addr::Abs(va)) if size == 8 && self.image.imports.contains_key(&va) => Val::known(Known::Import(va)),
            _ => Val::UNKNOWN,
        }
    }

    /// A write of `size` bytes (0: unknown) through a memory operand with this base, index and displacement: the
    /// frame slots it may overwrite are forgotten. With an index, that is every slot from the base up: below the entry
    /// `rsp`, up to it (no frame array reaches the return address and the arguments above it); at or above it, all.
    fn forget_written(st: &mut State, base: Register, index: Register, disp: u64, size: usize) {
        let Some(Known::Stack(d)) = gpr(base).map(|g| st.regs[g].known) else { return };
        let off = d.wrapping_add(disp as i64);
        if index == Register::None {
            st.forget_slots(off, if size == 0 { 8 } else { size as i64 });
        } else if off < 0 {
            st.forget_slots(off, -off);
        } else {
            // An array in the home or argument area: everything above it.
            let top = st.slots.keys().next_back().map_or(off + 8, |k| k + 8).max(off + 8 * i64::from(MAX_ARGS) + 8);
            st.forget_slots(off, top - off);
        }
    }

    /// The value of operand `n`.
    fn operand(&self, st: &State, i: &Instruction, n: u32) -> Val {
        match i.op_kind(n) {
            OpKind::Register => {
                let r = i.op_register(n);
                match gpr(r) {
                    Some(g) if r.size() == 8 => st.regs[g],
                    Some(g) if r.size() == 4 => match st.regs[g].known {
                        Known::Const(c) => Val::known(Known::Const(c & 0xffff_ffff)),
                        _ => Val::UNKNOWN,
                    },
                    _ => Val::UNKNOWN,
                }
            }
            OpKind::Memory => self.load(st, Self::mem_addr(st, i), i.memory_size().size()),
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => Val::known(Known::Const(i.immediate(n))),
            _ => Val::UNKNOWN,
        }
    }

    fn write_reg(st: &mut State, r: Register, v: Val) {
        let Some(g) = gpr(r) else { return };
        let v = match (r.size(), v.known) {
            (8, _) => v,
            (4, Known::Const(c)) => Val::known(Known::Const(c & 0xffff_ffff)),
            _ => Val::UNKNOWN,
        };
        st.set_reg(g, v);
    }

    /// Writes `v` to operand 0, a register or memory.
    fn write_dest(&self, st: &mut State, i: &Instruction, v: Val) {
        match i.op_kind(0) {
            OpKind::Register => Self::write_reg(st, i.op_register(0), v),
            OpKind::Memory => {
                let size = i.memory_size().size();
                match Self::mem_addr(st, i) {
                    Some(Addr::Stack(off)) => {
                        st.forget_slots(off, size.max(1) as i64);
                        if size == 8 {
                            st.slots.insert(off, v);
                        }
                    }
                    Some(Addr::Abs(_)) => {}
                    None => Self::forget_written(st, i.memory_base(), i.memory_index(), i.memory_displacement64(), size),
                }
            }
            _ => {}
        }
    }

    /// What an instruction does the scan does not model: every register and frame slot it writes is forgotten.
    fn forget_writes(&mut self, st: &mut State, i: &Instruction) {
        let info = self.info.info(i);
        let writes = |a: OpAccess| matches!(a, OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite);
        let regs: Vec<Register> = info.used_registers().iter().filter(|u| writes(u.access())).map(|u| u.register()).collect();
        let mems: Vec<(Register, Register, u64, usize)> =
            info.used_memory().iter().filter(|m| writes(m.access())).map(|m| (m.base(), m.index(), m.displacement(), m.memory_size().size())).collect();
        for (base, index, disp, size) in mems {
            Self::forget_written(st, base, index, disp, size);
        }
        for r in regs {
            if let Some(g) = gpr(r) {
                st.set_reg(g, Val::UNKNOWN);
            }
        }
    }

    /// A direct call: what the callee does with its parameters and stack arguments, seen from here. The registers it
    /// keeps. A callee [`MAX_CALL_DEPTH`] calls deep is scanned for its frame effects only (a stack probe there still
    /// counts), and what it passes on is recorded as [`DEPTH_LIMIT`]; its own calls are not scanned.
    fn call_direct(&mut self, st: &mut State, target: u64, depth: u32, flow: &mut Flow) -> u16 {
        if depth > MAX_CALL_DEPTH {
            flow.record_register_args(DEPTH_LIMIT, st);
            return 0;
        }
        let callee = self.flow(target, depth + 1);
        flow.incomplete |= callee.incomplete;
        for ((name, arg), reached) in &callee.sinks {
            for k in 0..MAX_ARGS {
                if reached.params & (1 << k) != 0 {
                    if depth < MAX_CALL_DEPTH {
                        flow.record((name, *arg), st.arg(k));
                    } else {
                        flow.record((DEPTH_LIMIT, k), st.arg(k));
                    }
                }
            }
        }
        // The callee's own writes to its stack arguments, which are this frame's [rsp + 8k].
        if let Known::Stack(d) = st.regs[RSP].known {
            for k in 4..MAX_ARGS {
                if callee.writes_args & (1 << k) != 0 {
                    st.forget_slots(d + 8 * i64::from(k), 8);
                }
            }
        }
        callee.kept
    }

    /// Where a call or jump through operand 0 goes: the import (by its table slot), or `None` for anything else.
    fn indirect_import(&self, st: &State, i: &Instruction) -> Option<u64> {
        let v = match i.op0_kind() {
            OpKind::Memory => match Self::mem_addr(st, i) {
                Some(Addr::Abs(va)) if self.image.imports.contains_key(&va) => return Some(va),
                Some(Addr::Abs(va)) if self.image.cfg_dispatch == Some(va) => st.regs[RAX],
                addr => self.load(st, addr, 8),
            },
            _ => self.operand(st, i, 0),
        };
        match v.known {
            Known::Import(slot) => Some(slot),
            _ => None,
        }
    }

    /// A call or jump through operand 0: records what reaches the import's (or the unknown callee's) register
    /// arguments (the scan cannot tell how many arguments it takes, so its stack ones are not looked at). A jump that
    /// is not to an import is a tail call only with the frame gone (`rsp` back at its entry): then what its target
    /// finds at `[rsp]` is recorded too; a jump within a frame is a computed jump (a switch) and records nothing. The
    /// registers the callee keeps: Control Flow Guard's check keeps all but rax, r10 and r11, anything else none.
    fn call_indirect(&self, st: &State, i: &Instruction, flow: &mut Flow) -> u16 {
        if i.op0_kind() == OpKind::Memory {
            if let (Some(Addr::Abs(va)), Some(check)) = (Self::mem_addr(st, i), self.image.cfg_check) {
                if va == check {
                    return !((1 << RAX) | (1 << 10) | (1 << 11));
                }
            }
        }
        let tail = i.flow_control() == FlowControl::IndirectBranch;
        match self.indirect_import(st, i).and_then(|slot| self.image.imports.get(&slot)) {
            Some(name) => flow.record_register_args(name, st),
            None if !tail => flow.record_register_args(INDIRECT_CALL, st),
            None if st.regs[RSP].known == Known::Stack(0) => {
                flow.record_register_args(INDIRECT_CALL, st);
                flow.record((INDIRECT_JUMP, 0), st.slots.get(&0).copied().unwrap_or(Val::UNKNOWN));
            }
            None => {}
        }
        0
    }

    /// Runs one instruction on `st`; the addresses that follow it.
    fn step(&mut self, st: &mut State, i: &Instruction, depth: u32, flow: &mut Flow) -> Vec<u64> {
        let chkstk = st.chkstk.take();
        let next = i.next_ip();
        match i.flow_control() {
            FlowControl::Return => {
                flow.kept &= !st.written;
                return vec![];
            }
            FlowControl::Interrupt | FlowControl::Exception | FlowControl::XbeginXabortXend => return vec![],
            FlowControl::UnconditionalBranch => return vec![i.near_branch_target()],
            FlowControl::ConditionalBranch => return vec![i.near_branch_target(), next],
            FlowControl::IndirectBranch => {
                // A tail call through a pointer, or a computed jump within the function (a switch table): neither
                // is followed.
                self.call_indirect(st, i, flow);
                flow.kept = 0;
                return vec![];
            }
            FlowControl::Call | FlowControl::IndirectCall => {
                // The callee sees rsp - 8, so its stack arguments are this frame's [rsp + 8k], as State::arg reads them.
                let rax = st.regs[RAX];
                let kept = if i.flow_control() == FlowControl::Call {
                    self.call_direct(st, i.near_branch_target(), depth, flow)
                } else {
                    self.call_indirect(st, i, flow)
                };
                for r in VOLATILE {
                    if kept & (1 << r) == 0 {
                        st.set_reg(r, Val::UNKNOWN);
                    }
                }
                // The callee's home space for the register arguments, within this function's own frame (a call
                // made before the frame is set up, as `__chkstk`'s, does not write it).
                if let Known::Stack(d) = st.regs[RSP].known {
                    st.forget_slots(d, (0x20).min(-d));
                }
                // A stack probe (`__chkstk`) keeps rax, the frame's size, for the `sub rsp, rax` after it.
                if let (true, Known::Const(c)) = (kept & (1 << RAX) != 0, rax.known) {
                    st.chkstk = Some(c);
                }
                return vec![next];
            }
            FlowControl::Next => {}
        }
        match i.mnemonic() {
            Mnemonic::Push if i.op_count() == 1 => {
                let v = if i.op0_kind() == OpKind::Register && i.op0_register().size() != 8 { Val::UNKNOWN } else { self.operand(st, i, 0) };
                if let Known::Stack(d) = st.regs[RSP].known {
                    st.forget_slots(d - 8, 8);
                    st.slots.insert(d - 8, v);
                    st.regs[RSP] = Val::known(Known::Stack(d - 8));
                }
            }
            Mnemonic::Pop if i.op_count() == 1 => {
                let v = match st.regs[RSP].known {
                    Known::Stack(d) => {
                        st.regs[RSP] = Val::known(Known::Stack(d + 8));
                        st.slots.get(&d).copied().unwrap_or(Val::UNKNOWN)
                    }
                    _ => Val::UNKNOWN,
                };
                self.write_dest(st, i, v);
            }
            Mnemonic::Mov => {
                let v = self.operand(st, i, 1);
                self.write_dest(st, i, v);
            }
            Mnemonic::Lea => {
                let v = if i.op0_register().size() != 8 {
                    Val::UNKNOWN
                } else {
                    match Self::mem_addr(st, i) {
                        Some(Addr::Stack(off)) => Val::known(Known::Stack(off)),
                        Some(Addr::Abs(va)) => Val::known(Known::Const(va)),
                        None => Val::UNKNOWN,
                    }
                };
                Self::write_reg(st, i.op0_register(), v);
            }
            Mnemonic::Add | Mnemonic::Sub if i.op0_kind() == OpKind::Register => {
                let mut by = self.operand(st, i, 1).known;
                if by == Known::Unknown && i.mnemonic() == Mnemonic::Sub && i.op1_kind() == OpKind::Register && i.op1_register() == Register::RAX {
                    by = chkstk.map_or(Known::Unknown, Known::Const);
                }
                let r = i.op0_register();
                let cur = gpr(r).map_or(Known::Unknown, |g| st.regs[g].known);
                let sub = i.mnemonic() == Mnemonic::Sub;
                let v = match (cur, by) {
                    (Known::Stack(d), Known::Const(c)) if r.size() == 8 => Known::Stack(if sub { d.wrapping_sub(c as i64) } else { d.wrapping_add(c as i64) }),
                    (Known::Const(a), Known::Const(c)) => Known::Const(if sub { a.wrapping_sub(c) } else { a.wrapping_add(c) }),
                    _ => Known::Unknown,
                };
                Self::write_reg(st, r, Val::known(v));
            }
            // A conditional move keeps either value; an exchange swaps.
            Mnemonic::Cmova
            | Mnemonic::Cmovae
            | Mnemonic::Cmovb
            | Mnemonic::Cmovbe
            | Mnemonic::Cmove
            | Mnemonic::Cmovg
            | Mnemonic::Cmovge
            | Mnemonic::Cmovl
            | Mnemonic::Cmovle
            | Mnemonic::Cmovne
            | Mnemonic::Cmovno
            | Mnemonic::Cmovnp
            | Mnemonic::Cmovns
            | Mnemonic::Cmovo
            | Mnemonic::Cmovp
            | Mnemonic::Cmovs
                if i.op0_register().size() == 8 =>
            {
                let src = self.operand(st, i, 1);
                let r = i.op0_register();
                let cur = gpr(r).map_or(Val::UNKNOWN, |g| st.regs[g]);
                Self::write_reg(st, r, cur.join(src));
            }
            Mnemonic::Xchg if i.op0_kind() == OpKind::Register && i.op1_kind() == OpKind::Register && i.op0_register().size() == 8 => {
                let (a, b) = (self.operand(st, i, 0), self.operand(st, i, 1));
                Self::write_reg(st, i.op0_register(), b);
                Self::write_reg(st, i.op1_register(), a);
            }
            Mnemonic::Xor if i.op0_kind() == OpKind::Register && i.op1_kind() == OpKind::Register && i.op0_register() == i.op1_register() => {
                Self::write_reg(st, i.op0_register(), Val::known(Known::Const(0)));
            }
            _ => self.forget_writes(st, i),
        }
        vec![next]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::code_asm::*;
    use iced_x86::BlockEncoderOptions;

    const BASE: u64 = 0x1_8000_1000;

    /// The import table slots the tests' code calls through, placed after the code.
    #[derive(Clone, Copy)]
    struct Iat {
        gmhe_a: CodeLabel,
        gmfn_w: CodeLabel,
        gmhe_w: CodeLabel,
        other: CodeLabel,
        /// Control Flow Guard's dispatch pointer (not an import).
        cfg: CodeLabel,
        /// Control Flow Guard's check pointer.
        check: CodeLabel,
    }

    /// Assembles `build`'s code at [`BASE`] and scans the function at the label it returns.
    fn scan(build: impl FnOnce(&mut CodeAssembler, Iat) -> Result<CodeLabel, IcedError>) -> Flow {
        let mut a = CodeAssembler::new(64).unwrap();
        let iat = Iat {
            gmhe_a: a.create_label(),
            gmfn_w: a.create_label(),
            gmhe_w: a.create_label(),
            other: a.create_label(),
            cfg: a.create_label(),
            check: a.create_label(),
        };
        let entry = build(&mut a, iat).unwrap();
        let mut slots = [iat.gmhe_a, iat.gmfn_w, iat.gmhe_w, iat.other, iat.cfg, iat.check];
        for slot in &mut slots {
            a.set_label(slot).unwrap();
            a.dq(&[0]).unwrap();
        }
        let result = a.assemble_options(BASE, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS).unwrap();
        let entry_va = result.label_ip(&entry).unwrap();
        let mut image = Image::default();
        for (slot, name) in slots.iter().zip(["GetModuleHandleExA", "GetModuleFileNameW", "GetModuleHandleExW", "VirtualQuery"]) {
            image.add_import(result.label_ip(slot).unwrap(), name);
        }
        image.set_cfg_dispatch(result.label_ip(&slots[4]).unwrap());
        image.set_cfg_check(result.label_ip(&slots[5]).unwrap());
        let code = result.inner.code_buffer;
        image.add_code(BASE, &code);
        Scanner::new(&image).function(entry_va)
    }

    #[test]
    fn the_return_address_read_after_the_prologue() {
        // CUDA_Init's shape in driver 620.73: pushes, a frame, then [rsp + frame] into rdx.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.mov(qword_ptr(rsp + 0x18), rbx)?;
            a.push(rbp)?;
            a.push(rsi)?;
            a.sub(rsp, 0x40)?;
            a.mov(rdx, qword_ptr(rsp + 0x50))?;
            a.mov(ecx, 6)?;
            a.lea(r8, qword_ptr(rsp + 0x30))?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.add(rsp, 0x40)?;
            a.pop(rsi)?;
            a.pop(rbp)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
        assert!(!f.incomplete);
    }

    #[test]
    fn its_own_address_is_not_its_caller() {
        // CUDA_Init also asks for the core's own module, with the address of the function itself.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.lea(rdx, qword_ptr(entry))?;
            a.mov(ecx, 6)?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.mov(rcx, qword_ptr(rsp + 0x28))?;
            a.call(qword_ptr(iat.gmfn_w))?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(!f.reads_caller(), "{f:?}");
        // Handing the return address to another import is reported, not taken for reading the caller.
        assert_eq!(f.return_address_sinks(), BTreeSet::from([("GetModuleFileNameW".to_string(), 0)]));
    }

    #[test]
    fn through_a_helper_that_takes_it_as_an_argument() {
        // GetFeatureRequirements' shape: the return address in rcx to a helper, which hands its rcx to rdx.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            let mut helper = a.create_label();
            a.set_label(&mut helper)?;
            a.push(rbx)?;
            a.sub(rsp, 0x30)?;
            a.mov(rbx, rcx)?;
            a.mov(qword_ptr(rsp + 0x28), rbx)?;
            a.mov(rdx, qword_ptr(rsp + 0x28))?;
            a.mov(ecx, 6)?;
            a.call(qword_ptr(iat.gmhe_w))?;
            a.add(rsp, 0x30)?;
            a.pop(rbx)?;
            a.ret()?;
            a.set_label(&mut entry)?;
            a.push(rsi)?;
            a.sub(rsp, 0xa0)?;
            a.test(ecx, ecx)?;
            let mut out = a.create_label();
            a.js(out)?;
            a.mov(rcx, qword_ptr(rsp + 0xa8))?;
            a.call(helper)?;
            a.set_label(&mut out)?;
            a.add(rsp, 0xa0)?;
            a.pop(rsi)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
    }

    #[test]
    fn through_a_stack_argument_and_a_tail_jump() {
        // The return address as a callee's fifth argument; the entry is a jump to the real function.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            let mut body = a.create_label();
            let mut helper = a.create_label();
            a.set_label(&mut helper)?;
            a.mov(rdx, qword_ptr(rsp + 0x28))?; // its fifth argument
            a.xor(ecx, ecx)?;
            a.jmp(qword_ptr(iat.gmhe_a))?; // a tail call to the import
            a.set_label(&mut entry)?;
            a.jmp(body)?;
            a.set_label(&mut body)?;
            a.mov(rax, qword_ptr(rsp))?;
            a.sub(rsp, 0x38)?;
            a.mov(qword_ptr(rsp + 0x20), rax)?;
            a.call(helper)?;
            a.add(rsp, 0x38)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
    }

    #[test]
    fn a_large_frame_from_chkstk() {
        // mov eax, size; call __chkstk; sub rsp, rax: the frame's size survives the call.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            let mut chkstk = a.create_label();
            a.set_label(&mut chkstk)?;
            a.ret()?;
            a.set_label(&mut entry)?;
            a.push(rbx)?;
            a.mov(eax, 0x2010)?;
            a.call(chkstk)?;
            a.sub(rsp, rax)?;
            a.mov(rdx, qword_ptr(rsp + 0x2018))?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.add(rsp, 0x2010)?;
            a.pop(rbx)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
    }

    #[test]
    fn the_return_address_on_one_path_counts() {
        // One path loads the return address into rdx, the other zero: GetModuleHandleExA may see the caller's.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            let mut other = a.create_label();
            let mut join = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.test(ecx, ecx)?;
            a.je(other)?;
            a.xor(edx, edx)?;
            a.jmp(join)?;
            a.set_label(&mut other)?;
            a.mov(rdx, qword_ptr(rsp + 0x28))?;
            a.set_label(&mut join)?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
    }

    #[test]
    fn a_call_overwrites_rdx() {
        // rdx after a call is the callee's leftover, not the return address loaded before it.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.mov(rdx, qword_ptr(rsp + 0x28))?;
            a.call(qword_ptr(iat.other))?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(!f.reads_caller(), "{f:?}");
        assert_eq!(f.return_address_sinks(), BTreeSet::from([("VirtualQuery".to_string(), 1)]));
    }

    #[test]
    fn an_unmodelled_write_forgets_the_slot() {
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.mov(rax, qword_ptr(rsp + 0x28))?;
            a.mov(qword_ptr(rsp + 0x20), rax)?;
            a.inc(qword_ptr(rsp + 0x20))?;
            a.mov(rdx, qword_ptr(rsp + 0x20))?;
            a.call(qword_ptr(iat.other))?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.return_address_sinks().is_empty(), "{f:?}");
    }

    #[test]
    fn calls_deeper_than_the_limit_are_not_followed() {
        // A chain of helpers each passing rcx on: found within MAX_CALL_DEPTH, not beyond.
        let chain = |n: u32| {
            scan(move |a, iat| {
                let mut labels: Vec<CodeLabel> = (0..=n).map(|_| a.create_label()).collect();
                // labels[0] calls GetModuleHandleExA with its rcx; labels[k] calls labels[k - 1].
                a.set_label(&mut labels[0])?;
                a.mov(rdx, rcx)?;
                a.jmp(qword_ptr(iat.gmhe_a))?;
                for k in 1..=n as usize {
                    let callee = labels[k - 1];
                    a.set_label(&mut labels[k])?;
                    a.sub(rsp, 0x28)?;
                    if k == n as usize {
                        a.mov(rcx, qword_ptr(rsp + 0x28))?;
                    }
                    a.call(callee)?;
                    a.add(rsp, 0x28)?;
                    a.ret()?;
                }
                Ok(labels[n as usize])
            })
        };
        assert!(chain(MAX_CALL_DEPTH).reads_caller());
        let beyond = chain(MAX_CALL_DEPTH + 1);
        assert!(!beyond.reads_caller());
        // Not silently: the return address is reported as passed to a call the scan did not follow.
        assert_eq!(beyond.return_address_sinks(), BTreeSet::from([(DEPTH_LIMIT.to_string(), 0)]));
        // The deepest function followed keeps its frame through a stack probe it calls.
        let probed = |n: u32| {
            scan(move |a, iat| {
                let mut chk = a.create_label();
                a.set_label(&mut chk)?;
                a.ret()?;
                let mut labels: Vec<CodeLabel> = (0..=n).map(|_| a.create_label()).collect();
                a.set_label(&mut labels[0])?;
                a.mov(qword_ptr(rsp + 8), rcx)?;
                a.mov(eax, 0x2010)?;
                a.call(chk)?;
                a.sub(rsp, rax)?;
                a.mov(rdx, qword_ptr(rsp + 0x2018))?;
                a.call(qword_ptr(iat.gmhe_a))?;
                a.add(rsp, 0x2010)?;
                a.ret()?;
                for k in 1..=n as usize {
                    let callee = labels[k - 1];
                    a.set_label(&mut labels[k])?;
                    a.sub(rsp, 0x28)?;
                    if k == n as usize {
                        a.mov(rcx, qword_ptr(rsp + 0x28))?;
                    }
                    a.call(callee)?;
                    a.add(rsp, 0x28)?;
                    a.ret()?;
                }
                Ok(labels[n as usize])
            })
        };
        for n in 1..=MAX_CALL_DEPTH {
            let f = probed(n);
            assert!(f.reads_caller() && !f.return_address_sinks().iter().any(|(name, _)| name == DEPTH_LIMIT), "{n}: {f:?}");
        }
    }

    #[test]
    fn recursion_and_loops_end() {
        let f = scan(|a, _iat| {
            let mut entry = a.create_label();
            let mut top = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.set_label(&mut top)?;
            a.dec(ecx)?;
            a.call(entry)?;
            a.jne(top)?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(!f.reads_caller() && !f.incomplete, "{f:?}");
    }

    #[test]
    fn calls_through_control_flow_guard_and_registers_reach_the_import() {
        // mov rax, [IAT]; call [guard dispatch]: the call goes to rax.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.mov(rdx, qword_ptr(rsp + 0x28))?;
            a.mov(rax, qword_ptr(iat.gmhe_a))?;
            a.call(qword_ptr(iat.cfg))?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
        // mov r10, [IAT]; call r10, and the pointer kept in a frame slot: call [rsp + 0x20].
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.mov(r10, qword_ptr(iat.gmhe_w))?;
            a.mov(qword_ptr(rsp + 0x20), r10)?;
            a.mov(rdx, qword_ptr(rsp + 0x28))?;
            a.call(r10)?;
            a.mov(rdx, qword_ptr(rsp + 0x28))?;
            a.call(qword_ptr(rsp + 0x20))?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert_eq!(f.sinks.get(&("GetModuleHandleExW".to_string(), 1)).map(|r| r.ret), Some(true), "{f:?}");
        // A call through some other pointer is not taken for the import rax happens to hold.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.mov(rax, qword_ptr(iat.gmhe_a))?;
            a.mov(rdx, qword_ptr(rsp + 0x28))?;
            a.call(qword_ptr(rbx + 0x10))?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(!f.reads_caller(), "{f:?}");
        assert_eq!(f.return_address_sinks(), BTreeSet::from([(INDIRECT_CALL.to_string(), 1)]));
    }

    #[test]
    fn an_rbp_frame() {
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.push(rbp)?;
            a.mov(rbp, rsp)?;
            a.and(rsp, -32)?; // the frame is aligned: rsp is unknown from here, rbp is not
            a.sub(rsp, 0x40)?;
            a.mov(rdx, qword_ptr(rbp + 8))?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.mov(rsp, rbp)?;
            a.pop(rbp)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
    }

    #[test]
    fn an_unresolved_tail_jump_hands_on_the_return_address() {
        // GetScratchBufferSize's shape: the frame popped, then jmp through a pointer from a table.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.push(rdi)?;
            a.sub(rsp, 0x20)?;
            a.mov(rax, qword_ptr(r9 + 0x2538))?;
            a.add(rsp, 0x20)?;
            a.pop(rdi)?;
            a.jmp(qword_ptr(iat.cfg))?;
            Ok(entry)
        });
        assert_eq!(f.return_address_sinks(), BTreeSet::from([(INDIRECT_JUMP.to_string(), 0)]));
        // A computed jump inside a frame (a switch) hands nothing on.
        let f = scan(|a, _iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.jmp(rax)?;
            Ok(entry)
        });
        assert!(f.return_address_sinks().is_empty(), "{f:?}");
    }

    #[test]
    fn writes_that_overwrite_a_saved_return_address() {
        // A callee that writes its fifth argument, where the caller kept the return address.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            let mut helper = a.create_label();
            a.set_label(&mut helper)?;
            a.mov(qword_ptr(rsp + 0x28), 0)?;
            a.ret()?;
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x38)?;
            a.mov(rax, qword_ptr(rsp + 0x38))?;
            a.mov(qword_ptr(rsp + 0x20), rax)?;
            a.call(helper)?;
            a.mov(rdx, qword_ptr(rsp + 0x20))?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.add(rsp, 0x38)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(!f.reads_caller(), "{f:?}");
        // An indexed write into the frame.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x38)?;
            a.mov(rax, qword_ptr(rsp + 0x38))?;
            a.mov(qword_ptr(rsp + 0x20), rax)?;
            a.xor(ecx, ecx)?;
            a.mov(qword_ptr(rsp + rcx + 0x20), 0)?;
            a.mov(rdx, qword_ptr(rsp + 0x20))?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.add(rsp, 0x38)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(!f.reads_caller(), "{f:?}");
        // A byte written next to it leaves it alone.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x38)?;
            a.mov(rax, qword_ptr(rsp + 0x38))?;
            a.mov(qword_ptr(rsp + 0x20), rax)?;
            a.inc(byte_ptr(rsp + 0x1f))?;
            a.mov(rdx, qword_ptr(rsp + 0x20))?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.add(rsp, 0x38)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
    }

    #[test]
    fn only_a_stack_probe_keeps_the_frame_size_in_rax() {
        // An ordinary callee returns something else in rax: sub rsp, rax leaves the frame unknown.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            let mut helper = a.create_label();
            a.set_label(&mut helper)?;
            a.mov(eax, 0x20)?;
            a.ret()?;
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.mov(eax, 0x10)?;
            a.call(helper)?;
            a.sub(rsp, rax)?;
            a.mov(rdx, qword_ptr(rsp + 0x38))?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.ret()?;
            Ok(entry)
        });
        assert!(!f.reads_caller(), "{f:?}");
    }

    #[test]
    fn conditional_moves_and_exchanges_keep_it() {
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.test(rdx, rdx)?;
            a.cmove(rdx, qword_ptr(rsp + 0x28))?; // p ? p : _ReturnAddress()
            a.call(qword_ptr(iat.gmhe_a))?;
            a.mov(rax, qword_ptr(rsp + 0x28))?;
            a.xchg(rax, r8)?;
            a.mov(rcx, r8)?;
            a.call(qword_ptr(iat.other))?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
        assert!(f.return_address_sinks().contains(&("VirtualQuery".to_string(), 0)), "{f:?}");
    }

    #[test]
    fn a_stack_probe_keeps_the_argument_registers() {
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            let mut chk = a.create_label();
            a.set_label(&mut chk)?;
            a.ret()?;
            a.set_label(&mut entry)?;
            a.mov(rcx, qword_ptr(rsp))?;
            a.mov(eax, 0x2000)?;
            a.call(chk)?;
            a.sub(rsp, rax)?;
            a.mov(rdx, rcx)?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.add(rsp, 0x2000)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
    }

    #[test]
    fn a_switch_and_the_guard_check_hand_nothing_on() {
        // A computed jump within the frame with the return address in rcx: not a tail call.
        let f = scan(|a, _iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.mov(rcx, qword_ptr(rsp + 0x28))?;
            a.jmp(rax)?;
            Ok(entry)
        });
        assert!(f.return_address_sinks().is_empty(), "{f:?}");
        // Control Flow Guard's check (rcx = the target) keeps rdx for the call that follows it.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.sub(rsp, 0x28)?;
            a.mov(rdx, qword_ptr(rsp + 0x28))?;
            a.mov(rcx, qword_ptr(iat.gmhe_a))?;
            a.mov(rbx, rcx)?;
            a.call(qword_ptr(iat.check))?;
            a.call(rbx)?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(f.reads_caller(), "{f:?}");
        assert_eq!(f.return_address_sinks().len(), 1, "{f:?}");
    }

    #[test]
    fn an_indexed_write_above_the_entry_forgets_what_is_there() {
        // The return address kept in the home space, then an array write into the home space.
        let f = scan(|a, iat| {
            let mut entry = a.create_label();
            a.set_label(&mut entry)?;
            a.mov(rax, qword_ptr(rsp))?;
            a.mov(qword_ptr(rsp + 0x10), rax)?;
            a.mov(qword_ptr(rsp + rcx * 8 + 8), 0)?;
            a.sub(rsp, 0x28)?;
            a.mov(rdx, qword_ptr(rsp + 0x38))?;
            a.call(qword_ptr(iat.gmhe_a))?;
            a.add(rsp, 0x28)?;
            a.ret()?;
            Ok(entry)
        });
        assert!(!f.reads_caller(), "{f:?}");
    }
}
