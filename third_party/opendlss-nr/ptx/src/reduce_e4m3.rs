//! Split-K reduction + publication (gemm_reduce.comp): f16 partials [S][rows][N] summed in slice order
//! (((p0 + p1) + p2) + p3 ...), then SiLU / E4M3 / f16 outputs. One thread per 8 columns.
//!
//! Arguments: `<splits> <flags>`        flags: SILU=2 E4=4 F16=8

use std::collections::HashMap;

use crate::ptxgen::Ptx;
use crate::swin::*;

pub const F_SILU: usize = 2;
pub const F_E4: usize = 4;
pub const F_F16: usize = 8;

/// The kernel's name and PTX.
pub fn generate(splits: usize, flags: usize) -> (String, String) {
    let (silu_f, out_e4, out_f16) = (flags & F_SILU != 0, flags & F_E4 != 0, flags & F_F16 != 0);
    let name = format!("reduce_e4m3_s{splits}_f{flags}");
    let p = Ptx::new();
    let params = [
        ("u64", "pPartial"),
        ("u64", "pOut"),
        ("u64", "pOut16"),
        ("u32", "rows"),
        ("u32", "N"),
        ("u32", "splitStride"),
        ("u32", "outputStride"),
        ("u32", "outputColumnOffset"),
    ];
    p.entry(&name, &params, 0, Some(256), None, false);
    let pm: HashMap<&str, String> = params.iter().map(|&(t, n)| (n, if t == "u64" { p.load_param_u64(n) } else { p.load_param_u32(n) })).collect();
    let tid = p.special("tid.x");
    let ctaid = p.special("ctaid.x");
    let group = p.add32(&tid, p.shl32(&ctaid, 8)); // 8-column group index
    let groups_per_row = p.shr32(&pm["N"], 3);
    let row = p.reg("b32");
    p.emit(format!("div.u32 {row}, {group}, {groups_per_row};"));
    let col = p.shl32(p.sub32(&group, p.mul32(&row, &groups_per_row)), 3);
    let p_ok = p.setp("lt.u32", &row, &pm["rows"]);
    let end_l = p.label("END");
    p.emit(format!("@!{p_ok} bra {end_l};"));
    let base = p.add64(&pm["pPartial"], p.widen(p.shl32(p.add32(p.mul32(&row, &pm["N"]), &col), 1)));
    let mut total = p.regs("b32", 4);
    p.emit(format!("ld.global.cg.v4.b32 {{{}, {}, {}, {}}}, [{base}];", total[0], total[1], total[2], total[3]));
    for s in 1..splits {
        let part = p.regs("b32", 4);
        let addr = p.add64(&base, p.widen(p.mul32(&pm["splitStride"], p.imm32(2 * s))));
        p.emit(format!("ld.global.cg.v4.b32 {{{}, {}, {}, {}}}, [{addr}];", part[0], part[1], part[2], part[3]));
        for i in 0..4 {
            total[i] = hadd2(&p, &total[i], &part[i]);
        }
    }
    if silu_f {
        let k = SiluConsts::new(&p);
        for i in 0..4 {
            total[i] = silu(&p, &total[i], &k);
        }
    }
    let out_idx = p.add32(p.mul32(&row, &pm["outputStride"]), p.add32(&pm["outputColumnOffset"], &col));
    if out_e4 {
        let codes: Vec<String> = total.iter().map(|v| cvt_e4x2(&p, v)).collect();
        let w0 = pack16(&p, &codes[0], &codes[1]);
        let w1 = pack16(&p, &codes[2], &codes[3]);
        let addr = p.add64(&pm["pOut"], p.widen(&out_idx));
        p.emit(format!("st.global.v2.b32 [{addr}], {{{w0}, {w1}}};"));
    }
    if out_f16 {
        let addr = p.add64(&pm["pOut16"], p.widen(p.shl32(&out_idx, 1)));
        p.emit(format!("st.global.v4.b32 [{addr}], {{{}, {}, {}, {}}};", total[0], total[1], total[2], total[3]));
    }
    p.emit(format!("{end_l}:"));
    p.emit("ret;");
    (name, p.finish())
}

/// `<splits> <flags>`
pub fn cli(args: &[&str]) -> String {
    let (splits, flags) = (crate::req(args, 0), crate::req(args, 1));
    generate(splits, flags).1
}
