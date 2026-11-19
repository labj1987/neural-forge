//! OpenDLSS-NR's PTX kernel generators: every PTX kernel the native backend embeds is emitted here, as text,
//! by plain Rust (no CUDA toolkit; the driver JIT-compiles the PTX). `crates/native/build.rs` calls
//! [`kernels`] and embeds the result.
//!
//! One module per generator, each with the `generate` function that emits the kernel and a `cli` that takes the
//! generator's arguments as strings (the same order and defaults as its command-line form in
//! upstream's `scripts/ptx/*.py`, without the output path). `ptxgen` is the emitter, `swin` the shared fragment-level
//! building blocks. `cargo run -p opendlss-nr-ptx --example write_ptx DIR` writes every kernel to DIR, to diff the
//! generated PTX before and after a change.
//!
//! Ported from OpenDLSS-NR by maan (<https://github.com/maanHimself/OpenDLSS-NR>, `scripts/ptx`), MIT: like the rest
//! of `third_party/opendlss-nr`, this crate stays under that licence (`../LICENSE`, `../NOTICE`).

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

pub mod block32_e4m3;
pub mod ffn_e4m3;
pub mod gemm2_e4m3;
pub mod gemmt_e4m3;
pub mod gemmv_e4m3;
pub mod global_attention_e4m3;
pub mod global_attention_stream_e4m3;
pub mod mlp_e4m3;
pub mod ptxgen;
pub mod qkv_e4m3;
pub mod reduce_e4m3;
pub mod swin;

/// One kernel: its name (the embedded asset's name), the generator and the generator's arguments.
pub struct Kernel {
    pub stem: String,
    pub generator: &'static str,
    pub args: Vec<String>,
}

impl Kernel {
    /// The kernel's PTX.
    pub fn generate(&self) -> String {
        let args: Vec<&str> = self.args.iter().map(String::as_str).collect();
        let cli = match self.generator {
            "block32_e4m3" => block32_e4m3::cli,
            "ffn_e4m3" => ffn_e4m3::cli,
            "gemm2_e4m3" => gemm2_e4m3::cli,
            "gemmt_e4m3" => gemmt_e4m3::cli,
            "gemmv_e4m3" => gemmv_e4m3::cli,
            "global_attention_e4m3" => global_attention_e4m3::cli,
            "global_attention_stream_e4m3" => global_attention_stream_e4m3::cli,
            "mlp_e4m3" => mlp_e4m3::cli,
            "qkv_e4m3" => qkv_e4m3::cli,
            "reduce_e4m3" => reduce_e4m3::cli,
            other => panic!("unknown PTX generator {other}"),
        };
        cli(&args)
    }
}

/// A generator argument as a number (the command-line forms' `int(...)`).
pub(crate) fn int(arg: &str) -> usize {
    arg.parse().unwrap_or_else(|_| panic!("PTX generator argument {arg:?} is not a non-negative integer"))
}

/// A required generator argument as a number: a clear error when it is missing.
pub(crate) fn req(args: &[&str], index: usize) -> usize {
    int(req_str(args, index))
}

/// A required generator argument as text: a clear error when it is missing.
pub(crate) fn req_str<'a>(args: &[&'a str], index: usize) -> &'a str {
    args.get(index).copied().unwrap_or_else(|| panic!("PTX generator needs argument {} (got {} arguments)", index + 1, args.len()))
}

/// A register cap: a `u32`, an error when it does not fit.
pub(crate) fn reg(arg: &str) -> u32 {
    arg.parse().unwrap_or_else(|_| panic!("PTX generator register cap {arg:?} is not a 32-bit non-negative integer"))
}

/// Every kernel the native backend embeds. The same generator invocations, in the same order, as upstream's
/// build_shaders.ps1, except one register cap: the 128-channel FFN kernels are capped at 64 registers (upstream:
/// none), so two of their 512-thread workgroups fit on an SM instead of one (88 and 84 registers); bit-exact, and
/// 0.20 ms (full network) and 0.06 ms (small) faster at 1486x836 on the RTX 5070 (docs/DLSS_KERNEL_CATALOGUE.md,
/// "Register caps"); and the 256-channel FFN kernels at 3 row tiles left out: they run at 4 since 4 is faster
/// ("Row tiles").
pub fn kernels() -> Vec<Kernel> {
    let mut v = Vec::new();
    let mut add = |generator: &'static str, stem: String, args: &[&str]| {
        v.push(Kernel { stem, generator, args: args.iter().map(|a| a.to_string()).collect() });
    };
    for k in ["64", "128", "256"] {
        add("mlp_e4m3", format!("mlp_e4m3_K{k}"), &[k, "3", "1", "1"]);
    }
    for c in ["64", "128", "256", "512"] {
        add("qkv_e4m3", format!("qkv_e4m3_K{c}"), &[c, "80"]);
    }
    for k in ["32", "64", "128", "256", "512", "1024"] {
        for f in ["5", "13", "4", "8", "6"] {
            add("gemm2_e4m3", format!("gemm2_e4m3_K{k}_f{f}"), &[k, f]);
        }
    }
    for [c, r, x] in [["64", "4", "0"], ["128", "4", "64"], ["256", "4", "64"]] {
        add("ffn_e4m3", format!("ffn_e4m3_C{c}_R{r}"), &[c, r, x]);
        add("ffn_e4m3", format!("ffn_e4m3_C{c}_R{r}_proj"), &[c, r, x, "1"]);
    }
    for [k, f, p, s] in [["4096", "1", "32", "4"], ["1024", "0", "16", "2"], ["1024", "1", "8", "4"], ["1024", "0", "8", "4"]] {
        add("gemmt_e4m3", format!("gemmt_e4m3_K{k}_f{f}_p{p}_s{s}"), &[k, f, p, s]);
    }
    for [s, f] in [["4", "4"], ["2", "8"], ["4", "8"]] {
        add("reduce_e4m3", format!("reduce_e4m3_s{s}_f{f}"), &[s, f]);
    }
    for [k, f, s] in [["1024", "6", "1"], ["4096", "5", "4"], ["1024", "8", "2"], ["1024", "5", "4"]] {
        add("gemmv_e4m3", format!("gemmv_e4m3_K{k}_f{f}_s{s}"), &[k, f, s]);
        add("gemmv_e4m3", format!("gemmv_e4m3_K{k}_f{f}_s{s}_m96"), &[k, f, s, "4", "6", "0", "0", "0", "96"]);
    }
    for padded in ["64", "128", "192", "256"] {
        add("global_attention_e4m3", format!("global_attention_e4m3_p{padded}"), &[padded]);
    }
    add("global_attention_stream_e4m3", "global_normalize_e4m3".into(), &["normalize"]);
    add("global_attention_stream_e4m3", "global_attention_stream_e4m3".into(), &["attention", "4"]);
    for [k, f] in [["512", "4"], ["512", "5"], ["512", "13"], ["256", "5"], ["4096", "5"], ["1024", "8"], ["1024", "5"]] {
        add("gemmv_e4m3", format!("gemmv_e4m3_K{k}_f{f}_s1_m96"), &[k, f, "1", "4", "6", "0", "0", "0", "96"]);
    }
    for f in ["2", "66", "74", "130", "48"] {
        add("block32_e4m3", format!("block32_e4m3_f{f}"), &[f]);
    }
    v
}
