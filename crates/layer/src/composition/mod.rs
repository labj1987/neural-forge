//! The composition pass: the colour math (`color.rs`), its CPU reference (`apply.rs`, which the
//! GPU tests compare against), the proxy encode (`encode.rs`, `encode_pass.rs`) and the GPU
//! pipeline that runs it on every present (`gpu.rs`, `shaders/compose.comp`). See `color.rs`'s
//! module doc comment for what is rederived from where.
#![allow(dead_code)]

pub mod apply;
// The colour matrices are the published constants digit for digit (and match the GLSL twins in
// `compose.comp`), more digits than an f32 holds. Kept as published so they can be checked
// against their sources at a glance; the compiler rounds them the same way either way.
#[allow(clippy::excessive_precision)]
pub mod color;
pub mod encode;
pub mod encode_pass;
pub mod gpu;
