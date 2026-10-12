//! The composition pass: the colour math (`color.rs`), the proxy encode (`encode.rs`,
//! `encode_pass.rs`) and the GPU pipeline that runs it on every present (`gpu.rs`,
//! `shaders/compose.comp`). See `color.rs`'s module doc comment for what is rederived from where.

// The colour matrices are the published constants digit for digit (and match the GLSL twins in
// `compose.comp`), more digits than an f32 holds. Kept as published so they can be checked
// against their sources at a glance; the compiler rounds them the same way either way.
#[allow(clippy::excessive_precision)]
// The clean-room expression of the colour math that `compose.comp` is hand-translated from
// (ATTRIBUTION.md): kept, with its tests, as the reference the shader's provenance rests on. Only
// `luminance` is called from Rust; nothing composes on the CPU.
#[allow(dead_code)]
pub mod color;
pub mod encode;
pub mod encode_pass;
pub mod gpu;
