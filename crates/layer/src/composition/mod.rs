//! The composition pass: color math (`color.rs`, `downscale.rs`) plus, eventually,
//! the GPU pipeline that runs it (compute shader, descriptor sets, image resources).
//! See the plan's milestone 4 and `color.rs`'s module doc comment for what's
//! rederived from where.
//!
//! As of 2026-09-10, `apply.rs` wires `color.rs`'s math into `capture.rs`'s real
//! write-back, on the CPU (see that module's own doc comment for exactly what it does
//! and doesn't handle, and the crate's `CLAUDE.md` "composition" entry for the GPU
//! path this is a correctness-first stand-in for). `downscale.rs`'s functions remain
//! genuinely unused outside `#[cfg(test)]` -- nothing calls into the working-scale
//! down-leg yet.
#![allow(dead_code)]

pub mod apply;
// The colour matrices are the published constants digit for digit (and match the GLSL twins in
// `compose.comp`), more digits than an f32 holds. Kept as published so they can be checked
// against their sources at a glance; the compiler rounds them the same way either way.
#[allow(clippy::excessive_precision)]
pub mod color;
pub mod downscale;
pub mod encode;
pub mod encode_pass;
pub mod gpu;
