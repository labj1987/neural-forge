# vk-layer-for-rust, taken into Neural Forge

The framework Neural Forge's Vulkan layer (`crates/layer`) is built on: Google's `vulkan-layer` crate, which
writes a Vulkan layer in safe Rust (dispatch tables, the loader's layer interface, and hooks generated for every
command in the Vulkan registry).

- Upstream: https://github.com/google/vk-layer-for-rust
- Taken from commit `102d87cd9781c045b232e77eef198eed75ce8a8c` (2025-05-27, upstream's newest when it was taken
  in on 2026-10-22).
- License: Apache-2.0 (`LICENSE`; shipped in the AppImage's `THIRD_PARTY_LICENSES.txt`). Every file changed here
  says so in a "Modified for Neural Forge" note under its licence header, as the licence asks.

Until 3.8.6 the layer used it as a git dependency at that commit. Since then this directory holds it and it is
this project's own code: upstream is no longer tracked.

## What is here

- `vulkan-layer/`: the framework. `src/bindings/vk_layer/generated*`, `src/layer_trait/generated.rs` and
  `src/global_simple_intercept/generated.rs` are generated; everything else is written by hand.
- `vulkan-layer-macros/`: its procedural macros (`auto_deviceinfo_impl` and the others).
- `scripts/`: the generator: `codegen.rs` (bindgen for `vk_layer.h`, then the Python generators in
  `vulkan_layer_genvk/` over `vk.xml`, then rustfmt).
- `Cargo.toml`, `Cargo.lock`: their own workspace, apart from Neural Forge's (the root workspace excludes this
  directory and depends on `vulkan-layer` by path). Its tests need their own build settings (they unwind; the
  root workspace aborts on panic) and run one at a time: several share mock expectations held in statics.
- `LICENSE`, `README.md` (upstream's), `rustfmt.toml`.

## Working on it

- Edit the files here directly. Mark each file changed for the first time with a "Modified for Neural Forge"
  note under its licence header, and list notable changes in `CHANGELOG.md`. Credit anything ported from a later
  upstream commit by that commit.
- Check: `cargo clippy --locked --workspace --all-targets [--all-features] -- -D warnings` and
  `cargo test --locked --workspace --all-features -- --test-threads=1`, in this directory. CI runs both.
- The generated files are never edited by hand: change the generator and run `bash scripts/vk-layer-codegen.sh`
  from the repository root (`--check` compares instead of writing; CI runs that). It fetches Vulkan-Headers at
  commit `38899bf6` (1.3.281, the headers ash 0.38.0 was generated from; SHA-256 pinned) once into `tools/native/`,
  and needs libclang and the pinned toolchain's rustfmt. The generated code is written against ash 0.38.0: newer
  headers need an ash generated from them too (ash's own registry must have every type the hooks name).

## Left out when it was taken in

- `examples/hello-world`, the `Android.bp` files, upstream's CI and formatting setup (`.github/`,
  `Makefile.toml`, `Pipfile`, `pyproject.toml`, `taplo.toml`, `yamlfmt.yml`), `CONTRIBUTING.md` (Google's CLA),
  and the `third_party/Vulkan-Headers` submodule (the generator fetches it instead).
- From `scripts/`: `ci.rs` and `list_files/` (upstream's CI runner and licence-header lister) and the
  dependencies only they used.

## Changes made when it was taken in

1. **Workspace.** The members are `vulkan-layer`, `vulkan-layer-macros` and `scripts`; `Cargo.lock` is upstream's
   with the entries of what was left out removed, every other version as upstream had it.
2. **Manifests.** `vulkan-layer` and `vulkan-layer-macros` declare `license = "Apache-2.0"` and upstream as their
   `repository`, with a copy of `LICENSE` in each crate's directory, so the licence listing and cargo-deny find
   both. All three crates are `publish = false`.
3. **Warnings under Rust 1.99** (CI's `-D warnings`; as a git dependency its warnings were not shown):
   `LazyCollection::get` returns `Cow<'_, T>`; the tests compare function pointers by address (`as usize`), as
   upstream's other tests already did; `unstable_api` re-exports `Feature` and `CheckEmpty` only with the
   `unstable` feature, where the module is public (without it nothing used them).
4. **The generator uses only pinned tools.** It formatted with whatever nightly rustfmt was installed; it uses the
   pinned toolchain's now. `rustfmt.toml` keeps only its stable option, and the generator wraps its comment list of
   unhandled commands at 100 columns itself (the one thing nightly's `wrap_comments` changed). The copyright year
   in generated files is fixed at upstream's 2025 instead of the current year. Regenerated, every generated file is
   byte-identical to upstream's.

## Changes since

1. **ash 0.38 and Vulkan 1.3.281** (2026-10-22; was ash 0.37.3 and 1.3.251). The generator names ash 0.38's
   per-extension function tables (`ash::khr::swapchain::{InstanceFn, DeviceFn}` in place of
   `vk::KhrSwapchainFn`), giving instance and device dispatch each its own; it reads `len="1"` (a pointer to one
   element, which newer registries spell out) and knows QNX's `_screen_buffer`. Regenerated, the hooks cover the
   commands of the 30 header versions in between. The hand-written code follows ash 0.38: structures carry the
   lifetime of what they point at (`vk::BaseInStructure<'a>` in the chain iterators, `'static` where the layer's
   loader structures hold raw pointers), the builders' `::builder()`/`.build()` are `::default()` and gone, the
   core function tables are `ash::DeviceFnV1_0` and the others, and the tests' mocks name every lifetime the
   trait elides.
2. **Newer dependencies** (2026-10-26). `Cargo.lock` follows the weekly dependency round like the root one.
   thiserror 2, mockall 0.15, parking_lot 0.13, syn 3, bindgen 0.73 (with prettyplease 0.3) and imara-diff 0.2
   needed no change to the hand-written code except the generator's diff printer (imara-diff 0.2's
   `Diff::compute` and `unified_diff`). bindgen 0.73 writes the bindings' layout checks as compile-time
   assertions in place of `#[test]` functions; regenerated, nothing else in the generated files changed.
3. **Fixes from the 2026-10-28 review.** `vkEnumeratePhysicalDeviceGroups` reads the groups only when the caller
   passed an array for them: a count query (the array NULL, as an application asks how many there are) read
   through the NULL pointer. The mock driver enumerates one group per physical device, and a test makes both
   queries. With a create-device hook that returns `Unhandled`, the extension list the next `vkCreateDevice` was
   given (the application's, without the layer's own) is kept until `create_device_info` has read the same create
   info: it pointed at a list already freed. `vkDestroyDevice` and `vkDestroyInstance` drop the layer's own
   device and instance data before they call the next layer's: its Drop (how a layer sees the destroy) ran on a
   device or instance already destroyed. Tests with two layers check the order.
