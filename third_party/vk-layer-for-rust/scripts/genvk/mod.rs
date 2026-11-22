// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified for Neural Forge (2026): ported from Python (vulkan_layer_genvk/__main__.py) to Rust
// (UPSTREAM.md).

//! The generators of the framework's sources from the Vulkan XML API registry (`vk.xml`).

mod global_simple_intercept;
mod layer_trait;
mod registry;
mod vk_xml;

#[cfg(test)]
mod tests;

/// A file generated from `vk.xml`.
#[derive(Clone, Copy)]
pub enum Target {
    /// `layer_trait/generated.rs`
    LayerTrait,
    /// `global_simple_intercept/generated.rs`
    GlobalSimpleIntercept,
}

impl Target {
    pub const ALL: [Target; 2] = [Target::LayerTrait, Target::GlobalSimpleIntercept];

    /// The file's path under `vulkan-layer/src`.
    pub fn path(self) -> &'static str {
        match self {
            Target::LayerTrait => "layer_trait/generated.rs",
            Target::GlobalSimpleIntercept => "global_simple_intercept/generated.rs",
        }
    }
}

/// The contents of `target` (before rustfmt) generated from the registry `vk_xml`.
pub fn generate(vk_xml: &str, target: Target) -> String {
    let document = registry::Registry::parse_document(vk_xml);
    let registry = registry::Registry::new(&document);
    match target {
        Target::LayerTrait => layer_trait::generate(&registry),
        Target::GlobalSimpleIntercept => global_simple_intercept::generate(&registry),
    }
}
