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
// Modified for Neural Forge (2026): a hook method whose command has success codes other than
// VK_SUCCESS that come with its output says it cannot return them; ported from Python
// (layer_trait_generator.py) to Rust (UPSTREAM.md).

//! Generates `layer_trait/generated.rs`: the `InstanceHooks` and `DeviceHooks` traits, with a
//! method for every command, and the `VulkanCommand` enum.

use super::registry::{Command, Registry};
use super::vk_xml::{
    generate_unhandled_command_comments, snake_case_to_upper_camel_case, write_preamble,
    DispatchChainType, OrderedMap, RustMethod, UnhandledCommand, VkXmlCommand,
    VkXmlToRustMethodInfo, VulkanAliases,
};
use std::collections::HashMap;

#[derive(Default)]
struct LayerTraitGenerator {
    all_commands: Vec<(DispatchChainType, OrderedMap<RustMethod>)>,
    command_aliases: VulkanAliases,
    unhandled_commands: OrderedMap<UnhandledCommand>,
    /// A command's success codes besides VK_SUCCESS, and VK_INCOMPLETE (which the framework
    /// returns itself for a count and array pair).
    other_success_codes: HashMap<String, Vec<String>>,
}

impl LayerTraitGenerator {
    /// The doc comment of a hook whose command reports a success code other than VK_SUCCESS
    /// together with its output (VK_SUBOPTIMAL_KHR with an image index,
    /// VK_PIPELINE_COMPILE_REQUIRED with pipelines): the hook's VkResult<T> cannot carry both.
    fn limitation_doc_lines(&self, command: &RustMethod) -> Vec<String> {
        let codes = self
            .other_success_codes
            .get(&command.vk_xml_cmd.name)
            .map(Vec::as_slice)
            .unwrap_or_default();
        if command.vk_xml_cmd.return_type != "VkResult" || codes.is_empty() {
            return vec![];
        }
        let return_info =
            VkXmlToRustMethodInfo::from_vk_xml_command(&command.vk_xml_cmd).return_info;
        if return_info.main_source_vk_xml_params.is_empty() {
            return vec![];
        }
        let codes_text: Vec<String> = codes.iter().map(|code| format!("`{code}`")).collect();
        vec![
            format!(
                "    /// `{}` can also succeed with {}, but a hook",
                command.vk_xml_cmd.name,
                codes_text.join(", ")
            ),
            "    /// cannot return one of those with the output: `Ok` reaches the application as".to_owned(),
            "    /// `VK_SUCCESS`, and `Err(code)` returns the code without the output (an array of".to_owned(),
            "    /// handles the application sized is set to `VK_NULL_HANDLE`). A hook that needs such a".to_owned(),
            "    /// code with its output must return `LayerResult::Unhandled`.".to_owned(),
        ]
    }

    fn begin_file(&self, out: &mut String) {
        write_preamble(out);
        out.push('\n');
        out.push_str("// This file is generated from the Vulkan XML API registry.\n");
        out.push_str("#![allow(clippy::too_many_arguments)]\n");
        out.push_str(
            &[
                "use std::{ffi::{c_int, c_void}, mem::MaybeUninit};",
                "",
                "use ash::{vk, prelude::VkResult};",
                "",
                "use crate::VkLayerDeviceLink;",
                "use super::{LayerResult, TryFromVulkanCommandError};",
            ]
            .join("\n"),
        );
        out.push_str("\n\n");
    }

    fn end_file(&self, out: &mut String) {
        let mut not_aliased_commands = std::collections::HashSet::new();
        for (_, commands) in &self.all_commands {
            for command in commands.keys() {
                let represent_name = self
                    .command_aliases
                    .get_represent_name(command)
                    .unwrap_or_else(|| panic!("{command} is never added to the command aliases"));
                if &represent_name == command {
                    not_aliased_commands.insert(command.clone());
                }
            }
        }

        out.push_str(&generate_unhandled_command_comments(
            self.unhandled_commands.values(),
        ));

        let mut trait_lines: Vec<(DispatchChainType, Vec<String>)> = vec![
            (
                DispatchChainType::Instance,
                vec!["pub trait InstanceHooks: Send + Sync {".to_owned()],
            ),
            (
                DispatchChainType::Device,
                vec!["pub trait DeviceHooks: Send + Sync {".to_owned()],
            ),
        ];
        let mut command_enum: Vec<String> = vec![
            "#[derive(Debug, PartialOrd, Ord, PartialEq, Eq, Hash, Clone)]".to_owned(),
            "pub enum VulkanCommand {".to_owned(),
        ];
        let mut try_from_command_impl: Vec<String> = [
            "impl TryFrom<&str> for VulkanCommand {",
            "    type Error = TryFromVulkanCommandError;",
            "",
            "    fn try_from(value: &str) -> Result<Self, Self::Error> {",
            "        match value {",
        ]
        .map(str::to_owned)
        .to_vec();

        for (dispatch_type, commands) in &self.all_commands {
            for (name, command) in commands.iter() {
                if !not_aliased_commands.contains(name) {
                    continue;
                }
                if let Some((_, lines)) = trait_lines.iter_mut().find(|(ty, _)| ty == dispatch_type)
                {
                    lines.extend(self.limitation_doc_lines(command));
                    lines.push(format!("    {} {{", command.to_rust_string()));
                    lines.push("        LayerResult::Unhandled".to_owned());
                    lines.push("    }".to_owned());
                }
                let original_name = &command.vk_xml_cmd.name;
                let enum_variant_name = snake_case_to_upper_camel_case(&command.name);
                command_enum.push(format!("    {enum_variant_name},"));
                try_from_command_impl.push(format!(
                    "            \"{original_name}\" => Ok(VulkanCommand::{enum_variant_name}),"
                ));
            }
        }
        command_enum.push("}".to_owned());
        try_from_command_impl.extend(
            [
                "            _ => Err(TryFromVulkanCommandError::UnknownCommand(value.to_owned())),",
                "        }",
                "    }",
                "}",
            ]
            .map(str::to_owned),
        );
        out.push_str(&command_enum.join("\n"));
        out.push_str("\n\n");
        out.push_str(&try_from_command_impl.join("\n"));
        out.push_str("\n\n");

        trait_lines.sort_by_key(|(dispatch_type, _)| *dispatch_type);
        for (_, lines) in trait_lines {
            out.push_str(&lines.join("\n"));
            out.push('\n');
            out.push_str("}\n");
            out.push('\n');
        }
    }

    fn gen_cmd(&mut self, command: &Command, registry: &Registry) {
        let name = command.name.as_str();
        if let Some(unhandled_command) = UnhandledCommand::find(name) {
            self.unhandled_commands
                .insert(&unhandled_command.name.clone(), unhandled_command);
            return;
        }

        const SHOULD_SKIP: [&str; 9] = [
            // Should be intercepted through Drop trait of InstanceInfo and DeviceInfo
            "vkDestroyInstance",
            "vkDestroyDevice",
            // Should be intercepted when the PhysicalDeviceInfo is created
            "vkEnumeratePhysicalDevices",
            "vkEnumeratePhysicalDeviceGroups",
            // Layer and extension properties should be provided via trait consts
            "vkEnumerateDeviceLayerProperties",
            "vkEnumerateDeviceExtensionProperties",
            // TODO: pre-instance functions are not supported now, but such support should be added.
            "vkEnumerateInstanceExtensionProperties",
            "vkEnumerateInstanceLayerProperties",
            "vkEnumerateInstanceVersion",
        ];
        if SHOULD_SKIP.contains(&name) {
            return;
        }

        let vk_xml_cmd = VkXmlCommand::from_command(command, registry);
        let Some(dispatch_chain_type) = vk_xml_cmd.get_dispatch_chain_type() else {
            self.unhandled_commands.insert(
                name,
                UnhandledCommand::new(name, "Unknown dispatch chain type"),
            );
            return;
        };

        self.command_aliases
            .add_alias(name, command.alias.as_deref());
        self.other_success_codes.insert(
            name.to_owned(),
            command
                .success_codes
                .iter()
                .filter(|code| {
                    !code.is_empty() && *code != "VK_SUCCESS" && *code != "VK_INCOMPLETE"
                })
                .cloned()
                .collect(),
        );
        let rust_method = RustMethod::from_vk_xml_command(&vk_xml_cmd);
        match self
            .all_commands
            .iter_mut()
            .find(|(dispatch_type, _)| *dispatch_type == dispatch_chain_type)
        {
            Some((_, commands)) => commands.insert(name, rust_method),
            None => {
                let mut commands = OrderedMap::default();
                commands.insert(name, rust_method);
                self.all_commands.push((dispatch_chain_type, commands));
            }
        }
    }
}

/// The contents of `layer_trait/generated.rs`, before rustfmt.
pub fn generate(registry: &Registry) -> String {
    let mut generator = LayerTraitGenerator::default();
    let mut out = String::new();
    generator.begin_file(&mut out);
    for (_, commands) in &registry.features {
        for command in commands {
            generator.gen_cmd(command, registry);
        }
    }
    generator.end_file(&mut out);
    out
}
