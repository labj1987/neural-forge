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
// Modified for Neural Forge (2026): ported to ash 0.38; the command alias table; whether an extension is a
// device one; a hook's error leaves NULL in every entry of an output array of handles; ported from Python
// (global_simple_intercept_generator.py) to Rust (UPSTREAM.md).

//! Generates `global_simple_intercept/generated.rs`: the `Extension` enum, the dispatch tables, the
//! command alias table and the `extern "system"` entry point of every command, which calls the
//! layer's hook and falls through to the next layer when the hook leaves the command unhandled.

use super::registry::{children, Command, Feature, Registry};
use super::vk_xml::{
    camel_case_to_snake_case, decayed_type_to_rust_type, generate_unhandled_command_comments,
    snake_case_to_upper_camel_case, write_preamble, DispatchChainType, OrderedMap, RustMethod,
    RustParam, RustType, UnhandledCommand, VkXmlCommand, VkXmlLenKind, VkXmlParam,
    VkXmlToRustMethodInfo, VkXmlType, VulkanAliases,
};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::rc::Rc;

/// The extensions whose dispatch tables ash marks deprecated.
const DEPRECATED_EXTENSIONS: [&str; 1] = ["VK_EXT_debug_report"];

/// Where the commands of a core version or an extension are dispatched.
struct CommandDispatchInfo {
    /// A core version's (major, minor).
    core_info: Option<(u32, u32)>,
    extension_info: Option<ExtensionInfo>,
    commands: BTreeSet<String>,
}

struct ExtensionInfo {
    name: String,
    /// The registry's type="device" (else "instance").
    is_device: bool,
}

impl CommandDispatchInfo {
    fn from_feature(feature: &Feature) -> CommandDispatchInfo {
        let mut core_info = None;
        let mut extension_info = None;
        if feature.is_version {
            let mut version_number = feature.version_number.split('.');
            let mut part = || -> u32 {
                version_number
                    .next()
                    .and_then(|part| part.parse().ok())
                    .unwrap_or_else(|| {
                        panic!("{}: bad version {}", feature.name, feature.version_number)
                    })
            };
            core_info = Some((part(), part()));
        } else {
            let extension_type = feature.node.attribute("type");
            assert!(
                matches!(extension_type, Some("device" | "instance")),
                "{} is neither a device nor an instance extension",
                feature.name
            );
            extension_info = Some(ExtensionInfo {
                name: feature.name.clone(),
                is_device: extension_type == Some("device"),
            });
        }
        let mut commands = BTreeSet::new();
        for require in children(feature.node, "require") {
            for command in children(require, "command") {
                let name = command.attribute("name").unwrap_or_else(|| {
                    panic!(
                        "Failed to find the name of a command for {}",
                        feature.category
                    )
                });
                if UnhandledCommand::find(name).is_none() {
                    commands.insert(name.to_owned());
                }
            }
        }
        CommandDispatchInfo {
            core_info,
            extension_info,
            commands,
        }
    }

    fn get_dispatch_table_field_name(&self) -> String {
        match (&self.core_info, &self.extension_info) {
            (Some((major, minor)), _) => format!("core.fp_v{major}_{minor}()"),
            (None, Some(extension_info)) => extension_info.name[3..].to_lowercase(),
            (None, None) => unreachable!("CommandDispatchInfo should either be core or extension."),
        }
    }

    fn enum_variant_name(&self) -> String {
        let extension_info = self
            .extension_info
            .as_ref()
            .expect("Only extension is defined as enum variant");
        let name = &extension_info.name;
        snake_case_to_upper_camel_case(name.strip_prefix("VK_").unwrap_or(name))
    }

    /// The enum with the path, e.g. `Feature::Core(ApiVersion { major: 1, minor: 1})`.
    fn get_enum_name(&self) -> String {
        match &self.core_info {
            Some((major, minor)) => {
                format!("Feature::Core(ApiVersion {{ major: {major}, minor: {minor}}})")
            }
            None => format!(
                "Feature::Extension(Extension::{})",
                self.enum_variant_name()
            ),
        }
    }

    fn get_feature_enum_lines<'d>(
        dispatch_infos: impl Iterator<Item = &'d CommandDispatchInfo>,
    ) -> Vec<String> {
        let mut extension_enum_lines: Vec<String> = [
            "#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]",
            "pub enum Extension {",
        ]
        .map(str::to_owned)
        .to_vec();
        let mut extension_enum_try_from_lines: Vec<String> = [
            "impl TryFrom<&str> for Extension {",
            "    type Error = TryFromExtensionError;",
            "",
            "    fn try_from(value: &str) -> Result<Self, Self::Error> {",
            "        match value {",
        ]
        .map(str::to_owned)
        .to_vec();
        let mut str_from_extension_enum_lines: Vec<String> = [
            "impl From<Extension> for &'static str {",
            "    fn from(value: Extension) -> &'static str {",
            "         match value {",
        ]
        .map(str::to_owned)
        .to_vec();
        let mut device_variants = vec![];
        for dispatch_info in dispatch_infos {
            let Some(extension_info) = &dispatch_info.extension_info else {
                continue;
            };
            let variant_name = dispatch_info.enum_variant_name();
            extension_enum_lines.push(format!("    {variant_name},"));
            extension_enum_try_from_lines.push(format!(
                "            \"{}\" => Ok(Extension::{variant_name}),",
                extension_info.name
            ));
            str_from_extension_enum_lines.push(format!(
                "            Extension::{variant_name} => \"{}\",",
                extension_info.name
            ));
            if extension_info.is_device {
                device_variants.push(format!("Extension::{variant_name}"));
            }
        }

        extension_enum_lines.push("}".to_owned());
        extension_enum_try_from_lines.extend(
            [
                "            _ => Err(TryFromExtensionError::UnknownExtension(value.to_owned()))",
                "        }",
                "    }",
                "}",
            ]
            .map(str::to_owned),
        );
        str_from_extension_enum_lines.extend(["        }", "    }", "}"].map(str::to_owned));
        let is_device_lines = vec![
            "impl Extension {".to_owned(),
            "    /// Whether this is a device extension (the registry's `type=\"device\"`), not an instance one."
                .to_owned(),
            "    pub fn is_device_extension(&self) -> bool {".to_owned(),
            format!("        matches!(self, {})", device_variants.join(" | ")),
            "    }".to_owned(),
            "}".to_owned(),
        ];
        let separator = || vec!["\n".to_owned()];
        [
            extension_enum_lines,
            separator(),
            extension_enum_try_from_lines,
            separator(),
            str_from_extension_enum_lines,
            separator(),
            is_device_lines,
            separator(),
        ]
        .concat()
    }

    fn get_dispatch_tables_impl_lines(
        dispatch_infos: &[&CommandDispatchInfo],
        instance_commands: &OrderedMap<Rc<VulkanCommand>>,
        device_commands: &OrderedMap<Rc<VulkanCommand>>,
        skip_commands: &HashSet<String>,
    ) -> Vec<String> {
        let has_unskipped_commands = |info: &CommandDispatchInfo| {
            info.commands
                .iter()
                .any(|command| !skip_commands.contains(command))
        };
        let mut device_dispatch_def_lines: Vec<String> = [
            "pub(crate) struct DeviceDispatchTable {",
            "    pub core: Arc<ash::Device>,",
        ]
        .map(str::to_owned)
        .to_vec();
        let mut device_dispatch_impl_lines: Vec<String> = [
            "impl DeviceDispatchTable {",
            "    pub(crate) fn load(get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr, device: Arc<ash::Device>) -> Self {",
            "        let proc_addr_loader = get_device_proc_addr_loader(get_device_proc_addr, &device);",
            "        Self {",
            "            core: Arc::clone(&device),",
        ]
        .map(str::to_owned)
        .to_vec();
        let mut instance_dispatch_def_lines: Vec<String> = [
            "pub(crate) struct InstanceDispatchTable {",
            "    pub core: Arc<ash::Instance>,",
        ]
        .map(str::to_owned)
        .to_vec();
        let mut instance_dispatch_impl_lines: Vec<String> = [
            "impl InstanceDispatchTable {",
            "    pub(crate) fn load(get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr, instance: Arc<ash::Instance>) -> Self {",
            "        let proc_addr_loader = get_instance_proc_addr_loader(get_instance_proc_addr, &instance);",
            "        Self {",
            "            core: Arc::clone(&instance),",
        ]
        .map(str::to_owned)
        .to_vec();

        for dispatch_info in dispatch_infos
            .iter()
            .filter(|info| has_unskipped_commands(info))
        {
            if dispatch_info.core_info.is_some() {
                continue;
            }
            let extension_info = dispatch_info.extension_info.as_ref().expect(
                "A CommandDispatchInfo should be either a core dispatch or an extension dispatch.",
            );

            // Remove the VK_ prefix
            let field_name = dispatch_info.get_dispatch_table_field_name();
            // ash 0.38 keeps each extension's function tables in a module named by its vendor and the
            // rest of its name (VK_KHR_swapchain: ash::khr::swapchain), one for the instance-level
            // commands (InstanceFn) and one for the device-level ones (DeviceFn).
            let (vendor, module) = field_name
                .split_once('_')
                .unwrap_or_else(|| panic!("{field_name} has no vendor"));
            let module = if module.starts_with(|c: char| c.is_ascii_digit()) {
                format!("_{module}")
            } else {
                module.to_owned()
            };

            let lines = |table: &str| -> (Vec<String>, Vec<String>) {
                let field_type = format!("ash::{vendor}::{module}::{table}");
                let mut def_lines = vec![];
                if DEPRECATED_EXTENSIONS.contains(&extension_info.name.as_str()) {
                    def_lines.push("    #[allow(deprecated)]".to_owned());
                }
                def_lines.push(format!("    {field_name}: Arc<{field_type}>,"));
                let impl_lines = vec![format!(
                    "            {field_name}: Arc::new({field_type}::load(&proc_addr_loader)),"
                )];
                (def_lines, impl_lines)
            };

            if dispatch_info
                .commands
                .iter()
                .any(|command| instance_commands.contains_key(command))
            {
                let (def_lines, impl_lines) = lines("InstanceFn");
                instance_dispatch_def_lines.extend(def_lines);
                instance_dispatch_impl_lines.extend(impl_lines);
            }
            if dispatch_info
                .commands
                .iter()
                .any(|command| device_commands.contains_key(command))
            {
                let (def_lines, impl_lines) = lines("DeviceFn");
                device_dispatch_def_lines.extend(def_lines);
                device_dispatch_impl_lines.extend(impl_lines);
            }
        }

        device_dispatch_def_lines.push("}".to_owned());
        device_dispatch_impl_lines.extend(["        }", "    }", "}"].map(str::to_owned));
        instance_dispatch_def_lines.push("}".to_owned());
        instance_dispatch_impl_lines.extend(["        }", "    }", "}"].map(str::to_owned));

        [
            device_dispatch_def_lines,
            device_dispatch_impl_lines,
            instance_dispatch_def_lines,
            instance_dispatch_impl_lines,
        ]
        .concat()
    }
}

struct RustFfiFunction {
    name: String,
    return_type: String,
    parameters: Vec<RustParam>,
}

impl RustFfiFunction {
    fn from_vk_xml_command(vk_xml_cmd: &VkXmlCommand) -> RustFfiFunction {
        let c_return_type = vk_xml_cmd.return_type.as_str();
        let return_type = if c_return_type == "void" {
            "()".to_owned()
        } else {
            decayed_type_to_rust_type(c_return_type)
        };
        let name = &vk_xml_cmd.name;
        RustFfiFunction {
            name: camel_case_to_snake_case(name.strip_prefix("vk").unwrap_or(name)),
            return_type,
            parameters: vk_xml_cmd
                .parameters
                .iter()
                .map(RustParam::from_vk_xml_param_for_ffi)
                .collect(),
        }
    }

    fn get_def_str(&self) -> String {
        let params: Vec<String> = self
            .parameters
            .iter()
            .map(RustParam::to_rust_string)
            .collect();
        let fn_return_type = if self.return_type == "()" {
            String::new()
        } else {
            format!(" -> {}", self.return_type)
        };
        format!(
            "extern \"system\" fn {}({}){fn_return_type}",
            self.name,
            params.join(", ")
        )
    }
}

struct VulkanCommand {
    vk_xml_command: VkXmlCommand,
    rust_fn: RustFfiFunction,
}

#[derive(PartialEq)]
enum LastExprType {
    Unit,
    VkResult,
}

impl VulkanCommand {
    fn find_param_by_name(&self, name: &str) -> Option<(&VkXmlParam, &RustParam)> {
        self.vk_xml_command
            .parameters
            .iter()
            .zip(&self.rust_fn.parameters)
            .find(|(param, _)| param.name == name)
    }

    /// The lines that assign the variable `ret_var` to the FFI return parameter, e.g. the result of
    /// create_image to the p_image pointer of the FFI parameter.
    ///
    /// With `null_fill`, the lines for a hook's error instead: an output array of handles whose
    /// length the application gave (vkAllocateCommandBuffers, vkAllocateDescriptorSets,
    /// vkCreate*Pipelines, vkCreateSharedSwapchainsKHR) gets VK_NULL_HANDLE in every entry, as the
    /// spec asks of a failed allocation or creation (the application may free or destroy every entry
    /// of the array after the error); no lines for any other output.
    fn generate_assign_to_ret_param_lines(
        &self,
        ret_var: &str,
        ret_param_index: usize,
        last_expr_type: LastExprType,
        null_fill: bool,
    ) -> Vec<String> {
        assert!(
            ret_param_index < self.vk_xml_command.parameters.len(),
            "Parameter index out of range."
        );
        let ret_xml_param = &self.vk_xml_command.parameters[ret_param_index];
        let ret_rust_param = &self.rust_fn.parameters[ret_param_index];
        let ret_param_xml_type = &ret_xml_param.ty;
        let points_to = ret_param_xml_type.points_to().unwrap_or_else(|| {
            panic!("{} returns through a non-pointer", self.vk_xml_command.name)
        });
        match ret_param_xml_type.len {
            Some(VkXmlLenKind::Variable) => {
                let len_var = ret_xml_param
                    .len_var
                    .as_deref()
                    .expect("A variable length without a len");
                let mut need_to_assign_len_param = false;
                let len_expr = if len_var.contains("->") {
                    let len_components: Vec<&str> = len_var.split("->").collect();
                    assert!(
                        len_components.len() == 2,
                        "Unsupported length parameter: {len_var}"
                    );
                    let (len_struct_name, len_field_name) = (len_components[0], len_components[1]);
                    let (len_struct_xml_param, len_struct_rust_param) =
                        self.find_param_by_name(len_struct_name).unwrap_or_else(|| {
                            panic!("Failed to find the length parameter: {len_struct_name}")
                        });
                    assert!(
                        !len_struct_xml_param.ty.is_optional,
                        "The length of {}, {len_struct_name} in {len_var}, is optional. This is currently not supported.",
                        ret_xml_param.name
                    );
                    format!(
                        "unsafe {{ {}.as_ref().unwrap() }}.{}.try_into().unwrap()",
                        len_struct_rust_param.name,
                        camel_case_to_snake_case(len_field_name)
                    )
                } else {
                    let (len_xml_param, len_rust_param) =
                        self.find_param_by_name(len_var).unwrap_or_else(|| {
                            panic!("Failed to find the length parameter: {len_var}.")
                        });
                    let len_xml_type = &len_xml_param.ty;
                    assert!(
                        !len_xml_type.is_optional,
                        "Can't handle optional length for now."
                    );
                    if matches!(len_xml_type.name(), Some("uint32_t" | "size_t")) {
                        if len_xml_type.name() != Some("size_t") {
                            format!("{}.try_into().unwrap()", len_rust_param.name)
                        } else {
                            len_rust_param.name.clone()
                        }
                    } else if len_xml_type.points_to().is_some_and(|points_to| {
                        matches!(points_to.name(), Some("uint32_t" | "size_t"))
                    }) {
                        need_to_assign_len_param = true;
                        len_rust_param.name.clone()
                    } else {
                        panic!("Unhandled length parameter type: {len_xml_type:?}");
                    }
                };

                if null_fill {
                    if need_to_assign_len_param || points_to.name() == Some("void") {
                        return vec![];
                    }
                    return vec![format!(
                        "unsafe {{ std::slice::from_raw_parts_mut({}, {len_expr}) }}.fill(vk::Handle::from_raw(0));",
                        ret_rust_param.name
                    )];
                }
                let mut lines = vec![];
                if points_to.name() == Some("void") {
                    lines.push(format!("let {0} = {0} as *mut u8;", ret_rust_param.name));
                }
                if !need_to_assign_len_param {
                    lines.push(format!(
                        "unsafe {{ std::slice::from_raw_parts_mut({}, {len_expr}) }}.copy_from_slice(&{ret_var});",
                        ret_rust_param.name
                    ));
                    if last_expr_type == LastExprType::VkResult {
                        lines.push("vk::Result::SUCCESS".to_owned());
                    }
                    return lines;
                }
                let mut fill_array_lines = vec![
                    "    fill_vk_out_array(".to_owned(),
                    format!("        &{ret_var},"),
                    format!("        NonNull::new({len_expr}).unwrap(),"),
                    format!("        {},", ret_rust_param.name),
                    "    )".to_owned(),
                ];
                if last_expr_type == LastExprType::VkResult {
                    lines.push("unsafe {".to_owned());
                    lines.extend(fill_array_lines);
                    lines.push("}".to_owned());
                } else {
                    if let Some(last) = fill_array_lines.last_mut() {
                        last.push(';');
                    }
                    lines.push(format!(
                        "// We can't return INCOMPLETE from {}",
                        self.vk_xml_command.name
                    ));
                    lines.push("#[allow(unused_must_use)]".to_owned());
                    lines.push("unsafe {".to_owned());
                    lines.extend(fill_array_lines);
                    lines.push("}".to_owned());
                }
                lines
            }
            None => {
                if null_fill {
                    return vec![];
                }
                assert!(
                    !ret_param_xml_type.is_optional,
                    "Optional return type in {} is not supported.",
                    self.vk_xml_command.name
                );
                let mut ret_expr = ret_var.to_owned();
                if points_to.name() == Some("VkBool32") {
                    ret_expr = format!("if {ret_var} {{ vk::TRUE }} else {{ vk::FALSE }}");
                } else if let Some(inner) = points_to.points_to().filter(|_| points_to.is_optional)
                {
                    assert!(
                        !inner.is_const,
                        "Optional const pointer return value is not supported."
                    );
                    ret_expr = format!("{ret_var}.unwrap_or(std::ptr::null_mut())");
                }
                let mut lines = vec![format!(
                    "*unsafe {{ {}.as_mut() }}.unwrap() = {ret_expr};",
                    ret_rust_param.name
                )];
                if last_expr_type == LastExprType::VkResult {
                    lines.push("vk::Result::SUCCESS".to_owned());
                }
                lines
            }
            Some(VkXmlLenKind::NullTerminated) => {
                panic!("Unsupported return type with length type: NullTerminated")
            }
        }
    }

    fn get_rust_fn_impl_lines<'d>(
        &self,
        dispatch_infos: impl Iterator<Item = &'d CommandDispatchInfo>,
    ) -> Vec<String> {
        let mut lines = vec![
            "let global = Self::instance();".to_owned(),
            format!("// {}", self.vk_xml_command.name),
        ];
        assert!(
            !self.vk_xml_command.parameters.is_empty(),
            "Vulkan commands should have at least one parameter."
        );
        let dispatch_chain_var = match self.vk_xml_command.get_dispatch_chain_type() {
            Some(DispatchChainType::Device) => {
                lines.push(format!(
                    "let device_info = global.get_device_info({}).unwrap();",
                    self.rust_fn.parameters[0].name
                ));
                "device_info"
            }
            Some(DispatchChainType::Instance) => {
                lines.push(format!(
                    "let instance_info = global.get_instance_info({}).unwrap();",
                    self.rust_fn.parameters[0].name
                ));
                "instance_info"
            }
            other => panic!("Unhandled dispatch chain type: {other:?}"),
        };
        let mut dispatch_infos = dispatch_infos;
        let dispatch_info = dispatch_infos
            .find(|info| info.commands.contains(&self.vk_xml_command.name))
            .unwrap_or_else(|| {
                panic!(
                    "No dispatch info found for command {}",
                    self.vk_xml_command.name
                )
            });
        lines.push(format!(
            "let dispatch_table = &{dispatch_chain_var}.dispatch_table.{};",
            dispatch_info.get_dispatch_table_field_name()
        ));

        let vk_xml_to_rust_method_info =
            VkXmlToRustMethodInfo::from_vk_xml_command(&self.vk_xml_command);

        let mut intercept_params = vec![];
        for param in &vk_xml_to_rust_method_info.parameters {
            let rust_param = &param.rust_method_param;
            let xml_param = &param.main_source_vk_xml_param;
            let transformer = ParamTransformer::ALL
                .into_iter()
                .find(|transformer| transformer.type_matches(&rust_param.ty, &xml_param.ty))
                .unwrap_or_else(|| {
                    panic!(
                        "Failed to transform parameter {}, {} in {}.",
                        xml_param.name, rust_param.name, self.vk_xml_command.name
                    )
                });
            intercept_params.push(transformer.transform(
                rust_param,
                xml_param,
                &self.vk_xml_command,
            ));
        }

        let param_names: Vec<&str> = self
            .vk_xml_command
            .parameters
            .iter()
            .map(|param| param.name.as_str())
            .collect();
        let return_info = &vk_xml_to_rust_method_info.return_info;
        let ret_param_index = || {
            assert!(return_info.main_source_vk_xml_params.len() == 1);
            let name = &return_info.main_source_vk_xml_params[0].name;
            param_names
                .iter()
                .position(|param| param == name)
                .unwrap_or_else(|| panic!("{name} is not a parameter"))
        };
        let generate_ret_expr = |ret_var: &str| -> Vec<String> {
            match self.vk_xml_command.return_type.as_str() {
                "VkResult" => {
                    if return_info.main_source_vk_xml_params.is_empty() {
                        return [
                            format!("match {ret_var} {{"),
                            "    Ok(()) => vk::Result::SUCCESS,".to_owned(),
                            "    Err(e) => e,".to_owned(),
                            "}".to_owned(),
                        ]
                        .to_vec();
                    }
                    let param_index = ret_param_index();
                    let assign_to_ret_param_lines = self.generate_assign_to_ret_param_lines(
                        "res",
                        param_index,
                        LastExprType::VkResult,
                        false,
                    );
                    let null_fill_lines = self.generate_assign_to_ret_param_lines(
                        "res",
                        param_index,
                        LastExprType::VkResult,
                        true,
                    );
                    let mut err_lines = vec!["    Err(e) => e,".to_owned()];
                    if !null_fill_lines.is_empty() {
                        err_lines = vec!["    Err(e) => {".to_owned()];
                        err_lines
                            .extend(null_fill_lines.iter().map(|line| format!("        {line}")));
                        err_lines.push("        e".to_owned());
                        err_lines.push("    }".to_owned());
                    }
                    let mut lines =
                        vec![format!("match {ret_var} {{"), "    Ok(res) => {".to_owned()];
                    lines.extend(
                        assign_to_ret_param_lines
                            .iter()
                            .map(|line| format!("        {line}")),
                    );
                    lines.push("    }".to_owned());
                    lines.extend(err_lines);
                    lines.push("}".to_owned());
                    lines
                }
                "void" if !return_info.main_source_vk_xml_params.is_empty() => self
                    .generate_assign_to_ret_param_lines(
                        ret_var,
                        ret_param_index(),
                        LastExprType::Unit,
                        false,
                    ),
                "VkBool32" => vec![format!("if {ret_var} {{ vk::TRUE }} else {{ vk::FALSE }}")],
                _ => vec![ret_var.to_owned()],
            }
        };

        let rust_ffi_param_names: Vec<&str> = self
            .rust_fn
            .parameters
            .iter()
            .map(|param| param.name.as_str())
            .collect();
        lines.push(format!(
            "let layer_result = {dispatch_chain_var}.customized_info.borrow().hooks().{}({});",
            self.rust_fn.name,
            intercept_params.join(", ")
        ));
        lines.push("match layer_result {".to_owned());
        lines.push("    LayerResult::Handled(res) => {".to_owned());
        lines.extend(
            generate_ret_expr("res")
                .iter()
                .map(|line| format!("        {line}")),
        );
        lines.push("    }".to_owned());
        lines.push(format!(
            "    LayerResult::Unhandled => unsafe {{ (dispatch_table.{})({}) }},",
            self.rust_fn.name,
            rust_ffi_param_names.join(", ")
        ));
        lines.push("}".to_owned());
        lines
    }
}

/// How a hook's argument is made from the FFI parameter. The first that matches is used.
#[derive(Clone, Copy)]
enum ParamTransformer {
    Array,
    Bool,
    Primitive,
    NoLenPtr,
    NullTerminatedLenPtr,
    VoidDataSlice,
    BoolSlice,
    GeneralSlice,
}

impl ParamTransformer {
    const ALL: [ParamTransformer; 8] = [
        ParamTransformer::Array,
        ParamTransformer::Bool,
        ParamTransformer::Primitive,
        ParamTransformer::NoLenPtr,
        ParamTransformer::NullTerminatedLenPtr,
        ParamTransformer::VoidDataSlice,
        ParamTransformer::BoolSlice,
        ParamTransformer::GeneralSlice,
    ];

    fn type_matches(self, rust_type: &RustType, vk_xml_type: &VkXmlType) -> bool {
        let points_to = vk_xml_type.points_to();
        match self {
            ParamTransformer::Array => {
                !vk_xml_type.dimensions.is_empty()
                    && rust_type
                        .refers_to()
                        .and_then(RustType::array_length)
                        .is_some_and(|length| length == vk_xml_type.dimensions[0])
            }
            ParamTransformer::Bool => {
                rust_type.name() == Some("bool") && vk_xml_type.name() == Some("VkBool32")
            }
            ParamTransformer::Primitive => rust_type.name().is_some(),
            ParamTransformer::NoLenPtr => points_to.is_some() && vk_xml_type.len.is_none(),
            ParamTransformer::NullTerminatedLenPtr => {
                vk_xml_type.len == Some(VkXmlLenKind::NullTerminated)
                    && points_to.is_some_and(|points_to| {
                        points_to.name() == Some("char") && points_to.is_const
                    })
                    && !vk_xml_type.is_optional
                    && rust_type.refers_to().is_some_and(|refers_to| {
                        refers_to.name() == Some("str") && refers_to.is_const
                    })
            }
            ParamTransformer::VoidDataSlice => {
                vk_xml_type.len == Some(VkXmlLenKind::Variable)
                    && points_to.is_some_and(|points_to| points_to.name() == Some("void"))
            }
            ParamTransformer::BoolSlice => {
                vk_xml_type.len == Some(VkXmlLenKind::Variable)
                    && points_to.is_some_and(|points_to| {
                        points_to.name() == Some("VkBool32") && points_to.is_const
                    })
                    && !vk_xml_type.is_optional
            }
            ParamTransformer::GeneralSlice => {
                vk_xml_type.len == Some(VkXmlLenKind::Variable) && points_to.is_some()
            }
        }
    }

    fn get_ptr_to_slice_fn(vk_xml_type: &VkXmlType) -> &'static str {
        let element_xml_type = vk_xml_type.points_to().expect("A slice of a non-pointer");
        match (vk_xml_type.is_optional, element_xml_type.is_const) {
            (true, true) => "maybe_slice_from_raw_parts",
            (true, false) => "maybe_uninit_slice_from_raw_parts_mut",
            (false, true) => "slice_from_raw_parts",
            (false, false) => "uninit_slice_from_raw_parts_mut",
        }
    }

    fn transform(
        self,
        rust_param: &RustParam,
        vk_xml_param: &VkXmlParam,
        vk_xml_command: &VkXmlCommand,
    ) -> String {
        let ffi_name = RustParam::param_name_from_vk_xml_param_for_ffi;
        let param_name = ffi_name(&vk_xml_param.name);
        let ty = &vk_xml_param.ty;
        let params_by_name = |name: &str| {
            vk_xml_command
                .parameters
                .iter()
                .rev()
                .find(|param| param.name == name)
        };
        let len_var = || {
            vk_xml_param
                .len_var
                .as_deref()
                .expect("A slice without a length")
        };
        match self {
            ParamTransformer::Bool => format!("{param_name} == vk::TRUE"),
            ParamTransformer::Array => {
                assert!(!ty.is_optional);
                assert!(ty.is_const);
                format!("unsafe {{ {param_name}.as_ref() }}.unwrap()")
            }
            ParamTransformer::Primitive => param_name,
            ParamTransformer::NoLenPtr => {
                let points_to = ty.points_to().expect("A pointer without a pointee");
                // A parameter that is still a pointer in Rust is passed as it is.
                if rust_param.ty.points_to().is_some() {
                    return param_name;
                }
                let ptr_to_opt_ref_expr = if points_to.is_const {
                    format!("unsafe {{ {param_name}.as_ref() }}")
                } else {
                    format!("unsafe {{ ptr_as_uninit_mut({param_name}) }}")
                };
                if ty.is_optional {
                    ptr_to_opt_ref_expr
                } else {
                    format!("{ptr_to_opt_ref_expr}.unwrap()")
                }
            }
            ParamTransformer::NullTerminatedLenPtr => {
                format!("unsafe {{ CStr::from_ptr({param_name}) }}.to_str().unwrap()")
            }
            ParamTransformer::BoolSlice => {
                let len_var = len_var();
                let len_xml_param =
                    params_by_name(len_var).expect("The length of a bool slice is not a parameter");
                assert!(!len_var.contains("->"));
                assert!(matches!(
                    len_xml_param.ty.name(),
                    Some("uint32_t" | "size_t" | "VkDeviceSize")
                ));
                format!(
                    "unsafe {{ bool_iterator_from_raw_parts({param_name}, {}) }}",
                    ffi_name(len_var)
                )
            }
            ParamTransformer::VoidDataSlice => {
                let points_to = ty.points_to().expect("A pointer without a pointee");
                let len_var_xml_param =
                    params_by_name(len_var()).expect("The length of a slice is not a parameter");
                assert!(!len_var_xml_param.ty.is_optional);
                let ptr_to_slice_fn = Self::get_ptr_to_slice_fn(ty);
                let len_expr = ffi_name(&len_var_xml_param.name);
                if ty.is_optional && !points_to.is_const {
                    let len_points_to = len_var_xml_param
                        .ty
                        .points_to()
                        .expect("A length that is not a pointer");
                    assert!(
                        matches!(len_points_to.name(), Some("uint32_t" | "size_t")),
                        "Unsuppored length parameter {} in {}",
                        len_var_xml_param.name,
                        vk_xml_command.name
                    );
                } else {
                    assert!(matches!(
                        len_var_xml_param.ty.name(),
                        Some("uint32_t" | "size_t" | "VkDeviceSize")
                    ));
                }
                let data_ptr_type = if points_to.is_const {
                    "*const u8"
                } else {
                    "*mut u8"
                };
                format!(
                    "unsafe {{ {ptr_to_slice_fn}({param_name} as {data_ptr_type}, {len_expr}) }}"
                )
            }
            ParamTransformer::GeneralSlice => {
                let points_to = ty.points_to().expect("A pointer without a pointee");
                assert!(points_to.name().is_some());
                let len_var = len_var();
                let len_expr = if len_var == "(samples + 31) / 32"
                    && vk_xml_command.name == "vkCmdSetSampleMaskEXT"
                {
                    "samples.as_raw().div_ceil(32)".to_owned()
                } else if ty.is_optional && !points_to.is_const {
                    // A mutable optional slice: only supported with a mutable pointer type length
                    // variable.
                    let len_var_xml_param =
                        params_by_name(len_var).expect("The length of a slice is not a parameter");
                    let len_points_to = len_var_xml_param
                        .ty
                        .points_to()
                        .expect("A length that is not a pointer");
                    assert!(
                        matches!(len_points_to.name(), Some("size_t" | "uint32_t")),
                        "Unsupported length type: {} in {}",
                        len_var_xml_param.name,
                        vk_xml_command.name
                    );
                    ffi_name(&len_var_xml_param.name)
                } else if len_var.contains("->") {
                    // pBuildInfo->geometryCount in vkGetAccelerationStructureBuildSizesKHR.
                    let len_var_components: Vec<&str> = len_var.split("->").collect();
                    assert!(len_var_components.len() == 2, "Unknown len: {len_var}");
                    let len_xml_param = params_by_name(len_var_components[0])
                        .expect("Can't find the length parameter.");
                    assert!(
                        !len_xml_param.ty.is_optional,
                        "When handling length parameter, {} in {len_var} is optional. This is currently not supported.",
                        len_var_components[0]
                    );
                    let len_points_to = len_xml_param
                        .ty
                        .points_to()
                        .expect("A length struct that is not a pointer");
                    assert!(len_points_to.is_const);
                    // Safe, because len_var is a const pointer and it's not optional, so it must
                    // point to an initialized object.
                    format!(
                        "unsafe {{ {}.as_ref() }}.unwrap().{}",
                        ffi_name(len_var_components[0]),
                        ffi_name(len_var_components[1])
                    )
                } else {
                    let len_var_xml_param = params_by_name(len_var).unwrap_or_else(|| {
                        panic!(
                            "The length variable of {} {len_var} isn't a parameter of {}",
                            vk_xml_param.name, vk_xml_command.name
                        )
                    });
                    let rust_ffi_len_var_name = ffi_name(&len_var_xml_param.name);
                    if len_var_xml_param.ty.name() == Some("uint32_t") {
                        rust_ffi_len_var_name
                    } else if len_var_xml_param
                        .ty
                        .points_to()
                        .expect("Unsupported length variable type")
                        .name()
                        == Some("uint32_t")
                    {
                        assert!(!len_var_xml_param.ty.is_optional);
                        format!("*unsafe {{ {rust_ffi_len_var_name}.as_ref() }}.unwrap()")
                    } else {
                        panic!(
                            "Unsupported length variable type: {:?}",
                            len_var_xml_param.ty
                        );
                    }
                };
                let ptr_to_slice_fn = Self::get_ptr_to_slice_fn(ty);
                format!("unsafe {{ {ptr_to_slice_fn}({param_name}, {len_expr}) }}")
            }
        }
    }
}

#[derive(Default)]
struct GlobalSimpleInterceptGenerator {
    device_commands: OrderedMap<Rc<VulkanCommand>>,
    instance_commands: OrderedMap<Rc<VulkanCommand>>,
    unhandled_commands: OrderedMap<UnhandledCommand>,
    command_aliases: VulkanAliases,
    dispatch_infos: OrderedMap<CommandDispatchInfo>,
}

/// The commands the framework implements by hand.
const MANUALLY_IMPLEMENTED_COMMANDS: [&str; 9] = [
    "vkDestroyInstance",
    "vkEnumeratePhysicalDevices",
    "vkEnumeratePhysicalDeviceGroups",
    "vkGetInstanceProcAddr",
    "vkGetDeviceProcAddr",
    "vkCreateDevice",
    "vkDestroyDevice",
    "vkEnumerateDeviceExtensionProperties",
    "vkEnumerateDeviceLayerProperties",
];

/// The commands the layer always intercepts.
const ALWAYS_HOOKED_COMMANDS: [&str; 13] = [
    "vkEnumerateInstanceLayerProperties",
    "vkEnumerateInstanceExtensionProperties",
    "vkEnumerateDeviceLayerProperties",
    "vkEnumerateDeviceExtensionProperties",
    "vkGetInstanceProcAddr",
    "vkGetDeviceProcAddr",
    "vkCreateInstance",
    "vkDestroyInstance",
    "vkCreateDevice",
    "vkDestroyDevice",
    "vkEnumeratePhysicalDeviceGroups",
    "vkEnumeratePhysicalDeviceGroupsKHR",
    "vkEnumeratePhysicalDevices",
];

impl GlobalSimpleInterceptGenerator {
    fn begin_file(&self, out: &mut String) {
        write_preamble(out);
        out.push('\n');
        out.push_str("// This file is generated from the Vulkan XML API registry.\n");
        out.push_str(
            &[
                "#![allow(unused_unsafe)]",
                "use std::{borrow:: Borrow, ffi::{c_int, c_void, c_char, CStr}, ptr::NonNull, sync::Arc, collections::HashSet};",
                "use ash::vk;",
                "use smallvec::smallvec;",
                "",
                "use crate::{DeviceInfo, fill_vk_out_array, Global, InstanceInfo, Layer, LayerResult, LayerVulkanCommand, InstanceHooks, DeviceHooks, vk_utils::{slice_from_raw_parts, ptr_as_uninit_mut}};",
                "use super::{get_instance_proc_addr_loader, get_device_proc_addr_loader, VulkanCommand, TryFromExtensionError, ApiVersion, Feature, bool_iterator_from_raw_parts, maybe_slice_from_raw_parts, maybe_uninit_slice_from_raw_parts_mut, uninit_slice_from_raw_parts_mut};",
                "",
            ]
            .join("\n"),
        );
        out.push('\n');
    }

    fn merged_commands(&self) -> OrderedMap<Rc<VulkanCommand>> {
        let mut all_commands = OrderedMap::default();
        for (name, command) in self
            .instance_commands
            .iter()
            .chain(self.device_commands.iter())
        {
            all_commands.insert(name, Rc::clone(command));
        }
        all_commands
    }

    fn generate_vulkan_command_entries(
        &self,
        indent: usize,
        commands: &OrderedMap<Rc<VulkanCommand>>,
        hooked_commands_var: &str,
    ) -> String {
        let mut command_to_dispatch_infos: BTreeMap<&str, Vec<&CommandDispatchInfo>> =
            BTreeMap::new();
        for dispatch_info in self.dispatch_infos.values() {
            for command in &dispatch_info.commands {
                command_to_dispatch_infos
                    .entry(command)
                    .or_default()
                    .push(dispatch_info);
            }
        }
        let mut lines = vec![];
        let mut command_items: Vec<(&String, &Rc<VulkanCommand>)> = commands.iter().collect();
        command_items.sort_by(|a, b| a.0.cmp(b.0));
        for (proc_name, command) in command_items {
            // Use the actual name, because ash doesn't generate type names for aliased types.
            let fp_type_name = format!("vk::PFN_{}", command.vk_xml_command.name);
            let rust_fn_name = format!("Self::{}", command.rust_fn.name);
            let features: Vec<String> = command_to_dispatch_infos
                .get(proc_name.as_str())
                .unwrap_or_else(|| panic!("No feature has {proc_name}"))
                .iter()
                .map(|dispatch_info| dispatch_info.get_enum_name())
                .collect();
            assert!(
                !features.is_empty(),
                "Every command must have at least one dispatch_info associated."
            );
            let hook_command_variant_value = snake_case_to_upper_camel_case(&command.rust_fn.name);
            let hooked_expr = if ALWAYS_HOOKED_COMMANDS.contains(&proc_name.as_str()) {
                "true".to_owned()
            } else {
                format!("{hooked_commands_var}.contains(&LayerVulkanCommand::{hook_command_variant_value})")
            };
            lines.extend([
                "VulkanCommand {".to_owned(),
                format!("    name: \"{proc_name}\","),
                format!("    features: smallvec![{}],", features.join(", ")),
                format!("    hooked: {hooked_expr},"),
                format!(
                    "    proc: unsafe {{ std::mem::transmute::<{fp_type_name}, vk::PFN_vkVoidFunction>({rust_fn_name})}},"
                ),
                "},".to_owned(),
            ]);
        }
        lines
            .iter()
            .map(|line| format!("{}{line}\n", " ".repeat(indent)))
            .collect()
    }

    fn end_file(&mut self, out: &mut String) {
        let all_commands = self.merged_commands();
        // Handle aliases
        for name in all_commands.keys() {
            let represent_name = self
                .command_aliases
                .get_represent_name(name)
                .unwrap_or_else(|| panic!("{name} is never added to the command aliases."));
            let represent = Rc::clone(
                all_commands
                    .get(&represent_name)
                    .unwrap_or_else(|| panic!("{represent_name} is not a command")),
            );
            for commands in [&mut self.instance_commands, &mut self.device_commands] {
                if commands.contains_key(name) {
                    commands.insert(name, Rc::clone(&represent));
                }
            }
        }

        // Merge the 2 maps again, because the aliased commands are directed to point to the same
        // version.
        let all_commands = self.merged_commands();
        let aliased_commands: HashSet<String> = all_commands
            .iter()
            .filter(|(name, command)| &command.vk_xml_command.name != *name)
            .map(|(name, _)| name.clone())
            .collect();
        let not_aliased_commands: Vec<Rc<VulkanCommand>> = all_commands
            .iter()
            .filter(|(name, _)| !aliased_commands.contains(*name))
            .map(|(_, command)| Rc::clone(command))
            .collect();
        // Not all dispatch will be used. For some extension, e.g. VK_EXT_shader_object, all
        // commands are in other extensions like VK_EXT_extended_dynamic_state, so we don't need to
        // create a dispatch table for it.
        let mut used_dispatch_info: Vec<&CommandDispatchInfo> = vec![];
        for command in &not_aliased_commands {
            let command_name = &command.vk_xml_command.name;
            let dispatch_info = self
                .dispatch_infos
                .values()
                .find(|info| info.commands.contains(command_name))
                .unwrap_or_else(|| panic!("Can't find dispatch info for {command_name}"));
            if !used_dispatch_info
                .iter()
                .any(|used| std::ptr::eq(*used, dispatch_info))
            {
                used_dispatch_info.push(dispatch_info);
            }
        }

        out.push_str(
            &CommandDispatchInfo::get_feature_enum_lines(self.dispatch_infos.values()).join("\n"),
        );
        out.push_str("\n\n");

        out.push_str(
            &CommandDispatchInfo::get_dispatch_tables_impl_lines(
                &used_dispatch_info,
                &self.instance_commands,
                &self.device_commands,
                &aliased_commands,
            )
            .join("\n"),
        );
        out.push_str("\n\n");

        // An aliased command is dispatched through its representative's table entry, which is loaded
        // under the representative's name only. Below a layer or driver that resolves that name to
        // NULL (a core name on a device of an older version that enabled the extension), the loaders
        // try these names next.
        let mut alias_groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for name in &aliased_commands {
            let represent = all_commands
                .get(name)
                .map(|command| command.vk_xml_command.name.clone());
            alias_groups
                .entry(represent.unwrap_or_default())
                .or_default()
                .push(name.clone());
        }
        let mut alias_lines: Vec<String> = [
            "/// The other names of a command (its aliases in the registry, the extension names",
            "/// of a promoted command), NUL-terminated, tried in this order when the next layer",
            "/// resolves its name to NULL.",
            "pub(crate) fn command_aliases(name: &CStr) -> &'static [&'static [u8]] {",
            "    match name.to_bytes() {",
        ]
        .map(str::to_owned)
        .to_vec();
        for (represent, mut aliases) in alias_groups {
            aliases.sort_by_key(|alias| (!alias.ends_with("KHR"), alias.clone()));
            let aliases: Vec<String> = aliases
                .iter()
                .map(|alias| format!("b\"{alias}\\0\""))
                .collect();
            alias_lines.push(format!(
                "        b\"{represent}\" => &[{}],",
                aliases.join(", ")
            ));
        }
        alias_lines.extend(["        _ => &[],", "    }", "}", ""].map(str::to_owned));
        out.push_str(&alias_lines.join("\n"));
        out.push('\n');

        out.push_str(&generate_unhandled_command_comments(
            self.unhandled_commands.values(),
        ));

        out.push('\n');

        out.push_str(
            &[
                "impl<T: Layer> Global<T> {",
                "    pub(crate) fn create_device_commands(&self, instance_info: &T::InstanceInfo, device_info: Option<&T::DeviceInfo>) -> Box<[VulkanCommand]> {",
                "        let hooked_commands = self.layer_info",
                "            .hooked_device_commands(instance_info, device_info)",
                "            .collect::<HashSet<_>>();",
                "        Box::new([",
                "",
            ]
            .join("\n"),
        );
        out.push_str(&self.generate_vulkan_command_entries(
            12,
            &self.device_commands,
            "hooked_commands",
        ));
        out.push_str("        ])\n");
        out.push_str("    }\n");

        out.push_str(
            &[
                "    pub(crate) fn create_instance_commands(&self, instance_info: &T::InstanceInfo) -> Box<[VulkanCommand]> {",
                "        let hooked_commands = self.layer_info",
                "            .hooked_instance_commands(instance_info)",
                "            .collect::<HashSet<_>>();",
                "        Box::new([",
                "",
            ]
            .join("\n"),
        );
        out.push_str(&self.generate_vulkan_command_entries(
            12,
            &self.instance_commands,
            "hooked_commands",
        ));
        out.push_str("        ])\n");
        out.push_str("    }\n");

        for vulkan_command in &not_aliased_commands {
            if MANUALLY_IMPLEMENTED_COMMANDS.contains(&vulkan_command.vk_xml_command.name.as_str())
            {
                continue;
            }
            out.push_str(&format!(
                "    {} {{\n",
                vulkan_command.rust_fn.get_def_str()
            ));
            for line in vulkan_command.get_rust_fn_impl_lines(self.dispatch_infos.values()) {
                out.push_str(&format!("        {line}\n"));
            }
            out.push_str("    }\n");
        }
        out.push_str("}\n");
    }

    fn gen_cmd(&mut self, command: &Command, registry: &Registry) {
        let name = command.name.as_str();
        // The feature a command is generated in is not the one to dispatch it through (when handling
        // `vkGetRayTracingShaderGroupHandlesKHR` it is `VK_NV_ray_tracing` instead of
        // `VK_KHR_ray_tracing_pipeline`), so the dispatch info is connected with the command in
        // end_file.
        if let Some(unhandled_command) = UnhandledCommand::find(name) {
            self.unhandled_commands.insert(name, unhandled_command);
            return;
        }

        let vk_xml_cmd = VkXmlCommand::from_command(command, registry);
        let rust_fn = RustFfiFunction::from_vk_xml_command(&vk_xml_cmd);
        // Built for its checks: the hook method of every command must be expressible.
        RustMethod::from_vk_xml_command(&vk_xml_cmd);

        self.command_aliases
            .add_alias(name, command.alias.as_deref());

        let dispatch_chain_type = vk_xml_cmd.get_dispatch_chain_type();
        let vulkan_command = Rc::new(VulkanCommand {
            vk_xml_command: vk_xml_cmd,
            rust_fn,
        });
        match dispatch_chain_type {
            Some(DispatchChainType::Instance) => {
                self.instance_commands.insert(name, vulkan_command)
            }
            Some(DispatchChainType::Device) => self.device_commands.insert(name, vulkan_command),
            Some(DispatchChainType::Global) => {}
            None => self.unhandled_commands.insert(
                name,
                UnhandledCommand::new(name, "Unknown dispatch chain type"),
            ),
        }
    }

    fn begin_feature(&mut self, feature: &Feature) {
        assert!(
            !self.dispatch_infos.contains_key(&feature.name),
            "Duplicate entries for {}",
            feature.name
        );
        self.dispatch_infos
            .insert(&feature.name, CommandDispatchInfo::from_feature(feature));
    }
}

/// The contents of `global_simple_intercept/generated.rs`, before rustfmt.
pub fn generate(registry: &Registry) -> String {
    let mut generator = GlobalSimpleInterceptGenerator::default();
    let mut out = String::new();
    generator.begin_file(&mut out);
    for (feature, commands) in &registry.features {
        generator.begin_feature(feature);
        for command in commands {
            generator.gen_cmd(command, registry);
        }
    }
    generator.end_file(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::super::tests::vk_xml_command;
    use super::*;

    /// `transformer` takes the hook method's parameter `rust_index` (the command's parameter
    /// `xml_index`) and makes `expected` of it, and is the first that takes it.
    fn check(
        command: &str,
        transformer: ParamTransformer,
        rust_index: usize,
        xml_index: usize,
        expected: &str,
    ) {
        let command = vk_xml_command(command);
        let method = RustMethod::from_vk_xml_command(&command);
        let rust_param = &method.parameters[rust_index];
        let xml_param = &command.parameters[xml_index];
        assert!(transformer.type_matches(&rust_param.ty, &xml_param.ty));
        assert_eq!(
            transformer.transform(rust_param, xml_param, &command),
            expected
        );
        let first = ParamTransformer::ALL
            .into_iter()
            .find(|transformer| transformer.type_matches(&rust_param.ty, &xml_param.ty))
            .unwrap();
        assert_eq!(first.transform(rust_param, xml_param, &command), expected);
    }

    #[test]
    fn bool_param_transformer() {
        check(
            "vkCmdSetRepresentativeFragmentTestEnableNV",
            ParamTransformer::Bool,
            1,
            1,
            "representative_fragment_test_enable == vk::TRUE",
        );
    }

    #[test]
    fn array_param_transformer() {
        check(
            "vkCmdSetBlendConstants",
            ParamTransformer::Array,
            1,
            1,
            "unsafe { blend_constants.as_ref() }.unwrap()",
        );
    }

    #[test]
    fn primitive_param_transformer() {
        check(
            "vkCmdDraw",
            ParamTransformer::Primitive,
            0,
            0,
            "command_buffer",
        );
        check(
            "vkCmdDraw",
            ParamTransformer::Primitive,
            1,
            1,
            "vertex_count",
        );
    }

    #[test]
    fn primitive_param_transformer_keyword_argument() {
        check(
            "vkGetPhysicalDeviceImageFormatProperties",
            ParamTransformer::Primitive,
            2,
            2,
            "_type",
        );
    }

    #[test]
    fn c_expr_length_var_ptr_param_transformer() {
        check(
            "vkGetAccelerationStructureBuildSizesKHR",
            ParamTransformer::GeneralSlice,
            2,
            3,
            "unsafe { maybe_slice_from_raw_parts(p_max_primitive_counts, unsafe { p_build_info.as_ref() }.unwrap().geometry_count) }",
        );
    }

    #[test]
    fn no_length_not_optional_const_ptr_param_transformer() {
        check(
            "vkCreateImage",
            ParamTransformer::NoLenPtr,
            0,
            1,
            "unsafe { p_create_info.as_ref() }.unwrap()",
        );
    }

    #[test]
    fn no_length_optional_const_ptr_param_transformer() {
        check(
            "vkCreateImage",
            ParamTransformer::NoLenPtr,
            1,
            2,
            "unsafe { p_allocator.as_ref() }",
        );
    }

    #[test]
    fn null_terminated_length_const_ptr_param_transformer() {
        check(
            "vkDebugReportMessageEXT",
            ParamTransformer::NullTerminatedLenPtr,
            5,
            6,
            "unsafe { CStr::from_ptr(p_layer_prefix) }.to_str().unwrap()",
        );
    }

    #[test]
    fn boolean_slice_param_transformer() {
        check(
            "vkCmdSetColorWriteEnableEXT",
            ParamTransformer::BoolSlice,
            1,
            2,
            "unsafe { bool_iterator_from_raw_parts(p_color_write_enables, attachment_count) }",
        );
    }

    #[test]
    fn const_void_data_slice_param_transformer() {
        check(
            "vkCmdUpdateBuffer",
            ParamTransformer::VoidDataSlice,
            3,
            4,
            "unsafe { slice_from_raw_parts(p_data as *const u8, data_size) }",
        );
    }

    #[test]
    fn mutable_void_data_slice_param_transformer() {
        check(
            "vkGetQueryPoolResults",
            ParamTransformer::VoidDataSlice,
            3,
            5,
            "unsafe { uninit_slice_from_raw_parts_mut(p_data as *mut u8, data_size) }",
        );
    }

    #[test]
    fn required_const_slice_param_transformer() {
        check(
            "vkQueueSubmit",
            ParamTransformer::GeneralSlice,
            1,
            2,
            "unsafe { slice_from_raw_parts(p_submits, submit_count) }",
        );
    }

    #[test]
    fn required_mutable_slice_param_transformer() {
        check(
            "vkGetCalibratedTimestampsEXT",
            ParamTransformer::GeneralSlice,
            1,
            3,
            "unsafe { uninit_slice_from_raw_parts_mut(p_timestamps, timestamp_count) }",
        );
    }

    #[test]
    fn optional_mutable_slice_param_transformer() {
        check(
            "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR",
            ParamTransformer::GeneralSlice,
            2,
            3,
            "unsafe { maybe_uninit_slice_from_raw_parts_mut(p_counters, p_counter_count) }",
        );
    }

    #[test]
    fn cmd_set_sample_mask_ext_param_transformer() {
        check(
            "vkCmdSetSampleMaskEXT",
            ParamTransformer::GeneralSlice,
            2,
            2,
            "unsafe { slice_from_raw_parts(p_sample_mask, samples.as_raw().div_ceil(32)) }",
        );
    }

    #[test]
    fn no_length_not_optional_mutable_ptr_param_transformer() {
        check(
            "vkGetPhysicalDeviceFeatures2",
            ParamTransformer::NoLenPtr,
            1,
            1,
            "unsafe { ptr_as_uninit_mut(p_features) }.unwrap()",
        );
    }

    #[test]
    fn no_length_not_optional_mutable_opaque_type_ptr_param_transformer() {
        check(
            "vkGetPhysicalDeviceXcbPresentationSupportKHR",
            ParamTransformer::NoLenPtr,
            2,
            2,
            "connection",
        );
    }

    #[test]
    fn optional_mutable_void_slice_param_transformer() {
        check(
            "vkGetPipelineCacheData",
            ParamTransformer::VoidDataSlice,
            2,
            3,
            "unsafe { maybe_uninit_slice_from_raw_parts_mut(p_data as *mut u8, p_data_size) }",
        );
    }
}
