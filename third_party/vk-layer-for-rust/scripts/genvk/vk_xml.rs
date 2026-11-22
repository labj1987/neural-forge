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
// Modified for Neural Forge (2026): comments wrapped here, a fixed copyright year; newer registries; the
// VkBool32 array hooks' iterator is not 'static; ported from Python (vk_xml_util.py) to Rust
// (UPSTREAM.md).

//! What the generators share: commands and their parameters as `vk.xml` gives them, the Rust types
//! and methods they turn into, command aliases and the commands left out.

use super::registry::{child, children, et_tail, et_text, itertext, Command, Registry};
use roxmltree::Node;
use std::collections::HashMap;

/// A map that iterates in insertion order, as Python's `dict` does.
pub struct OrderedMap<V> {
    keys: Vec<String>,
    values: HashMap<String, V>,
}

impl<V> Default for OrderedMap<V> {
    fn default() -> Self {
        OrderedMap {
            keys: vec![],
            values: HashMap::new(),
        }
    }
}

impl<V> OrderedMap<V> {
    /// Sets the value of `key`; a new key goes last, an existing one keeps its place.
    pub fn insert(&mut self, key: &str, value: V) {
        if self.values.insert(key.to_owned(), value).is_none() {
            self.keys.push(key.to_owned());
        }
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.values.get(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.keys.iter()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.keys.iter().map(|key| (key, &self.values[key]))
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.keys.iter().map(|key| &self.values[key])
    }
}

/// A union-find of command names joined by their aliases. The representative of a set is picked by
/// priority, from the highest to the lowest: the core name, the KHR suffixed name, the EXT suffixed
/// name, any other suffixed name.
#[derive(Default)]
pub struct VulkanAliases {
    parents: HashMap<String, Option<String>>,
}

impl VulkanAliases {
    fn root(&self, name: &str) -> String {
        let mut name = name.to_owned();
        while let Some(Some(parent)) = self.parents.get(&name) {
            name = parent.clone();
        }
        name
    }

    pub fn add_alias(&mut self, name: &str, alias: Option<&str>) {
        self.parents.entry(name.to_owned()).or_insert(None);
        let Some(alias) = alias else {
            return;
        };
        self.parents.entry(alias.to_owned()).or_insert(None);

        let name_root = self.root(name);
        let alias_root = self.root(alias);

        let stem_len = name
            .chars()
            .zip(alias.chars())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| a.len_utf8())
            .sum::<usize>();
        let rank = |suffix: &str| -> usize {
            ["", "KHR", "EXT"]
                .iter()
                .position(|known| *known == suffix)
                .unwrap_or(3)
        };
        let name_rank = rank(&name[stem_len..]);
        let alias_rank = rank(&alias[stem_len..]);
        let (mut parent, mut child) = (name_root, alias_root);
        if name_rank > alias_rank || (name_rank == alias_rank && name > alias) {
            (parent, child) = (child, parent);
        }
        assert!(
            parent != child,
            "{name} and {alias} are already aliases of each other"
        );
        self.parents.insert(child, Some(parent));
    }

    /// The representative of `name`'s set (`vkWaitSemaphores` for `vkWaitSemaphoresKHR`), or `None`
    /// if `name` was never added.
    pub fn get_represent_name(&self, name: &str) -> Option<String> {
        self.parents.contains_key(name).then(|| self.root(name))
    }
}

#[derive(Clone)]
pub struct UnhandledCommand {
    pub name: String,
    pub reason: String,
}

impl UnhandledCommand {
    pub fn new(name: &str, reason: &str) -> Self {
        UnhandledCommand {
            name: name.to_owned(),
            reason: reason.to_owned(),
        }
    }

    /// The commands the generated code cannot handle, and why.
    pub fn find(name: &str) -> Option<UnhandledCommand> {
        const COMMANDS_UNABLE_TO_HANDLE: [(&str, &[&str]); 3] = [
            (
                "The ash Rust binding doesn't have proper bindings yet.",
                &[
                    "vkMapMemory2KHR",
                    "vkUnmapMemory2KHR",
                    "vkCreateShadersEXT",
                    "vkDestroyShaderEXT",
                    "vkGetShaderBinaryDataEXT",
                    "vkCmdBindShadersEXT",
                    "vkCmdDrawClusterHUAWEI",
                    "vkCmdDrawClusterIndirectHUAWEI",
                    "vkCmdSetAttachmentFeedbackLoopEnableEXT",
                    "vkCmdSetDiscardRectangleEnableEXT",
                    "vkCmdSetExclusiveScissorEnableNV",
                    "vkCmdSetDiscardRectangleModeEXT",
                ],
            ),
            // TODO: add a way to customize type generation for the ppBuildRangeInfos parameter of
            // vkCmdBuildAccelerationStructuresKHR. It should be
            // &[&[vk::AccelerationStructureBuildRangeInfoKHR]]. Now it is generated as
            // &[&vk::AccelerationStructureBuildRangeInfoKHR]. Other similar commands include but not
            // limit to vkCmdBuildAccelerationStructuresIndirectKHR.
            (
                "Dynamic multi-dimensional array bindings are not supported yet.",
                &[
                    "vkCmdBuildAccelerationStructuresKHR",
                    "vkBuildAccelerationStructuresKHR",
                    "vkCmdBuildAccelerationStructuresIndirectKHR",
                ],
            ),
            // TODO: for these cases, we may need to manually generate them.
            (
                "The length info and the data pointer are nested in structs.",
                &["vkGetDeviceFaultInfoEXT"],
            ),
        ];
        COMMANDS_UNABLE_TO_HANDLE
            .iter()
            .find(|(_, names)| names.contains(&name))
            .map(|(reason, _)| UnhandledCommand::new(name, reason))
    }
}

pub fn camel_case_to_snake_case(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    if chars.len() <= 1 {
        return input.to_lowercase();
    }
    // Handling continuous capitalized letters like EXT, KHR in the suffix, or IOS, FB, in the middle.
    let mut result: String = chars[0].to_lowercase().collect();
    for window in chars.windows(3) {
        let (prev, cur, next) = (window[0], window[1], window[2]);
        if cur.is_uppercase() && (next.is_lowercase() || !prev.is_uppercase()) {
            result.push('_');
        }
        result.extend(cur.to_lowercase());
    }
    result.extend(chars[chars.len() - 1].to_lowercase());
    result
}

pub fn snake_case_to_upper_camel_case(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    if chars.len() <= 1 {
        return input.to_uppercase();
    }
    let mut output: String = chars[0].to_uppercase().collect();
    for window in chars.windows(3) {
        let (prev, cur, next) = (window[0], window[1], window[2]);
        if cur == '_' && next != '_' {
            continue;
        }
        if prev == '_' {
            output.extend(cur.to_uppercase());
            continue;
        }
        output.push(cur);
    }
    output.push(chars[chars.len() - 1]);
    output
}

fn escape_rust_keywords(id: &str) -> String {
    if id == "type" {
        format!("_{id}")
    } else {
        id.to_owned()
    }
}

/// The year in the generated files' copyright notice: Google's, from when upstream last generated
/// them (2025). Not the current year: regenerating must not rewrite the notice.
const COPYRIGHT_YEAR: u32 = 2025;

pub fn write_preamble(out: &mut String) {
    out.push_str(&format!(
        "// Copyright {COPYRIGHT_YEAR} Google LLC\n\
         //\n\
         // Licensed under the Apache License, Version 2.0 (the \"License\");\n\
         // you may not use this file except in compliance with the License.\n\
         // You may obtain a copy of the License at\n\
         //\n\
         //     http://www.apache.org/licenses/LICENSE-2.0\n\
         //\n\
         // Unless required by applicable law or agreed to in writing, software\n\
         // distributed under the License is distributed on an \"AS IS\" BASIS,\n\
         // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n\
         // See the License for the specific language governing permissions and\n\
         // limitations under the License.\n\
         //\n\
         // Modified for Neural Forge (2026): generated for ash 0.38 and Vulkan 1.3.281 (UPSTREAM.md).\n\
         \n\
         #![allow(missing_docs)]\n"
    ));
}

/// Pointers to these types stay pointers.
const OPAQUE_TYPES: [(&str, &str); 11] = [
    ("void", "c_void"),
    // from "xcb/xcb.h"
    ("xcb_connection_t", "vk::xcb_connection_t"),
    // from "wayland-client.h"
    ("wl_display", "vk::wl_display"),
    ("wl_surface", "vk::wl_surface"),
    // from "screen/screen.h"
    ("_screen_context", "vk::_screen_context"),
    ("_screen_window", "vk::_screen_window"),
    ("_screen_buffer", "vk::_screen_buffer"),
    // from "windows.h"
    ("SECURITY_ATTRIBUTES", "vk::SECURITY_ATTRIBUTES"),
    ("ANativeWindow", "vk::ANativeWindow"),
    ("AHardwareBuffer", "vk::AHardwareBuffer"),
    ("CAMetalLayer", "vk::CAMetalLayer"),
];

fn is_opaque_type(name: &str) -> bool {
    OPAQUE_TYPES.iter().any(|(opaque, _)| *opaque == name)
}

const PRIMITIVE_TYPES: [(&str, &str); 8] = [
    ("uint64_t", "u64"),
    ("uint32_t", "u32"),
    ("uint16_t", "u16"),
    ("int32_t", "i32"),
    ("char", "c_char"),
    ("size_t", "usize"),
    ("float", "f32"),
    ("int", "c_int"),
];

const MANUAL_TYPES: [(&str, &str); 17] = [
    // from "X11/Xlib.h"
    ("Display", "vk::Display"),
    ("VisualID", "vk::VisualID"),
    ("Window", "vk::Window"),
    // from "X11/extensions/Xrandr.h"
    ("RROutput", "vk::RROutput"),
    // from "windows.h"
    ("HINSTANCE", "vk::HINSTANCE"),
    ("HWND", "vk::HWND"),
    ("HMONITOR", "vk::HMONITOR"),
    ("HANDLE", "vk::HANDLE"),
    ("DWORD", "vk::DWORD"),
    ("LPCWSTR", "vk::LPCWSTR"),
    // from "xcb/xcb.h"
    ("xcb_visualid_t", "vk::xcb_visualid_t"),
    ("xcb_window_t", "vk::xcb_window_t"),
    // from "directfb.h"
    ("IDirectFB", "vk::IDirectFB"),
    ("IDirectFBSurface", "vk::IDirectFBSurface"),
    // from "zircon/types.h"
    ("zx_handle_t", "vk::zx_handle_t"),
    // from "ggp_c/vulkan_types.h"
    ("GgpStreamDescriptor", "vk::GgpStreamDescriptor"),
    ("GgpFrameToken", "vk::GgpFrameToken"),
];

pub fn decayed_type_to_rust_type(decayed_type: &str) -> String {
    if let Some((_, rust_type)) = MANUAL_TYPES
        .iter()
        .chain(OPAQUE_TYPES.iter())
        .chain(PRIMITIVE_TYPES.iter())
        .find(|(c_type, _)| *c_type == decayed_type)
    {
        return (*rust_type).to_owned();
    }
    if let Some(without_vk) = decayed_type.strip_prefix("Vk") {
        let suffix = ["KHR", "EXT"]
            .into_iter()
            .find(|suffix| decayed_type.ends_with(suffix))
            .unwrap_or("");
        let without_suffix = &decayed_type[..decayed_type.len() - suffix.len()];
        // ash binding doesn't have the flag bits types.
        if without_suffix.ends_with("FlagBits") {
            let base = &without_suffix[2..];
            let base = base.strip_suffix("FlagBits").unwrap_or(base);
            return format!("vk::{base}Flags{suffix}");
        }
        return format!("vk::{without_vk}");
    }
    if decayed_type.starts_with("PFN_vk") {
        return format!("vk::{decayed_type}");
    }
    panic!("Unknown type: {decayed_type}");
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DispatchChainType {
    Device = 1,
    Instance = 2,
    Global = 3,
}

// TODO: add support for "1" kind
// TODO: add support for expression type which starts with "latexmath:"
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VkXmlLenKind {
    Variable,
    NullTerminated,
}

#[derive(Clone, Debug)]
pub struct VkXmlType {
    pub is_const: bool,
    pub is_optional: bool,
    /// A pointer type only has `points_to`. Otherwise, the type only has `name`.
    pub points_to: Option<Box<VkXmlType>>,
    pub name: Option<String>,
    /// Whether the named type is a struct.
    is_struct: bool,
    pub len: Option<VkXmlLenKind>,
    pub dimensions: Vec<u32>,
}

impl VkXmlType {
    pub fn decayed_type(&self) -> &VkXmlType {
        match &self.points_to {
            Some(points_to) if self.name.is_none() => points_to.decayed_type(),
            _ => self,
        }
    }

    pub fn is_struct(&self) -> bool {
        self.is_struct
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn points_to(&self) -> Option<&VkXmlType> {
        self.points_to.as_deref()
    }
}

fn param_decayed_type(param: Node) -> String {
    let type_node = child(param, "type").expect("A parameter without a <type>");
    itertext(type_node).trim().to_owned()
}

#[derive(Clone, Debug)]
pub struct VkXmlParam {
    pub ty: VkXmlType,
    pub name: String,
    pub len_var: Option<String>,
}

impl VkXmlParam {
    fn from_param_element(param: Node, is_struct: bool) -> VkXmlParam {
        let type_node = child(param, "type").expect("A parameter without a <type>");
        let name_node = child(param, "name").expect("A parameter without a <name>");

        let param_name = itertext(name_node).trim().to_owned();
        let decayed_type = param_decayed_type(param);

        let mut dimensions = vec![];
        if let Some(tail) = et_tail(name_node).filter(|tail| !tail.trim().is_empty()) {
            let tail = tail.trim();
            let digits = tail
                .strip_prefix('[')
                .and_then(|tail| tail.strip_suffix(']'));
            assert!(
                digits.is_some_and(
                    |digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
                ),
                "Unexpected name tail: {tail}"
            );
            dimensions = digits
                .unwrap_or_default()
                .split("][")
                .map(|dimension| dimension.parse().expect("A dimension is not a number"))
                .collect();
        }

        let mut is_optionals = vec![];
        if let Some(optional) = param.attribute("optional") {
            for is_optional in optional.split(',').map(str::trim) {
                match is_optional {
                    "true" => is_optionals.push(true),
                    "false" => is_optionals.push(false),
                    _ => panic!("Unexpected optional attribute: {optional}"),
                }
            }
        }
        let mut is_consts = vec![];
        let tail = et_tail(type_node).expect("No text after a parameter's <type>");
        let mut const_or_stars = tail.trim();
        while !const_or_stars.is_empty() {
            is_consts.push(const_or_stars.ends_with("const"));
            const_or_stars = const_or_stars
                .strip_suffix("const")
                .unwrap_or(const_or_stars)
                .trim();
            assert!(
                const_or_stars.ends_with('*'),
                "Unknown type: {}",
                itertext(param)
            );
            const_or_stars = const_or_stars
                .strip_suffix('*')
                .unwrap_or(const_or_stars)
                .trim();
        }

        let mut last_is_const = false;
        if let Some(text) = et_text(param) {
            let prefix = text.trim();
            let prefix = prefix.strip_suffix("struct").unwrap_or(prefix).trim();
            if !prefix.is_empty() {
                assert!(prefix == "const", "Unknown type: {}", itertext(param));
                last_is_const = true;
            }
        }
        is_consts.push(last_is_const);

        assert!(
            is_optionals.len() <= is_consts.len(),
            "Length of optional attributes is greater than the levels of pointers unexpectedly in param: {}.",
            itertext(param)
        );
        is_optionals.resize(is_consts.len(), false);
        let mut vk_xml_type = VkXmlType {
            is_const: is_consts.pop().unwrap_or_default(),
            is_optional: is_optionals.pop().unwrap_or_default(),
            points_to: None,
            name: Some(decayed_type),
            is_struct,
            len: None,
            dimensions,
        };
        for _ in 0..is_consts.len() {
            vk_xml_type = VkXmlType {
                is_const: is_consts.pop().unwrap_or_default(),
                is_optional: is_optionals.pop().unwrap_or_default(),
                points_to: Some(Box::new(vk_xml_type)),
                name: None,
                is_struct: false,
                len: None,
                dimensions: vec![],
            };
        }

        let mut len_attr = param.attribute("len");
        // len="1" (newer registries, e.g. vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI) says the
        // pointer is to a single element, which a parameter without a len attribute already means.
        if len_attr.is_some_and(|len| len.trim() == "1") {
            len_attr = None;
        }
        let mut len_var = None;
        if let Some(len) = len_attr.map(str::trim) {
            if len.starts_with("latexmath:") {
                let altlen = param
                    .attribute("altlen")
                    .expect("Expect latexmath len comes with altlen");
                len_var = Some(altlen.to_owned());
                vk_xml_type.len = Some(VkXmlLenKind::Variable);
            } else {
                assert!(!len.contains(','), "Multi-dimensional array unimplemented.");
                if len == "null-terminated" {
                    vk_xml_type.len = Some(VkXmlLenKind::NullTerminated);
                } else {
                    vk_xml_type.len = Some(VkXmlLenKind::Variable);
                    len_var = Some(len.to_owned());
                }
            }
        }

        VkXmlParam {
            ty: vk_xml_type,
            name: param_name,
            len_var,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VkXmlCommand {
    pub return_type: String,
    pub name: String,
    pub parameters: Vec<VkXmlParam>,
}

impl VkXmlCommand {
    pub fn from_command(command: &Command, registry: &Registry) -> VkXmlCommand {
        let mut parameters = vec![];
        for param in children(command.node, "param") {
            let type_name = param_decayed_type(param);
            let type_info = registry
                .type_info(&type_name)
                .unwrap_or_else(|| panic!("Unknown type: {type_name}"));
            parameters.push(VkXmlParam::from_param_element(param, type_info.is_struct));
        }
        let return_type = child(command.node, "proto")
            .and_then(|proto| child(proto, "type"))
            .and_then(et_text)
            .unwrap_or_else(|| panic!("{} has no return type", command.name))
            .trim()
            .to_owned();
        VkXmlCommand {
            return_type,
            name: command.name.clone(),
            parameters,
        }
    }

    pub fn get_dispatch_chain_type(&self) -> Option<DispatchChainType> {
        const GLOBAL_COMMANDS: [&str; 4] = [
            "vkCreateInstance",
            "vkEnumerateInstanceExtensionProperties",
            "vkEnumerateInstanceLayerProperties",
            "vkEnumerateInstanceVersion",
        ];
        if GLOBAL_COMMANDS.contains(&self.name.as_str()) {
            return Some(DispatchChainType::Global);
        }
        match self.parameters.first()?.ty.name()? {
            "VkInstance" | "VkPhysicalDevice" => Some(DispatchChainType::Instance),
            "VkDevice" | "VkCommandBuffer" | "VkQueue" => Some(DispatchChainType::Device),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum RustKind {
    Name(String),
    RefersTo(Box<RustType>),
    PointsTo(Box<RustType>),
    SliceOf(Box<RustType>),
    Array { element: Box<RustType>, length: u32 },
    Template { name: String, params: Vec<RustType> },
}

#[derive(Clone, Debug)]
pub struct RustType {
    pub is_const: bool,
    pub is_optional: bool,
    pub kind: RustKind,
}

impl RustType {
    fn new(is_const: bool, is_optional: bool, kind: RustKind) -> RustType {
        RustType {
            is_const,
            is_optional,
            kind,
        }
    }

    pub fn named(name: &str) -> RustType {
        RustType::new(true, false, RustKind::Name(name.to_owned()))
    }

    pub fn name(&self) -> Option<&str> {
        match &self.kind {
            RustKind::Name(name) => Some(name),
            _ => None,
        }
    }

    pub fn refers_to(&self) -> Option<&RustType> {
        match &self.kind {
            RustKind::RefersTo(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn points_to(&self) -> Option<&RustType> {
        match &self.kind {
            RustKind::PointsTo(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn slice_of(&self) -> Option<&RustType> {
        match &self.kind {
            RustKind::SliceOf(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn array_length(&self) -> Option<u32> {
        match &self.kind {
            RustKind::Array { length, .. } => Some(*length),
            _ => None,
        }
    }

    /// The type parameter of `MaybeUninit<T>`.
    fn maybe_uninit_param(&self) -> &RustType {
        match &self.kind {
            RustKind::Template { name, params } if name == "MaybeUninit" => &params[0],
            _ => panic!("Not a MaybeUninit<T>: {self:?}"),
        }
    }

    fn handle_array_type(
        vk_xml_type: &VkXmlType,
        from_vk_xml_type: fn(&VkXmlType) -> RustType,
    ) -> RustType {
        assert!(
            !vk_xml_type.dimensions.is_empty(),
            "vk_xml_type must be an array type"
        );
        let mut dimensions = vk_xml_type.dimensions.clone();
        let length = dimensions.pop().unwrap_or_default();
        assert!(
            !vk_xml_type.is_optional,
            "Optional array is not supported yet"
        );
        let element_vk_xml_type = VkXmlType {
            dimensions,
            ..vk_xml_type.clone()
        };
        let element = if element_vk_xml_type.dimensions.is_empty() {
            from_vk_xml_type(&element_vk_xml_type)
        } else {
            RustType::handle_array_type(&element_vk_xml_type, from_vk_xml_type)
        };
        RustType::new(
            vk_xml_type.is_const,
            vk_xml_type.is_optional,
            RustKind::Array {
                element: Box::new(element),
                length,
            },
        )
    }

    pub fn from_vk_xml_type(vk_xml_type: &VkXmlType) -> RustType {
        if !vk_xml_type.dimensions.is_empty() {
            return RustType::new(
                true,
                false,
                RustKind::RefersTo(Box::new(RustType::handle_array_type(
                    vk_xml_type,
                    RustType::from_vk_xml_type,
                ))),
            );
        }

        if let Some(name) = vk_xml_type.name() {
            let name = if name == "VkBool32" {
                "bool".to_owned()
            } else {
                decayed_type_to_rust_type(name)
            };
            return RustType::new(
                vk_xml_type.is_const,
                vk_xml_type.is_optional,
                RustKind::Name(name),
            );
        }

        let points_to = vk_xml_type
            .points_to()
            .expect("A VkXmlType is either a basic type or a pointer type.");

        match vk_xml_type.len {
            Some(VkXmlLenKind::NullTerminated) => {
                assert!(
                    points_to.name() == Some("char") && points_to.is_const,
                    "null-terminated only supports const char* type."
                );
                return RustType::new(
                    true,
                    vk_xml_type.is_optional,
                    RustKind::RefersTo(Box::new(RustType::named("str"))),
                );
            }
            Some(VkXmlLenKind::Variable) => {
                let mut element = if points_to.name() == Some("void") {
                    // void* is actually an array of zero-able bytes
                    RustType::new(points_to.is_const, true, RustKind::Name("u8".to_owned()))
                } else {
                    RustType::from_vk_xml_type(points_to)
                };
                // For mutable pointer parameters in vk.xml, they can be uninitialized (e.g.
                // pPhysicalDevices in vkEnumeratePhysicalDevices), which makes it invalid to use the
                // slice type (&mut [T]) to be the Rust interface type. The Rust slice type requires
                // the underlying data be initialized. We use &mut [MaybeUninit<T>] here instead.
                if !points_to.is_const {
                    element = RustType::new(
                        false,
                        false,
                        RustKind::Template {
                            name: "MaybeUninit".to_owned(),
                            params: vec![element],
                        },
                    );
                }
                return RustType::new(
                    vk_xml_type.is_const,
                    vk_xml_type.is_optional,
                    RustKind::SliceOf(Box::new(element)),
                );
            }
            None => {}
        }

        if points_to.name().is_some_and(is_opaque_type) {
            // A pointer to an opaque type will translate to a pointer
            return RustType::new(
                vk_xml_type.is_const,
                vk_xml_type.is_optional,
                RustKind::PointsTo(Box::new(RustType::from_vk_xml_type(points_to))),
            );
        }
        let mut refers_to = RustType::from_vk_xml_type(points_to);
        if !points_to.is_const {
            // For mutable pointer parameters in vk.xml, they can be uninitialized (e.g. pDevice in
            // vkCreateDevice), which makes it invalid to use the reference type (&mut T) to be the
            // Rust interface type. The Rust reference type requires the underlying data be
            // initialized. We use &mut MaybeUninit<T> here.
            refers_to = RustType::new(
                points_to.is_const,
                false,
                RustKind::Template {
                    name: "MaybeUninit".to_owned(),
                    params: vec![refers_to],
                },
            );
        }
        RustType::new(
            vk_xml_type.is_const,
            vk_xml_type.is_optional,
            RustKind::RefersTo(Box::new(refers_to)),
        )
    }

    pub fn from_vk_xml_type_for_ffi(vk_xml_type: &VkXmlType) -> RustType {
        if !vk_xml_type.dimensions.is_empty() {
            return RustType::new(
                true,
                vk_xml_type.is_optional,
                RustKind::PointsTo(Box::new(RustType::handle_array_type(
                    vk_xml_type,
                    RustType::from_vk_xml_type_for_ffi,
                ))),
            );
        }

        if let Some(name) = vk_xml_type.name() {
            // Suppress the generation of Option<T> at the FFI boundary.
            return RustType::new(
                vk_xml_type.is_const,
                false,
                RustKind::Name(decayed_type_to_rust_type(name)),
            );
        }

        let points_to = vk_xml_type
            .points_to()
            .expect("A VkXmlType is either a basic type or a pointer type.");
        // Suppress the generation of Option<T>
        RustType::new(
            vk_xml_type.is_const,
            false,
            RustKind::PointsTo(Box::new(RustType::from_vk_xml_type_for_ffi(points_to))),
        )
    }

    fn wrap_option(&self, inner: String) -> String {
        if self.is_optional {
            format!("Option<{inner}>")
        } else {
            inner
        }
    }

    pub fn to_rust_string(&self) -> String {
        let inner = match &self.kind {
            // We do nothing on is_optional, optional=false on scalar type doesn't necessarily mean
            // the value must be non-zero.
            RustKind::Name(name) => return name.clone(),
            RustKind::Template { name, params } => {
                let params: Vec<String> = params.iter().map(RustType::to_rust_string).collect();
                return self.wrap_option(format!("{name}<{}>", params.join(", ")));
            }
            RustKind::RefersTo(inner) | RustKind::PointsTo(inner) | RustKind::SliceOf(inner) => {
                inner
            }
            RustKind::Array { element, .. } => element,
        };
        assert!(
            inner.name() != Some("bool"),
            "This interface breaks the ABI, and won't be efficient."
        );
        let text = match &self.kind {
            RustKind::PointsTo(points_to) => {
                if points_to.is_const {
                    format!("*const {}", points_to.to_rust_string())
                } else {
                    format!("*mut {}", points_to.to_rust_string())
                }
            }
            RustKind::SliceOf(slice_of) => {
                let mut_mod = if slice_of.is_const { "" } else { "mut " };
                format!("&{mut_mod}[{}]", slice_of.to_rust_string())
            }
            RustKind::RefersTo(refers_to) => {
                let mut_mod = if refers_to.is_const { "" } else { "mut " };
                format!("&{mut_mod}{}", refers_to.to_rust_string())
            }
            RustKind::Array { element, length } => {
                format!("[{}; {length}]", element.to_rust_string())
            }
            RustKind::Name(_) | RustKind::Template { .. } => unreachable!(),
        };
        self.wrap_option(text)
    }
}

#[derive(Clone, Debug)]
pub struct RustParam {
    pub name: String,
    pub ty: RustType,
}

impl RustParam {
    pub fn from_vk_xml_param(vk_xml_param: &VkXmlParam) -> RustParam {
        RustParam {
            name: format!("_{}", camel_case_to_snake_case(&vk_xml_param.name)),
            ty: RustType::from_vk_xml_type(&vk_xml_param.ty),
        }
    }

    pub fn param_name_from_vk_xml_param_for_ffi(vk_xml_param_name: &str) -> String {
        escape_rust_keywords(&camel_case_to_snake_case(vk_xml_param_name))
    }

    pub fn from_vk_xml_param_for_ffi(vk_xml_param: &VkXmlParam) -> RustParam {
        RustParam {
            name: RustParam::param_name_from_vk_xml_param_for_ffi(&vk_xml_param.name),
            ty: RustType::from_vk_xml_type_for_ffi(&vk_xml_param.ty),
        }
    }

    pub fn to_rust_string(&self) -> String {
        format!("{}: {}", self.name, self.ty.to_rust_string())
    }
}

pub struct ReturnInfo {
    pub main_source_vk_xml_params: Vec<VkXmlParam>,
    pub rust_method_return_type: String,
}

pub struct ParamInfo {
    pub main_source_vk_xml_param: VkXmlParam,
    pub rust_method_param: RustParam,
}

/// How a command becomes a hook method: its return type and the parameters the method keeps.
pub struct VkXmlToRustMethodInfo {
    pub return_info: ReturnInfo,
    pub rust_method_name: String,
    pub parameters: Vec<ParamInfo>,
}

impl VkXmlToRustMethodInfo {
    pub fn from_vk_xml_command(vk_xml_cmd: &VkXmlCommand) -> VkXmlToRustMethodInfo {
        let mut vk_xml_params = vk_xml_cmd.parameters.clone();
        let return_info = Self::return_info(vk_xml_cmd, &mut vk_xml_params);
        let mut rust_params = Self::parameters(&vk_xml_params);

        if vk_xml_cmd.name == "vkCreateDevice" {
            // Also pass the current VkLayerDeviceLink and the pDevice to the layer.
            rust_params.insert(
                2,
                ParamInfo {
                    main_source_vk_xml_param: vk_xml_params[1].clone(),
                    rust_method_param: RustParam {
                        name: "_layer_device_link".to_owned(),
                        ty: RustType::new(
                            true,
                            false,
                            RustKind::RefersTo(Box::new(RustType::named("VkLayerDeviceLink"))),
                        ),
                    },
                },
            );
        }
        let name = &vk_xml_cmd.name;
        VkXmlToRustMethodInfo {
            return_info,
            rust_method_name: camel_case_to_snake_case(name.strip_prefix("vk").unwrap_or(name)),
            parameters: rust_params,
        }
    }

    /// The return info; removes the parameters it returns from `vk_xml_params`.
    fn return_info(vk_xml_cmd: &VkXmlCommand, vk_xml_params: &mut Vec<VkXmlParam>) -> ReturnInfo {
        let return_c_type = vk_xml_cmd.return_type.as_str();
        let none = |rust_method_return_type: String| ReturnInfo {
            main_source_vk_xml_params: vec![],
            rust_method_return_type,
        };
        if return_c_type == "VkBool32" {
            // We use bool instead of vk::Bool32 for VkBool32 in our layer trait interfaces.
            return none("LayerResult<bool>".to_owned());
        }
        if !["void", "VkResult"].contains(&return_c_type) {
            // Handle return types that can't be combined with extra out parameters, e.g.
            // vkGetBufferOpaqueCaptureAddress which returns uint64_t.
            return none(format!(
                "LayerResult<{}>",
                decayed_type_to_rust_type(return_c_type)
            ));
        }

        assert!(
            !vk_xml_params.is_empty(),
            "Unexpected command with 0 parameters."
        );
        // TODO: add support to multiple out parameters.
        let last_param_type = &vk_xml_cmd.parameters[vk_xml_cmd.parameters.len() - 1].ty;

        let get_rust_method_return_type = |inner_type: &str| -> String {
            if return_c_type == "VkResult" {
                format!("LayerResult<VkResult<{inner_type}>>")
            } else {
                format!("LayerResult<{inner_type}>")
            }
        };

        let unit = || none(get_rust_method_return_type("()"));
        if last_param_type.decayed_type().is_struct() {
            // For struct type, we don't treat it as a return type given the complication of pNext
            // chain of the return parameter, and possible input fields.
            return unit();
        }
        if vk_xml_cmd.name == "vkCreateDevice" {
            // We don't treat VkDevice just as a return value, because the layer is supposed to pass
            // through the passed in pDevice parameter.
            return unit();
        }
        let Some(points_to) = last_param_type.points_to() else {
            // If the last parameter is not a pointer, it's not an out parameter, and won't be
            // included in the return type
            return unit();
        };
        // The last parameter is a C pointer.
        if points_to.is_const {
            // The last parameter is not a mutable pointer, so it's not an out parameter.
            return unit();
        }

        let last_rust_method_type = RustType::from_vk_xml_type(last_param_type);

        if last_param_type.len.is_none() {
            // If the last parameter is a mutable pointer of a simple type. It's part of the return
            // type. If the last parameter is void*, it's possible that the spec is missing the len
            // attribute(e.g. vkGetSamplerOpaqueCaptureDescriptorDataEXT), but we are sure, so we
            // don't transform the parameter type, and it won't be part of the return type.
            if points_to.name() == Some("void") {
                return unit();
            }
            let refers_to = last_rust_method_type
                .refers_to()
                .expect("A mutable pointer must be a reference");
            let rust_return_type = refers_to.maybe_uninit_param();
            let return_type = get_rust_method_return_type(&rust_return_type.to_rust_string());
            return ReturnInfo {
                main_source_vk_xml_params: vec![vk_xml_params
                    .pop()
                    .unwrap_or_else(|| unreachable!())],
                rust_method_return_type: return_type,
            };
        }

        // The last parameter is a C array.
        if points_to.name() == Some("void") {
            // For an array of void, we don't return a Vec, as the return value can be large, and we
            // should use the passed-in memory as the storage.
            return unit();
        }

        // TODO: If the len_var is a const parameter, we should still pass the slice as the
        // interface. In this case, if we return a Vec and remove this parameter, we don't know the
        // length to use to call into the Vulkan API.
        // TODO: returning Vec<_> could cause extra copy here compared to directly passing a mutable
        // slice to the layer trait. Even worse, the layer won't be able to access the length of the
        // passed in write buffer. Restrict the use of Vec<_> to only return handles
        let len_var = vk_xml_params[vk_xml_params.len() - 1].len_var.clone();
        if let Some(len_param_index) = vk_xml_params
            .iter()
            .position(|param| Some(&param.name) == len_var.as_ref())
        {
            vk_xml_params.remove(len_param_index);
        }
        let slice_of = last_rust_method_type
            .slice_of()
            .expect("A mutable array must be a slice");
        let element_type = slice_of.maybe_uninit_param();
        let return_type =
            get_rust_method_return_type(&format!("Vec<{}>", element_type.to_rust_string()));
        ReturnInfo {
            main_source_vk_xml_params: vec![vk_xml_params.pop().unwrap_or_else(|| unreachable!())],
            rust_method_return_type: return_type,
        }
    }

    /// The method's parameters once the returned ones are removed. Some parameters do not appear in
    /// the method, because they are included in another parameter's type (e.g. in
    /// get_ray_tracing_shader_group_handles_khr p_data is a slice. data_size is the length of
    /// p_data, which is included in the length of the slice).
    fn parameters(vk_xml_params: &[VkXmlParam]) -> Vec<ParamInfo> {
        let len_params: Vec<&String> = vk_xml_params
            .iter()
            .filter_map(|param| param.len_var.as_ref())
            .collect();
        let mut result = vec![];
        for (i, param) in vk_xml_params.iter().enumerate() {
            // Remove the leading VkInstance or VkDevice parameter, since it should be included in
            // &self.
            if i == 0 && matches!(param.ty.name(), Some("VkInstance" | "VkDevice")) {
                continue;
            }
            // Any non-pointer length parameters are removed, as they are always included in the
            // correspondent slice parameter
            if len_params.contains(&&param.name) && param.ty.name().is_some() {
                continue;
            }
            result.push(ParamInfo {
                main_source_vk_xml_param: param.clone(),
                rust_method_param: RustParam::from_vk_xml_param(param),
            });
        }
        result
    }
}

pub struct RustMethod {
    pub return_type: String,
    pub name: String,
    pub parameters: Vec<RustParam>,
    pub vk_xml_cmd: VkXmlCommand,
}

/// The generic parameters a method takes for its VkBool32 arrays: Python's generator of `T`, `U`,
/// `V`, `W`, which `next()` advances and `send()` gives the bound of the name it yielded last.
struct TypeParams {
    started: bool,
    position: usize,
    used: Vec<(&'static str, Option<&'static str>)>,
}

impl TypeParams {
    const NAMES: [&'static str; 4] = ["T", "U", "V", "W"];

    fn resume(&mut self, restriction: Option<&'static str>) -> &'static str {
        if !self.started {
            self.started = true;
        } else {
            self.used.push((Self::NAMES[self.position], restriction));
            self.position += 1;
        }
        assert!(
            self.position < Self::NAMES.len(),
            "Too many generic type parameters."
        );
        Self::NAMES[self.position]
    }
}

impl RustMethod {
    pub fn from_vk_xml_command(vk_xml_cmd: &VkXmlCommand) -> RustMethod {
        let info = VkXmlToRustMethodInfo::from_vk_xml_command(vk_xml_cmd);
        RustMethod {
            name: info.rust_method_name,
            return_type: info.return_info.rust_method_return_type,
            parameters: info
                .parameters
                .into_iter()
                .map(|param| param.rust_method_param)
                .collect(),
            vk_xml_cmd: vk_xml_cmd.clone(),
        }
    }

    pub fn to_rust_string(&self) -> String {
        let mut type_params = TypeParams {
            started: false,
            position: 0,
            used: vec![],
        };
        // TODO: move this logic to RustMethod generation process so that the Rust type
        // representation is more natural
        let mut params = vec![];
        for param in &self.parameters {
            if param.ty.slice_of().and_then(RustType::name) == Some("bool") {
                let type_param = type_params.resume(None);
                // Not 'static: the iterator reads the application's array, which lives only for the
                // call, so a hook must not be able to keep it (a safe use after free).
                type_params.resume(Some("Iterator<Item = bool>"));
                // Generic type for bool for efficient ABI breaking API.
                params.push(format!("{}: {type_param}", param.name));
            } else {
                params.push(param.to_rust_string());
            }
        }
        // Prevent the Rust linter from complaining explicit -> ()
        assert!(
            self.return_type != "()",
            "Unexpected rust return type: (). All return type should be LayerResult<T>"
        );
        let mut generic_params = String::new();
        if !type_params.used.is_empty() {
            let used: Vec<String> = type_params
                .used
                .iter()
                .map(|(name, restriction)| format!("{name}: {}", restriction.unwrap_or("None")))
                .collect();
            generic_params = format!("<{}>", used.join(", "));
        }
        format!(
            "fn {}{generic_params}(&self, {}) -> {}",
            self.name,
            params.join(", "),
            self.return_type
        )
    }
}

/// `textwrap.wrap` with words broken only at spaces, never inside a word.
fn wrap(text: &str, width: usize, initial_indent: &str, subsequent_indent: &str) -> Vec<String> {
    let mut lines = vec![];
    let mut line = String::new();
    let mut indent = initial_indent;
    for word in text.split_whitespace() {
        if line.is_empty() {
            line = format!("{indent}{word}");
        } else if line.len() + 1 + word.len() <= width {
            line.push(' ');
            line.push_str(word);
        } else {
            lines.push(std::mem::take(&mut line));
            indent = subsequent_indent;
            line = format!("{indent}{word}");
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

pub fn generate_unhandled_command_comments<'c>(
    unhandled_commands: impl Iterator<Item = &'c UnhandledCommand>,
) -> String {
    let mut lines = vec!["// Unhandled commands:".to_owned()];
    // Wrapped at the comment width in rustfmt.toml (100), as nightly rustfmt's wrap_comments did,
    // so the pinned stable rustfmt formats the output the same.
    for command in unhandled_commands {
        lines.extend(wrap(
            &format!("{}: {}", command.name, command.reason),
            100,
            "// * ",
            "//   ",
        ));
    }
    if lines.len() == 1 {
        return String::new();
    }
    lines.iter().map(|line| format!("{line}\n")).collect()
}

#[cfg(test)]
mod tests {
    use super::super::tests::vk_xml_command;
    use super::*;

    fn method(name: &str) -> RustMethod {
        RustMethod::from_vk_xml_command(&vk_xml_command(name))
    }

    fn param_types(method: &RustMethod) -> Vec<String> {
        method
            .parameters
            .iter()
            .map(|param| param.ty.to_rust_string())
            .collect()
    }

    #[test]
    fn get_image_sparse_memory_requirements2() {
        let method = method("vkGetImageSparseMemoryRequirements2");
        assert_eq!(method.return_type, "LayerResult<()>");
        assert_eq!(
            param_types(&method),
            [
                "&vk::ImageSparseMemoryRequirementsInfo2",
                "&mut MaybeUninit<u32>",
                "Option<&mut [MaybeUninit<vk::SparseImageMemoryRequirements2>]>",
            ]
        );
    }

    #[test]
    fn allocate_descriptor_sets() {
        let method = method("vkAllocateDescriptorSets");
        assert_eq!(param_types(&method)[0], "&vk::DescriptorSetAllocateInfo");
        assert_eq!(
            method.return_type,
            "LayerResult<VkResult<Vec<vk::DescriptorSet>>>"
        );
    }

    #[test]
    fn get_device_queue() {
        let method = method("vkGetDeviceQueue");
        assert_eq!(method.parameters.len(), 2);
        assert_eq!(method.return_type, "LayerResult<vk::Queue>");
    }

    #[test]
    fn queue_submit() {
        let method = method("vkQueueSubmit");
        assert_eq!(
            param_types(&method),
            ["vk::Queue", "&[vk::SubmitInfo]", "vk::Fence"]
        );
    }

    #[test]
    fn get_ray_tracing_shader_group_handles_khr() {
        let method = method("vkGetRayTracingShaderGroupHandlesKHR");
        assert_eq!(
            param_types(&method),
            ["vk::Pipeline", "u32", "u32", "&mut [MaybeUninit<u8>]"]
        );
        assert_eq!(method.return_type, "LayerResult<VkResult<()>>");
    }

    #[test]
    fn get_sampler_opaque_capture_descriptor_data_ext() {
        let method = method("vkGetSamplerOpaqueCaptureDescriptorDataEXT");
        assert_eq!(
            param_types(&method),
            ["&vk::SamplerCaptureDescriptorDataInfoEXT", "*mut c_void"]
        );
    }

    #[test]
    fn ffi_cmd_set_blend_constants() {
        let command = vk_xml_command("vkCmdSetBlendConstants");
        let param = RustParam::from_vk_xml_param_for_ffi(&command.parameters[1]);
        assert_eq!(param.to_rust_string(), "blend_constants: *const [f32; 4]");
    }

    #[test]
    fn enumerate_physical_device_queue_family_performance_query_counters_khr() {
        let method = method("vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
        let types = param_types(&method);
        assert_eq!(types[2], "&mut MaybeUninit<u32>");
        assert_eq!(
            types[3],
            "Option<&mut [MaybeUninit<vk::PerformanceCounterKHR>]>"
        );
        assert_eq!(
            types[4],
            "Option<&mut [MaybeUninit<vk::PerformanceCounterDescriptionKHR>]>"
        );
    }

    #[test]
    fn map_memory() {
        let method = method("vkMapMemory");
        assert_eq!(method.parameters.len(), 4);
        assert_eq!(
            method.return_type,
            "LayerResult<VkResult<Option<*mut c_void>>>"
        );
    }

    #[test]
    fn get_physical_device_surface_support_khr() {
        let command = vk_xml_command("vkGetPhysicalDeviceSurfaceSupportKHR");
        let return_info = VkXmlToRustMethodInfo::from_vk_xml_command(&command).return_info;
        assert_eq!(return_info.main_source_vk_xml_params.len(), 1);
        assert_eq!(return_info.main_source_vk_xml_params[0].name, "pSupported");
    }

    #[test]
    fn bool_array_hook_takes_a_generic_iterator() {
        let method = method("vkCmdSetColorWriteEnableEXT");
        assert_eq!(
            method.to_rust_string(),
            "fn cmd_set_color_write_enable_ext<T: Iterator<Item = bool>>(&self, _command_buffer: vk::CommandBuffer, \
             _p_color_write_enables: T) -> LayerResult<()>"
        );
    }

    #[test]
    fn case_conversions() {
        assert_eq!(
            camel_case_to_snake_case("GetPhysicalDeviceXcbPresentationSupportKHR"),
            "get_physical_device_xcb_presentation_support_khr"
        );
        assert_eq!(camel_case_to_snake_case("pIOSSurface"), "p_ios_surface");
        assert_eq!(
            snake_case_to_upper_camel_case("cmd_set_color_write_enable_ext"),
            "CmdSetColorWriteEnableExt"
        );
        assert_eq!(
            snake_case_to_upper_camel_case("KHR_8bit_storage"),
            "KHR8bitStorage"
        );
    }

    #[test]
    fn alias_representative_prefers_core_then_khr_then_ext() {
        let mut aliases = VulkanAliases::default();
        aliases.add_alias("vkCmdSetCullMode", None);
        aliases.add_alias("vkCmdSetCullModeEXT", Some("vkCmdSetCullMode"));
        aliases.add_alias("vkFooEXT", None);
        aliases.add_alias("vkFooKHR", Some("vkFooEXT"));
        assert_eq!(
            aliases.get_represent_name("vkCmdSetCullModeEXT").as_deref(),
            Some("vkCmdSetCullMode")
        );
        assert_eq!(
            aliases.get_represent_name("vkFooEXT").as_deref(),
            Some("vkFooKHR")
        );
        assert_eq!(aliases.get_represent_name("vkUnknown"), None);
    }

    #[test]
    fn unhandled_command_comments_wrap_at_100_columns() {
        let comments = generate_unhandled_command_comments(
            [UnhandledCommand::find("vkCmdBuildAccelerationStructuresIndirectKHR").unwrap()].iter(),
        );
        assert_eq!(
            comments,
            "// Unhandled commands:\n\
             // * vkCmdBuildAccelerationStructuresIndirectKHR: Dynamic multi-dimensional array bindings are not\n\
             //   supported yet.\n"
        );
    }
}
