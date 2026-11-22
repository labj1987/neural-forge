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
// Modified for Neural Forge (2026): the part of Khronos's registry framework (Vulkan-Headers'
// registry/reg.py and generator.py, Copyright 2013-2024 The Khronos Group Inc., Apache-2.0) the
// generators used, written in Rust (UPSTREAM.md).

//! Reads `vk.xml` and works out which commands the generators see, in which order: what Khronos's
//! `Registry.apiGen()` did with the options the generators ran with (API `vulkan`, no profile,
//! every version, the extensions whose `supported` names `vulkan`, none added or removed, all
//! emitted, the default feature sort, dependencies followed, command aliases not required).

use roxmltree::{Document, Node};
use std::collections::{HashMap, HashSet};

const API_NAME: &str = "vulkan";

/// reg.py's `apiNameMatch`: whether an `api`/`supported` attribute names the API (no attribute
/// always matches).
fn api_name_match(supported: Option<&str>) -> bool {
    supported.is_none_or(|supported| supported.split(',').any(|api| api == API_NAME))
}

/// Whether an element survives reg.py's `stripNonmatchingAPIs`, which deletes every element whose
/// `api` attribute does not name the API (with everything below it).
fn kept(node: &Node) -> bool {
    api_name_match(node.attribute("api"))
}

/// The child elements named `tag` (ElementTree's `findall(tag)` on the stripped tree).
pub fn children<'a, 'i>(
    node: Node<'a, 'i>,
    tag: &'static str,
) -> impl Iterator<Item = Node<'a, 'i>> {
    node.children()
        .filter(move |child| child.is_element() && child.has_tag_name(tag) && kept(child))
}

/// The first child element named `tag` (ElementTree's `find(tag)`).
pub fn child<'a, 'i>(node: Node<'a, 'i>, tag: &'static str) -> Option<Node<'a, 'i>> {
    children(node, tag).next()
}

/// The descendant elements named `tag`, in document order (ElementTree's `findall('.//tag')`).
pub fn descendants<'a, 'i>(node: Node<'a, 'i>, tag: &'static str) -> Vec<Node<'a, 'i>> {
    fn walk<'a, 'i>(node: Node<'a, 'i>, tag: &str, found: &mut Vec<Node<'a, 'i>>) {
        for child in node
            .children()
            .filter(|child| child.is_element() && kept(child))
        {
            if child.has_tag_name(tag) {
                found.push(child);
            }
            walk(child, tag, found);
        }
    }
    let mut found = vec![];
    walk(node, tag, &mut found);
    found
}

/// Concatenates the text nodes in `nodes` up to the first element, skipping comments (ElementTree's
/// parser drops comments and joins the text around them).
fn text_until_element<'a, 'i: 'a>(nodes: impl Iterator<Item = Node<'a, 'i>>) -> Option<String> {
    let mut text: Option<String> = None;
    for node in nodes {
        if node.is_element() {
            break;
        }
        if let Some(part) = node.is_text().then(|| node.text()).flatten() {
            text.get_or_insert_with(String::new).push_str(part);
        }
    }
    text
}

/// ElementTree's `elem.text`: the text before the element's first child element.
pub fn et_text(node: Node) -> Option<String> {
    text_until_element(node.children())
}

/// ElementTree's `elem.tail`: the text after the element, before its next sibling element.
pub fn et_tail(node: Node) -> Option<String> {
    text_until_element(node.next_siblings().skip(1))
}

/// ElementTree's `"".join(elem.itertext())`: all the text inside the element.
pub fn itertext(node: Node) -> String {
    fn walk(node: Node, text: &mut String) {
        for child in node.children() {
            if let Some(part) = child.is_text().then(|| child.text()).flatten() {
                text.push_str(part);
            } else if child.is_element() && kept(&child) {
                walk(child, text);
            }
        }
    }
    let mut text = String::new();
    walk(node, &mut text);
    text
}

/// A `<feature>` (a core version) or an `<extension>` (reg.py's `FeatureInfo`).
pub struct Feature<'a, 'i> {
    pub node: Node<'a, 'i>,
    pub name: String,
    /// `<feature>`, not `<extension>`.
    pub is_version: bool,
    /// The version's `number` attribute ("1.3"); "0" for an extension.
    pub version_number: String,
    /// The extension's vendor (the second `_` part of its name); `VERSION` for a version.
    pub category: String,
    sortorder: i64,
    number: i64,
}

impl<'a, 'i> Feature<'a, 'i> {
    fn new(node: Node<'a, 'i>) -> Self {
        let name = node
            .attribute("name")
            .unwrap_or_else(|| panic!("A feature or extension without a name"))
            .to_owned();
        let sortorder = node.attribute("sortorder").map_or(0, |order| {
            order.parse().expect("sortorder is not an integer")
        });
        if node.has_tag_name("feature") {
            Feature {
                node,
                version_number: node.attribute("number").unwrap_or_default().to_owned(),
                name,
                is_version: true,
                category: "VERSION".to_owned(),
                sortorder,
                number: 0,
            }
        } else {
            let category = name
                .split('_')
                .nth(1)
                .unwrap_or_else(|| panic!("{name} has no vendor"))
                .to_owned();
            Feature {
                node,
                name,
                is_version: false,
                version_number: "0".to_owned(),
                category,
                sortorder,
                number: node.attribute("number").map_or(0, |number| {
                    number.parse().expect("number is not an integer")
                }),
            }
        }
    }

    /// generator.py's `regSortFeatures` key: explicit sort order, then versions (VKSC ones after
    /// the others), Khronos extensions, the other extensions, then version, then extension number.
    fn sort_key(&self) -> (i64, u8, f64, i64) {
        let category = if self.is_version {
            if self.name.starts_with("VKSC") {
                1
            } else {
                0
            }
        } else if ["ARB", "KHR", "OES"].contains(&self.category.to_uppercase().as_str()) {
            2
        } else {
            3
        };
        let version: f64 = self.version_number.parse().unwrap_or_else(|_| {
            panic!(
                "{}: version {} is not a number",
                self.name, self.version_number
            )
        });
        (self.sortorder, category, version, self.number)
    }
}

struct TypeEntry<'a, 'i> {
    node: Node<'a, 'i>,
    required: bool,
}

struct EnumEntry {
    alias: Option<String>,
    required: bool,
}

/// A command as reg.py's `CmdInfo` has it. An alias (`<command name="vkFooKHR" alias="vkFoo"/>`)
/// gets a copy of the aliased command's definition under its own name.
struct CmdEntry<'a, 'i> {
    /// The element with the `<proto>` and the `<param>`s.
    node: Node<'a, 'i>,
    alias: Option<String>,
    required: bool,
    declared: bool,
}

/// A command handed to a generator (an `OutputGenerator.genCmd` call).
pub struct Command<'a, 'i> {
    pub name: String,
    pub alias: Option<String>,
    pub node: Node<'a, 'i>,
    /// `successcodes`, less the codes of what was not required (`stripUnsupportedAPIs`).
    pub success_codes: Vec<String>,
}

/// What `genType`/`genGroup` gave a generator for a type it was handed.
pub struct TypeInfo {
    /// A `<type category="struct">`.
    pub is_struct: bool,
}

/// The registry after `apiGen()`'s selection: the features generated, in order, each with the
/// commands generated for it.
pub struct Registry<'a, 'i> {
    types: HashMap<String, TypeEntry<'a, 'i>>,
    groups: HashSet<String>,
    enums: HashMap<String, EnumEntry>,
    commands: HashMap<String, CmdEntry<'a, 'i>>,
    pub features: Vec<(Feature<'a, 'i>, Vec<Command<'a, 'i>>)>,
}

/// reg.py's `matchAPIProfile` for API `vulkan` and no profile.
fn match_api_profile(node: Node) -> bool {
    if let Some(api) = node.attribute("api") {
        if api != API_NAME {
            return false;
        }
    }
    if let Some(profile) = node.attribute("profile") {
        panic!("No profile requested, but 'profile' attribute is present with value '{profile}'");
    }
    true
}

impl<'a, 'i> Registry<'a, 'i> {
    pub fn parse_document(xml: &'i str) -> Document<'i> {
        Document::parse(xml).unwrap_or_else(|e| panic!("vk.xml does not parse: {e}"))
    }

    pub fn new(document: &'a Document<'i>) -> Self {
        let root = document.root_element();
        let mut registry = Registry {
            types: HashMap::new(),
            groups: HashSet::new(),
            enums: HashMap::new(),
            commands: HashMap::new(),
            features: vec![],
        };
        registry.parse_tree(root);
        let features = registry.select_features(root);
        registry.tag_features(&features);
        registry.features = registry.generate(features);
        registry
    }

    /// reg.py's `parseTree`: the dictionaries of types, groups, enums, commands and features. The
    /// first definition of a name wins.
    fn parse_tree(&mut self, root: Node<'a, 'i>) {
        for types in children(root, "types") {
            for node in children(types, "type") {
                let name = match node.attribute("name") {
                    Some(name) => name.to_owned(),
                    None => child(node, "name")
                        .and_then(et_text)
                        .filter(|name| !name.is_empty())
                        .expect("Type without a name!"),
                };
                self.types.entry(name).or_insert(TypeEntry {
                    node,
                    required: false,
                });
            }
        }
        for enums in children(root, "enums") {
            self.groups
                .insert(enums.attribute("name").unwrap_or_default().to_owned());
            let required = enums.attribute("type").is_some();
            for node in children(enums, "enum") {
                self.add_enum(node, required);
            }
        }

        let mut aliases = vec![];
        for commands in children(root, "commands") {
            for node in children(commands, "command") {
                let name = match node.attribute("name") {
                    Some(name) => name.to_owned(),
                    None => child(node, "proto")
                        .and_then(|proto| child(proto, "name"))
                        .and_then(et_text)
                        .filter(|name| !name.is_empty())
                        .expect("Command without a name!"),
                };
                let alias = node.attribute("alias").map(str::to_owned);
                if let Some(alias) = &alias {
                    aliases.push((name.clone(), alias.clone()));
                }
                self.commands.entry(name).or_insert(CmdEntry {
                    node,
                    alias,
                    required: false,
                    declared: false,
                });
            }
        }
        // An alias gets a copy of what it aliases, under its own name.
        for (name, alias) in aliases {
            let Some(target) = self.commands.get(&alias) else {
                continue;
            };
            assert!(
                child(target.node, "proto")
                    .and_then(|proto| child(proto, "name"))
                    .is_some(),
                "{name} aliases {alias}, which has no definition"
            );
            let node = target.node;
            self.commands.insert(
                name,
                CmdEntry {
                    node,
                    alias: Some(alias),
                    required: false,
                    declared: false,
                },
            );
        }

        // Enums defined in <require> blocks.
        let extensions =
            children(root, "extensions").flat_map(|extensions| children(extensions, "extension"));
        for feature in children(root, "feature").chain(extensions) {
            for require in children(feature, "require") {
                for node in children(require, "enum") {
                    let defines = ["value", "bitpos", "alias"].iter().any(|attribute| {
                        !node.attribute(*attribute).unwrap_or_default().is_empty()
                    });
                    if node.attribute("extends").is_some() || defines {
                        self.add_enum(node, false);
                    }
                }
            }
        }
    }

    fn add_enum(&mut self, node: Node, required: bool) {
        let name = node.attribute("name").unwrap_or_default().to_owned();
        self.enums.entry(name).or_insert(EnumEntry {
            alias: node.attribute("alias").map(str::to_owned),
            required,
        });
    }

    /// The versions and extensions generated, sorted (`apiGen`'s selection and `regSortFeatures`).
    fn select_features(&self, root: Node<'a, 'i>) -> Vec<Feature<'a, 'i>> {
        let mut features = vec![];
        let mut seen = HashSet::new();
        for node in children(root, "feature") {
            let feature = Feature::new(node);
            if seen.insert(feature.name.clone()) {
                features.push(feature);
            }
        }
        let mut extensions = vec![];
        let mut seen = HashSet::new();
        for extensions_node in children(root, "extensions") {
            for node in children(extensions_node, "extension") {
                let extension = Feature::new(node);
                if seen.insert(extension.name.clone()) {
                    extensions.push(extension);
                }
            }
        }
        extensions.sort_by_key(|extension| extension.number);
        for extension in extensions {
            let mut supported = extension.node.attribute("supported");
            // The generator allows VK_ANDROID_native_buffer, which the registry does not mark as
            // supported.
            if extension.name == "VK_ANDROID_native_buffer" && supported.is_some() {
                supported = Some(API_NAME);
            }
            if api_name_match(supported) {
                features.push(extension);
            }
        }
        features.sort_by(|a, b| {
            a.sort_key()
                .partial_cmp(&b.sort_key())
                .expect("NaN version")
        });
        features
    }

    /// `apiGen`'s passes 1 and 2: tag what each feature requires, then what it removes.
    fn tag_features(&mut self, features: &[Feature<'a, 'i>]) {
        for feature in features {
            for require in
                children(feature.node, "require").filter(|require| match_api_profile(*require))
            {
                self.check_feature_dictionary(require);
                self.mark_required(require, true);
            }
        }
        for feature in features {
            for remove in
                children(feature.node, "remove").filter(|remove| match_api_profile(*remove))
            {
                self.mark_required(remove, false);
            }
        }
    }

    /// The failures of reg.py's `fillFeatureDictionary` (its dictionary itself is not used): a
    /// required type whose alias is unknown or whose category it has no entry for, an enum or a
    /// command it does not know.
    fn check_feature_dictionary(&self, require: Node) {
        const CATEGORIES: [&str; 11] = [
            "enumconstant",
            "command",
            "enum",
            "struct",
            "handle",
            "basetype",
            "include",
            "define",
            "bitmask",
            "union",
            "funcpointer",
        ];
        for type_node in children(require, "type") {
            let name = type_node.attribute("name").unwrap_or_default();
            let Some(mut info) = self.types.get(name) else {
                continue;
            };
            let mut alias = type_node
                .attribute("alias")
                .or_else(|| info.node.attribute("alias"));
            while let Some(name) = alias {
                info = self
                    .types
                    .get(name)
                    .unwrap_or_else(|| panic!("type {name} is not defined"));
                alias = info.node.attribute("alias");
            }
            let category = info.node.attribute("category");
            assert!(
                category.is_some_and(|category| CATEGORIES.contains(&category)),
                "type {name} has the category {category:?}, which the registry has no list for"
            );
        }
        for enum_node in children(require, "enum") {
            let name = enum_node.attribute("name").unwrap_or_default();
            assert!(
                enum_node.attribute("alias").is_some() || self.enums.contains_key(name),
                "{name} is not a known name"
            );
        }
        for command_node in children(require, "command") {
            let name = command_node.attribute("name").unwrap_or_default();
            assert!(
                command_node.attribute("alias").is_some() || self.commands.contains_key(name),
                "{name} is not a known name"
            );
        }
    }

    /// reg.py's `markRequired`.
    fn mark_required(&mut self, require: Node, required: bool) {
        for type_node in children(require, "type") {
            self.mark_type_required(type_node.attribute("name").unwrap_or_default(), required);
        }
        for enum_node in children(require, "enum") {
            self.mark_enum_required(enum_node.attribute("name").unwrap_or_default(), required);
        }
        for command_node in children(require, "command") {
            self.mark_cmd_required(command_node.attribute("name").unwrap_or_default(), required);
        }
    }

    /// reg.py's `markTypeRequired`: a required type requires the types and enums it is defined with.
    fn mark_type_required(&mut self, name: &str, required: bool) {
        let Some(info) = self.types.get(name) else {
            return;
        };
        let node = info.node;
        if required {
            for attribute in ["requires", "alias"] {
                if let Some(dependency) = node.attribute(attribute) {
                    if dependency != name {
                        self.mark_type_required(dependency, required);
                    }
                }
            }
            for subtype in descendants(node, "type") {
                let subtype = et_text(subtype).expect("A <type> without text");
                if subtype != name {
                    self.mark_type_required(&subtype, required);
                }
            }
            for subenum in descendants(node, "enum") {
                let subenum = et_text(subenum).expect("An <enum> without text");
                self.mark_enum_required(&subenum, required);
            }
            if let Some(bitvalues) = node.attribute("bitvalues") {
                self.mark_type_required(bitvalues, required);
            }
        }
        if let Some(info) = self.types.get_mut(name) {
            info.required = required;
        }
    }

    /// reg.py's `markEnumRequired`: an enum's alias goes with it.
    fn mark_enum_required(&mut self, name: &str, required: bool) {
        let Some(info) = self.enums.get_mut(name) else {
            return;
        };
        info.required = required;
        if let Some(alias) = info.alias.clone() {
            self.mark_enum_required(&alias, required);
        }
    }

    /// reg.py's `markCmdRequired`: a required command requires the types of its parameters.
    fn mark_cmd_required(&mut self, name: &str, required: bool) {
        let Some(info) = self.commands.get_mut(name) else {
            return;
        };
        info.required = required;
        let node = info.node;
        if required {
            for type_node in descendants(node, "type") {
                let type_name = et_text(type_node).expect("A <type> without text");
                self.mark_type_required(&type_name, required);
            }
        }
    }

    /// `apiGen`'s pass 3: each feature's required commands not generated before, in the order of
    /// its `<require>` blocks, an alias after the command it aliases (`generateFeature`).
    fn generate(
        &mut self,
        features: Vec<Feature<'a, 'i>>,
    ) -> Vec<(Feature<'a, 'i>, Vec<Command<'a, 'i>>)> {
        let mut generated = vec![];
        for feature in features {
            let mut commands = vec![];
            for require in children(feature.node, "require") {
                for command_node in children(require, "command") {
                    let name = command_node.attribute("name").unwrap_or_default();
                    self.generate_command(name, &mut commands);
                }
            }
            generated.push((feature, commands));
        }
        generated
    }

    fn generate_command(&mut self, name: &str, commands: &mut Vec<Command<'a, 'i>>) {
        let Some(info) = self.commands.get_mut(name) else {
            return;
        };
        if !info.required || info.declared {
            return;
        }
        info.declared = true;
        let alias = info.alias.clone();
        let node = info.node;
        if let Some(alias) = &alias {
            self.generate_command(alias, commands);
        }
        let success_codes = node
            .attribute("successcodes")
            .map(|codes| {
                codes
                    .split(',')
                    .filter(|code| self.enums.get(*code).is_some_and(|info| info.required))
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        commands.push(Command {
            name: name.to_owned(),
            alias,
            node,
            success_codes,
        });
    }

    /// The type a generator was handed under this name before it was handed a command using it:
    /// none for an unknown or unrequired type, or an enum type without a group of values.
    pub fn type_info(&self, name: &str) -> Option<TypeInfo> {
        let info = self.types.get(name)?;
        if !info.required {
            return None;
        }
        if info.node.attribute("category") == Some("enum") {
            match info.node.attribute("alias") {
                Some(alias) => assert!(
                    self.groups.contains(alias),
                    "{name} aliases {alias}, which has no group of values"
                ),
                None if !self.groups.contains(name) => return None,
                None => {}
            }
            return Some(TypeInfo { is_struct: false });
        }
        Some(TypeInfo {
            is_struct: info.node.attribute("category") == Some("struct"),
        })
    }
}
