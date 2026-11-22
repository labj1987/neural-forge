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
// Modified for Neural Forge (2026): the Python tests' registry helper (vk_xml_util.py's TestUtils), in
// Rust, on a fixture taken from vk.xml instead of the fetched headers (UPSTREAM.md).

//! The generators' test helpers: commands from `test_registry.xml`, the commands the tests use as
//! Vulkan-Headers 1.3.281's `vk.xml` defines them.

use super::registry::Registry;
use super::vk_xml::VkXmlCommand;

const TEST_REGISTRY: &str = include_str!("test_registry.xml");

/// The command `name` of the test registry, as the generators see it.
pub fn vk_xml_command(name: &str) -> VkXmlCommand {
    let document = Registry::parse_document(TEST_REGISTRY);
    let registry = Registry::new(&document);
    let command = registry
        .features
        .iter()
        .flat_map(|(_, commands)| commands)
        .find(|command| command.name == name)
        .unwrap_or_else(|| panic!("{name} is not in the test registry"));
    VkXmlCommand::from_command(command, &registry)
}

#[test]
fn an_alias_follows_the_command_it_aliases() {
    let document = Registry::parse_document(TEST_REGISTRY);
    let registry = Registry::new(&document);
    let names: Vec<&str> = registry.features[0]
        .1
        .iter()
        .map(|command| command.name.as_str())
        .collect();
    let alias = names
        .iter()
        .position(|name| *name == "vkGetCalibratedTimestampsEXT")
        .unwrap();
    let target = names
        .iter()
        .position(|name| *name == "vkGetCalibratedTimestampsKHR")
        .unwrap();
    assert!(target < alias);
    let command = &registry.features[0].1[alias];
    assert_eq!(
        command.alias.as_deref(),
        Some("vkGetCalibratedTimestampsKHR")
    );
    // An alias has the definition of the command it aliases, under its own name.
    assert_eq!(
        vk_xml_command("vkGetCalibratedTimestampsEXT")
            .parameters
            .len(),
        5
    );
}
