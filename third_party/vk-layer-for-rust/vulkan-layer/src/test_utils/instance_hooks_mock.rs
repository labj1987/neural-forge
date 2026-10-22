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
// Modified for Neural Forge (2026): ported to ash 0.38 (UPSTREAM.md).

use std::mem::MaybeUninit;

use mockall::mock;

use crate::{InstanceHooks, LayerResult, VkLayerDeviceLink};
use ash::{prelude::VkResult, vk};

// We don't automock the original trait, because that hurts compilation speed significantly.
mock! {
    pub InstanceHooks {}
    impl InstanceHooks for InstanceHooks {
        fn destroy_surface_khr<'a, 'b>(
            &self,
            surface: vk::SurfaceKHR,
            p_allocator: Option<&'a vk::AllocationCallbacks<'b>>,
        ) -> LayerResult<()>;

        fn create_device<'a, 'b, 'c, 'd>(
            &self,
            _physical_device: vk::PhysicalDevice,
            _p_create_info: &'c vk::DeviceCreateInfo<'d>,
            _layer_device_link: &VkLayerDeviceLink,
            _p_allocator: Option<&'a vk::AllocationCallbacks<'b>>,
            _p_device: &mut MaybeUninit<vk::Device>,
        ) -> LayerResult<VkResult<()>>;
    }
}
