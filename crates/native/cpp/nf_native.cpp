// The layer's side of the vendored OpenDLSS-NR network (nf_native.h).
#include "nf_native.h"

#include "nf_assets.h"

#include <volk.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "kernels.h"
#include "nr_graph.h"
#include "nr_model.h"
#include "numeric.h"
#include "reference.h"
#include "vk_context.h"

namespace {

// A network of this process fell back to barriers (nf_native_fall_back_to_barriers): counter chaining
// faulted or timed out here, so no network opened later in the process chains again.
std::atomic<bool> g_fellBack{false};

void say(char* out, size_t len, const std::string& text) {
  if (!out || !len) return;
  snprintf(out, len, "%s", text.c_str());
}

// One feature flag the kernels need: where it lives in the Vulkan 1.1/1.2/1.3 aggregate structure
// (if it was promoted) and in its own structure, with the extension that structure belongs to.
struct Need {
  const char* name;
  VkStructureType aggregate;   // 0 when there is none
  size_t aggregateOffset;
  VkStructureType single;
  size_t singleSize;
  size_t singleOffset;
  const char* extension;
};

#define NEED_PROMOTED(field, AGG, aggType, SINGLE, singleType, ext)                                            \
  Need { #field, AGG, offsetof(aggType, field), SINGLE, sizeof(singleType), offsetof(singleType, field), ext }
#define NEED_EXTENSION(field, SINGLE, singleType, ext) \
  Need { #field, VkStructureType(0), 0, SINGLE, sizeof(singleType), offsetof(singleType, field), ext }

const Need kNeeds[] = {
    NEED_PROMOTED(storageBuffer16BitAccess, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, VkPhysicalDeviceVulkan11Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, VkPhysicalDevice16BitStorageFeatures,
                  VK_KHR_16BIT_STORAGE_EXTENSION_NAME),
    NEED_PROMOTED(uniformAndStorageBuffer16BitAccess, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                  VkPhysicalDeviceVulkan11Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
                  VkPhysicalDevice16BitStorageFeatures, VK_KHR_16BIT_STORAGE_EXTENSION_NAME),
    NEED_PROMOTED(storageBuffer8BitAccess, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, VkPhysicalDevice8BitStorageFeatures,
                  VK_KHR_8BIT_STORAGE_EXTENSION_NAME),
    NEED_PROMOTED(uniformAndStorageBuffer8BitAccess, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                  VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
                  VkPhysicalDevice8BitStorageFeatures, VK_KHR_8BIT_STORAGE_EXTENSION_NAME),
    NEED_PROMOTED(shaderFloat16, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, VkPhysicalDeviceShaderFloat16Int8Features,
                  VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME),
    NEED_PROMOTED(shaderInt8, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, VkPhysicalDeviceShaderFloat16Int8Features,
                  VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME),
    NEED_PROMOTED(vulkanMemoryModel, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, VkPhysicalDeviceVulkanMemoryModelFeatures,
                  VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME),
    NEED_PROMOTED(vulkanMemoryModelDeviceScope, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                  VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
                  VkPhysicalDeviceVulkanMemoryModelFeatures, VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME),
    NEED_PROMOTED(hostQueryReset, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, VkPhysicalDeviceHostQueryResetFeatures,
                  VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME),
    NEED_PROMOTED(bufferDeviceAddress, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, VkPhysicalDeviceBufferDeviceAddressFeatures,
                  VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME),
    NEED_PROMOTED(subgroupSizeControl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, VkPhysicalDeviceVulkan13Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, VkPhysicalDeviceSubgroupSizeControlFeatures,
                  VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME),
    NEED_PROMOTED(computeFullSubgroups, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, VkPhysicalDeviceVulkan13Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, VkPhysicalDeviceSubgroupSizeControlFeatures,
                  VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME),
    NEED_PROMOTED(synchronization2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, VkPhysicalDeviceVulkan13Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, VkPhysicalDeviceSynchronization2Features,
                  VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME),
    NEED_PROMOTED(maintenance4, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, VkPhysicalDeviceVulkan13Features,
                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, VkPhysicalDeviceMaintenance4Features,
                  VK_KHR_MAINTENANCE_4_EXTENSION_NAME),
    NEED_EXTENSION(cooperativeMatrix, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
                   VkPhysicalDeviceCooperativeMatrixFeaturesKHR, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME),
    NEED_EXTENSION(cooperativeMatrixWorkgroupScope, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
                   VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME),
    NEED_EXTENSION(cooperativeMatrixFlexibleDimensions, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
                   VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME),
    NEED_EXTENSION(cooperativeMatrixReductions, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
                   VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME),
    NEED_EXTENSION(cooperativeMatrixConversions, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
                   VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME),
    NEED_EXTENSION(cooperativeMatrixPerElementOperations, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
                   VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME),
    NEED_EXTENSION(cooperativeMatrixTensorAddressing, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
                   VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME),
    NEED_EXTENSION(cooperativeMatrixBlockLoads, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
                   VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME),
    NEED_EXTENSION(shaderFloat8, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
                   VkPhysicalDeviceShaderFloat8FeaturesEXT, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME),
    NEED_EXTENSION(shaderFloat8CooperativeMatrix, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
                   VkPhysicalDeviceShaderFloat8FeaturesEXT, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME),
};

// Extensions with no feature flag of their own that the kernels still need.
const char* const kExtensions[] = {VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME};

// Core VkPhysicalDeviceFeatures the kernels need.
const size_t kCoreNeeds[] = {offsetof(VkPhysicalDeviceFeatures, shaderInt16), offsetof(VkPhysicalDeviceFeatures, shaderInt64)};
const char* const kCoreNames[] = {"shaderInt16", "shaderInt64"};

VkBool32& flag(void* structure, size_t offset) { return *reinterpret_cast<VkBool32*>(static_cast<char*>(structure) + offset); }

VkBaseOutStructure* find(const void* chain, VkStructureType type) {
  for (auto* s = static_cast<VkBaseOutStructure*>(const_cast<void*>(chain)); s; s = s->pNext)
    if (s->sType == type) return s;
  return nullptr;
}

struct InstanceFns {
  PFN_vkGetPhysicalDeviceFeatures2 features2 = nullptr;
  PFN_vkGetPhysicalDeviceProperties properties = nullptr;
  PFN_vkEnumerateDeviceExtensionProperties extensions = nullptr;
  InstanceFns(PFN_vkGetInstanceProcAddr gipa, VkInstance instance) {
    features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(gipa(instance, "vkGetPhysicalDeviceFeatures2"));
    if (!features2)
      features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(gipa(instance, "vkGetPhysicalDeviceFeatures2KHR"));
    properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(gipa(instance, "vkGetPhysicalDeviceProperties"));
    extensions = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(gipa(instance, "vkEnumerateDeviceExtensionProperties"));
  }
};

std::vector<std::string> deviceExtensions(const InstanceFns& fns, VkPhysicalDevice physical) {
  uint32_t count = 0;
  fns.extensions(physical, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> props(count);
  fns.extensions(physical, nullptr, &count, props.data());
  std::vector<std::string> names;
  for (const auto& p : props) names.emplace_back(p.extensionName);
  return names;
}

bool has(const std::vector<std::string>& names, const char* name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

// The first missing requirement, or empty when the device has them all.
std::string missingRequirement(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical) {
  InstanceFns fns(gipa, instance);
  if (!fns.features2 || !fns.properties || !fns.extensions) return "vkGetPhysicalDeviceFeatures2";
  VkPhysicalDeviceProperties properties;
  fns.properties(physical, &properties);
  if (VK_API_VERSION_MINOR(properties.apiVersion) < 3 && VK_API_VERSION_MAJOR(properties.apiVersion) == 1)
    return "Vulkan 1.3 (the device reports " + std::to_string(VK_API_VERSION_MAJOR(properties.apiVersion)) + "." +
           std::to_string(VK_API_VERSION_MINOR(properties.apiVersion)) + ")";
  const std::vector<std::string> extensions = deviceExtensions(fns, physical);
  for (const char* name : kExtensions)
    if (!has(extensions, name)) return name;
  for (const Need& need : kNeeds)
    if (need.aggregate == 0 && !has(extensions, need.extension)) return need.extension;
  // Query every single structure once.
  std::vector<std::unique_ptr<char[]>> storage;
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  for (const Need& need : kNeeds) {
    if (find(features.pNext, need.single)) continue;
    storage.emplace_back(new char[need.singleSize]());
    auto* s = reinterpret_cast<VkBaseOutStructure*>(storage.back().get());
    s->sType = need.single;
    s->pNext = static_cast<VkBaseOutStructure*>(features.pNext);
    features.pNext = s;
  }
  fns.features2(physical, &features);
  for (size_t i = 0; i < 2; ++i)
    if (!flag(&features.features, kCoreNeeds[i])) return kCoreNames[i];
  for (const Need& need : kNeeds)
    if (!flag(find(features.pNext, need.single), need.singleOffset)) return need.name;
  return "";
}

// What nf_native_device_extend changed and owns.
struct Extension {
  std::vector<std::pair<VkBool32*, VkBool32>> restore;   // application flags set in place, with their old values
  std::vector<std::unique_ptr<char[]>> prepended;        // our own feature structures
  std::vector<const char*> extensionNames;
  VkPhysicalDeviceFeatures coreCopy{};
};

void setFlag(Extension& x, VkBool32* at) {
  if (*at) return;
  x.restore.emplace_back(at, *at);
  *at = VK_TRUE;
}

}  // namespace

// ---- The network -----------------------------------------------------------------------------

struct NfNative {
  std::unique_ptr<vk::Context> context;
  std::unique_ptr<nr::Model> model;
  std::unique_ptr<nr::Kernels> kernels;
  std::unique_ptr<nr::Graph> graph;
  nr::Geometry geometry{};
  nr::Activation* features = nullptr;
  VkCommandPool secondaryPool = VK_NULL_HANDLE;
  VkCommandBuffer secondary = VK_NULL_HANDLE;
  VkCommandPool ownPool = VK_NULL_HANDLE;
  VkCommandBuffer ownSecondary = VK_NULL_HANDLE;
  float blendScale = 1.0f;
  uint64_t fenceTimeoutNs = 5'000'000'000ull;

  bool stalled() const { return context && context->stalled(); }

  void dropGraph() {
    if (stalled()) {
      // The GPU may still run the graph: its command buffers and activations are leaked, not freed.
      secondary = ownSecondary = VK_NULL_HANDLE;
      (void)graph.release();
      features = nullptr;
      return;
    }
    if (secondary) vkFreeCommandBuffers(context->device(), secondaryPool, 1, &secondary);
    secondary = VK_NULL_HANDLE;
    if (ownSecondary) vkFreeCommandBuffers(context->device(), ownPool, 1, &ownSecondary);
    ownSecondary = VK_NULL_HANDLE;
    graph.reset();
    features = nullptr;
    // Nothing recorded holds the split-K scratch's address now: the next build sizes it for its own graph
    // (a larger frame needs more than the first build's).
    if (kernels) kernels->releaseScratch();
  }

  // Build (or rebuild) the graph for `geometry`: a warm-up run on the layer's queue, then the
  // secondary recording.
  void build(uint32_t width, uint32_t height) {
    dropGraph();
    geometry = nr::Geometry::fromValid(width, height);
    graph = std::make_unique<nr::Graph>(*context, *model, *kernels, geometry, nr::Graph::Options{});
    features = graph->allocate("input features", geometry.fullWidth * geometry.fullHeight, 16, nr::Format::F32);
    context->fillZero(features->buffer);
    // The first recording uploads the weights' re-laid matrices (each its own bounded submit) and
    // its run makes the driver compile every PTX module.
    context->resetDescriptorPool(0);
    VkCommandBuffer warm = context->beginCommands();
    graph->record(warm, *features);
    context->endAndSubmit(warm, true);
    const nr::Kernels::ChainTimeouts timeouts = kernels->chainTimeouts();
    if (timeouts.waits) throw std::runtime_error("the warm-up run's counter chain timed out at " + timeouts.counter);
    // The frame recordings (one per queue family they run on): descriptor pool 1 stays reserved for them.
    context->resetDescriptorPool(1);
    secondary = recordSecondary(secondaryPool);
    ownSecondary = recordSecondary(ownPool);
  }

  VkCommandBuffer recordSecondary(VkCommandPool pool) {
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    allocate.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(context->device(), &allocate, &commands));
    context->initDispatchable(commands);
    VkCommandBufferInheritanceInfo inheritance{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    begin.pInheritanceInfo = &inheritance;
    VK_CHECK(vkBeginCommandBuffer(commands, &begin));
    context->computeBarrier(commands);
    graph->record(commands, *features);
    context->computeBarrier(commands);
    VK_CHECK(vkEndCommandBuffer(commands));
    return commands;
  }
};

extern "C" {

uint32_t nf_native_abi_version(void) { return 1; }

uint32_t nf_native_asset_count(const char* kind) { return nf_native::assetCount(kind); }

uint32_t nf_native_device_supported(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical,
                                    char* missing, size_t missing_len) {
  try {
    const std::string first = missingRequirement(gipa, instance, physical);
    say(missing, missing_len, first);
    return first.empty() ? 1 : 0;
  } catch (const std::exception& e) {
    say(missing, missing_len, e.what());
    return 0;
  }
}

uint32_t nf_native_device_extend(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical,
                                 const VkDeviceCreateInfo* in, VkDeviceCreateInfo* out, void** state, char* err,
                                 size_t err_len) {
  *out = *in;
  *state = nullptr;
  try {
    const std::string first = missingRequirement(gipa, instance, physical);
    if (!first.empty()) {
      say(err, err_len, "the device lacks " + first);
      return 0;
    }
    auto x = std::make_unique<Extension>();
    InstanceFns fns(gipa, instance);
    const std::vector<std::string> supported = deviceExtensions(fns, physical);
    for (uint32_t i = 0; i < in->enabledExtensionCount; ++i) x->extensionNames.push_back(in->ppEnabledExtensionNames[i]);
    auto addExtension = [&](const char* name) {
      for (const char* have : x->extensionNames)
        if (!strcmp(have, name)) return;
      if (has(supported, name)) x->extensionNames.push_back(name);
    };
    for (const char* name : kExtensions) addExtension(name);
    void* head = const_cast<void*>(in->pNext);
    for (const Need& need : kNeeds) {
      VkBaseOutStructure* aggregate = need.aggregate ? find(in->pNext, need.aggregate) : nullptr;
      if (aggregate) {
        setFlag(*x, &flag(aggregate, need.aggregateOffset));
        continue;
      }
      VkBaseOutStructure* single = find(head, need.single);   // the application's, or one prepended already
      if (!single) {
        x->prepended.emplace_back(new char[need.singleSize]());
        single = reinterpret_cast<VkBaseOutStructure*>(x->prepended.back().get());
        single->sType = need.single;
        single->pNext = static_cast<VkBaseOutStructure*>(head);
        head = single;
      }
      VkBool32* at = &flag(single, need.singleOffset);
      const bool ours = std::any_of(x->prepended.begin(), x->prepended.end(),
                                    [&](const std::unique_ptr<char[]>& p) { return p.get() == reinterpret_cast<char*>(single); });
      if (ours) *at = VK_TRUE;
      else setFlag(*x, at);
      addExtension(need.extension);
    }
    // Core features: in the application's VkPhysicalDeviceFeatures2 (in place), else in a copy of
    // its pEnabledFeatures.
    if (VkBaseOutStructure* f2 = find(in->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)) {
      auto* features = &reinterpret_cast<VkPhysicalDeviceFeatures2*>(f2)->features;
      for (size_t offset : kCoreNeeds) setFlag(*x, &flag(features, offset));
    } else {
      if (in->pEnabledFeatures) x->coreCopy = *in->pEnabledFeatures;
      for (size_t offset : kCoreNeeds) flag(&x->coreCopy, offset) = VK_TRUE;
      out->pEnabledFeatures = &x->coreCopy;
    }
    out->pNext = head;
    out->enabledExtensionCount = (uint32_t)x->extensionNames.size();
    out->ppEnabledExtensionNames = x->extensionNames.data();
    *state = x.release();
    return 1;
  } catch (const std::exception& e) {
    *out = *in;
    say(err, err_len, e.what());
    return 0;
  }
}

void nf_native_device_restore(void* state) {
  auto* x = static_cast<Extension*>(state);
  if (!x) return;
  for (auto it = x->restore.rbegin(); it != x->restore.rend(); ++it) *it->first = it->second;
  delete x;
}

NfNative* nf_native_open(const NfNativeOpen* open, uint32_t* stalled, char* err, size_t err_len) {
  if (stalled) *stalled = 0;
  auto n = std::make_unique<NfNative>();
  try {
    n->fenceTimeoutNs = (uint64_t)open->fence_timeout_ms * 1'000'000ull;
    nf_native::installAssetLoader();
    nr::Kernels::setChainEnabled(open->chain != 0 && !g_fellBack.load());
    n->context = std::make_unique<vk::Context>(open->gipa, open->instance, open->physical, open->device, open->queue_family,
                                               open->queue_index, n->fenceTimeoutNs, open->init_dispatchable, open->init_user,
                                               open->frame_family);
    n->model = std::make_unique<nr::Model>(*n->context, open->model_dir, true);
    n->kernels = std::make_unique<nr::Kernels>(*n->context, std::string());
    n->kernels->setSiluTable(ref::siluTable());
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = open->frame_family;
    VK_CHECK(vkCreateCommandPool(open->device, &pool, nullptr, &n->secondaryPool));
    pool.queueFamilyIndex = open->queue_family;
    VK_CHECK(vkCreateCommandPool(open->device, &pool, nullptr, &n->ownPool));
    const nr::Tensor& blend = n->model->tensor(70, 0, "blend_scale");
    if (blend.byteLength >= 2) {
      uint16_t half;
      memcpy(&half, blend.bytes, 2);
      n->blendScale = std::clamp(num::f16ToF32(half), 0.0f, 1.0f);
    }
    return n.release();
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
    if (stalled) *stalled = n->stalled() ? 1 : 0;
    if (n->context && !n->stalled()) {
      if (n->secondaryPool) vkDestroyCommandPool(n->context->device(), n->secondaryPool, nullptr);
      if (n->ownPool) vkDestroyCommandPool(n->context->device(), n->ownPool, nullptr);
    }
    // Stalled, each of these leaks what it made instead of destroying it.
    n->kernels.reset();
    n->model.reset();
    n->context.reset();
    return nullptr;
  }
}

uint32_t nf_native_build(NfNative* n, uint32_t valid_width, uint32_t valid_height, char* err, size_t err_len) {
  if (n->stalled()) {
    say(err, err_len, "the network's queue stalled earlier; it takes no more work");
    return 0;
  }
  try {
    n->build(valid_width, valid_height);
    return 1;
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
    n->dropGraph();
    return 0;
  }
}

uint32_t nf_native_frame(const NfNative* n, NfNativeFrame* frame) {
  if (!n->graph || !n->secondary) return 0;
  frame->field_width = n->geometry.fullWidth;
  frame->field_height = n->geometry.fullHeight;
  frame->features = n->features->buffer.buffer;
  frame->features_bytes = n->features->buffer.size;
  frame->head = n->graph->head().buffer.buffer;
  frame->head_bytes = n->graph->head().buffer.size;
  frame->blend_scale = n->blendScale;
  frame->chained = n->graph->chained() ? 1 : 0;
  return 1;
}

uint32_t nf_native_stalled(const NfNative* n) { return n->stalled() ? 1 : 0; }

VkCommandBuffer nf_native_graph_commands(const NfNative* n) { return n->secondary; }

VkCommandBuffer nf_native_graph_commands_own(const NfNative* n) { return n->ownSecondary; }

uint32_t nf_native_record_graph(NfNative* n, VkCommandBuffer primary, char* err, size_t err_len) {
  try {
    n->context->resetDescriptorPool(0);
    n->graph->record(primary, *n->features);
    return 1;
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
    return 0;
  }
}

uint32_t nf_native_chain_timeouts(const NfNative* n, char* where, size_t where_len) {
  const nr::Kernels::ChainTimeouts timeouts = n->kernels->chainTimeouts();
  say(where, where_len, timeouts.counter);
  return timeouts.waits;
}

void nf_native_reset_chain_timeouts(NfNative* n) { n->kernels->resetChainTimeouts(); }

uint32_t nf_native_fall_back_to_barriers(NfNative* n, char* err, size_t err_len) {
  g_fellBack.store(true);
  nr::Kernels::setChainEnabled(false);
  n->kernels->resetChainTimeouts();
  return nf_native_build(n, n->geometry.validWidth, n->geometry.validHeight, err, err_len);
}

void nf_native_close(NfNative* n) {
  if (!n) return;
  try {
    n->dropGraph();
    if (!n->stalled()) {
      if (n->secondaryPool) vkDestroyCommandPool(n->context->device(), n->secondaryPool, nullptr);
      if (n->ownPool) vkDestroyCommandPool(n->context->device(), n->ownPool, nullptr);
    }
    n->kernels.reset();
    n->model.reset();
    n->context.reset();
  } catch (...) {
  }
  delete n;
}

}  // extern "C"
