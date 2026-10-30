// The layer's side of the vendored OpenDLSS-NR network (nf_native.h).
#include "nf_native.h"

#include "nf_assets.h"

#include <volk.h>
#include <vulkan/vk_layer.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
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

// One live network per process: volk's function table (loaded for one device), the chaining switch and
// the asset loader are process-global, so a second network would repoint the first one's calls.
std::atomic<bool> g_live{false};

void say(char* out, size_t len, const char* text) {
  if (!out || !len) return;
  snprintf(out, len, "%s", text);
}

void say(char* out, size_t len, const std::string& text) { say(out, len, text.c_str()); }

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

// Why an instance created for `api` (the application's VkApplicationInfo.apiVersion; 0 is 1.0) cannot run
// the network, or empty. A device's API version is the lower of its instance's and the GPU's: below 1.3
// the kernels' SPIR-V 1.6 is not allowed and Vulkan 1.3's core commands, which the vendored code calls by
// those names (vkCmdPipelineBarrier2, vkGetBufferDeviceAddress, ...), are not there, extensions or not.
std::string instanceRequirement(uint32_t api) {
  if (api == 0) api = VK_API_VERSION_1_0;
  const uint32_t major = VK_API_VERSION_MAJOR(api), minor = VK_API_VERSION_MINOR(api);
  if (major > 1 || (major == 1 && minor >= 3)) return "";
  return "the application's instance asks for Vulkan " + std::to_string(major) + "." + std::to_string(minor) +
         "; the network needs an instance for Vulkan 1.3";
}

// The first missing requirement of the device, or empty when it has them all.
std::string missingRequirement(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical) {
  InstanceFns fns(gipa, instance);
  if (!fns.features2) return "vkGetPhysicalDeviceFeatures2 (or its KHR name)";
  if (!fns.properties) return "vkGetPhysicalDeviceProperties";
  if (!fns.extensions) return "vkEnumerateDeviceExtensionProperties";
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

// The size of every structure a VkDeviceCreateInfo's pNext chain may carry, by sType: generated by build.rs
// from the pinned Vulkan-Headers' registry (vk.xml, the structures that extend VkDeviceCreateInfo and
// vulkan_core.h defines).
struct ChainStruct {
  VkStructureType type;
  size_t size;
};
const ChainStruct kDeviceChain[] = {
#include "nf_device_chain.inc"
};

size_t deviceChainSize(VkStructureType type) {
  // The loader's own nodes, which every device create info reaching a layer carries (its layer link and
  // device callbacks): not in the registry as extending VkDeviceCreateInfo.
  if (type == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO) return sizeof(VkLayerInstanceCreateInfo);
  if (type == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO) return sizeof(VkLayerDeviceCreateInfo);
  for (const ChainStruct& s : kDeviceChain)
    if (s.type == type) return s.size;
  return 0;
}

// What nf_native_device_extend owns: a copy of the application's chain with the network's flags set in
// it, the structures it adds, the extension list and the core features. The application's own create
// info and chain are only read (they are const: written in place they could be read-only memory).
struct Extension {
  std::vector<std::unique_ptr<char[]>> nodes;   // the chain's copies and the structures added
  std::vector<const char*> extensionNames;
  VkPhysicalDeviceFeatures coreCopy{};
};

// The application's chain, copied node by node (shallow: what the nodes point at stays the
// application's and is only read), up to the first structure of a size this build does not know: that
// one and everything after it stay linked as the application's own (`*tail`), never written.
void* copyChain(Extension& x, const void* chain, const void** tail) {
  void* head = nullptr;
  void** link = &head;
  *tail = nullptr;
  for (auto* s = static_cast<const VkBaseInStructure*>(chain); s; s = s->pNext) {
    const size_t size = deviceChainSize(s->sType);
    if (!size) {
      *link = const_cast<VkBaseInStructure*>(s);
      *tail = s;
      break;
    }
    x.nodes.emplace_back(new char[size]);
    memcpy(x.nodes.back().get(), s, size);
    auto* copy = reinterpret_cast<VkBaseOutStructure*>(x.nodes.back().get());
    copy->pNext = nullptr;
    *link = copy;
    link = reinterpret_cast<void**>(&copy->pNext);
  }
  return head;
}

}  // namespace

// ---- The network -----------------------------------------------------------------------------

// Why nf_native_build_sizes cannot build for `sizes`, or empty: checked before anything is dropped.
static std::string invalidSizes(const uint32_t* sizes, uint32_t count) {
  if (count == 0 || !sizes) return "no frame size to build the network for";
  for (uint32_t i = 0; i < count; ++i)
    if (sizes[2 * i] == 0 || sizes[2 * i + 1] == 0)
      return "no network for a " + std::to_string(sizes[2 * i]) + "x" + std::to_string(sizes[2 * i + 1]) + " frame";
  return "";
}

// Hold slots nf_native_record_graph_at serves (nf_native_hold_slots; the layer's slots per device).
constexpr uint32_t kHoldPools = 16;

// A warm-up whose counter chain timed out: the build falls back to barriers and runs again.
struct WarmUpChainTimeout : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// One frame size's graph: its activations, the layer's input features and the two recordings.
struct NfSlot {
  nr::Geometry geometry{};
  std::unique_ptr<nr::Graph> graph;
  nr::Activation* features = nullptr;
  VkCommandBuffer secondary = VK_NULL_HANDLE;
  VkCommandBuffer ownSecondary = VK_NULL_HANDLE;
  // The descriptor sets one recording of the graph allocates at most: its dispatches, counted at the
  // warm-up's recording (PTX launches are counted too and take none).
  uint32_t sets = 0;
};

struct NfNative {
  std::unique_ptr<vk::Context> context;
  std::unique_ptr<nr::Model> model;
  std::unique_ptr<nr::Kernels> kernels;
  // The graphs of the sizes built together (nf_native_build_sizes), in the order asked for. They share the
  // weights, the kernels and the split-K scratch (sized by the largest, built first), and run one after
  // another on one queue, never at the same time (each opens and closes with compute barriers).
  std::vector<NfSlot> slots;
  // The activations of every slot's graph that its warm-up used, placed in this one block by lifetime
  // (shareActivations); null with NEURAL_FORGE_NATIVE_SHARED_ACTIVATIONS=0 (each activation its own allocation).
  VkDeviceMemory activationArena = VK_NULL_HANDLE;
  VkDeviceSize activationArenaBytes = 0, activationBytesBefore = 0;
  VkCommandPool secondaryPool = VK_NULL_HANDLE;
  VkCommandPool ownPool = VK_NULL_HANDLE;
  // The descriptor pools of the recordings into the application's command buffers
  // (nf_native_record_graph_at), one per hold slot, made on first use: a slot's sets stay valid while the
  // buffer recorded with them may still be submitted, and are reset only when that slot records again.
  std::vector<VkDescriptorPool> holdPools;
  std::vector<uint32_t> holdPoolSets;   // the sets each of those pools was made for
  // nf_native_fake_chain_timeout's: waits added to the watchdog's count, and one for the next warm-up.
  uint32_t fakeTimeouts = 0;
  bool fakeWarmUpTimeout = false;
  float blendScale = 1.0f;
  uint64_t rawLoaded = 0;
  uint64_t fenceTimeoutNs = 5'000'000'000ull;
  // The graphs' options for the next builds: captures only for tests (nf_native_set_captures).
  nr::Graph::Options graphOptions{};

  bool stalled() const { return context && context->stalled(); }

  // The watchdog's count (nf_native_chain_timeouts) and its reset.
  nr::Kernels::ChainTimeouts timeouts() const {
    nr::Kernels::ChainTimeouts t = kernels->chainTimeouts();
    if (fakeTimeouts && !t.waits) t.counter = "a faked timeout";
    t.waits += fakeTimeouts;
    return t;
  }
  void resetTimeouts() {
    kernels->resetChainTimeouts();
    fakeTimeouts = 0;
  }

  // Barriers between every launch from now on, for this network and every one opened later in the process.
  void fallBack() {
    g_fellBack.store(true);
    nr::Kernels::setChainEnabled(false);
    resetTimeouts();
  }

  // The most descriptor sets a recording of any built slot allocates.
  uint32_t holdSets() const {
    uint32_t most = 0;
    for (const NfSlot& slot : slots) most = std::max(most, slot.sets);
    return most;
  }

  void dropGraph() {
    if (stalled()) {
      // The GPU may still run the graphs: their command buffers and activations are leaked, not freed.
      for (NfSlot& slot : slots) (void)slot.graph.release();
      slots.clear();
      activationArena = VK_NULL_HANDLE;   // leaked with them
      return;
    }
    for (NfSlot& slot : slots) {
      if (slot.secondary) vkFreeCommandBuffers(context->device(), secondaryPool, 1, &slot.secondary);
      if (slot.ownSecondary) vkFreeCommandBuffers(context->device(), ownPool, 1, &slot.ownSecondary);
      slot.graph.reset();
    }
    slots.clear();
    // The graphs destroyed their buffers in it: nothing is bound to the block now.
    context->freeMemory(activationArena);
    activationArena = VK_NULL_HANDLE;
    activationArenaBytes = activationBytesBefore = 0;
    // Nothing recorded holds the split-K scratch's address now: the next build sizes it for its own graphs
    // (a larger frame needs more than the first build's).
    if (kernels) kernels->releaseScratch();
  }

  // Build (or rebuild) the graphs for `sizes` (pairs of valid width and height, `count` of them): each a
  // warm-up run on the layer's queue, then the secondary recordings of all of them. The largest is built
  // first: the split-K scratch is sized by the first recording that needs it, and the smaller graphs' fit
  // in it. The sizes are valid (invalidSizes).
  void build(const uint32_t* sizes, uint32_t count) {
    dropGraph();
    // A count the frames of the graph just dropped left (the caller drained them, but reads the count only at
    // its next hold, which a rebuild for another size can come before): counter chaining gave up on this
    // device, so the build falls back as the caller would. Never left standing: a stale count would fail
    // every warm-up from now on.
    if (timeouts().waits) {
      if (nr::Kernels::chainEnabled()) fallBack();
      resetTimeouts();
    }
    try {
      buildGraphs(sizes, count);
    } catch (const WarmUpChainTimeout&) {
      // The warm-up waited on the GPU, which runs nothing of the network now.
      dropGraph();
      fallBack();
      buildGraphs(sizes, count);
    }
  }

  void buildGraphs(const uint32_t* sizes, uint32_t count) {
    slots.resize(count);
    std::vector<uint32_t> order(count);
    for (uint32_t i = 0; i < count; ++i) order[i] = i;
    auto area = [&](uint32_t i) { return uint64_t(sizes[2 * i]) * sizes[2 * i + 1]; };
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return area(a) > area(b); });
    // With the activations sharing memory by lifetime (shareActivations), each size's graph is first recorded only to
    // track the lifetimes, its activations aliased in one scratch block (nr::Graph::setAliasing) and never run; then
    // they are placed in the shared block and the warm-ups run on it. So a build never holds every activation's own
    // memory at once: that took about twice what the build keeps, briefly, and when the game had filled its VRAM
    // budget the driver moved the game's memory to system memory for good (docs/LESSONS.md, October 29).
    const bool share = sharingActivations();
    for (uint32_t i : order) {
      NfSlot& slot = slots[i];
      slot.geometry = nr::Geometry::fromValid(sizes[2 * i], sizes[2 * i + 1]);
      slot.graph = std::make_unique<nr::Graph>(*context, *model, *kernels, slot.geometry, graphOptions);
      slot.features = slot.graph->allocate("input features", slot.geometry.fullWidth * slot.geometry.fullHeight, 16, nr::Format::F32);
      context->fillZero(slot.features->buffer);
      if (!share) continue;
      // The first recording uploads the weights' re-laid matrices (each its own bounded submit).
      slot.graph->setAliasing(true);
      context->resetDescriptorPool(0);
      VkCommandBuffer tracked = context->beginCommands();
      try {
        // The context's use tracker points into this graph: switched off on every way out of the recording, so
        // a recording that throws (and the graph freed after it) never leaves it to the next use of a buffer.
        struct Untrack {
          nr::Graph& graph;
          ~Untrack() { graph.trackLifetimes(false); }
        } untrack{*slot.graph};
        slot.graph->trackLifetimes(true);
        slot.graph->record(tracked, *slot.features);
      } catch (...) {
        context->discardCommands(tracked);
        throw;
      }
      context->discardCommands(tracked);
    }
    if (share) {
      // The scratch blocks go before the shared block is allocated (the aliased buffers stay until rebound below).
      for (NfSlot& slot : slots) slot.graph->releaseScratch();
      shareActivations();
      for (NfSlot& slot : slots) slot.graph->setAliasing(false);
    }
    for (uint32_t i : order) {
      NfSlot& slot = slots[i];
      // The warm-up: its run makes the driver compile every PTX module (and, without sharing, its recording uploads the
      // weights' re-laid matrices and tracks the lifetimes, unused then).
      context->resetDescriptorPool(0);
      VkCommandBuffer warm = context->beginCommands();
      try {
        slot.graph->record(warm, *slot.features);
      } catch (...) {
        context->discardCommands(warm);
        throw;
      }
      slot.sets = kernels->dispatchCount();
      // This warm-up's count alone: nothing else of the network runs (the previous warm-up was waited on, and
      // the caller drained every frame of the previous build).
      resetTimeouts();
      if (fakeWarmUpTimeout) {
        fakeWarmUpTimeout = false;
        fakeTimeouts = 1;
      }
      context->endAndSubmit(warm, true);
      const nr::Kernels::ChainTimeouts t = timeouts();
      if (t.waits) {
        if (slot.graph->chained()) throw WarmUpChainTimeout("the warm-up run's counter chain timed out at " + t.counter);
        throw std::runtime_error("the warm-up run (with barriers) reports a counter-chain timeout at " + t.counter);
      }
      // The warm-up built every matrix the graph uses and has completed: the records' raw buffers now keep only the
      // bytes the kernels read through them (`Model::compactAux`). The recordings below use those.
      model->compactAux();
    }
    // The frame recordings (one per queue family they run on), every size's: descriptor pool 1 stays reserved
    // for them.
    context->resetDescriptorPool(1);
    for (NfSlot& slot : slots) {
      slot.secondary = recordSecondary(secondaryPool, slot);
      slot.ownSecondary = recordSecondary(ownPool, slot);
    }
  }

  // Activation memory (docs/NATIVE_BACKEND.md, "Activation memory"). Each activation had its own allocation, alive
  // for the whole build. After the warm-ups (nothing recorded yet runs): the activations a slot's warm-up never
  // touched (allocated for routes it did not take) get a minimal buffer, and the ones it used share one block,
  // placed first fit, largest first, so that two overlap only when their barrier intervals do not (a barrier orders
  // every launch before it against every one after, chained or not). The slots share the block as they share the
  // split-K scratch: they never run at the same time. The features and the head stay their own: the layer
  // writes and reads them outside the graph. So do a graph's captures (nf_native_set_captures), read after a run
  // (nf_native_capture); their copies' sources are placed like any activation, alive until the copy (a use the
  // warm-up reported). The block starts zeroed, as the activations did.
  static bool sharingActivations() {
    const char* off = getenv("NEURAL_FORGE_NATIVE_SHARED_ACTIVATIONS");
    return !(off && std::string(off) == "0");
  }

  void shareActivations() {
    struct Placement {
      nr::Graph* graph;
      nr::Activation* activation;
      uint32_t first, last;
      VkDeviceSize offset, bytes;
    };
    std::vector<Placement> placements;
    VkDeviceSize arena = 0, alignment = 256, before = 0;
    uint32_t types = ~0u;
    for (NfSlot& slot : slots) {
      std::vector<nr::Activation*> unused;
      std::vector<nr::Graph::Lifetime> used = slot.graph->lifetimes(unused);
      const auto& captures = slot.graph->boundaries();
      auto own = [&](const nr::Activation* a) {
        return a == slot.features || a == &slot.graph->head() ||
               std::any_of(captures.begin(), captures.end(), [a](const auto& c) { return c.second == a; });
      };
      for (nr::Activation* a : unused) {
        if (own(a)) continue;
        before += a->buffer.size;
        slot.graph->rebind(*a, context->createBuffer(16, false, a->label.c_str()));
      }
      std::vector<Placement> mine;
      for (const nr::Graph::Lifetime& l : used) {
        if (own(l.activation)) continue;
        const VkMemoryRequirements r = context->bufferRequirements(l.activation->buffer.size);
        alignment = std::max(alignment, r.alignment);
        types &= r.memoryTypeBits;
        before += l.activation->buffer.size;
        mine.push_back({slot.graph.get(), l.activation, l.first, l.last, 0, r.size});
      }
      std::stable_sort(mine.begin(), mine.end(), [](const Placement& a, const Placement& b) { return a.bytes > b.bytes; });
      std::vector<Placement> placed;
      for (Placement p : mine) {
        // The lowest offset clear of every placed activation alive in an overlapping interval.
        std::vector<std::pair<VkDeviceSize, VkDeviceSize>> taken;
        for (const Placement& q : placed)
          if (q.first <= p.last && p.first <= q.last) taken.emplace_back(q.offset, q.offset + q.bytes);
        std::sort(taken.begin(), taken.end());
        VkDeviceSize offset = 0;
        for (const auto& [start, end] : taken) {
          if (offset + p.bytes <= start) break;
          offset = std::max(offset, (end + alignment - 1) / alignment * alignment);
        }
        p.offset = offset;
        arena = std::max(arena, offset + p.bytes);
        placed.push_back(p);
      }
      placements.insert(placements.end(), placed.begin(), placed.end());
    }
    activationBytesBefore = before;
    if (placements.empty()) return;
    activationArena = context->allocateDeviceMemory(arena, types);
    activationArenaBytes = arena;
    for (const Placement& p : placements)
      p.graph->rebind(*p.activation, context->createBufferAt(activationArena, p.offset, p.bytes, p.activation->label.c_str()));
    vk::Buffer whole = context->createBufferAt(activationArena, 0, arena, "activation arena");
    context->fillZero(whole);
    context->destroyBuffer(whole);
  }

  VkCommandBuffer recordSecondary(VkCommandPool pool, NfSlot& slot) {
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
    slot.graph->record(commands, *slot.features);
    context->computeBarrier(commands);
    VK_CHECK(vkEndCommandBuffer(commands));
    return commands;
  }
};

extern "C" {

uint32_t nf_native_abi_version(void) { return 6; }

uint32_t nf_native_hold_slots(void) { return kHoldPools; }

uint32_t nf_native_asset_count(const char* kind) {
  try {
    return nf_native::assetCount(kind);
  } catch (...) {
    return 0;
  }
}

uint32_t nf_native_device_supported(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, uint32_t instance_api_version,
                                    VkPhysicalDevice physical, char* missing, size_t missing_len) {
  try {
    std::string first = instanceRequirement(instance_api_version);
    if (first.empty()) first = missingRequirement(gipa, instance, physical);
    say(missing, missing_len, first);
    return first.empty() ? 1 : 0;
  } catch (const std::exception& e) {
    say(missing, missing_len, e.what());
    return 0;
  } catch (...) {
    say(missing, missing_len, "an unknown C++ exception");
    return 0;
  }
}

uint32_t nf_native_device_extend(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, uint32_t instance_api_version,
                                 VkPhysicalDevice physical, const VkDeviceCreateInfo* in, VkDeviceCreateInfo* out,
                                 void** state, char* err, size_t err_len) {
  *out = *in;
  *state = nullptr;
  try {
    const std::string old = instanceRequirement(instance_api_version);
    if (!old.empty()) {
      say(err, err_len, old);
      return 0;
    }
    const std::string first = missingRequirement(gipa, instance, physical);
    if (!first.empty()) {
      say(err, err_len, "the device lacks " + first);
      return 0;
    }
    auto x = std::make_unique<Extension>();
    InstanceFns fns(gipa, instance);
    const std::vector<std::string> supported = deviceExtensions(fns, physical);
    for (uint32_t i = 0; i < in->enabledExtensionCount; ++i) x->extensionNames.push_back(in->ppEnabledExtensionNames[i]);
    // VK_EXT_buffer_device_address excludes VK_KHR_buffer_device_address and the core feature the
    // network's kernels need (VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-03328, -pNext-04748).
    for (const char* have : x->extensionNames)
      if (!strcmp(have, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
        throw std::runtime_error("the application enables VK_EXT_buffer_device_address, which excludes the buffer device "
                                 "address the network needs");
    auto addExtension = [&](const char* name) {
      for (const char* have : x->extensionNames)
        if (!strcmp(have, name)) return;
      if (has(supported, name)) x->extensionNames.push_back(name);
    };
    for (const char* name : kExtensions) addExtension(name);
    // Every flag is set in the copy of the chain (the application's structure's copy, or one prepended).
    const void* tail = nullptr;
    void* head = copyChain(*x, in->pNext, &tail);
    // A structure this side sets that lies in the uncopied tail could only be written in place, or
    // given twice: refused.
    auto copied = [&](VkStructureType type) -> VkBaseOutStructure* {
      if (tail && find(tail, type))
        throw std::runtime_error("the device create info chains a structure the network sets (sType " +
                                 std::to_string(static_cast<int>(type)) + ") after one this build does not know");
      return find(head, type);
    };
    for (const Need& need : kNeeds) {
      if (VkBaseOutStructure* aggregate = need.aggregate ? copied(need.aggregate) : nullptr) {
        flag(aggregate, need.aggregateOffset) = VK_TRUE;
        continue;
      }
      VkBaseOutStructure* single = copied(need.single);   // the application's copy, or one prepended already
      if (!single) {
        x->nodes.emplace_back(new char[need.singleSize]());
        single = reinterpret_cast<VkBaseOutStructure*>(x->nodes.back().get());
        single->sType = need.single;
        single->pNext = static_cast<VkBaseOutStructure*>(head);
        head = single;
      }
      flag(single, need.singleOffset) = VK_TRUE;
      addExtension(need.extension);
    }
    // Core features: in the copy of the application's VkPhysicalDeviceFeatures2, else in a copy of its
    // pEnabledFeatures.
    if (VkBaseOutStructure* f2 = copied(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)) {
      auto* features = &reinterpret_cast<VkPhysicalDeviceFeatures2*>(f2)->features;
      for (size_t offset : kCoreNeeds) flag(features, offset) = VK_TRUE;
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
  } catch (...) {
    *out = *in;
    say(err, err_len, "an unknown C++ exception");
    return 0;
  }
}

void nf_native_device_restore(void* state) {
  delete static_cast<Extension*>(state);   // no throwing destructors in Extension
}

NfNative* nf_native_open(const NfNativeOpen* open, uint32_t* stalled, char* err, size_t err_len) {
  if (stalled) *stalled = 0;
  bool free = false;
  if (!g_live.compare_exchange_strong(free, true)) {
    say(err, err_len, "another network is open in this process (one at a time: its Vulkan function table is process-wide)");
    return nullptr;
  }
  std::unique_ptr<NfNative> n;
  // What a failed open made: destroyed, or leaked when a wait timed out (each part's own rule).
  auto fail = [&](const char* why) -> NfNative* {
    say(err, err_len, why);
    if (n) {
      if (stalled) *stalled = n->stalled() ? 1 : 0;
      if (n->context && !n->stalled()) {
        if (n->secondaryPool) vkDestroyCommandPool(n->context->device(), n->secondaryPool, nullptr);
        if (n->ownPool) vkDestroyCommandPool(n->context->device(), n->ownPool, nullptr);
      }
      n->kernels.reset();
      n->model.reset();
      n->context.reset();
    }
    g_live.store(false);
    return nullptr;
  };
  try {
    n = std::make_unique<NfNative>();
    n->fenceTimeoutNs = (uint64_t)open->fence_timeout_ms * 1'000'000ull;
    nf_native::installAssetLoader();
    nr::Kernels::setChainEnabled(open->chain != 0 && !g_fellBack.load());
    n->context = std::make_unique<vk::Context>(open->gipa, open->instance, open->physical, open->device, open->queue_family,
                                               open->queue_index, n->fenceTimeoutNs, open->init_dispatchable, open->init_user,
                                               open->frame_family);
    if (!open->model || (open->model->tensor_count && !open->model->tensors)) throw std::runtime_error("no model");
    std::vector<nr::Tensor> records(open->model->tensor_count);
    for (size_t i = 0; i < records.size(); ++i) {
      const NfNativeTensor& from = open->model->tensors[i];
      nr::Tensor& to = records[i];
      to.name = from.name;
      to.block = from.block;
      to.layer = from.layer;
      to.parameter = from.parameter;
      to.stage = from.stage;
      to.stageOffset = from.stage_offset;
      to.byteLength = from.byte_length;
      to.bytes = from.bytes;
    }
    n->model = std::make_unique<nr::Model>(*n->context, open->model->block_count, std::move(records));
    n->rawLoaded = n->model->rawBytes();
    n->kernels = std::make_unique<nr::Kernels>(*n->context, std::string());
    n->kernels->setSiluTable(ref::siluTable());
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = open->frame_family;
    VK_CHECK(vkCreateCommandPool(open->device, &pool, nullptr, &n->secondaryPool));
    pool.queueFamilyIndex = open->queue_family;
    VK_CHECK(vkCreateCommandPool(open->device, &pool, nullptr, &n->ownPool));
    // The post block, the last one: 70 in the full network, 30 in the small one.
    const nr::Tensor& blend = n->model->tensor((int)n->model->blockCount() - 1, 0, "blend_scale");
    if (blend.byteLength >= 2) {
      uint16_t half;
      memcpy(&half, blend.bytes, 2);
      n->blendScale = std::clamp(num::f16ToF32(half), 0.0f, 1.0f);
    }
    return n.release();
  } catch (const std::exception& e) {
    return fail(e.what());
  } catch (...) {
    return fail("an unknown C++ exception");
  }
}

uint32_t nf_native_build(NfNative* n, uint32_t valid_width, uint32_t valid_height, char* err, size_t err_len) {
  const uint32_t size[2] = {valid_width, valid_height};
  return nf_native_build_sizes(n, size, 1, err, err_len);
}

uint32_t nf_native_build_sizes(NfNative* n, const uint32_t* sizes, uint32_t count, char* err, size_t err_len) {
  if (n->stalled()) {
    say(err, err_len, "the network's queue stalled earlier; it takes no more work");
    return 0;
  }
  // Refused with the current build kept: nothing of it needs to be drained for a call that does nothing.
  if (const std::string invalid = invalidSizes(sizes, count); !invalid.empty()) {
    say(err, err_len, invalid);
    return 0;
  }
  try {
    n->build(sizes, count);
    return 1;
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
  } catch (...) {
    say(err, err_len, "an unknown C++ exception");
  }
  try {
    n->dropGraph();
  } catch (...) {
  }
  return 0;
}

uint32_t nf_native_frame(const NfNative* n, NfNativeFrame* frame) { return nf_native_frame_at(n, 0, frame); }

uint32_t nf_native_frame_at(const NfNative* n, uint32_t slot, NfNativeFrame* frame) {
  if (slot >= n->slots.size()) return 0;
  const NfSlot& s = n->slots[slot];
  if (!s.graph || !s.secondary) return 0;
  frame->field_width = s.geometry.fullWidth;
  frame->field_height = s.geometry.fullHeight;
  frame->features = s.features->buffer.buffer;
  frame->features_bytes = s.features->buffer.size;
  frame->head = s.graph->head().buffer.buffer;
  frame->head_bytes = s.graph->head().buffer.size;
  frame->blend_scale = n->blendScale;
  frame->chained = s.graph->chained() ? 1 : 0;
  return 1;
}

uint32_t nf_native_stalled(const NfNative* n) { return n->stalled() ? 1 : 0; }

void nf_native_weight_bytes(const NfNative* n, uint64_t* raw_loaded, uint64_t* raw, uint64_t* matrices) {
  *raw_loaded = n->rawLoaded;
  *raw = n->model->rawBytes();
  *matrices = n->model->matrixBytes();
}

VkCommandBuffer nf_native_graph_commands(const NfNative* n) { return nf_native_graph_commands_at(n, 0); }

VkCommandBuffer nf_native_graph_commands_own(const NfNative* n) { return nf_native_graph_commands_own_at(n, 0); }

VkCommandBuffer nf_native_graph_commands_at(const NfNative* n, uint32_t slot) {
  return slot < n->slots.size() ? n->slots[slot].secondary : VK_NULL_HANDLE;
}

VkCommandBuffer nf_native_graph_commands_own_at(const NfNative* n, uint32_t slot) {
  return slot < n->slots.size() ? n->slots[slot].ownSecondary : VK_NULL_HANDLE;
}

uint32_t nf_native_record_graph(NfNative* n, VkCommandBuffer primary, char* err, size_t err_len) {
  if (n->stalled()) {
    say(err, err_len, "the network's queue stalled earlier; it takes no more work");
    return 0;
  }
  try {
    if (n->slots.empty() || !n->slots[0].graph) throw std::runtime_error("the network is not built");
    n->context->resetDescriptorPool(0);
    n->slots[0].graph->record(primary, *n->slots[0].features);
    return 1;
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
    return 0;
  } catch (...) {
    say(err, err_len, "an unknown C++ exception");
    return 0;
  }
}

uint32_t nf_native_profile_begin(NfNative* n, VkCommandBuffer primary, uint32_t max_dispatches, char* err, size_t err_len) {
  try {
    if (!n->kernels) throw std::runtime_error("the network is not open");
    n->kernels->beginProfile(primary, max_dispatches);
    return 1;
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
    return 0;
  } catch (...) {
    say(err, err_len, "an unknown C++ exception");
    return 0;
  }
}

uint32_t nf_native_profile_end(NfNative* n, char* out, size_t out_len) {
  if (out_len) out[0] = 0;
  try {
    if (!n->kernels) return 0;
    std::vector<nr::Kernels::ProfileEntry> entries = n->kernels->endProfile();
    std::string text;
    for (const nr::Kernels::ProfileEntry& e : entries) text += e.label + "\t" + std::to_string(e.milliseconds) + "\n";
    if (out_len) {
      size_t len = std::min(text.size(), out_len - 1);
      std::memcpy(out, text.data(), len);
      out[len] = 0;
    }
    return static_cast<uint32_t>(entries.size());
  } catch (...) {
    return 0;
  }
}

uint32_t nf_native_record_graph_at(NfNative* n, uint32_t slot, uint32_t hold, VkCommandBuffer primary, char* err,
                                   size_t err_len) {
  if (n->stalled()) {
    say(err, err_len, "the network's queue stalled earlier; it takes no more work");
    return 0;
  }
  try {
    if (slot >= n->slots.size() || !n->slots[slot].graph) throw std::runtime_error("the network is not built for that size");
    if (hold >= kHoldPools) throw std::runtime_error("no such hold slot");
    if (n->holdPools.size() <= hold) {
      n->holdPools.resize(hold + 1, VK_NULL_HANDLE);
      n->holdPoolSets.resize(hold + 1, 0);
    }
    VkDescriptorPool& pool = n->holdPools[hold];
    // Sized for one recording of the largest graph built (its dispatches, counted at its warm-up), and made
    // again when a later build needs more: nothing recorded with this hold slot is pending (the contract).
    const uint32_t sets = std::max(1u, n->holdSets());
    if (pool && n->holdPoolSets[hold] < sets) {
      vkDestroyDescriptorPool(n->context->device(), pool, nullptr);
      pool = VK_NULL_HANDLE;
    }
    if (!pool) {
      VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sets * vk::kGenericBindings};
      VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      info.maxSets = sets;
      info.poolSizeCount = 1;
      info.pPoolSizes = &size;
      VK_CHECK(vkCreateDescriptorPool(n->context->device(), &info, nullptr, &pool));
      n->holdPoolSets[hold] = sets;
    } else {
      VK_CHECK(vkResetDescriptorPool(n->context->device(), pool, 0));
    }
    // The context allocates the recording's sets from the hold's pool; its own pool is put back whatever
    // happens.
    struct Restore {
      vk::Context& context;
      VkDescriptorPool previous;
      ~Restore() { context.useDescriptorPool(previous); }
    } restore{*n->context, n->context->useDescriptorPool(pool)};
    n->context->computeBarrier(primary);
    n->slots[slot].graph->record(primary, *n->slots[slot].features);
    n->context->computeBarrier(primary);
    return 1;
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
    return 0;
  } catch (...) {
    say(err, err_len, "an unknown C++ exception");
    return 0;
  }
}

uint32_t nf_native_chain_timeouts(const NfNative* n, char* where, size_t where_len) {
  try {
    const nr::Kernels::ChainTimeouts timeouts = n->timeouts();
    say(where, where_len, timeouts.counter);
    return timeouts.waits;
  } catch (...) {
    say(where, where_len, "unreadable");
    return 0;
  }
}

void nf_native_reset_chain_timeouts(NfNative* n) { n->resetTimeouts(); }

void nf_native_fall_back_to_barriers(NfNative* n) {
  try {
    n->fallBack();
    // The chained recording is of no more use; the next nf_native_build records the graph with barriers.
    n->dropGraph();
  } catch (...) {
  }
}

void nf_native_forget_fall_back(void) { g_fellBack.store(false); }

void nf_native_set_captures(NfNative* n, uint32_t boundaries, uint32_t intermediates) {
  n->graphOptions.captureBoundaries = boundaries != 0;
  n->graphOptions.captureIntermediates = intermediates != 0;
}

uint32_t nf_native_captures(const NfNative* n, uint32_t slot, char* out, size_t out_len) {
  if (out_len) out[0] = 0;
  if (slot >= n->slots.size() || !n->slots[slot].graph) return 0;
  try {
    const auto& captures = n->slots[slot].graph->boundaries();
    std::string text;
    for (const auto& [name, activation] : captures)
      text += name + "\t" + std::to_string(activation->validBytes()) + "\t" + std::to_string(activation->buffer.size) + "\n";
    say(out, out_len, text);
    return static_cast<uint32_t>(captures.size());
  } catch (...) {
    return 0;
  }
}

uint32_t nf_native_capture(NfNative* n, uint32_t slot, const char* name, void* out, size_t out_len, char* err,
                           size_t err_len) {
  if (n->stalled()) {
    say(err, err_len, "the network's queue stalled earlier; it takes no more work");
    return 0;
  }
  if (slot >= n->slots.size() || !n->slots[slot].graph) {
    say(err, err_len, "no such slot");
    return 0;
  }
  try {
    const auto& captures = n->slots[slot].graph->boundaries();
    const auto found = captures.find(name);
    if (found == captures.end()) {
      say(err, err_len, std::string("no capture named ") + name);
      return 0;
    }
    const nr::Activation& capture = *found->second;
    if (out_len != capture.validBytes()) {
      say(err, err_len, "capture " + found->first + " holds " + std::to_string(capture.validBytes()) + " bytes, not " +
                            std::to_string(out_len));
      return 0;
    }
    const std::vector<uint8_t> bytes = n->context->download(capture.buffer, out_len);
    std::memcpy(out, bytes.data(), out_len);
    return 1;
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
    return 0;
  }
}

void nf_native_fake_chain_timeout(NfNative* n, uint32_t in_warm_up) {
  if (in_warm_up)
    n->fakeWarmUpTimeout = true;
  else
    ++n->fakeTimeouts;
}

void nf_native_close(NfNative* n) {
  if (!n) return;
  try {
    n->dropGraph();
    if (!n->stalled()) {
      if (n->secondaryPool) vkDestroyCommandPool(n->context->device(), n->secondaryPool, nullptr);
      if (n->ownPool) vkDestroyCommandPool(n->context->device(), n->ownPool, nullptr);
      for (VkDescriptorPool pool : n->holdPools)
        if (pool) vkDestroyDescriptorPool(n->context->device(), pool, nullptr);
    }
    n->kernels.reset();
    n->model.reset();
    n->context.reset();
  } catch (...) {
  }
  delete n;
  g_live.store(false);
}

}  // extern "C"
