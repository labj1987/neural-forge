// The C interface between the layer (Rust) and the vendored OpenDLSS-NR network (C++).
//
// The layer owns everything about the game's frame: when to run, which images, the proxy encode,
// history and composite. This side owns the network: the model, the kernels, the graph for one
// field size, recorded once into a secondary command buffer the layer executes every frame.
//
// Every Vulkan call made here goes through the next layer's dispatch (the `get_instance_proc_addr`
// given at open), never through the loader, so none of it is seen by the layer's own hooks.
// Functions return 1 on success and 0 on failure with a message in `err` (always terminated).
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t nf_native_abi_version(void);
uint32_t nf_native_asset_count(const char* kind);

// ---- Device creation -------------------------------------------------------------------------

// Whether `physical` can run the network: every extension and feature the kernels need. On 0,
// `missing` names the first missing one.
uint32_t nf_native_device_supported(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical,
                                    char* missing, size_t missing_len);

// Extends `in` (the create info about to go to the next layer's vkCreateDevice) with the network's
// extensions and features, writing the result to `out`. Feature flags the application's own chain
// already carries a structure for are set in that structure in place and put back by
// nf_native_device_restore; everything else is prepended in memory owned by `*state`. `out`, its
// extension list and its chain stay valid until nf_native_device_restore(*state), which must be
// called once vkCreateDevice has returned, whatever it returned.
uint32_t nf_native_device_extend(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical,
                                 const VkDeviceCreateInfo* in, VkDeviceCreateInfo* out, void** state, char* err,
                                 size_t err_len);
void nf_native_device_restore(void* state);

// ---- The network on an adopted device --------------------------------------------------------

typedef struct NfNative NfNative;

typedef struct NfNativeOpen {
  PFN_vkGetInstanceProcAddr gipa;   // the next layer's
  VkInstance instance;
  VkPhysicalDevice physical;
  VkDevice device;
  uint32_t queue_family;            // the family of a queue the layer added for the network (uploads, warm-up)
  uint32_t queue_index;             // that queue's index; nothing else may use it
  uint32_t frame_family;            // the family of the game's queue the recorded graph runs on
  const char* model_dir;            // the extract-model output
  uint32_t chain;                   // 1: counter chaining between PTX launches; 0: barriers
  uint32_t fence_timeout_ms;        // bound on every wait this side makes
  // Called on every dispatchable object this side gets from below the loader (its queue, its command
  // buffers), to set the loader's dispatch pointer; null outside a layer.
  void (*init_dispatchable)(void* user, VkDevice device, void* object);
  void* init_user;
} NfNativeOpen;

// Loads and verifies the model (SHA-256), builds the kernels. Slow (seconds): run off the game's
// threads. Uses only `queue_index`, which nothing else may use while any nf_native call runs. On
// failure `*stalled` (when not null) says whether a wait on the queue timed out (nf_native_stalled).
NfNative* nf_native_open(const NfNativeOpen* open, uint32_t* stalled, char* err, size_t err_len);

// The graph for a `valid_width` x `valid_height` frame: allocates the activations, runs it once on
// the layer's queue (the driver compiles the PTX then, and the weights' device copies are made),
// then records it into the secondary command buffer nf_native_graph_commands returns. Slow. Any
// previous build's secondary must not be pending.
uint32_t nf_native_build(NfNative* n, uint32_t valid_width, uint32_t valid_height, char* err, size_t err_len);

typedef struct NfNativeFrame {
  uint32_t field_width, field_height;   // the padded field the network runs on
  VkBuffer features;                    // f32 [field_height][field_width][16], written by the layer
  VkDeviceSize features_bytes;
  VkBuffer head;                        // f32 [field_height][field_width][4], read by the layer
  VkDeviceSize head_bytes;
  float blend_scale;                    // block70.layer0.blend_scale, clamped to [0, 1]
  uint32_t chained;                     // the recorded graph links its launches by counters
} NfNativeFrame;

uint32_t nf_native_frame(const NfNative* n, NfNativeFrame* frame);

// The recorded graph: a secondary command buffer (simultaneous use, no render pass) of `frame_family`,
// reading `features` and writing `head`. It opens and closes with compute barriers only; the caller
// orders its own writes of `features` before it and its reads of `head` after it.
VkCommandBuffer nf_native_graph_commands(const NfNative* n);

// The same graph recorded a second time, as a secondary of `queue_family` (the network's own queue's
// family): for work the layer submits on a queue of that family (the after-the-upscaler path). Both
// recordings share the activations: they must never run at the same time.
VkCommandBuffer nf_native_graph_commands_own(const NfNative* n);

// Records the graph straight into `primary` (descriptor pool 0, which the warm-up also uses). For
// measurements beside the secondary; the frame path uses nf_native_graph_commands.
uint32_t nf_native_record_graph(NfNative* n, VkCommandBuffer primary, char* err, size_t err_len);

// Counter-chain waits that gave up since the last reset (the watchdog). Read only once a frame that
// executed the graph has completed. `where` names the first stuck counter.
uint32_t nf_native_chain_timeouts(const NfNative* n, char* where, size_t where_len);
void nf_native_reset_chain_timeouts(NfNative* n);

// Rebuilds the graph with barriers between every launch (process-wide, permanent). The previous
// secondary must not be pending. Same cost as nf_native_build.
uint32_t nf_native_fall_back_to_barriers(NfNative* n, char* err, size_t err_len);

// 1 once a bounded wait on the network's queue timed out: its work may still be running, so the network
// takes no more work (every later build fails) and destroys nothing (close leaks what it made). The
// caller treats the network as dead for this device.
uint32_t nf_native_stalled(const NfNative* n);

// Destroys everything this side made. Nothing of it may be pending.
void nf_native_close(NfNative* n);

#ifdef __cplusplus
}
#endif
