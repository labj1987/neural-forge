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
// The hold slots nf_native_record_graph_at serves (`hold` 0 up to this, exclusive).
uint32_t nf_native_hold_slots(void);

// ---- Device creation -------------------------------------------------------------------------

// Whether `physical` can run the network: every extension and feature the kernels need, and an instance
// created for Vulkan 1.3 or later (`instance_api_version`: the application's VkApplicationInfo.apiVersion,
// 0 when it gave none, which is 1.0). The kernels are SPIR-V 1.6 and the glue calls Vulkan 1.3's core
// commands, which a device made from an older instance does not have whatever the GPU supports. On 0,
// `missing` names the first missing one.
uint32_t nf_native_device_supported(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, uint32_t instance_api_version,
                                    VkPhysicalDevice physical, char* missing, size_t missing_len);

// Extends `in` (the create info about to go to the next layer's vkCreateDevice) with the network's
// extensions and features, writing the result to `out`. `in` and its chain are only read: `out`'s chain
// is a copy of the application's (each structure's size from the Vulkan registry, the loader's own
// link and callback nodes included; from the first structure this build does not know on, the rest stays
// the application's, linked as it is and never written) with the flags set in the copies, and the structures the
// application has none of prepended, all in memory owned by `*state`. Fails where
// nf_native_device_supported does (an instance older than 1.3 says so as is; a missing device requirement
// as "the device lacks <it>"), and when the application enables VK_EXT_buffer_device_address (it excludes the core feature the kernels need). `out`, its
// extension list and its chain stay valid until nf_native_device_restore(*state), which frees them
// and must be called once vkCreateDevice has returned, whatever it returned.
uint32_t nf_native_device_extend(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, uint32_t instance_api_version,
                                 VkPhysicalDevice physical,
                                 const VkDeviceCreateInfo* in, VkDeviceCreateInfo* out, void** state, char* err,
                                 size_t err_len);
void nf_native_device_restore(void* state);

// ---- The network on an adopted device --------------------------------------------------------

typedef struct NfNative NfNative;

// One tensor record of the model, read and checked by the Rust side (crates/native/src/model_file.rs): `bytes`
// is its `byte_length` bytes at `stage_offset` in stage `stage`'s file.
typedef struct NfNativeTensor {
  const char* name;
  const char* parameter;
  const char* stage;
  int32_t block;
  int32_t layer;
  uint32_t stage_offset;
  uint32_t byte_length;
  const uint8_t* bytes;
} NfNativeTensor;

typedef struct NfNativeModel {
  uint32_t block_count;             // the manifest's totals.blockCount
  const NfNativeTensor* tensors;
  size_t tensor_count;
} NfNativeModel;

typedef struct NfNativeOpen {
  PFN_vkGetInstanceProcAddr gipa;   // the next layer's
  VkInstance instance;
  VkPhysicalDevice physical;
  VkDevice device;
  uint32_t queue_family;            // the family of a queue the layer added for the network (uploads, warm-up)
  uint32_t queue_index;             // that queue's index; nothing else may use it
  uint32_t frame_family;            // the family of the game's queue the recorded graph runs on
  // The extract-model output, loaded and verified (SHA-256): read during the call, and the records' strings and
  // bytes stay valid and unchanged until nf_native_close (the weights are re-laid out from them on demand).
  const NfNativeModel* model;
  uint32_t chain;                   // 1: counter chaining between PTX launches; 0: barriers
  uint32_t fence_timeout_ms;        // bound on every wait this side makes
  // Called on every dispatchable object this side gets from below the loader (its queue, its command
  // buffers), to set the loader's dispatch pointer; null outside a layer.
  void (*init_dispatchable)(void* user, VkDevice device, void* object);
  void* init_user;
} NfNativeOpen;

// Uploads the model's records, builds the kernels. Slow (seconds): run off the game's
// threads. Uses only `queue_index`, which nothing else may use while any nf_native call runs. On
// failure `*stalled` (when not null) says whether a wait on the queue timed out (nf_native_stalled).
// One network per process at a time (volk's function table is process-global): fails while another
// is open, until nf_native_close.
NfNative* nf_native_open(const NfNativeOpen* open, uint32_t* stalled, char* err, size_t err_len);

// The graph for a `valid_width` x `valid_height` frame: allocates the activations, runs it once on
// the layer's queue (the driver compiles the PTX then, and the weights' device copies are made),
// then records it into the secondary command buffer nf_native_graph_commands returns. Slow. Any
// previous build's secondary must not be pending. Invalid sizes (none, or a zero width or height) fail
// before anything is dropped: the previous build stays as it was. Counter-chain timeouts left from frames
// of the previous build (unread by nf_native_chain_timeouts), or a warm-up's own, switch the network to
// barriers as nf_native_fall_back_to_barriers does, and the build records with them (`chained` 0); every
// warm-up starts from a count of 0.
uint32_t nf_native_build(NfNative* n, uint32_t valid_width, uint32_t valid_height, char* err, size_t err_len);

// nf_native_build for `count` frame sizes at once (`sizes`: valid width, height pairs): one graph each, slot i
// for sizes[i] (slot 0 is what nf_native_frame and nf_native_graph_commands return). Each is warmed up as
// nf_native_build's (the largest first), then all are recorded. The graphs share the weights and the kernels'
// scratch, and run one after another on one queue, never at the same time. Any previous build's secondaries must
// not be pending. Slow (one warm-up per size).
uint32_t nf_native_build_sizes(NfNative* n, const uint32_t* sizes, uint32_t count, char* err, size_t err_len);

typedef struct NfNativeFrame {
  uint32_t field_width, field_height;   // the padded field the network runs on
  VkBuffer features;                    // f32 [field_height][field_width][16], written by the layer
  VkDeviceSize features_bytes;
  VkBuffer head;                        // f32 [field_height][field_width][4], read by the layer
  VkDeviceSize head_bytes;
  float blend_scale;                    // the post block's blend_scale (block 70, 30 small), clamped to [0, 1]
  uint32_t chained;                     // the recorded graph links its launches by counters
} NfNativeFrame;

uint32_t nf_native_frame(const NfNative* n, NfNativeFrame* frame);
// Slot `slot`'s frame (nf_native_build_sizes); 0 past the last slot.
uint32_t nf_native_frame_at(const NfNative* n, uint32_t slot, NfNativeFrame* frame);

// The recorded graph: a secondary command buffer (simultaneous use, no render pass) of `frame_family`,
// reading `features` and writing `head`. It opens and closes with compute barriers only; the caller
// orders its own writes of `features` before it and its reads of `head` after it.
VkCommandBuffer nf_native_graph_commands(const NfNative* n);

// The same graph recorded a second time, as a secondary of `queue_family` (the network's own queue's
// family): for work the layer submits on a queue of that family (the after-the-upscaler path). Both
// recordings share the activations: they must never run at the same time.
VkCommandBuffer nf_native_graph_commands_own(const NfNative* n);
// Slot `slot`'s recordings (nf_native_build_sizes); null past the last slot.
VkCommandBuffer nf_native_graph_commands_at(const NfNative* n, uint32_t slot);
VkCommandBuffer nf_native_graph_commands_own_at(const NfNative* n, uint32_t slot);

// Records the graph straight into `primary`, a primary command buffer of `frame_family` the caller is
// recording on this thread (descriptor pool 0, which the warm-up also uses, reset first: no buffer
// recorded by an earlier call may still be pending). For measurements beside the secondary; the frame path
// uses nf_native_graph_commands.
uint32_t nf_native_record_graph(NfNative* n, VkCommandBuffer primary, char* err, size_t err_len);

// Measurement only (the rig's per-kernel profile): a timestamp after every dispatch that the next
// nf_native_record_graph records into `primary`, up to `max_dispatches`. Call it on `primary` just before that
// recording, and nf_native_profile_end once the buffer has completed.
uint32_t nf_native_profile_begin(NfNative* n, VkCommandBuffer primary, uint32_t max_dispatches, char* err, size_t err_len);

// The profile's dispatches, one "label\tmilliseconds\n" line each (the label names the block and the kernel),
// written into `out` (truncated to `out_len`, NUL-terminated). Returns how many dispatches were timed.
uint32_t nf_native_profile_end(NfNative* n, char* out, size_t out_len);

// Records slot `slot`'s graph (nf_native_build_sizes) straight into `primary`, a primary command buffer of
// `frame_family` the caller is recording (the application's: the hold inside DLSS's command buffer), between
// two compute barriers. Its descriptor sets come from hold slot `hold`'s own pool (0..15, made on first use),
// which this call resets first: the caller guarantees that no command buffer recorded with that hold slot
// may still be submitted or pending. The recording reads the build's activations, so the build must not be
// dropped (a rebuild, nf_native_fall_back_to_barriers, nf_native_close) while any such buffer may still be
// submitted. Binds compute pipelines and descriptor sets and pushes constants: the caller restores its own
// state after it. On failure (0) part of the graph may be recorded.
uint32_t nf_native_record_graph_at(NfNative* n, uint32_t slot, uint32_t hold, VkCommandBuffer primary, char* err,
                                   size_t err_len);

// Counter-chain waits that gave up since the last reset (the watchdog). Read only once a frame that
// executed the graph has completed. `where` names the first stuck counter.
uint32_t nf_native_chain_timeouts(const NfNative* n, char* where, size_t where_len);
void nf_native_reset_chain_timeouts(NfNative* n);

// Switches to barriers between every launch (process-wide, permanent: a network opened later in the
// process does not chain either, whatever its `chain`) and drops the graph recorded with chaining: the
// next nf_native_build records it with barriers. The previous secondary must not be pending. Cheap.
void nf_native_fall_back_to_barriers(NfNative* n);

// Forgets a fall back to barriers for networks opened from now on (the live one keeps what it has): for a
// test of the fall back that shares its process with other tests.
void nf_native_forget_fall_back(void);

// For tests of the graph's captures (OpenDLSS-NR's parity copies): the graphs of the next builds copy every block's
// and transition's output as they run (`boundaries`), and with `intermediates` (and `boundaries`) the tensors inside
// each block too, which runs the blocks unfused. Both 0 (no captures) until set. Takes effect at the next build.
void nf_native_set_captures(NfNative* n, uint32_t boundaries, uint32_t intermediates);

// Slot `slot`'s captures, one "name\tbytes\tbuffer_bytes\n" line each (`bytes` what a run copies into it,
// `buffer_bytes` the size of the buffer it is copied into), written into `out` (truncated to `out_len`,
// NUL-terminated). Returns how many there are (0 past the last slot).
uint32_t nf_native_captures(const NfNative* n, uint32_t slot, char* out, size_t out_len);

// Reads capture `name` of slot `slot` back into `out`: exactly its `bytes` (`out_len`), as the last run of the slot's
// graph left them. Only once that run has completed. Waits for the network's queue (a download on it).
uint32_t nf_native_capture(NfNative* n, uint32_t slot, const char* name, void* out, size_t out_len, char* err,
                           size_t err_len);

// For tests of the watchdog: as if a counter wait gave up, in the graph's last frame (`in_warm_up` 0: the
// count nf_native_chain_timeouts reads) or in the next build's first warm-up (1). Cleared as a real one is.
void nf_native_fake_chain_timeout(NfNative* n, uint32_t in_warm_up);

// 1 once a bounded wait on the network's queue timed out: its work may still be running, so the network
// takes no more work (every later build fails) and destroys nothing (close leaks what it made). The
// caller treats the network as dead for this device.
uint32_t nf_native_stalled(const NfNative* n);

// GPU bytes of the weights: the records' raw buffers as loaded and now (after the first build only the scale
// vectors the kernels read through them), and the matrices built from the records.
void nf_native_weight_bytes(const NfNative* n, uint64_t* raw_loaded, uint64_t* raw, uint64_t* matrices);

// Destroys everything this side made. Nothing of it may be pending: no submitted work that executes a
// recording (nf_native_graph_commands*, nf_native_record_graph*) may still run.
void nf_native_close(NfNative* n);

#ifdef __cplusplus
}
#endif
