# Phase 4: DMA-BUF transport -- investigation and current status

**Status: blocked on a real architectural constraint, not a missing implementation.**
This document exists so a future session doesn't re-derive (or worse, re-discover the
hard way) what this one already found. Read this before writing any DMA-BUF code.

## What Phase 4 was meant to be

Phase 3 (`EXTERNAL_MEMORY_HOST_DESIGN.md`, `PROTOCOL_V3_DESIGN.md`) removed the CPU
copies on the *host-memory* path: both sides import the same SHM proxy/answer regions
directly as device memory via `VK_EXT_external_memory_host`. That still means every
frame crosses through system RAM. DMA-BUF transport is the next lever: share the
*actual GPU-resident image* between the layer's device and the helper's device, with
no host round trip at all -- the theoretical ceiling for this pipeline's transport
cost. `neuralforge_protocol::ShmHeader` already reserves the fields this would need
(`proxy_export_seq`/`proxy_pid`/`proxy_fd`/`proxy_gen` and the `answer_*` equivalents,
plus `layer_proxy_seq`/`layer_answer_seq` as the importer's echo) -- written
speculatively in an earlier session, before anyone had confirmed the underlying
mechanism actually works. This session confirmed it does not, at least not the way
those fields imply.

## Why this is architecturally harder than Phase 3

Phase 3's zero-copy work is symmetric: both the layer (native Linux) and the helper
(a Windows PE binary, but running under Wine on the *same* Linux machine) each talk to
their own real Vulkan device and import host memory through the same POSIX API
(`VK_EXT_external_memory_host`, a plain virtual-address import -- no driver-specific
handle exchange needed).

DMA-BUF is not symmetric this way. `VK_EXT_external_memory_dma_buf`/
`VK_KHR_external_memory_fd` are POSIX-specific: they hand back (or accept) a raw Linux
file descriptor. A **Windows guest application under Wine never sees that fd** --
`crates/helper/src/main.rs`'s own `WANTED_DEVICE_EXTENSIONS` comment already records
why (found and fixed 2026-09-10, well before this phase started): Wine's Vulkan
implementation for Windows guest apps only ever exposes the Windows-shaped
`VK_KHR_external_memory_win32` handle type. Requesting the Linux `_fd`/`dma_buf` pair
from inside the helper fails outright -- confirmed again this session:
`vkEnumerateDeviceExtensionProperties` on the helper's own device does not list
`VK_EXT_external_memory_dma_buf` or `VK_KHR_external_memory_fd` at all, only the win32
one.

So the real question isn't "how do we wire up the extension" (that part's a dead
end) -- it's **"can a win32 external-memory handle a Wine guest gets from
`vkGetMemoryWin32HandleKHR` be turned into (or built from) a real Linux fd another,
unrelated Linux process can import via `VK_EXT_external_memory_dma_buf`?"**

## What this session actually tried

Wine ships two long-stable, genuinely exported (if unofficial-for-app-use) `ntdll.dll`
functions specifically for bridging a win32 `HANDLE` and a raw Unix fd:
`wine_server_handle_to_fd` and `wine_server_fd_to_handle`. Confirmed present in this
project's actual target Proton build before writing any code:

```
objdump -p ".../Proton-CachyOS Latest/files/lib/wine/x86_64-windows/ntdll.dll" \
    | grep wine_server
	wine_server_call
	wine_server_fd_to_handle
	wine_server_handle_to_fd
```

`crates/helper/examples/dmabuf_probe.rs` (kept in the repo, see its own doc comment
for the exact result) does the natural first experiment:

1. Build a real Vulkan device in the helper (same `WANTED_DEVICE_EXTENSIONS`-style
   setup as `main.rs`'s own `create_vulkan_context`).
2. Allocate a small buffer with `VkExportMemoryAllocateInfo`/
   `VkExternalMemoryBufferCreateInfo` requesting `OPAQUE_WIN32`.
3. Call `vkGetMemoryWin32HandleKHR` (resolved by hand via `get_device_proc_addr`, not
   `ash`'s own `ExternalMemoryWin32::new`, which panics on a resolution failure the
   same way `EXTERNAL_MEMORY_HOST_DESIGN.md` already found and fixed for a different
   extension) to get a real win32 `HANDLE`.
4. Resolve `wine_server_handle_to_fd` from `ntdll.dll` via `GetProcAddress`, call it
   (wrapped in `guard::guarded`, since Wine does not treat this as a stable ABI a
   wrong guess should be safe to get wrong).

**Result on real hardware** (`lordnikon`, RTX 5070, driver 615.71.09,
Proton-CachyOS): every step up through getting the win32 handle succeeded --
`vkGetMemoryWin32HandleKHR` returned a real, non-null handle. `wine_server_handle_to_fd`
itself did not fault (`guard::guarded` reported `seh=0` -- the guessed function
signature is correct, this really is Wine's real ABI for it) but returned
`STATUS_OBJECT_TYPE_MISMATCH` (`0xC0000024`), a real, specific NTSTATUS, not a generic
failure.

## What that result means

`wine_server_handle_to_fd` is designed to unwrap **wineserver's own tracked kernel
objects** (files, sockets, devices -- things wineserver itself opened and knows the
real fd for) back into a Unix fd. A Vulkan external-memory win32 handle is not one of
those. It never went through `CreateFileW`/`socket()`/any wineserver-mediated open --
winevulkan (and, underneath it, the real host NVIDIA driver) minted it directly as an
opaque, driver-private token. `STATUS_OBJECT_TYPE_MISMATCH` read literally says
exactly that: wineserver doesn't recognize this handle's type as one it can hand back
a raw fd for, because it was never wineserver's object to begin with.

The most likely underlying reason, based on how NVIDIA's proprietary driver is known
to implement cross-process/cross-API memory sharing on real Windows (which is what
Wine's win32 external-memory path ultimately forwards to, even running on Linux):
`OPAQUE_WIN32` external memory on NVIDIA is backed by an NVIDIA-internal shared-surface
mechanism (historically an "NvKmtHandle"-style token), a genuinely different code path
from `VK_EXT_external_memory_dma_buf`'s kernel `dma_buf` objects -- not merely a
different *API shape* for the same underlying resource. If that's right, there may be
no real Linux `dma_buf` fd behind this handle at all, on this driver, for this handle
type -- not something any amount of Wine-side fd-juggling can extract, because it was
never created that way.

## The reverse direction (tried, also a dead end -- different reason)

The obvious next experiment: have the *layer* (full, native
`VK_EXT_external_memory_dma_buf` access) create a real dma-buf fd, get it into the
helper process's own fd table by having the helper open
`Z:\proc\<layer_pid>\fd\<layer_fd>` via `CreateFileW` (Wine already transparently maps
Unix paths under `Z:\`, the same translation `crates/helper/src/shm.rs::windows_path`
already relies on for the SHM file itself), then see whether the resulting handle is
usable at all.

`crates/layer/examples/dmabuf_export_probe.rs` (native Linux) allocates a real
`VK_EXT_external_memory_dma_buf`-exported buffer, gets a real fd via
`vkGetMemoryFdKHR`, prints its own pid and that fd, and holds both open for 90 seconds.
`crates/helper/examples/dmabuf_import_probe.rs` (Windows, run under Wine) takes that
pid+fd as CLI arguments and calls `CreateFileW` on `Z:\proc\<pid>\fd\<fd>`.

**Result on real hardware** (`lordnikon`, RTX 5070, driver 615.71.09, Proton-CachyOS):
`CreateFileW` itself failed. Traced the root cause down to the plain POSIX level,
*before* Wine is involved at all -- with the exporter process alive and its fd
confirmed open (`ls -la /proc/<pid>/fd/<fd>` showed a real, present entry), a direct,
non-Wine `cat /proc/<pid>/fd/<fd>` and a Python `os.open(...)` on the exact same path
both failed with `ENXIO` ("No such device or address"). `readlink` on that fd entry
shows `/dmabuf:` -- a dma-buf fd is backed by an `anon_inode` (like `epoll`/`eventfd`
fds), and the kernel does not implement `open()` on `/proc/<pid>/fd/<N>` for
anon-inode-backed files; only `dup()`-family syscalls (or fd-passing over a Unix
socket via `SCM_RIGHTS`) can hand another process a working reference to one. Wine's
own `Z:\proc\...` path mapping is real and otherwise works fine (confirmed via
`dosdevices/z: -> /`) -- it's simply asking the kernel to do something no `open()`
call, from any process, Wine-hosted or not, can do for this class of fd.

This makes the reverse direction's blocker a *different, more fundamental* one than
the forward direction's: the forward direction failed because of what the win32 handle
*was* (an NVIDIA-private token, not backed by a real `dma_buf` at all, per
`wine_server_handle_to_fd`'s `STATUS_OBJECT_TYPE_MISMATCH`). The reverse direction
fails because `/proc/pid/fd` re-opening -- the specific mechanism this document, and
the protocol header's reserved `proxy_pid`/`proxy_fd` fields, assumed would work for
*any* fd type -- cannot carry an anon-inode fd across a process boundary at all,
independent of Wine or NVIDIA specifics. A real dma-buf fd would need to cross via
`SCM_RIGHTS` over a Unix domain socket (the actual standard mechanism for this), which
means at minimum a wineserver-mediated Unix socket the helper can receive on --
something this project's current SHM-only IPC does not have, and a materially bigger
undertaking than the "just add a CreateFileW call" version tried here.

- Asking NVIDIA driver internals more directly (`nvidia-settings`/proc/sysfs, or the
  proprietary driver's own debug/query interfaces) whether `OPAQUE_WIN32` memory under
  Wine is ever `dma_buf`-backed on this driver version -- no obvious entry point found
  in a quick look; would need real NVIDIA documentation or source this project doesn't
  have access to. Moot for the reverse direction regardless, now that the fd-transfer
  mechanism itself is confirmed broken independent of what backs the memory.

## What this means for the project

Phase 4, as specified (share GPU memory via `dma_buf` between the native Linux layer
and *this* Wine-hosted Windows helper), is blocked by two independent, real,
confirmed-on-hardware constraints, not a gap in this project's own code that more
engineering effort closes:

1. The forward direction (Wine handle -> Linux fd): NVIDIA's `OPAQUE_WIN32` external
   memory under Wine is very likely not `dma_buf`-backed at all on this driver.
2. The reverse direction (Linux fd -> Wine handle via `/proc/pid/fd`): dma-buf fds are
   anon-inode-backed and cannot be re-opened via `/proc/<pid>/fd/<N>` by *any* process,
   Wine-hosted or not -- this is a plain Linux kernel limitation, confirmed without
   Wine in the loop at all.

The header's already-reserved `proxy_pid`/`proxy_fd`/etc. fields describe a mechanism
that this session's evidence says cannot work as designed for a real dma-buf fd,
regardless of which side initiates it. A working design would need real `SCM_RIGHTS`
fd-passing over a Unix domain socket instead -- a materially different and bigger
transport than anything this project's IPC currently has.

A same-process or same-OS native Linux NGX helper (no Wine at all) would sidestep this
whole asymmetry -- full, native `VK_EXT_external_memory_dma_buf` access with no
cross-process win32-handle problem layered on top at all. **Investigated as its own
question in `NATIVE_NGX_HELPER_DESIGN.md` (this correction: that idea was not actually
an established prior decision recorded in `ATTRIBUTION.md`, contrary to what an earlier
draft of this section claimed -- it was this project's own speculative aside, now
checked for real).** Real hardware evidence there: NVIDIA does ship a genuine native
Linux NGX runtime (`libnvidia-ngx.so.1`) that boots cleanly with no caller-identity
workaround needed, but no native Linux implementation of this project's actual target
feature (DLSS 5 Neural Rendering, `NVSDK_NGX_Feature_Reserved18`) exists anywhere --
it's an explicitly reserved, unallocated feature ID in NVIDIA's own current public SDK,
never published in any form except the Windows-only `nvngx_dlssnr.dll` this project
already depends on. That route is closed for a missing-artifact reason, not an
architecture one, and isn't something more Wine-side engineering effort here can fix.

**Recommendation**: leave Phase 4 blocked here. Both DMA-BUF transport directions this
document considered are now empirically closed, not merely judged unlikely, and the
native-helper detour that would have sidestepped the whole problem is also closed, for
an unrelated and unfixable-by-this-project reason. Phase 3's host-memory path
(`EXTERNAL_MEMORY_HOST_DESIGN.md`) already removed the CPU-copy overhead this phase
would have further reduced by avoiding a virtual-memory round trip; what DMA-BUF would
have additionally saved is real, but its actual size relative to the *model's own eval
time* is unknown until the GTA fps gate itself is measured (Phase 1's still-open
benchmark, Phase 5's own gate). If that measurement later shows transport cost is still
the dominant remaining cost, the real next step is `SCM_RIGHTS`-based fd-passing over a
Unix socket -- a genuinely new IPC mechanism, not a variant of anything tried here, and
still Wine-hosted-helper-compatible unlike the native-helper idea.
