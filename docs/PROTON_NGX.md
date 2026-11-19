# DLSS's own parameters through Proton: the NGX wrapper

The layer needs DLSS's colour input, depth, motion vectors, output, exposure, camera jitter, motion vector units and
reset to place and run its holds (`preupscale.rs`). The game hands NGX all of these by name, in the same process, on
the Windows side of Proton. This document is how Neural Forge reads them there, and how the layer uses them: the
records are its only source of them. Until 4.0 the layer inferred them from the CUDA kernels DLSS launches, with rules
that every new engine needed changed (PRE_UPSCALER_DESIGN.md and LESSONS.md keep that history).

## How it fits together

```
game / sl.dlss.dll ──NGX's static loader──► C:\windows\system32\_nvngx.dll   (WinVerifyTrust on NVIDIA's signed file)
                                              │ Wine's load order: _nvngx=b
                                              ▼
                        Neural Forge's builtin _nvngx.dll (crates/ngx-wrapper)
                              │ forwards all 75 exports to NVIDIA's core: its copy of the driver's (NVIDIA_WINE_DLL_DIR)
                              │   in C:\windows\system32\neural-forge\_nvngx.dll
                              │ a record per CreateFeature, two per EvaluateFeature (before and after NVIDIA's call)
                              │   (only with NEURAL_FORGE_ENABLE=1) ──► /dev/shm/neural-forge-ngx-<pid>.ring
                              ▼
            vkCmdCuLaunchKernelNVX ──► VK_LAYER_neuralforge_neural: ngx_ring.rs reads the ring
                                        records = the only source of what DLSS works on
```

- **Where NGX is loaded.** Proton copies NVIDIA's `_nvngx.dll` and `nvngx.dll` from the driver's `nvidia/wine`
  folder into the prefix's `system32` and sets `NVIDIA_WINE_DLL_DIR` (the `proton` script's `setup_prefix` and
  `init_session`). The prefix's registry names `C:\Windows\System32` as NGXCore's `FullPath`. NGX's static loader,
  linked into the game or into Streamline's `sl.common.dll`, checks that file's signature with `WinVerifyTrust` and
  loads it; NVIDIA's thin `nvngx.dll` is not used by it (Cyberpunk 2077, GTA V Enhanced, Crimson Desert, Red Dead
  Redemption 2).
- **Why a builtin passes the signature check.** Wine applies a DLL override to a module in the system directory by its
  base name (`dlls/ntdll/unix/loadorder.c`, `get_load_order`), and when the file found is a native PE and the load
  order says builtin, maps the builtin of that name instead (`dlls/ntdll/unix/loader.c`, `load_builtin`). The module
  keeps the requested path. So `WinVerifyTrust` reads NVIDIA's unchanged signed file, and `LoadLibrary` of that path
  returns the wrapper. A PE is a builtin when the 32 bytes at file offset 0x40 read `Wine builtin DLL`
  (`server/mapping.c`), which `scripts/build-ngx-wrapper.sh` writes after the build, as `winebuild --builtin` does.
- **Loading NVIDIA's core.** The wrapper copies `Z:<NVIDIA_WINE_DLL_DIR>/_nvngx.dll` to
  `C:\windows\system32\neural-forge\_nvngx.dll` (a folder that holds only that file) and loads the copy
  (`crates/ngx-wrapper/src/core_copy.rs`), so the core runs, as under stock Proton, from a folder without DLSS feature
  DLLs and under its own name. NVIDIA's core treats the folder it runs from as a place of its own (see "Where the core
  looks for feature DLLs" below), and the driver's `nvidia/wine` folder holds NVIDIA's `nvngx_dlssg.dll` and
  `nvngx_dlssd.dll`; Indiana Jones and the Great Circle crashed only with the core loaded from there (#51). The copy
  keeps the name `_nvngx.dll`: NVIDIA's feature DLLs (`nvngx_dlss.dll`, `nvngx_dlssd.dll`, `nvngx_dlssg.dll`; 310.4 to 310.9 and the driver's
  copies, read in Ghidra) check who calls each of their NGX exports: `GetModuleHandleExA(FROM_ADDRESS)` on the return
  address, `GetModuleFileNameW`, and a case-sensitive `wcsstr(path, L"nvngx.dll")`; a caller whose path lacks it gets
  `0xBAD00002` and the log line "Error: Not called from NGX runtime". So the core's path must contain `nvngx.dll` in
  lower case (in any folder): as `_nvngx_core.dll`, `nvngx_core.dll` or `_NVNGX.DLL` DLSS never starts. The wrapper says so in its text log, its trace and its
  NGX versions record (so `doctor` warns) when the core it loaded breaks the rule (`core_copy::name_warning`: one
  named by `NEURAL_FORGE_NGX_CORE`), and `doctor` warns about such a `NEURAL_FORGE_NGX_CORE` in a game's launch
  options. Nothing else depends on the name: the game's NGX loader (in `sl.common.dll`) compares no name after the
  load, the core uses only its folder, and no code in them calls `GetModuleHandle("_nvngx.dll")` (the same Ghidra
  pass), so the wrapper and the core
  sharing a base name is safe. Wine would hand a module of that name to the wrapper
  (`_nvngx=b`), so the Proton copy also sets `C:\windows\system32\neural-forge\_nvngx=n`: a full-path key is
  looked up before the base name. Each time Neural Forge renews the wrapper in its copies (`refresh_wrapper`, at start and
  after an update), it also makes a copy's `proton` script again from its source's when they differ; a copy whose source is no
  longer installed keeps its script, and that start and `doctor` say to make the copy again. The copy is compared with the driver's
  file on every load (about 2 ms) and rewritten, through a temporary file renamed over it, only when they differ (a
  new driver); a copy another process of the game put in place first is taken as it is. If the copy cannot be
  written, or loading it returns the wrapper (a Proton copy without the new line), the wrapper loads the driver's
  file by its full path as before, with its own override `Z:\...\nvidia\wine\_nvngx=n` (without it Wine hands the
  wrapper back to itself; the wrapper checks: a load that returns its own module is not taken). The text log and the
  trace say which core it loaded and why. `NEURAL_FORGE_NGX_CORE` names another core to load instead (a Windows path,
  or a Linux one through `Z:`), for diagnosis. `uninstall --purge` removes the copies from the prefixes of every
  Steam library.
- **Exports the core lacks.** The wrapper hooks ten exports: `NVSDK_NGX_{D3D11,D3D12,VULKAN}_{CreateFeature,
  EvaluateFeature,ReleaseFeature}` and `NVSDK_NGX_VULKAN_CreateFeature1` (`core_exports.rs`, `HOOKED`). It calls NVIDIA's
  function of each, so a core without one of them fails the wrapper's load, and NGX reports itself unavailable: the
  game runs without DLSS rather than with a half-forwarded core. Every other export is a jump through the pointer
  resolved at load; one the core lacks (an older driver) jumps to a stub that returns `NVSDK_NGX_Result_Fail`
  (`0xBAD00000`), as NGX answers a call it cannot serve, and the log names it. The stub takes no arguments, which is
  safe whatever the export's signature: on x64 Windows the caller passes arguments in registers and its own stack
  space and cleans up after the call. Since 4.2.0 such an export is also taken out of the wrapper's export table at
  load (its address set to 0, `pe.rs`; Wine's `find_ordinal_export` answers that with `STATUS_PROCEDURE_NOT_FOUND`),
  so `GetProcAddress` finds it missing, as it would in NVIDIA's core, and a caller that checks for an entry point
  takes the same path as without the wrapper (#51: the reporter's driver, 615.71, is older than the 620.70 export
  list). Only a caller that bound it before the wrapper's `DllMain` ended (none: NGX is loaded with `LoadLibrary`)
  would meet the stub.
- **Loading the core from `DllMain`.** The wrapper loads NVIDIA's core and resolves the exports in its `DllMain`,
  under the loader lock, which Microsoft's guidance for `DllMain` advises against. It is needed and safe here: the
  passthrough exports are bare jumps through the resolved pointers, so they must be set before any export can be
  called, and nothing else runs before that; Wine's loader lock is recursive, so the nested load (and the core's own
  `DllMain`) runs on the same thread without waiting; the core is loaded by its full path with a native override,
  so it cannot be the wrapper again (checked); and the wrapper's `DllMain` waits on no other thread. It does not list
  the process's modules for the NGX versions record (below): it starts a thread for that, which Wine runs only once the
  loader lock is free, and only a core that did not load is recorded from `DllMain` itself (one small file write).
- **What it records.** For D3D12 the `ID3D12Resource`s become VkImages through vkd3d-proton's
  `ID3D12DXVKInteropDevice1::GetVulkanResourceInfo1` (and the layout DLSS reads them in through
  `GetVulkanImageLayout`); for D3D11 through DXVK's `IDXGIVkInteropSurface::GetVulkanImageInfo`; Vulkan games hand
  NGX `NVSDK_NGX_Resource_VK` with the VkImage. winevulkan passes VkImage handles through unwrapped, so they are the
  handles the layer saw at `vkCreateImage`. The interop device is asked for per resource and released after the
  lookup, so the wrapper keeps no device alive and a game that makes a new device is asked through the new one.
  Per evaluate: `Color`, `MotionVectors`, `Depth`, `Output`,
  `ExposureTexture`, `Jitter.Offset`, `MV.Scale`, `DLSS.Pre.Exposure`, `DLSS.Exposure.Scale`, `Reset`, the extents,
  DLSS's subrect (`DLSS.Render.Subrect.Dimensions`, and where the rectangle starts in each image:
  `DLSS.Input.Color.Subrect.Base`, `DLSS.Input.MV.Subrect.Base`, `DLSS.Input.Depth.Subrect.Base` and
  `DLSS.Output.Subrect.Base`, X and Y, which NVIDIA's SDK helpers set with `SetUI`; 0 when not given), the feature's
  output size from its creation (`OutWidth`/`OutHeight`), and the feature (Super Resolution 1, Frame Generation 11,
  Ray Reconstruction 13) with its creation flags (`IsHDR`, `MVLowRes`, `MVJittered`, `DepthInverted`, `AutoExposure`).
  Looking an image up calls into the resource the parameter names, so the wrapper does it only for parameters
  NVIDIA's own call reads, which the game must keep alive: the images only for Super Resolution and Ray
  Reconstruction (the features the layer uses; Frame Generation's records carry none), and `ExposureTexture` not
  with `AutoExposure` (DLSS ignores it then, so a game may leave it pointing at a freed resource). The other images
  the text log lists are logged as their bare pointers. A creation NVIDIA failed (`NVSDK_NGX_FAILED`) is logged
  only: its handle out-parameter is not NVIDIA's then, and is not read.
- **The ring.** `crates/ngx-wrapper/src/ring.rs` (version 3, since the subrect fields; 4.0's wrapper writes version 2):
  a 32-byte header, 1024 slots of 168 bytes. A writer reserves a slot with an atomic add, zeroes its `seq`, writes it,
  then sets `seq` to its number; the reader takes a slot only while `seq` reads the same before and after copying it.
  The file is named by the Linux pid (the wrapper asks Wine with `NtQueryInformationProcess(ProcessWineUnixPid)`;
  `Z:\proc\self` would name the wineserver, which opens files for Wine), which the layer gets from `getpid()`. The layer
  maps it read-only and removes the file, so nothing is left behind. A ring no layer opened is removed once its game is
  gone: when Neural Forge (the app or the CLI) starts, and by `uninstall --purge`; never a running process's
  (`proton_tool::sweep_dead_rings`). `doctor` reports a running game's ring of another version than the layer reads (a
  copy with an older wrapper). An evaluate's record is written before NVIDIA's `EvaluateFeature` is called, so before
  NGX records any of the evaluate's launches, on the same thread; once the call returned, the wrapper writes the same
  record again with `KIND_END` (bit 16 of `api`; the API is the low 16 bits). Every launch of the evaluate lies between
  the two.
- **The NGX versions.** The wrapper lists the process's modules (`modules.rs`; `K32EnumProcessModules`, which Wine
  serves), keeps `_nvngx.dll` and `nvngx*.dll` (itself, NVIDIA's core, the snippets `nvngx_dlss.dll`,
  `nvngx_dlssg.dll`, `nvngx_dlssd.dll`, `nvngx_dlssnr.dll`, ...), whatever path each was loaded from, and reads each
  file's `VS_FIXEDFILEINFO` version through `version.dll` (loaded when needed, not imported). It logs them when the
  text log is on and, in a game the layer is on in, writes `$XDG_STATE_HOME/neural-forge/ngx-modules/<app id>-<exe>.txt`
  (the format is in `modules.rs`), with which core it loaded and why (the text log's "NVIDIA's core" part) and when it
  wrote it (`stage=`). It writes three times at most: once NVIDIA's core is loaded (`core-loaded`: a thread
  `DllMain` starts, `note_core_loaded_later`, which Wine runs once the `LoadLibrary` that loaded the wrapper has
  returned; it holds a reference to the wrapper while it runs, so a game freeing NGX meanwhile cannot unmap its code);
  after a feature is created (`feature-created`, `note_loaded`), which replaces it with the snippets loaded by then,
  and again only when the set of paths changes; or, instead of both, when the core did not load (`core-failed`, with
  the reason and no modules, from `DllMain` before it fails the load). A core-loaded note that runs after a feature
  was created writes nothing, so the later list wins; nor does one that finds the file already holding a
  feature-created record of the same process (`process=`, the Windows process id and its creation time, since Wine gives a game the
  same small id run after run: a game that frees NGX and loads it again, whose wrapper starts with no memory of what it
  wrote). A run replaces the file. So a game that never creates a DLSS
  feature (#51) still leaves a record of which core it ran, and `doctor` and the report say that its last run created
  none. `<app id>` is Steam's `SteamAppId` (else
  `SteamGameId`), else `x` and a hash of the executable's path, so games whose executables share a name keep apart.
  Wine passes `HOME` and every `XDG_*` variable into the game's Windows environment only with `WINE_HOST_` before the
  name (`dlls/ntdll/unix/env.c`, `is_special_env_var`), so the wrapper reads `XDG_STATE_HOME` and `HOME` under either
  name (until 4.6.2 it missed `XDG_STATE_HOME` and wrote under the home), and without both takes the home from
  `WINEHOMEDIR` (`\??\Z:\home\<user>`): the same directory the supervisor uses. It writes through `Z:` for a `Z:` home, and through the NT path (`\\?\unix\home\<user>\.local\state\...`, or the other drive's) for a `\??\unix\...` or other-drive `WINEHOMEDIR`, so it needs no `Z:` drive. `doctor`
  and the report (`ngx_versions.rs`) show that record with how long ago it was written, and the NGX DLLs present in each switched-on game's folder, its prefix's
  `ProgramData/NVIDIA/NGX` and `system32`, and the driver's core, with their versions read from the PE resources on
  the Linux side (`model::dll_file_version`: headers and resource section only). The record is for a bug report's
  sake: nothing reads it to decide anything.
- **Only where the layer is on.** The wrapper is in every game a Neural Forge Proton runs, but it creates the ring
  only with `NEURAL_FORGE_ENABLE=1` (the layer's `enable_environment`), without `NEURAL_FORGE_DISABLE=1`, and not with
  `NEURAL_FORGE_NGX_RING=off`: the layer is the ring's only reader and the one that removes the file, so in any
  other game the file would stay in `/dev/shm` until `uninstall --purge`.
- **Why `/dev/shm`.** Both sides need one path without talking to each other. Wine maps `Z:\dev\shm` to the host's
  `/dev/shm`, and Steam's pressure-vessel container shares the host's `/dev/shm` in every runtime, so the layer
  (inside the container) and the wrapper (Wine in the same container) see the same file. `XDG_RUNTIME_DIR` is not
  reliably the same path or directory inside the Steam runtime, so it is not used.
- **A stalled slot.** If the layer reads nothing (not loaded, or slow to poll), the writer laps the ring: 1024
  slots, records overwritten oldest first, the file never grows. The reader takes record `n` only when its slot's
  `seq` reads `n + 1`; a slot whose `seq` is still 0 (its writer between reserving and finishing) stops the reader
  there until the next poll, and once the writer has reserved 1024 records past it, the reader jumps ahead and the
  slot's record is lost, never misread. A slot overwritten while copied fails the `seq` check after the copy and is
  skipped. Writers never wait on the reader.

## The call trace (`NEURAL_FORGE_NGX_TRACE`, `crates/ngx-wrapper/src/trace.rs`)

For a game that stalls or crashes with the wrapper in and logs nothing in `NEURAL_FORGE_NGX_LOG` (#51: Indiana Jones
and the Great Circle, Vulkan with Streamline, `vkQueuePresentKHR` returning `VK_TIMEOUT` about 5 s after its
swapchain, before any feature was created). `NEURAL_FORGE_NGX_TRACE=<path>` (a Windows path, or a Linux one taken
through `Z:`; `{pid}` becomes the Linux pid, without it the pid goes before the extension, and a path ending in a
separator gets `ngx-trace-<pid>.txt` in that folder) makes the wrapper log:

- its load: the environment Proton set (`NVIDIA_WINE_DLL_DIR`, `WINEDLLOVERRIDES`), how long `LoadLibraryW` of
  NVIDIA's core took, every export's resolved address (and which the core lacks), the ring, `DllMain`'s end, and
  `DLL_PROCESS_DETACH`;
- every call into any of the 75 exports: `> name(rcx, rdx, r8, r9)` on entry, `< name = result (ms)` on return, with
  the thread id and the time since load. The hooked exports log in their Rust hooks; the passthrough exports, which
  are bare jumps otherwise, jump to a per-export entry (`exports.rs`, `traced`) that puts the export's index in r11
  and goes to `win::trace_thunk`: it saves the argument registers, logs the entry, replaces the caller's return
  address with `win::trace_return` (the real one kept on a per-thread stack of 64 frames), restores the registers
  and jumps to NVIDIA's function, so every signature passes through unchanged and the stack arguments stay where the
  caller put them. `trace_return` logs the result and goes back to the kept address. A call deeper than 64 is
  logged without its exit. (Not for a function a C++ exception crosses, which NGX's C exports do not let out.)
- the text log's lines too (`log: ...`), so one file has all of it.

The file is 64 MB, mapped once at load (shared, file-backed): a line is formatted on the stack and copied into the
mapping at a position reserved with one atomic add, and a fixed 128-byte header line is rewritten with the next
position. Writing a trace line and tracing a passthrough call take no system call, no lock, no allocation and no
flush (a test counts allocations); the hooked calls' detail lines (creation, evaluate, release) are formatted on the heap
first, as for the text log, and a line holds up to 2 KB, so the longest evaluate line is kept whole. The kernel
keeps the mapping's pages when the process dies, so the lines up to a crash are in the file. When the ring is full it
wraps; the header says `wrapped` and about where the next line goes, but writers on several threads rewrite it
without ordering, so it is a hint, not authoritative. The file's order is the write order (the atomic add); a line's
time is read before its place is reserved, so sorting by the time column reorders lines of different threads. Read
it in file order: `tail -c +129 FILE | tr -d '\0'` (the ring after the 128-byte header; unwritten bytes are zero),
and once wrapped from `next` on: `{ tail -c +$((129 + NEXT)) FILE; tail -c +129 FILE | head -c NEXT; } | tr -d '\0'`.
The game's pid is in the file's name (`indy-trace.txt` is written as `indy-trace-<pid>.txt`).
NVIDIA's core reads the module that called some of its exports from their return address (every `*_Init*` and
`*_GetFeatureRequirements`, and `NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements`: `core_exports.rs`,
`READS_CALLER`) and searches that module's folder for feature DLLs. For those the trace logs the entry only (`[exit
not traced: the core reads its caller]`) and leaves the return address alone: with `trace_return` in its place, the
core would take the wrapper in `system32` for the caller and miss a feature DLL in the game's folder, so the trace
would change what NGX finds (#51's first traces were taken that way). While tracing, the return addresses of the other passthrough calls point at `trace_return`, so a crash
dump's or a debugger's stack walk through an NGX call is not reliable; the wrapper's text log line says whether the
trace file opened (its
path, or why not). A call that never returned shows
as an entry line without its exit: in a stalled present, that names the NGX call the game's thread is in.

## What the layer does with the records (`crates/layer/src/ngx_ring.rs`, `preupscale.rs`)

- **Which buffer is an evaluate's.** A command buffer belongs to an evaluate when one of its launches names (through
  the registered views' handles) the colour input of a Super Resolution or Ray Reconstruction evaluate whose end
  record has not been read: the newest such evaluate. Its later launches up to the evaluate's end are the evaluate's
  too; a launch naming the colour input after the end (DLSS Frame Generation reading the frame) is not. A launch whose
  parameters the layer cannot read is matched to nothing.
- **The identification.** The newest Super Resolution evaluate whose colour input, depth and motion vectors are images
  of the device: those images (and every image of the colour input's extent and format, and every depth image of the
  depth's extent) are watched for their layouts. Another colour image of the same shape is the same identification;
  each buffer is held with its own evaluate's images.
- **Where the holds go.** Super Resolution's buffer is the hold point: the split in front of it, or the hold inside it
  at its first launch naming the colour input. Ray Reconstruction's (by NGX's feature) is never held before the
  upscaler. The hold after DLSS goes after the evaluate's last launch naming its output: as many as the last evaluate
  that ended had between its record and its end.
- **Jitter, reset, motion vector units:** the record's `Jitter.Offset`, `Reset` and `MV.Scale` (carried to the output's
  pixels for the hold after DLSS).
- **Exposure image:** the record's `ExposureTexture`; without one the exposure is measured from the frame.
- **Pre-exposure:** where the encode uses the game's exposure as it is, the paper white is multiplied by
  `DLSS.Pre.Exposure / DLSS.Exposure.Scale` (the colour input is the scene times the pre-exposure; DLSS scales the
  exposure image's value). Encode and decode read the same paper white, so the write-back stays exact.
- **Subrect:** a game may hand DLSS larger images than it renders and say which part to read: the render size
  (`DLSS.Render.Subrect.Dimensions`) at each input's base. Unreal Engine does (its scene textures are rounded up and
  kept at the largest size asked for; Expedition 33 gave 1485x835 of a 1488x836 colour input, #51). The hold before the
  upscaler then works on that rectangle alone (`ngx_ring::Record::crop`): it copies the colour input's rectangle from
  its base, the motion vectors' and the depth's from theirs, to the corner of staging images of the stage's working
  extent (the largest rectangle held in that image so far, rounded up to 64 pixels and within the image: a 1920x1080
  rectangle of a 3840x2160 image works at 1920x1088, not on 4x the pixels), fills the rest of them with the rectangle's
  edge (last column, then last row, replicated; motion vectors and depth too), runs the network there, and writes back
  over the rectangle only, so nothing DLSS does not read is fed to the network or written. The scene-cut detector and
  the measured exposure read the rectangle alone, not the padding (a new rectangle size has no thumbnail to compare
  with). The split hold
  works on the whole image, so it refuses such frames (`Hazard::Subrect`) and the hold inside DLSS's buffer takes them.
  A rectangle outside the colour input refuses the hold; motion vectors or depth whose rectangle leaves their image are
  not read. Their rectangle is the render size where the feature was created with `MVLowRes` or they are as large as the
  colour input (Hogwarts Legacy's 2562x1441 motion vectors for a 2564x1444 colour input), else (high-resolution ones)
  the feature's output size, at their base. The hold after DLSS works on the output's rectangle (the feature's output
  size at `DLSS.Output.Subrect.Base`), with the render-size motion vectors from their base (also where the colour
  input's extent is unknown but the render size is known: none where it does not fit their image). A smaller rectangle
  inside the same image (dynamic resolution) builds nothing again, and a larger one only once it grows past the working
  extent; a new size resets the network's history (its pixels stand for other points of the screen). A new image size builds the hold's resources for it. The log says the subrect
  once per feature (`[ngx] DLSS (...) is given part of its inputs: ...`).
- **HDR:** a record with `IsHDR` clear is said once in the log (the encode has no path for a display-referred input
  yet).

A summary every 5 s logs the records read and the evaluates matched to the buffers that carry them; `shmctl status`
shows them as `ngx_records` and `ngx_matched`. Without a ring (no wrapper, `NEURAL_FORGE_NGX_RING=off`) nothing is held
before or after DLSS: as soon as DLSS Super Resolution's or Ray Reconstruction's kernels launch without one, the log
and the status say the game is not on the Neural Forge Proton (or, when `WINEDLLOVERRIDES` shows the wrapper was asked
for, that it is on one but no ring reached the layer, with the ring's path), and the model stays off while they launch;
it runs on the finished frame only where the placement puts it there. Frame Generation's launches are no sign. The
layer looks for the ring from its first device's creation on (a thread, every 500 ms for 3 minutes, then every 5 s)
and removes the file once mapped, so nothing else can take a live game's ring.

## The Proton copy (`crates/supervisor/src/proton_tool.rs`, `neural-forge-cli proton-tool`, the Setup tab)

Nothing of Proton is rebuilt. The app copies an installed Proton (GE-Proton, Proton-CachyOS, Valve's Proton or Proton
Experimental) into Steam's `compatibilitytools.d` as `proton_copy::copy_name(<version>)`: the source's version name
made safe for a folder, with `Proton-` before it when it does not say "proton", and `-NeuralForge`
(`GE-Proton11-12-NeuralForge`, `Proton-experimental-11.0-20261105c-x86_64-NeuralForge`; `Proton-NeuralForge` for a
version with nothing left once made safe). Steam's client maps the Windows folders Steam Cloud saves live in (AppData,
Saved Games, Documents) into the game's prefix only for a compatibility tool whose name contains "proton" (any case); for
any other it logs "Unable to resolve path with root WinAppDataLocal" and syncs none of those saves (#71), so every copy's
name says Proton (`proton_copy::cloud_name`). It is made with hard links where the source is on the same
filesystem (no space), and writes three files new: `proton` with the two overrides after Valve's `NVIDIA_WINE_DLL_DIR`
block (refused unless that block is there exactly once), `compatibilitytool.vdf` with the new name, and
`files/lib/wine/x86_64-windows/_nvngx.dll`. The `version` file stays the source's, so a game's prefix is not upgraded
when it switches between the two. `PROTON_NO_NEURAL_FORGE_NGX=1` in a game's launch options turns the wrapper off for
it. The copy is built as `.<name>.tmp` and swapped into place in one step (`renameat2(RENAME_EXCHANGE)`); a copy a
running game uses (a process whose executable or arguments lie in it) is neither replaced nor removed.

- **Wine 10 or newer only.** The wrapper names its ring by the game's Linux pid, which it asks Wine for
  (`ProcessWineUnixPid`, class 1101); Wine 9 does not answer (Proton 9.0's `ntdll.so` never compares with 1101).
  A source's Wine version is read from the `wine-<major>.<minor>` string in its `ntdll.so`; a Proton older than 10, or
  one whose version cannot be read, is refused, and the version is kept in the marker (`wine=`). `doctor` reports a
  copy on older Wine as a failure.
- **Which Proton is copied** when a game is switched on: the one Steam runs it with (its `CompatToolMapping` entry,
  else the default entry `0`), found by the name Steam uses: Valve's Protons through their app's manifest
  (`proton_9` is app 2805730, installed in `Proton 9.0 (Beta)`), other tools by the key of their
  `compatibilitytool.vdf`. A chosen Proton that is not installed is refused, never replaced by another; with nothing
  chosen, Proton Experimental. A native Linux game (its installed launch executable is the Linux one) with no Proton
  chosen is left without Proton. The game's earlier choice is recorded (`config.vdf.neural-forge.compat`) before
  `config.vdf` is edited; a `config.vdf` without a `CompatToolMapping` block gets one.
- **Updates.** A copy is made again (`proton_tool::remake`, the one check behind switching a game on, Make again,
  `proton-tool update`, `doctor`, the Status tab's **Proton copies out of date** group and the Setup tab) for one of two
  reasons. `Remake::Source`: its source changed (a new GE-Proton in a "Latest" folder, a Proton Experimental update;
  `proton_copy::newer_source`, in `crates/protocol/src/proton_copy.rs`: the copy's marker names its source and the
  source's version then, the source's version now is its `version` file's second word). `Remake::Name`: its name lacks
  "Proton" (made by Neural Forge 4.2.0 or older from Proton Experimental, Proton Hotfix or Proton-CachyOS), whether or
  not its source is still there. Whether the source is there is one test too, `proton_tool::source_of` (the folder the
  marker names, read directly whatever library holds it, listed or not). `proton-tool update` (or Make again) then makes
  a copy of the source's current version; with Steam closed, every game set to the old copy (the default entry `0`
  too) is moved to the new one in one edit of `config.vdf` and the old copy is removed. A new version or a name that
  now says Proton is a new name (the games' `CompatToolMapping` moves with it); the same name (a wrapper-only update)
  is made again in place. With Steam running, the new copy is made and the old one kept, nothing moved; the next update
  with Steam closed finishes. A copy named without Proton whose source is gone is renamed in place instead
  (`proton_tool::rename_in`: it holds its own files): its `compatibilitytool.vdf` written new with `copy_name` of its
  own version first, then its folder renamed (a stop in between leaves a folder Make again renames; a manifest that
  cannot be written changes nothing), its games moved (folder and manifest put back exactly if they cannot be) and
  its wrapper renewed (a wrapper that cannot be renewed after the move is a note, Make again offers it again); refused
  while Steam runs, while a game runs with it, and when another tool has the new name (a Neural Forge copy of the same
  version there, made from the same source path or another, takes its games instead and the old copy goes:
  `proton_tool::rename_plan`, which `describe()` and `doctor` read too). Games a stopped rename left on the old name
  (no folder of that name, the renamed copy there: `compat_tool::orphaned_mappings`) are moved to the renamed copy by
  `proton-tool update`, by Make again and by switching any game on, and `doctor` reports them. Switching a game on that
  is set to a copy to be made again does the same for the copy; when the rename is refused (a game runs with the copy,
  another tool has the name) the game stays on its copy with its wrapper renewed and a note says the rename is pending. `compat_tool::stale_copies` adds the games
  set to each copy for the Status and Setup tabs; the Setup tab offers no **Make copy** for a source that has a copy to
  be made again (its Make again is the way). `doctor` names the copy Make again will give (`proton_tool::remade_name`).
  At launch only the layer runs (the launch options are environment variables), so it reads Steam's
  `STEAM_COMPAT_TOOL_PATHS` (the compat tool folders the game runs with), and for a copy among them logs once per
  process (`proton_notice.rs`, `proton_copy::stale_notice`) one of two lines: for a source with another version than
  the marker names, `[proton] this game runs with <copy>, made from <old> but <source> has <new> now: close Steam and
  use Make again ... to renew the copy (the new one is named <name>)`, which also says that Steam Cloud does not sync the
  saves in the prefix when the name lacks Proton; for a name without Proton alone, `[proton] this game runs with
  <copy>, whose name lacks "Proton": Steam Cloud does not sync the saves the game keeps in its Wine prefix with it;
  close Steam and use Make again ...: the copy is named <name> then and the game moves to it`. A source that is
  unreachable from the game's container or gone gives no first line. When Neural Forge itself is
  installed or updated (the app at start, `install --appdir`), each copy whose marker names another wrapper
  (`wrapper_sha256=`) gets the installed one; `doctor` warns about a copy that still holds another. A copy whose
  source folder is gone (ProtonUp replaced GE-Proton11-12 with 11-13) and whose name says Proton is not outdated: it
  holds its own files and keeps working, switching a game on and `update` only renew its wrapper, and `doctor` notes it.
- **Purge.** `uninstall --purge` is refused before it changes anything while a game runs with a copy (the games are
  named). Otherwise it puts back each recorded game's Proton, takes any other game still set to a copy (a marked one,
  or a `-NeuralForge` name whose folder is gone; never another tool with that suffix) off it, then removes only marked
  copies (and what a stopped copy left, `.<name>.tmp`).

- **What `doctor` checks in a copy** (`proton_tool::check_chain`, finding `proton-chain`, shown with the other
  findings in the Status tab's Diagnosis group). The trick depends on Wine's DLL load order, so each copy's chain is
  checked statically, reading files only: its wrapper (`files/lib/wine/x86_64-windows/_nvngx.dll`) is there, is a
  Wine builtin (the 16 bytes at 0x40) and is the file its marker records (and whether that is this Neural Forge's);
  its `proton` script sets `_nvngx=b` (a failure without it: games get NVIDIA's core), the copy of NVIDIA's core's
  full path native and the driver's core's full path native (warnings: the wrapper then falls back or fails); and its
  Wine's major version is one whose `get_load_order` (`dlls/ntdll/unix/loadorder.c`) was read
  (`VERIFIED_WINE_MAJORS`: 10 and 11, see Evidence; a newer one is a note). A wrapper file that is not the one its
  marker records is put back when Neural Forge starts (`refresh_wrapper` compares the file too). Nothing runs: a run
  of a test program through the copy's Wine that loads the full path and says which module it got would need a Wine
  prefix, which took 6.6 s and 640 MB to make with GE-Proton11-12 on the test machine (0.9 s per run once made),
  for each copy, and a prefix kept for it would go stale with the copy. The wrapper checks at run time that the core
  it loaded is not itself, and its NGX versions record says which core each game ran.
- **Hard links.** A copy shares most files with its source (same inode), so Steam updating the source in place changes
  the copy's files too. The marker records the shared files at the time (`linked=` and one `link_file=` line each:
  relative path, size, modification time; no device or inode numbers, no hashing); `doctor` warns when one of those
  files' own size or time changed while the source's version is unchanged (the Proton was rewritten in place), with
  Make again as the fix. A source that replaced a file (the copy then keeps the file it was made with) or was removed
  is no change and says nothing. A copy made before this was recorded, and a copy made across filesystems (full copies, `linked=0`), is not
  checked and says nothing.

**NGX updater (`nvngx_update.exe`).** Since driver R620 NVIDIA's `nvidia/wine` folder also ships `nvngx_update.exe`,
NGX's over-the-air updater, and Proton (e3cff01d, PR #10340) and proton-cachyos now copy it into the prefix's
`system32` next to `_nvngx.dll` and `nvngx.dll`. Copying does not run it: NVIDIA's core starts the updater under Wine
only when `PROTON_ENABLE_NGX_UPDATER=1` is set, which Proton does not do. Tested on 620.70.06 with Cyberpunk 2077 in a
Neural Forge copy of patched Proton Experimental: the updater never ran, `ProgramData/NVIDIA/NGX/models` stayed empty,
and no DLSS file in the prefix or the game folder changed. The wrapper works the same either way. A game that sets
`PROTON_ENABLE_NGX_UPDATER=1` lets NGX download newer snippets into `ProgramData/NVIDIA/NGX/models`. Neural Forge does
not depend on which DLSS snippet versions NGX loads.

Proton links many of a prefix's `system32` files to the Proton that last ran it; a removed copy's prefixes are set up
again by the next Proton a game runs with.

Why a local copy and no published Proton: the wrapper is Neural Forge's own code (not derived from Wine; it links only
the mingw-w64 runtime and libgcc, whose notices ship in the AppImage), so the copy made on the user's machine conveys no
LGPL code. Publishing ready-made GE-Proton tarballs would make Neural Forge a distributor of Wine, vkd3d-proton, FFmpeg
and the rest, owing their complete corresponding source (LGPL-2.1 section 6; GE's "Source code" archives lack the
submodules). The `proton` script is BSD-3-Clause (Valve); its notice stays in the copy.

## After a driver update (`neural-forge-cli ngx-audit`, `crates/supervisor/src/ngx_audit.rs`)

The wrapper is built for one driver's core. `crates/ngx-wrapper/src/core_exports.rs` holds what it assumes of it: the
75 exported names in ordinal order (`CORE_EXPORTS`: NVIDIA numbers them alphabetically from 1, and so does the
linker), the ten the wrapper hooks (`HOOKED`), the 20 whose code reads its caller (`READS_CALLER`), and the driver
and core file they were checked on (`CHECKED_DRIVER`, `CHECKED_CORE_SHA256`). The Linux crates compile the same file
(`neural_forge_supervisor::ngx_core_exports`, a `#[path]` module), so `doctor`, the Proton copies and the audit read
the wrapper's own lists; the wrapper's test checks that its `passthrough!` names and `HOOKED` together are
`CORE_EXPORTS`.

`neural-forge-cli ngx-audit [CORE] [--driver-dir DIR]` checks a core (by default the driver's
`nvidia/wine/_nvngx.dll`) against those lists, prints one `OK` or `FAIL` line per check and exits 1 on any `FAIL`.
Run it after every driver update; the maintainer's daily machine check runs the installed one against the installed
driver. It checks:

- **The exports**: a name the core has and the wrapper does not export (a game that looks it up does not find it), one
  the core no longer has (hooked: the wrapper refuses to load; passthrough: answered with NGX's failure code), and an
  ordinal that moved.
- **Which exports read their caller**, from the core's code (`ngx_audit/scan.rs`). An export reads its caller when the
  return address it was called with (`[rsp]` at its entry) reaches the address argument (`rdx`) of
  `GetModuleHandleExA` or `GetModuleHandleExW`, called through the import table: in the export's own code, in code it
  jumps to, or through up to four direct calls that pass it on as an argument (the `*_GetFeatureRequirements` exports
  hand it to a helper). The scan decodes the code with iced-x86 and follows registers and the function's frame slots
  (pushes, `sub rsp`, `__chkstk`'s `sub rsp, rax`, moves, `lea` of frame addresses) along every path, keeping a
  value's origin where paths meet; it forgets a value at any instruction it does not model and follows nothing
  through the heap or a computed jump. A hooked export that now reads its caller, a passthrough one that newly does,
  and a listed one that no longer does are all failures, and so is the return address reaching any other import's
  argument: that would be a new way of finding the caller. On 620.73.04 it finds exactly Ghidra's 20, at any call
  depth from 1 to 8 (at 0 it misses the five `GetFeatureRequirements`); its test repeats that wherever the file with
  `CHECKED_CORE_SHA256` is installed, and the other tests run it on small assembled functions and a hand-made PE.
- **The feature DLLs' caller check**: `nvngx_dlssg.dll` and `nvngx_dlssd.dll` (in DIR, by default the core's folder)
  still hold "Not called from NGX runtime", the message with which they refuse a caller whose path lacks lower-case
  `nvngx.dll` (see Evidence). Gone, their check changed, and what the wrapper does about the core's name needs
  another look.

A failure means the wrapper needs updating before games run on that driver: open the core (or the feature DLL) in
Ghidra at the exports the report names, confirm what changed, update `core_exports.rs` (and the `passthrough!` list
in `exports.rs` for an added or removed export), set `CHECKED_DRIVER` and `CHECKED_CORE_SHA256` to the new core, and
run the audit again. A path the scan cannot follow shows as "no longer reads its caller": confirm it in Ghidra
before changing the list.

## Maintenance

- GE-Proton releases about three times a month (GE-Proton11-1 2026-06-24 to 11-12 2026-11-03: twelve). A copy is made
  again in seconds. The wrapper needs Wine 10 or newer (`ProcessWineUnixPid`), nothing else of the Wine version.
- What can break: Valve rewrites the `NVIDIA_WINE_DLL_DIR` block (its TODO says the copies into `system32` go once
  NGXCore's `NGXPath` is set): the patch then refuses, and the anchor needs updating. `scripts/check_proton_scripts.py` (weekly, `proton-script-check.yml`) runs
  the patch on the newest Proton Experimental and proton-cachyos scripts and keeps one standing issue open while it does not apply. NVIDIA adds an export to its core:
  the wrapper lacks it and NGX's loader may refuse the core, so the game runs without DLSS. NVIDIA's core lacks one of
  the wrapper's exports (an older driver): a passthrough one answers `NVSDK_NGX_Result_Fail`, a hooked one fails the
  wrapper's load, so the game runs without DLSS (see "Exports the core lacks"). `build-ngx-wrapper.sh` compares the
  wrapper's exports (names and ordinals) with the driver's DLL wherever one is installed, `ngx-audit` checks a new
  driver's core against every assumption the wrapper makes of it (below), and `doctor` compares the exports
  with the user's driver both ways: a core export the wrapper does not forward, or one the wrapper hooks that the core
  lacks, is a failure; a forwarded export the core lacks is a warning. NGX starts verifying the loaded image instead of the file: the
  wrapper approach stops working (the fallback would be the in-memory export patch spike 1 used, from a builtin every
  game loads).

## Evidence

- Wine source: ValveSoftware/wine at `fda79a585939fb7512dc220a11249f9ffe825fcf` (GE-Proton11-12's submodule).
- DLL load order (2026-11-19, `dlls/ntdll/unix/loadorder.c` at gitlab.winehq.org wine/wine `wine-10.0`, `wine-11.0`,
  `wine-11.22` and master, and ValveSoftware/wine `fda79a5`): `get_load_order` takes `\??\` and `.dll` off the name;
  a module directly in `system32` is looked up by its base name, any other (a folder below `system32` too: 10.0 and
  11.0 strip the system folder only when no separator follows, 11.22's `is_system_dir_path` in `loader.c` returns
  false for one) first by its full path, then by `*` and its base name, then by its base name; each in
  `WINEDLLOVERRIDES` before the registry. 11.22 adds heuristics from the version resource (2026-03 and 04,
  `a31ec8da`, `8cf7a53e`) after those lookups, for a module none of them names; Valve's tree adds only Easy
  Anti-Cheat's bridge. So the copy's three keys mean the same in all of them.
- A scratch prefix under the copy: `WinVerifyTrust(C:\windows\system32\_nvngx.dll)` = 0, `LoadLibrary` of that path
  returns the wrapper (its `EvaluateFeature` in the wrapper's module), NVIDIA's core loaded from
  `Z:\run\host\usr\lib\x86_64-linux-gnu\nvidia\wine\_nvngx.dll`, no export missing.
- The wrapper costs nothing measurable (2026-11-10, RTX 5070, driver 620.70, 4K, the 3.14 layer, wrapper vs the same
  Proton without it): Cyberpunk 2077 benchmark with the layer off 120.2 vs 120.6 fps, on 69.3 vs 69.0; GTA V Enhanced
  fg4x-4k real 36.3 / 35.7 vs 36.1 / 35.9; Crimson Desert (Ray Reconstruction, HDR) within its scene noise.
- The layer from the records alone (2026-11-11, 4K, one run each): Cyberpunk 2077 RR off held before the upscaler,
  benchmark 90.9 fps; RR on held after DLSS, 58.3 fps; Crimson Desert RR, HDR, after DLSS ~63 fps; Red Dead
  Redemption 2 (Vulkan NGX, launched through Steam with the game switched on by `neural-forge-cli enable`, which set its
  Proton and put it back on `disable`) held before the upscaler, 71 fps; GTA San Andreas - The Definitive Edition
  (D3D11 through DXVK) held before the upscaler, ~50 fps. Crimson Desert with DLSS off in its own settings: the model on
  the finished frame (13.5 fps, as before); with DLSS on and no wrapper: the model off, 104 fps, the log says why.
- <a id="where-the-core-looks-for-feature-dlls"></a>**Where the core looks for feature DLLs, and its name**
  (2026-11-19, driver 620.73.04's `_nvngx.dll`, sha256 `637a9390…892a`, Ghidra through REA, NGX's own log and the
  wrapper's trace of Cyberpunk 2077 through the game sweep). The folders NGX searched, in order: the path the game's
  NGX library hands in, the folder of the module that called the export (`sl.common.dll`'s: the core finds it from
  the return address through `GetModuleHandleExA(FROM_ADDRESS)` and `GetModuleFileNameW` in every `*_Init*` and
  `*_GetFeatureRequirements` export), the "driver store", which under Wine is `NVIDIA_WINE_DLL_DIR` as Linux gives it
  (`/run/host/usr/lib/x86_64-linux-gnu/nvidia/wine`, read in `NGXGetPathUsingQAI` after the adapter query fails;
  Wine opens the Linux path, and NGX loaded the driver's `nvngx_dlssg.dll` and `nvngx_dlssd.dll` from it besides the
  game's), and `ngx-builtins` under that. So the driver's feature DLLs are offered whichever folder the core runs
  from. One more place depends on that folder: when the game's NGX library sets a flag in the `InternalData` it
  passes, `NGXInitContext` adds the core's own folder (its error: "Failed to get NGX core path. Cannot load generic
  snippet from here."). Cyberpunk's library does not set it; which games do, and what NGX takes from there, is not
  read out. The core's file name matters too: Cyberpunk with the core loaded as `system32\neural-forge\_nvngx_core.dll`
  made the same NGX calls with the same results up to `GetCapabilityParameters` (trace and NGX log identical), and
  then never called `UpdateFeature` or `CreateFeature` (no DLSS, 2 of 2 runs); as `C:\nfc\_nvngx.dll` and as
  `system32\neural-forge\_nvngx.dll` it ran DLSS and Frame Generation as with the driver's file. That is #51's "DLSS
  never starts" with `C:\nfcore\nvngx_core.dll`. Why (Ghidra, the feature DLLs 310.4 to 310.9 and the driver's
  copies): every NGX export of `nvngx_dlss.dll`, `nvngx_dlssd.dll` and `nvngx_dlssg.dll` looks its caller's module up
  from the return address and refuses one whose path lacks `nvngx.dll` (case-sensitive `wcsstr`) with `0xBAD00002`,
  "Error: Not called from NGX runtime - %S". The game-side loader (`sl.common.dll`) compares no name after the load.
  The 20 exports `READS_CALLER` names are exactly 620.73's that read their caller (Ghidra; `ngx-audit`'s scan finds the
  same 20 from the core's code). In #51 (Indiana Jones and the Great Circle: Vulkan, Streamline with
  DLSS, Ray Reconstruction and Frame Generation) the core loaded from the driver's folder reported a third feature
  supported, `UpdateFeature` ran for each feature, and the game's present timed out (`VK_TIMEOUT`) about 5 s after its
  swapchain. Whether the core's folder alone (the stock layout the wrapper now uses) avoids that is for the reporter
  to try. The over-the-air updater does not depend on the path: it is decided by `__NGX_DISABLE_UPDATER` and
  `PROTON_ENABLE_NGX_UPDATER` (`NGXIsUpdaterEnabledBase`). NGX's own log: with `__NGX_LOG_LEVEL=2` in the launch
  options the core logs to `C:\ProgramData\NVIDIA\NGX\models\nvngx_update.log` in the game's prefix (rewritten each
  start: the folders searched, each DLL loaded and why a feature is refused); `__NGX_ENABLE_OVERRIDE_LOG_PATH=1
  __NGX_LOG_PATH_OVERRIDE=<folder>` moves only the game-side library's `nvsdk_ngx.log`.
