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
                              │ forwards all 75 exports to NVIDIA's core, loaded from NVIDIA_WINE_DLL_DIR
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
- **Loading NVIDIA's core.** The wrapper loads `Z:<NVIDIA_WINE_DLL_DIR>/_nvngx.dll` by its full path. A full-path key
  is looked up before the base name, so the override `Z:\...\nvidia\wine\_nvngx=n` keeps that one native; without it
  Wine hands the wrapper back to itself (the wrapper checks: a load that returns its own module refuses the load).
  NVIDIA's snippets (`nvngx_dlss.dll` and the others) load as before.
- **Exports the core lacks.** The wrapper hooks ten exports: `NVSDK_NGX_{D3D11,D3D12,VULKAN}_{CreateFeature,
  EvaluateFeature,ReleaseFeature}` and `NVSDK_NGX_VULKAN_CreateFeature1` (`exports.rs`, `HOOKED`). It calls NVIDIA's
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
  so it cannot be the wrapper again (checked); and the wrapper's `DllMain` waits on no other thread.
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
- **The NGX versions.** After a feature is created (`modules.rs`, `note_loaded`) the wrapper lists the process's
  modules (`K32EnumProcessModules`, which Wine serves), keeps `_nvngx.dll` and `nvngx*.dll` (itself, NVIDIA's core, the
  snippets `nvngx_dlss.dll`, `nvngx_dlssg.dll`, `nvngx_dlssd.dll`, `nvngx_dlssnr.dll`, ...), whatever path each was
  loaded from, and reads each file's `VS_FIXEDFILEINFO` version through `version.dll` (loaded when needed, not
  imported). It logs them when the text log is on and, in a game the layer is on in, writes
  `$XDG_STATE_HOME/neural-forge/ngx-modules/<app id>-<exe>.txt` (the format is in `modules.rs`; a run replaces the
  file, and it is rewritten only when the set of paths changes). `<app id>` is Steam's `SteamAppId` (else
  `SteamGameId`), else `x` and a hash of the executable's path, so games whose executables share a name keep apart.
  Wine does not pass `HOME` into the game's Windows environment (it gives `WINEHOMEDIR`, `\??\Z:\home\<user>`), so
  without `XDG_STATE_HOME` the wrapper takes the home from there, the same directory the supervisor uses. It writes through `Z:` for a `Z:` home, and through the NT path (`\\?\unix\home\<user>\.local\state\...`, or the other drive's) for a `\??\unix\...` or other-drive `WINEHOMEDIR`, so it needs no `Z:` drive. `doctor`
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
through `Z:`; `{pid}` becomes the Linux pid) makes the wrapper log:

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
position. No system call, no lock, no flush on any call path; the kernel keeps the mapping's pages when the process
dies, so the lines up to a crash are in the file. When the ring is full it wraps (the header says `wrapped` and
where the next line goes); `tr -d '\0' < FILE | sort -n` orders the lines by time. A call that never returned shows
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
Experimental) into Steam's `compatibilitytools.d` as `<version>-NeuralForge`, with hard links where it is on the same
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
- **Updates.** When the source changes (a new GE-Proton in a "Latest" folder, a Proton Experimental update), the copy
  is listed as outdated, and `proton-tool update` (or Make again) makes a copy of the new version; with Steam closed,
  every game set to the old copy is moved to the new one and the old copy is removed. The copy's name holds the
  source's version name, so a new version is a new name (the games' `CompatToolMapping` moves with it); the same
  name (a wrapper-only update) is made again in place. **One check, four places:** `proton_copy::newer_source`
  (`crates/protocol/src/proton_copy.rs`: the copy's marker names its source and the source's version then; the
  source's version now is its `version` file's second word) is behind `proton_tool::source_changed`, `doctor`, the
  Status tab's **Proton copies out of date** group and Setup's Make again (`compat_tool::stale_copies` adds the
  games set to each copy), and the layer's launch notice. At launch only the layer runs (the launch options are
  environment variables), so it reads Steam's `STEAM_COMPAT_TOOL_PATHS` (the compat tool folders the game runs with),
  and for a copy among them whose source has another version than its marker names, logs once per process
  (`proton_notice.rs`): `[proton] this game runs with <copy>, made from <old> but <source> has <new> now: close Steam
  and use Make again ...`. Nothing is logged when the source is unreachable from the game's container or gone. When Neural Forge itself is
  installed or updated (the app at start, `install --appdir`), each copy whose marker names another wrapper
  (`wrapper_sha256=`) gets the installed one; `doctor` warns about a copy that still holds another. A copy whose
  source folder is gone (ProtonUp replaced GE-Proton11-12 with 11-13) is not outdated: it holds its own files and keeps
  working, switching a game on and `update` only renew its wrapper, and `doctor` notes it.
- **Purge.** `uninstall --purge` is refused before it changes anything while a game runs with a copy (the games are
  named). Otherwise it puts back each recorded game's Proton, takes any other game still set to a copy (a marked one,
  or a `-NeuralForge` name whose folder is gone; never another tool with that suffix) off it, then removes only marked
  copies (and what a stopped copy left, `.<name>.tmp`).

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

## Maintenance

- GE-Proton releases about three times a month (GE-Proton11-1 2026-06-24 to 11-12 2026-11-03: twelve). A copy is made
  again in seconds. The wrapper needs Wine 10 or newer (`ProcessWineUnixPid`), nothing else of the Wine version.
- What can break: Valve rewrites the `NVIDIA_WINE_DLL_DIR` block (its TODO says the copies into `system32` go once
  NGXCore's `NGXPath` is set): the patch then refuses, and the anchor needs updating. `scripts/check_proton_scripts.py` (weekly, `proton-script-check.yml`) runs
  the patch on the newest Proton Experimental and proton-cachyos scripts and keeps one standing issue open while it does not apply. NVIDIA adds an export to its core:
  the wrapper lacks it and NGX's loader may refuse the core, so the game runs without DLSS. NVIDIA's core lacks one of
  the wrapper's exports (an older driver): a passthrough one answers `NVSDK_NGX_Result_Fail`, a hooked one fails the
  wrapper's load, so the game runs without DLSS (see "Exports the core lacks"). `build-ngx-wrapper.sh` compares the
  wrapper's exports (names and ordinals) with the driver's DLL wherever one is installed, and `doctor` compares them
  with the user's driver both ways: a core export the wrapper does not forward, or one the wrapper hooks that the core
  lacks, is a failure; a forwarded export the core lacks is a warning. NGX starts verifying the loaded image instead of the file: the
  wrapper approach stops working (the fallback would be the in-memory export patch spike 1 used, from a builtin every
  game loads).

## Evidence

- Wine source: ValveSoftware/wine at `fda79a585939fb7512dc220a11249f9ffe825fcf` (GE-Proton11-12's submodule).
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
- Where NVIDIA's core loads from does not change which DLSS files it picks (2026-11-12, driver 620.70's
  `_nvngx.dll`, Ghidra): its "driver store" path under Wine is `NVIDIA_WINE_DLL_DIR` read from the environment
  (`NGXGetPathUsingQAI`, after the adapter query fails), not its own module's folder, so the driver's
  `nvngx_dlssg.dll`/`nvngx_dlssd.dll` are found the same way with and without the wrapper; and its over-the-air updater
  is decided by `__NGX_DISABLE_UPDATER` and `PROTON_ENABLE_NGX_UPDATER` (`NGXIsUpdaterEnabledBase`), not by its path.
  The Indiana Jones stall in #51 is therefore not the core's load path; the call trace is there to find what it is.
