# Native Linux NGX helper — investigation and current status

**Status: not implementable today, for a data-availability reason, not an architecture
one.** This document exists so a future session doesn't re-derive (or re-spend real
GPU time re-discovering) what this one already found. Read this before spending time
on "skip Wine entirely" as an architecture direction.

## What this was meant to be

`DMABUF_TRANSPORT_DESIGN.md` closed out Phase 4 (DMA-BUF transport) by naming a
longer-term idea as the real fix for the underlying asymmetry both DMA-BUF directions
ran into: a **native Linux NGX helper** — the layer (or a sibling native-Linux process)
calling NVIDIA's NGX API directly, with no Wine, no Windows PE binary, no cross-process
handle/fd bridging at all. If that were possible, Phase 4's whole DMA-BUF problem
disappears (same-process or same-OS memory sharing, no win32-handle asymmetry), and the
helper's entire Wine dependency goes away.

This session tested whether that's actually possible, rather than continuing to treat
it as an assumed future direction.

## What this session found: NVIDIA does ship a real native Linux NGX runtime

`lordnikon`'s driver package (615.71.09) includes a genuine native Linux ELF library at
`/usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.1`, exporting the same `NVSDK_NGX_VULKAN_*`
C ABI as Windows' `nvngx.dll` (confirmed via `nm -D`: `Init_Ext`, `CreateFeature`,
`EvaluateFeature`, `AllocateParameters`, etc., all present).

This isn't incidental — NVIDIA's own public SDK repository
(`github.com/NVIDIA/DLSS`, fetched this session via `gh api`) confirms it's a real,
documented, cross-platform component:

- `include/nvsdk_ngx_loader.h` defines `NGX_CORE_LIBRARY_NAME "libnvidia-ngx.so.1"` for
  the non-Windows branch and loads it with a plain `dlopen` — the exact file found
  installed on `lordnikon`.
- `lib/Linux_x86_64/rel/` ships real, redistributable per-feature snippet `.so`s for
  every *officially released* NGX feature: `libnvidia-ngx-dlss.so` (Super Resolution),
  `libnvidia-ngx-dlssd.so` (Ray Reconstruction), `libnvidia-ngx-dlssg.so` (Frame
  Generation).
- `/usr/share/doc/NVIDIA_GLX-1.0/html/ngx.html` (the driver's own shipped
  documentation, read this session) confirms this is a real, first-party-supported
  three-part system: **NGX Core** (the interface apps use), **NGX Updater** (downloads
  feature models into a configurable `ngx_models_path`, default
  `/usr/share/nvidia/ngx`), and **NGX Core for Proton** ("a copy of NGX Core built as a
  DLL for use with Proton" — this is what
  `/usr/lib/x86_64-linux-gnu/nvidia/wine/{nvngx,_nvngx}.dll` actually are, not
  incidental leftovers). The same doc also names `__NV_SIGNED_LOAD_CHECK=none` as a
  real, documented environment variable that disables Core's feature-signature check.

None of this was assumed — it's what NVIDIA itself ships and documents for exactly this
class of problem (Linux/Proton NGX consumption), independent of anything this project
has built.

## What this session actually tried, on real hardware

`crates/layer/examples/native_ngx_probe.rs` (new, native Linux, no Wine): `dlopen`s
`libnvidia-ngx.so.1`, creates a real Vulkan instance/device (`ash`, same pattern as
every other probe in this repo), and calls the real NGX bootstrap sequence directly —
`NVSDK_NGX_VULKAN_Init_Ext`, `AllocateParameters`, `CreateFeature`. Every call is
wrapped in a native signal-based guard (`sigsetjmp`/`siglongjmp` around
`SIGSEGV`/`SIGBUS`/`SIGILL`/`SIGFPE`, this project's Linux-native counterpart to
`crates/helper/src/guard.rs`'s Windows VEH+`setjmp`), since this calls a real, stripped,
undocumented-for-this-exact-purpose proprietary library with a partly-guessed argument
sequence, the same discipline as every Windows-side NGX call in this project. One
concrete, easy-to-get-wrong ABI detail handled deliberately: NVIDIA's own header
declares the app-data-path parameter as `const wchar_t*`, and `wchar_t` is **4 bytes on
Linux** (UTF-32) vs. **2 bytes on Windows** (UTF-16) — the existing Windows helper's
UTF-16 encoding is not portable to this call as-is.

**Result on real hardware** (`lordnikon`, RTX 5070, driver 615.71.09):

```
NVSDK_NGX_VULKAN_Init_Ext            -> 0x1        (SUCCESS, no fault)
NVSDK_NGX_VULKAN_AllocateParameters  -> 0x1        (SUCCESS, no fault)
NVSDK_NGX_VULKAN_CreateFeature(18)   -> 0xbad0000b (FAIL_UNABLE_TO_INITIALIZE_FEATURE, no fault)
```

Two things worth calling out about this before the negative result:

- **`Init_Ext` and `AllocateParameters` both succeeded cleanly, with *no* identity
  spoofing at all.** This is genuinely better than the Windows path, where
  `AllocateParameters` returns `FAIL_PLATFORM_ERROR` (`0xbad00002`) without the
  caller-identity spoof from `crates/helper/src/spoof.rs` installed first (see
  `ngx.rs`'s own doc comments). Whatever caller-identity mechanism Core uses on
  Windows, its native Linux build either doesn't apply the same check to this call
  path, or this probe's caller identity (a plain native ELF executable, not something
  masquerading as anything) already satisfies it. Either way: the native bootstrap
  itself is not the blocker.
- `CreateFeature(18)`'s result, `0xbad0000b`, is the *exact* code
  `crates/helper/src/abi.rs`'s own long-standing comment already names: "Core's
  `CreateFeature(18)` without going through the signed-snippet route." Getting the
  identical, well-known failure signature from the native library is itself a form of
  confirmation that this probe's calling sequence is basically right, not garbled.

**Control experiment**, run in the same session: the identical `CreateFeature` call
against `NVSDK_NGX_Feature_SuperSampling` (= 1, the genuinely public, officially
released "DLSS" feature) returned the **identical** `0xbad0000b`. This was expected
once `/usr/share/nvidia/ngx` (the default `ngx_models_path`) was confirmed to not exist
at all on this machine (`ls` — no such directory) and no per-feature `.so` was found
anywhere on the filesystem in the earlier DMA-BUF-session's full search either: this
machine has never had *any* NGX feature snippet installed, so Core can't dispatch to
one for *any* feature ID, reserved or not. This result alone doesn't distinguish
"Feature 18 is unimplementable on Linux" from "this box just never installed any NGX
snippet" — see the follow-up experiment below for that.

**Follow-up experiment**: downloaded NVIDIA's own real, official, redistributable
`libnvidia-ngx-dlss.so.310.9.1` (Super Resolution, from the public
`github.com/NVIDIA/DLSS` repo — not anything proprietary-but-unreleased, ordinary use
of NVIDIA's own published SDK) and pointed Core at a user-owned copy of it via
`__NGX_CONF_FILE` (a real, documented config-search mechanism) with `ngx_models_path`
set to a scratch directory holding the file — deliberately not writing to the
system-wide `/usr/share/nvidia/ngx` default, since that's a shared system path this
session didn't have standing permission to modify. Re-running `CreateFeature(1)` against
this configuration produced the **same** `0xbad0000b` — dropping the raw `.so` file
into a configured models path was not, by itself, enough to make Core discover and load
it. This is an honest, real negative result, not a dead end papered over: it means
Core's real feature-loading mechanism needs something more than "the right bytes in the
right directory" (likely version-specific subdirectory naming, or bookkeeping the real
NGX Updater writes that this session didn't replicate) — genuinely unresolved, and out
of scope to chase further without a clearer reason to (see Recommendation).

## What this means

1. **The native bootstrap works.** `libnvidia-ngx.so.1`'s `Init_Ext`/
   `AllocateParameters` succeed cleanly on real hardware, from a real native Linux
   process, no Wine involved, no caller-identity workaround needed. If this project
   ever had a genuine native Linux payload for its target feature, wiring it up through
   this native Core would very plausibly work — the mechanism itself is real and
   functional, not a dead end.
2. **There is no native Linux payload for Feature 18 anywhere.** NVIDIA's own current
   public header (`nvsdk_ngx_defs.h`) lists it as `NVSDK_NGX_Feature_Reserved18 = 18` —
   explicitly reserved and unallocated, with no shipped implementation on *any*
   platform's public SDK. The only place a working implementation of this feature has
   ever been observed to exist at all, across every machine and archive this project's
   sessions have searched, is the leaked/privately-distributed Windows PE
   `nvngx_dlssnr.dll` this project already depends on. No `.so` counterpart of it exists
   anywhere this project has looked.
3. **Even loading an officially released feature's real snippet natively isn't a solved
   problem for this project yet** — the models-path/`__NGX_CONF_FILE` follow-up
   experiment shows there's real, undocumented (from what this session found)
   bookkeeping Core needs beyond "the file exists," that a real NGX Updater run
   presumably produces and this session didn't reverse-engineer.

Put together: a "native Linux NGX helper" cannot evaluate DLSS 5 Neural Rendering
today, for a reason outside this project's own code or architecture entirely — the
model/feature implementation this project's whole purpose depends on has never been
published for Linux, in any form, by NVIDIA or anyone else. This is a different kind of
blocker than Phase 4's DMA-BUF findings (those were about *mechanism* feasibility, and
came out clearly negative on real evidence). This one is about *artifact availability*:
the mechanism to run a native snippet looks sound; the actual snippet this project
needs simply doesn't exist outside a Windows DLL.

The only ways this could ever change, none of them in this project's own control or
already-decided scope:

- NVIDIA eventually publishes (leaks or officially releases) a native Linux build of
  the DLSS 5 Neural Rendering feature. Nothing to build for that today; revisit if it
  ever happens.
- This project chooses to extract/reimplement the feature's actual inference logic
  from the Windows `nvngx_dlssnr.dll` binary itself, rather than calling into NVIDIA's
  own compiled implementation of it. This is a **materially different and larger legal
  and technical undertaking** than anything this project has done so far — the
  caller-identity spoof Alex already accepted (`ATTRIBUTION.md`, `spoof.rs`) is about
  *calling* NVIDIA's own compiled code under a false caller identity; this would be
  about *extracting and reusing the proprietary model/inference code itself*, a
  different category of exposure. Not something to start without Alex explicitly
  deciding to cross that line, the same way the caller-identity spoof was an explicit,
  named decision rather than something assumed.

## Recommendation

Keep the current architecture (Wine-hosted helper running the real
`nvngx_dlssnr.dll`). There is no native alternative available today, and the gap isn't
one more Wine/NVIDIA-API trick away from closing — it's a missing artifact nobody but
NVIDIA can produce, or a decision to reimplement proprietary model internals that
hasn't been made and shouldn't be assumed. Don't restart this investigation without new
information (NVIDIA publishing something for Linux, or Alex deciding to take on the
reimplementation question directly). If DMA-BUF-style zero-copy transport still matters
after Phase 1's real GTA fps measurement, revisit it as its own problem
(`DMABUF_TRANSPORT_DESIGN.md`'s own recommendation: real `SCM_RIGHTS` fd-passing) rather
than waiting on this.
