# Neural Forge

Neural Forge runs NVIDIA's DLSS 5 Neural Rendering model in Linux and Proton games. A Vulkan layer
inside the game runs the network on the game's own GPU (before DLSS upscales the frame wherever the
game uses DLSS) and writes the result back into the picture; there is no Windows helper and no Wine
prefix. You extract the model once from your own copy of NVIDIA's `nvngx_dlssnr.dll`, and a GTK4
app and a CLI set it up and tune it.

This is experimental, personal-use software that runs NVIDIA's proprietary model weights outside
NVIDIA's own runtime. Read [Legal](#legal) before you use it.

Repository: [labj1987/neural-forge](https://github.com/labj1987/neural-forge).

| Model settings | Setup | Status |
|---|---|---|
| ![Model tab, scrolled to the middle: Placement (Auto), Model resolution (Full), Model brightness (Game) and its scale, Per-pass settings, Keep the game's sky, More mask rules, the toggle key and the Colour heading; above are the switch, style, strength and passes, below the rest of the Colour group and the HUD on the finished frame](screenshots/settings.png) | ![Setup tab: the model present (the full and the small network of build 310.9.16.0), the Network choice (Full network), Extract from DLL, then the Steam games with their switches, the executables they target and an edit button each; below them, the launch option to copy by hand](screenshots/setup.png) | ![Status tab: telemetry (model, game, frame rate and the graph of the frame time and the model's GPU time), then the layer state, the Model placement line and the model (the full and the small network of build 310.9.16.0)](screenshots/status.png) |

## Requirements

- x86_64 Linux with glibc 2.39 or newer (Ubuntu 24.04, Fedora 40 or later), GTK 4.12 and
  libadwaita 1.5 or newer from the system.
- An NVIDIA RTX 40-series or newer GPU (the network uses FP8 tensor cores) with a driver that offers
  `VK_NV_cuda_kernel_launch`, `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2` and
  `VK_EXT_shader_float8` (tested with 615.71.09, 615.78.08 and 620.70.06). On other GPUs the layer does nothing.
- 64-bit games. A game that uses DLSS through Vulkan (under Proton: vkd3d-proton or DXVK, with
  DXVK-NVAPI) gets the model before the upscaler; other games get it on the finished frame.
- Your own `nvngx_dlssnr.dll` (from a game that ships it, or NVIDIA's SDK), once: build 310.9.16 has
  both the full and the small network, older builds the full one. This project does not ship the DLL
  or the weights.

## Install

Download the AppImage from [Releases](https://github.com/labj1987/neural-forge/releases), make it
executable and run it:

```bash
chmod +x neural-forge-*-x86_64.AppImage && ./neural-forge-*-x86_64.AppImage
```

Each start installs or updates the Vulkan layer under `~/.local/share/neural-forge` and
`~/.local/share/vulkan/implicit_layer.d`. Nothing needs root.

Then extract the model on the Setup tab with **Extract from DLL**, or:

```bash
~/.local/share/neural-forge/bin/neural-forge-cli extract-model /path/to/nvngx_dlssnr.dll
```

The DLL is not needed after that.

## Usage

1. Close Steam, then switch the game on in the Setup tab's Steam games list. That adds
   `NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=<the game's executable> %command%` to its launch
   options. With Steam running, copy the launch option from the Setup tab into the game's
   Properties instead.
2. In the game, use DLSS Super Resolution (Quality, Balanced or Performance): the model then runs
   before the upscaler, which is the fastest and best-looking place. With Ray Reconstruction on, it
   runs right after DLSS. Turn on the game's own DLSS Frame Generation if you want it.
3. Play. The model starts after 5 seconds of steady rendering. The app's settings apply live, and
   the app can be opened or closed at any time.

**F11** switches the model off and on. It reads the keyboard through evdev, which needs your user
in the `input` group (log out and in again afterwards); without it, it falls back to X11:

```bash
sudo usermod -aG input "$USER"
```

The Setup tab's **Network** row picks the full network (the default) or the small one, which takes
about half the GPU time and changes the picture less. When video memory runs short, the network
pauses and comes back when there is room.

Settings can differ per game: **Settings for**, under the tabs, picks All games or one game, and a game keeps
only the settings you change for it (**Restore defaults** removes them). A game without its own uses the
settings for all games.

The Model tab's **Colour** group shapes the model's change rather than the picture: its saturation,
contrast and detail, and protections that keep the frame's own hues, greys, shadows or highlights
(all off by default).

The Model tab's **Sharpening** group adds AMD FidelityFX CAS (adaptive sharpening) and Clarity (local
contrast) to the finished frame, in every game, after the model; both are off by default, and the HUD
regions keep them off the HUD.

The game's HDR works with the model before the upscaler and after DLSS. On the finished frame, HDR10
works too: the network runs on an SDR copy of the frame and its change is put back onto the HDR frame.
Highlights above about 185 nits get less and less of the change and none above about 295 nits; a
float16 (scRGB) or 10-bit SDR swapchain is presented untouched.

A game whose folder ships anti-cheat (Easy Anti-Cheat, BattlEye and others) asks before it is switched
on. Neural Forge does not hide its layer from anti-cheat: the game may refuse to start or go online
with it, and the anti-cheat can ban the account, most of all in online play.

After an update, restart the game: the layer and the app must be the same version.

Every setting, the other placements and all environment variables: [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## Known issues

- **GTA V Enhanced sometimes crashes during "Game Init".** A bug in the game, not Neural Forge's;
  relaunch.
- **The game's DLSS Frame Generation does not always switch on**, with or without Neural Forge. In
  GTA V Enhanced, set a frame rate limit (`FrameLimit` 120 in its settings.xml) and it stays on.
- **At 4K a 12 GB card is at its video memory limit.** Close programs that hold video memory
  (browsers with hardware acceleration) before playing at 4K: once the game runs short, it can stay
  slow until it restarts. With Ray Reconstruction the model runs on the 4K output and takes the most
  memory: Cyberpunk 2077 with path tracing at 4K runs short with the model on.
- **NVIDIA `Xid 109` or `Xid 119` GPU hangs** are a known NVIDIA driver bug under Proton, also in
  games without Neural Forge ([NVIDIA/open-gpu-kernel-modules#1097](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1097)).
  Workarounds other users report: [docs/HARDWARE_VALIDATION.md](docs/HARDWARE_VALIDATION.md) (2026-09-16).
- **Unreal Engine 5 games are only partly tested.** Lords of the Fallen and Remnant II ran; the Black
  Myth: Wukong Benchmark Tool faulted the GPU intermittently (`Xid 13`, `Xid 32`) with frame
  generation on.
- **Hogwarts Legacy turns Ray Reconstruction back off** unless you switch it on in its menu.
- **On the finished frame, Find the HUD can treat a first-person weapon as HUD** and keep the model
  off it. Switch Find the HUD off in the Model tab where that shows.
- **HDR on the finished frame** works with HDR10 games (tested in Crimson Desert). It needs the model
  resolution at 100%, takes no captures, and leaves the brightest highlights alone; float16 (scRGB) HDR
  is presented untouched.

## Troubleshooting

Start with **Diagnose** on the Status tab (or `neural-forge-cli doctor`): it checks the install, GPU,
driver, video memory, GPU faults, the last game's log and your launch options, and says what to do.
To report a bug, use the issue form and attach the zip **Save report** writes to your desktop, ideally with the
game still running: it holds the layer's log, the live status, your settings, the game's entry in the tested games
and the HDR verdict (personal names are removed, and of each game's launch options only Neural Forge's part is
included).

- **No effect in game.** Check the launch option, that the model is extracted, and give the game 5 s
  of play. The Status tab's Model placement line says where the model runs, or why it does not.
- **"Waiting for DLSS Super Resolution" never changes.** The game is not using DLSS Super Resolution
  (native resolution counts as not); the model runs on the finished frame instead.
- **"DLSS input identified, not holding".** Neural Forge found the frame DLSS upscales but cannot hold it,
  and says why on the same line; the model runs on the finished frame. Attach **Save report**'s zip to an issue.
- **HDR does not come on.** The Status tab's HDR line says what the game asked for and why, and whether the
  desktop's HDR is on: "game asked for SDR although HDR metadata was sent" with "the window offered no HDR colour
  space" means the game could not get HDR (under Proton add `PROTON_ENABLE_WAYLAND=1 PROTON_ENABLE_HDR=1` to its
  launch options, or start gamescope with `--hdr-enabled`); "HDR10 was available" means the game's own HDR
  setting is off.
- **Something looks wrong.** Press F11 to compare. `NEURAL_FORGE_PREUPSCALE=off` in the launch
  options keeps the network from loading at all.
- **F11 does nothing.** Add yourself to the `input` group (see [Usage](#usage)).
- **Flatpak Steam** cannot see the layer: use your distribution's Steam package.
- **Steam Linux Runtime:** if the game cannot reach Neural Forge, add
  `PRESSURE_VESSEL_FILESYSTEMS_RW=/tmp/neural-forge-$UID` to the launch options.

## More

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md): how it works, every setting, tested games, environment
  variables, CLI, files, uninstall.
- [docs/MEASURED_RESULTS.md](docs/MEASURED_RESULTS.md): frame rates and GPU time in GTA V Enhanced,
  Crimson Desert and Hogwarts Legacy.
- Building from source: [docs/RUNNING_AND_MEASURING.md](docs/RUNNING_AND_MEASURING.md), section 1.
- Everything else: [docs/README.md](docs/README.md).

## Legal

Neural Forge does not load or run NVIDIA's DLL, and it contains no workaround for NGX's caller
check. `extract-model` reads the model's weights out of the resource data of your own copy of
`nvngx_dlssnr.dll`: no code is read, nothing is decrypted, no check is bypassed. The weights are
NVIDIA's; they are not distributed here, and running them outside NVIDIA's runtime may be restricted
by NVIDIA's NGX licence, depending on jurisdiction and how you use it. There's no licence grant
here for NVIDIA's model and none implied. Use it at your own legal risk, for personal,
non-commercial use.

This project reads and adapts source from
[DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (AGPL-3.0); see
[ATTRIBUTION.md](ATTRIBUTION.md) for what is taken and from where. Upstream's license is AGPL-3.0,
which is why this project's license is AGPL-3.0-or-later as
well: a requirement for the adapted code, not just a preference.

## License

This project's own code is licensed under the **GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later)**; see [LICENSE](LICENSE). NVIDIA's model weights, which the user extracts from
NVIDIA's DLL, are not covered by it and are not redistributed here. The network's source in
`third_party/opendlss-nr` keeps its MIT licence (`third_party/opendlss-nr/UPSTREAM.md`); the Rust
crates and their licences are listed in [THIRD_PARTY_CRATES.md](THIRD_PARTY_CRATES.md).

## Acknowledgements

- [DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (bmitch87): the architecture, the
  composition strategy and many ported functions ([ATTRIBUTION.md](ATTRIBUTION.md)).
- [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) (maan): the network implementation the
  layer runs (in `third_party/opendlss-nr`).
- [vk-layer-for-rust](https://github.com/google/vk-layer-for-rust) (Google): the Vulkan layer
  framework (in `third_party/vk-layer-for-rust`).
- [OptiScaler](https://github.com/cdozdil/OptiScaler),
  [OptiScaler_DLSSNR](https://github.com/Dagherbou/OptiScaler_DLSSNR), RenoDX (clshortfuse), hhkbble
  and xenmods ([DLSSNR-Cost-Scaler](https://github.com/xenmods/DLSSNR-Cost-Scaler)): ideas and
  composition techniques.
- Development assistance: Claude Code (Anthropic) and Codex (OpenAI).
