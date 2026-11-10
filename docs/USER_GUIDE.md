# Neural Forge user guide

The [README](../README.md) covers installing and using Neural Forge. This guide has the rest, for the
newest release: how the model runs, every setting, the games it was tested in, the environment
variables, the CLI and the files it writes. Measurements are in [MEASURED_RESULTS.md](MEASURED_RESULTS.md).

## How it works

The network runs inside the game's process, in the Vulkan layer, on the game's GPU. NVIDIA's DLL is
not loaded at run time: the layer reads the model's weights from the folder `extract-model` wrote
(see [Install](../README.md#install)). Two networks come in NVIDIA's DLL; both are extracted, and the Setup
tab's **Network** row chooses which one runs:

- **Full network** (71 blocks), in every build of the DLL, and the default. Its shading and depth are
  stronger.
- **Small network** (31 blocks, the same kinds of blocks with fewer of each), in build 310.9.16. It
  takes about half the GPU time of the full one and changes the picture less. It is what NVIDIA's own
  runtime runs from build 310.9.16 at its default preset.

A game that is running switches to the chosen network at once: its next frames go to DLSS untouched
while the network loads, and the network's history starts afresh. When the chosen network is not
extracted (a DLL without the small one), the other one runs.

When the game and other programs leave too little video memory for the network, the driver would move
part of it to system memory, where it runs many times slower. Instead the network pauses and frames go
to DLSS untouched. The small network is no way out: it needs only about 45 MiB less video memory than the
full one. The network comes back once the memory has had room for it for 10 s (and no sooner than 30 s
after the pause, longer when pauses repeat). While paused, Neural Forge holds no video memory. The Status
tab's Model placement line says when the network is paused.

Each network's answer for an input frame is bit-exact with NGX's own
([NATIVE_BACKEND.md](NATIVE_BACKEND.md), 0.6 and "The small network").

The model runs in one of three places. With the Model tab's **Placement** at **Auto** (the default) the
layer picks the place for each game by itself:

- **Before the upscaler**, when the game uses DLSS Super Resolution (at Quality, Balanced,
  Performance or DLAA). The layer recognises the frame the game hands to
  DLSS: the scene at render resolution, in HDR (16-bit float), without the HUD. It holds that DLSS
  submission, runs the network on the frame (every frame while the camera is still, every second
  frame in motion, with the edit cache carrying it between), with DLSS's motion vectors and camera
  jitter for its history, and writes the answer back before DLSS upscales it. DLSS then upscales the
  enhanced frame. The model works on fewer pixels and gets the input it was made for. Games that
  record their frame in DLSS's own command buffer (Crimson Desert) are held inside that buffer.
- **After DLSS**, when the game runs DLSS Ray Reconstruction: Ray Reconstruction's input is the noisy
  ray-traced frame before its denoising, which is not a frame the model can improve, so the model runs
  right after DLSS instead, on its denoised, upscaled frame at the output resolution, before the game's
  post-processing and HUD, with DLSS's motion vectors for its history. Where DLSS's output cannot be
  held there, the model runs on the finished frame. Switching Ray Reconstruction on or off in a game
  moves the model within a few frames.
- **On the finished frame**, for everything else: native resolution, games without DLSS, and
  devices without NVIDIA's `VK_NVX_image_view_handle`. The finished
  frame is captured at present, answered by the network and composited back onto the same frame
  before it is shown. It has no history by default (no motion vectors there). The frame has the
  game's HUD in it: the layer finds the HUD and keeps the model's change off it (see
  [HUD on the finished frame](#settings)). An HDR10 frame (10-bit, PQ) is shown to the network as an
  SDR picture, its brightness divided by a reference white of 203 nits, and the network's change is put
  back onto the HDR frame in linear light. Highlights get less and less of the change from about 185 nits
  (in their brightest colour channel) and none from about 295 nits, pixels the network did not change keep
  their exact HDR values, and
  the model resolution must be 100% (below it the frame is presented untouched). A float16 (scRGB) or
  10-bit SDR swapchain is presented untouched, and captures of an HDR frame are refused.

The game's own **DLSS Frame Generation** works with the model before the upscaler and after DLSS.
Generated frames are built from enhanced frames, and the model runs once per real frame.

To run the model on the finished frame instead, set Placement to Finished frame in the Model tab (live,
no relaunch). `NEURAL_FORGE_PREUPSCALE=off` in the launch options loads no network at all: the game's
frames are presented as it made them, the rollback when something looks wrong.

How it works, step by step: [ARCHITECTURE.md](ARCHITECTURE.md).

## Settings

### The app's tabs

| Tab | What it has |
|---|---|
| Model | The Neural rendering switch, the placement (where the model runs in the frame), the model's style, strength presets (Light, Moderate, Reference, Overdrive, Remix), intensity, local tone, local structure, skin structure, auto mask, passes, pass feedback, model resolution and per-pass settings, model brightness and its scale, keep the game's sky, more mask rules (dark areas, fast motion), model every Nth frame (after the upscaler only), the toggle key, the Colour group: saturation, contrast and detail of the model's change, gamut guard, dither, hue, grey, shadow and highlight protection and a look-up table, the HUD group: finding the HUD and HUD regions on the finished frame, and the Sharpening group: adaptive sharpening (CAS) and Clarity on the finished frame. |
| Composition | How the model's answer is blended back on the after-the-upscaler path: detail and colour strength, highlight guard, model resolution after the upscaler (shown while the model runs there), the reversible proxy mode, transfer mode, colour trust, ratio smoothing, ghost guard, apply model edit, hold frame, and the white point; HDR input and Colour mode are shown but not used yet. |
| Debug | Compare views (side by side or wipe, split, zoom, swap), debug views (original/proxy, raw answer, amplified diff, colour trust, the composite before colour trust, and what the edit cache rejected before the upscaler), the edit cache at work (frames with and without the network, the share it rejected, its cost), the scene cuts the model started afresh on, and captures: take one (the next frame as before/after PNGs, with what happened) and open the captures folder. |
| Status | Telemetry (the model, the game, the frame rate, and a graph of the time per frame and the model's GPU time), layer state, **Model placement** (before the upscaler with DLSS's render size and missed frames, after DLSS with its output size, both, waiting for DLSS Super Resolution, DLSS's input found but not held and why, or after the upscaler), the model, reset and profiles, and **Diagnosis**: Diagnose (the same checks as `neural-forge-cli doctor`) and Save report. |
| Setup | The model (present, or extract it from `nvngx_dlssnr.dll`), which network runs (Full network, the default, or Small network; a running game switches at once), your installed Steam games with a Neural Forge switch each, the executable it targets and an edit button to choose another (a game whose folder ships anti-cheat says which, and asks before it is switched on), and the launch option to copy by hand for a game you pick. |

Before the upscaler the network runs on every frame while the camera is still and on every second
frame while it moves; the frame in between reuses its last result, moved with the game's motion and
checked against depth and colour (the edit cache). Where the picture changes completely from one frame
to the next (a cut in a cutscene, a respawn, a camera switch), the model starts afresh instead of blending
in the previous scene. The model's answer is written straight back (there is no
composition step), so "Model every Nth frame" and the Composition tab only change the after-the-upscaler
path. The Model tab's tuning and the switch apply to both.

**Strength presets** set intensity, local tone, local structure and skin structure together:

| Preset | Intensity | Local tone | Local structure | Skin structure | What it matches |
|---|---|---|---|---|---|
| Light | 0.45 | 0.55 | 0.25 | 0.15 | A light touch. |
| Moderate | 0.70 | 0.80 | 0.55 | 0.40 | In between. |
| Reference | 1.00 | 1.00 | 1.00 | follows local structure | NVIDIA's own defaults for the network; Neural Forge's default. |
| Overdrive | 1.30 | 1.25 | 1.45 | 1.20 | Stronger than NVIDIA's defaults. |
| Remix | 1.00 | 0.30 | 0.70 | 0.50 | RTX Remix's defaults for its DLSS neural rendering. |

**Keep the game's sky** (on by default) leaves the sky untouched before the upscaler: no edit where the
game's depth is at the far plane, fading in over the distance just in front of it. **More mask rules** can
also fade the edit out in dark areas or in fast motion (both off by default). Together they are a per-pixel
mask of the model's intensity and structure, after NVIDIA's control mask (the semantics come from
GrimsVerk's runtime research).

**Passes** (1 to 10, default 1) runs the model that many times over each frame, before or after the
upscaler: every pass after the first works on the previous pass's picture. Each pass costs about as much
GPU time as the first, and the effect builds up with every pass. In GTA V Enhanced at 1440p with frame
generation 4x (real frames per second, pass 4 of the built-in benchmark, one run each; the small
network's one pass is from 2026-10-19, the rest from 2026-10-28):

| Network | 1 pass | 2 passes | 3 passes |
|---|---|---|---|
| Small | 74.3 | 60.4 | 52.8 |
| Full | 62.1 | 50.7 | |

Two passes of the small network cost about as much as one pass of the full network, and look
over-processed: grittier stone and noisier roads. Each pass keeps its own history, so the passes do not
blend toward each other's pictures over time. **Per-pass settings** (Edit…) let each pass use its own
intensity, local tone, local structure, skin structure, style, auto mask, model resolution, colour of the
change or placement instead of the Model tab's.
**Pass feedback** (0 to 1, default 1) sets how much of a pass's picture the next pass works on; lower
values keep more of the pass's own input, so the passes add less to each other.

**Model resolution** (Model tab) sets the size the model runs at, as a share of the frame it works on
(DLSS's input before the upscaler, DLSS's output after it): **Full** (the default), **Quality** (66.7%),
**Balanced** (58%), **Performance** (50%) or **Ultra Performance** (33.3%). Below Full the model works on a
downscaled copy of the frame and its change is brought back to full size, following the frame's own edges,
so the frame keeps its own detail; the model costs less GPU time and memory but adds less detail of its own.
Before the upscaler the frame is already small (1486x836 in GTA V at 1440p) and the smaller sizes keep
little of the model's detail; after DLSS, on the full output (with Ray Reconstruction), Performance keeps
most of it for about 30 fps more (see [MEASURED_RESULTS.md](MEASURED_RESULTS.md)). Full is the frame exactly as
without the setting. Each pass can have its own model resolution in the per-pass settings: a pass works
on the previous pass's full-size picture at its own size. Changing it starts the model's history afresh
and shows a moment of untouched frames while the model is prepared for the new size.

**Model brightness** (Model tab) sets how bright the picture the model works on is, before the upscaler and
after DLSS; the game's own brightness is kept either way, because the picture is brought back exactly
afterwards. **Game** (the default) uses the exposure the game gives DLSS (measured from the frame instead where a game's is always 0, or too small to use, or, for as long as it does, where the game hands DLSS a frame it already brightened and its steady exposure would make the model's picture far too dark: Clair Obscur: Expedition 33; the layer log says so with `the colour input looks pre-exposed`, and `no longer looks pre-exposed` when it goes back). **Auto** measures it from the frame, within half
to 16 times the game's own so night stays night, for games that give the model a very dark picture: in daytime
Crimson Desert's has a median brightness of 0.14 of white with Game and 0.50 with Auto. How the model treats the
shadows changes with it, so compare the two in the game. **Model
brightness scale** (0.25 to 4, default 1) brightens (above 1) or darkens (below 1) what the model sees on top
of either. `neural-forge-cli shmctl set preupscale_exposure game|auto` and `shmctl set preupscale_brightness
1.5` do the same from a terminal.

**Colour** (Model tab) shapes the model's change to the frame instead of the frame, before and after the
upscaler. **Saturation**, **contrast** and **detail of the change** (0 to 4, default 1) scale how much the
model changes colour, how much it brightens or darkens, and its fine detail against its broad change; at 0
the frame keeps its own colours, its own brightness, or only the broad change. They apply to every pass, and
the per-pass settings can give a pass its own. **Gamut guard** (on) shortens a change that would push a
colour past what the frame can hold instead of clipping one channel, so the hue stays. **Dither** (off)
breaks up banding in smooth gradients after the upscaler. Four **protections** (0, off, to 1) keep parts of the
frame's own look against the model's change, judged by eye-matched colour (Oklab): **Hue protection** keeps each
colour's hue (the model may still make it more or less colourful, brighter or darker), **Grey protection** keeps
greys free of a tint, **Shadow protection** holds back darkening of what was already dark and **Highlight
protection** holds back brightening of what was already bright. Shadow protection also takes away much of the
detail the model adds in darker textures, since that detail is darkening. **Look-up table** applies a 3D `.cube` file (17 to
65 points; the Choose button or `neural-forge-cli shmctl lut FILE`) to the last pass's result where the model
changed the picture: its strength (0, off) grows with the size of the change and is full at **Look-up table
full at**, so a pixel the model left alone keeps its colour. A table that is removed or cannot be read stops
applying, before and after the upscaler alike; so does one whose `DOMAIN_MIN` to `DOMAIN_MAX` span in a channel
is below 0.000001 or too wide to give a usable grid, which the Choose button and `shmctl lut` refuse with the reason. At their defaults the picture is exactly the model's.

**Placement** (Model tab) sets where the model runs in the frame: **Auto** (the default; see
[How it works](#how-it-works)), **Before the upscaler** (with Ray Reconstruction on, the finished
frame), **After DLSS** (on DLSS's output, before the game's HUD and post-processing, at the output
resolution, so it costs more than before the upscaler; where a game's DLSS output cannot be held, it runs
where Auto would) or **Finished frame** (on the presented frame, HUD included; DLSS gets its input
untouched). Model resolution applies after DLSS too, as a share of the output size. Each pass can have its
own placement in the per-pass settings, so pass 1 can run before the upscaler and pass 2 after DLSS, each
with its own settings and history. Passes run in the frame's order: a pass placed earlier than the pass
before it runs where that one does, and passes on the finished frame after passes held in the game's frame
run after DLSS instead (the model serves one of the two at a time); the log says when a placement is moved.
`neural-forge-cli shmctl set placement auto|before|upscaled|present` (and `pass2_placement`, ...) does the
same from a terminal.

**HUD on the finished frame** (Model tab): where the model runs on the finished frame, the game's HUD is in
the picture. **Find the HUD** (on) keeps the model's change off what stays still and sharp while the
picture moves: text, icons, the minimap. It needs a few seconds of movement to find them, and a still
scene (a menu, a paused game) never adds to what it found. **HUD regions** are up to eight rectangles in
percent of the frame: **Keep** has no change inside, **Soften** keeps the change's strength (a share) inside;
the change comes back over a few pixels around each one. A region is cut at the frame's edges (Left at 90 %
leaves at most 10 % of width), and the list shows what is kept. With a history after the upscaler, the history is
not used under the HUD either, so changing HUD text leaves no trail. They are saved as `set_hud_regions` in
`config.ini` and can be set with `neural-forge-cli shmctl set hud_regions "0,0.88,0.3,0.12;0.7,0,0.3,0.1,soften,0.5"`
(left, top, width, height in 0-1, then `keep` or `soften` and the strength). Before the upscaler and after
DLSS the HUD is not drawn yet, so neither setting applies there. The regions also keep the sharpening off the
HUD (below), wherever the model runs.

**Sharpening** (Model tab) works on the finished frame in every game, whatever the placement: each presented
frame (frame generation's too) is sharpened just before it is shown, after the model, so the model never sees
it. Both filters are off at 0, the default. **Adaptive sharpening (CAS)** is AMD FidelityFX Contrast Adaptive
Sharpening: each pixel is sharpened by as much as its neighbours leave room for, so soft detail gets the most
and edges that are already sharp, and flat areas, get little; about 0.7 is CAS's own default and 1 its
strongest. **Clarity** (0 to 2) lifts local contrast of brightness: the difference between each pixel and a
blurred surround of **Clarity radius** pixels (at 1440p, scaled with the resolution; 2 to 64, default 16) is
added back times the strength, the same to all three colour channels, so colours keep their hue. **Clarity
midtones** (0.5) keeps it off the darkest and brightest parts (0 evenly, 1 midtones only), and **Clarity edge
protection** (0.5) stops the blur at strong brightness edges so they do not grow bright or dark halos. The
**HUD regions** apply: nothing inside a Keep region, the strength's share inside a Soften one (Find the HUD
does not apply here). Clarity does not use the game's depth: on the presented frame there is none to use.
GPU time a presented frame at 1440p on an RTX 5070: 0.29 ms for CAS, 0.33 ms for Clarity at radius 16 (0.43 ms at
64), 0.37 ms for both; most of it is copying the frame out and back. It is paid on every presented frame, so with
frame generation 4x four times a rendered frame. `neural-forge-cli shmctl set sharpen_cas 0.5` (and
`clarity_strength`, `clarity_radius`, `clarity_midtones`, `clarity_edges`) does the same from a terminal, and
a game can have its own sharpening like any other setting ([Settings for one game](#settings-for-one-game)); `NEURAL_FORGE_SHARPEN` overrides it for one launch, for comparisons. A capture (the Debug tab's Capture button) shows the frame
sharpened where the model runs in the game's frame (before the upscaler or after DLSS), and before the
sharpening where the model runs on the finished frame.

![Per-pass settings, scrolled to the end: an override switch for each of local tone, local structure, skin structure, style, auto mask, model resolution, edit saturation, edit contrast, edit detail and placement, each off](../screenshots/passes.png)

Settings are live: the GUI and the running layer share them through shared memory.
`neural-forge-cli shmctl status|set|toggle|capture` does the same from a terminal. The Debug tab's
Capture button (and `shmctl capture`) saves the next presented frame before and after Neural
Forge's composition and says what happened: where the files are and whether the effect changed the
frame, or why nothing was written; clicking again while it waits cancels it, and a request no game
takes is withdrawn after 5 seconds.

### Settings for one game

Every game uses the same settings unless it has some of its own. **Settings for**, the bar under the tab
switcher, picks whose settings the tabs show and change: **All games**, or one of your installed Steam games
(and any other game that has settings of its own or is running). A game's own settings are only the ones you
change while it is picked; every other setting follows All games, so a later change there still reaches it.
Setting a value back to the All games value removes it from the game's own. While a game is picked, each setting
it has its own value of is marked **This game**, with a button beside the mark that drops just that one (the
setting follows All games again); settings without the mark follow All games. **Restore defaults** removes all of
the game's own settings. The line beside the menu says how many it has and whether it is running.

When a game starts, its own settings are put on top of the settings for all games, and the menu switches to
it, so a change you make while it runs is saved for that game. A game without settings of its own runs with the
settings for all games. Picking another entry shows that entry's settings in the running game too, until you
pick the game again or start it again. Reset and loading a profile change the settings shown.

A Steam game is known by its app id; a game outside Steam by its executable's name (`exe:game.exe`). They are
saved in `~/.config/neural-forge/games.ini`, one `[570]` or `[exe:game.exe]` section per game holding only its own
`set_<name>=` lines. From a terminal: `neural-forge-cli game-settings list`, `show GAME`,
`set GAME NAME VALUE` (the names and values `shmctl set` takes) and `clear GAME [NAME]`. Neural Forge can ship
settings for games that need their own; none of the tested games does, so none are built in now. Your own
settings win over built-in ones, and an update never changes your files.

## Tested games

Each game was run on the test machine (RTX 5070, 2560x1440, Proton) with its own settings as found:
DLSS on, the game's frame generation on wherever it has it, no frame-rate cap. In each one the model
ran with no missed frames and no GPU fault.

| Game | Engine, API | DLSS used | Where the model runs |
|---|---|---|---|
| GTA V Enhanced | RAGE, DX12 | Super Resolution, Frame Generation 4x | Before the upscaler |
| Crimson Desert | BlackSpace, DX12 | Ray Reconstruction, dynamic Frame Generation | After DLSS with Ray Reconstruction on; before the upscaler, inside DLSS's buffer, with it off |
| GTA San Andreas - The Definitive Edition | Unreal Engine 4 | Super Resolution | Before the upscaler |
| Metro Last Light Redux | 4A Engine, DX12 | Super Resolution (the game's own DLSS 5 NR off) | Before the upscaler, inside DLSS's buffer |
| God of War (2018) | Santa Monica Studio, DX11 | Super Resolution | Before the upscaler |
| Marvel's Spider-Man Remastered | Insomniac, DX12 | Super Resolution, Frame Generation | Before the upscaler |
| Shadow Warrior 3: Definitive Edition | Unreal Engine 4, DX12 | Super Resolution | Before the upscaler |
| Lords of the Fallen (2023) | Unreal Engine 5, DX12 | Super Resolution, Frame Generation (launch option `-DLSSFG`) | Before the upscaler, inside DLSS's buffer |
| Remnant II | Unreal Engine 5, DX12 | Super Resolution, Frame Generation | Before the upscaler, inside DLSS's buffer |
| Hogwarts Legacy | Unreal Engine 4, DX12 | Super Resolution, Ray Reconstruction, Frame Generation | After DLSS with Ray Reconstruction on; before the upscaler, inside DLSS's buffer, with it off |
| Cyberpunk 2077 | REDengine 4, DX12 | Super Resolution, Frame Generation 3x (its built-in benchmark) | Before the upscaler, inside DLSS's buffer |
| Red Dead Redemption 2 | RAGE, Vulkan | Super Resolution (Quality) | Before the upscaler, inside DLSS's buffer, on a compute queue shared with the game; not on the finished frame |

Setup notes for these games:

- **Red Dead Redemption 2** runs on Vulkan (Settings > Graphics > Advanced > Graphics API). Start it through Steam as
  usual (its Rockstar launcher checks Steam). The model runs only before the upscaler there, so turn DLSS on.

- **GTA San Andreas - The Definitive Edition** resets its Frame Rate setting to 60 at every launch,
  whatever its settings files say (a game bug). Set it to Unlocked in Options > Graphics each time.
- **Shadow Warrior 3** greys out its NVIDIA DLSS option while FidelityFX CAS is on (Settings >
  Video): set CAS to Off and DLSS becomes selectable. Its first launch also benchmarks the machine and
  picks the Low preset under Proton; set Overall Quality yourself.
- **Marvel's Spider-Man Remastered** starts with DLAA-like dynamic resolution (target 60 fps), V-Sync
  on and frame generation off: set Upscale Quality to a fixed mode, Frame Generation to DLSS Frame
  Generation and V-Sync off (Settings > Display and Graphics).
- **Crimson Desert** compiles shaders on its first launch after a Neural Forge update; the effect
  starts once it is in the world.
- **Lords of the Fallen (2023)** has no Frame Generation option in its menu: the developers turned
  DLSS Frame Generation off in patch 1.009 (October 2023), and the launch option `-DLSSFG` turns it
  back on (patch 1.1.193): add `-DLSSFG` after `%command%` in the launch options Neural Forge wrote
  (`NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=<exe> %command% -DLSSFG`). While the game's settings file
  says frame generation is on (`DLSSFrameGenerationEnabled=True` in
  `AppData/Local/LOTF2/Saved/Config/Windows/GameUserSettings.ini`), VSync and Reflex are on and
  greyed out; set it to `False` with the game closed to change them, at the cost of frame generation.

## Environment variables

Set in a game's launch options unless the table says otherwise. Every variable was checked against
the code (`crates/`, `scripts/`).

| Variable | Read by | Effect |
|---|---|---|
| `NEURAL_FORGE_ENABLE=1` | layer (manifest) | Turns the layer on for this game. |
| `NEURAL_FORGE_DISABLE=1` | layer (manifest) | Forces it off, overriding `ENABLE`. |
| `NEURAL_FORGE_TARGET_EXE=a.exe,b.exe` | layer | Only these executables (case-insensitive basenames) may use the layer, even one whose name is on the built-in list of launchers and helpers (a game whose executable is `Launcher.exe`). Switching a game on sets it to the game's own executables. Without it, the first process not on that list to present steadily takes the layer. |
| `NEURAL_FORGE_PREUPSCALE=off` | layer | Rollback: the network is not loaded, so no frame is changed (the layer log says so once). The model on the finished frame is the Placement setting's Finished frame instead. Unset (or `model`) is the default. An unknown value is logged and treated as `off`. |
| `NEURAL_FORGE_PREUPSCALE=dump\|identity\|roundtrip` | layer | Diagnostics: write DLSS's input to `captures/`, hold without the model (raw copy back), or run the HDR encode and decode with no model. They wait for their own work inside the game's submit, so a submit that waits on an event or on a timeline value not reached yet goes to DLSS untouched. See [RUNNING_AND_MEASURING.md](RUNNING_AND_MEASURING.md). |
| `NEURAL_FORGE_EDIT_CACHE=off` | layer | Before the upscaler, run the network on every frame instead of the edit cache's cadence (every frame while the camera is still, every second frame while it moves), with none of the cache at all (the game's depth is neither made copyable nor copied). For A/B comparisons, benchmarks and rollback; unset (or `on`) is the default. An unknown value is logged and the cache stays on. |
| `NEURAL_FORGE_GAME_RESET=off` | layer | Before the upscaler, ignore the game's own signal that DLSS should start afresh (a camera cut the game knows of; read where Neural Forge knows how the game's DLSS passes it on, such as GTA V Enhanced), so the model keeps its history through it. For testing; unset (or `on`) is the default. |
| `NEURAL_FORGE_EXPOSURE_JUMP=off\|N` | layer | Before the upscaler, how much the frame's exposure must change between two frames of the model for it to start afresh: N stops, above 0 and up to 16 (default 1, twice or half as bright), or `off` to keep the history through any change. For testing; an unknown value is logged and the default stays. |
| `NEURAL_FORGE_SCENE_CUT=off\|N` | layer | Before the upscaler, how different two frames must be for the model to start afresh: the mean change of brightness, 1 to 255 (default 40), or `off` to never start afresh on a cut. For testing; an unknown value is logged and the default stays. |
| `NEURAL_FORGE_PREUPSCALE_PAPER_WHITE=3` | layer | The HDR encode's paper white (default 3; a positive number up to 1000). Tuning only. |
| `NEURAL_FORGE_PIPELINED=1` | layer | After the upscaler: a pipelined present. Higher frame rate, but answers land on later frames and ghost. HDR10 frames are presented untouched in this mode. |
| `NEURAL_FORGE_HDR_REFERENCE_NITS=203` | layer | On the finished frame of an HDR10 game, the brightness (in nits) the network's SDR picture takes as white before its highlight roll-off (default 203, clamped to 80-1000). Higher shows the network a darker picture and lets brighter highlights be changed. Tuning only. |
| `NEURAL_FORGE_HDR_PRESENT=off` | layer | Testing aid: present HDR10 frames on the finished frame untouched, as for float16 ones. |
| `NEURAL_FORGE_SHARPEN=off\|cas=0.5,clarity=0.4,...` | layer | The sharpening of the finished frame for this game, in place of the Model tab's: `off`, or comma-separated `cas`, `clarity`, `radius`, `midtones` and `edges` with their values (clamped to the settings' ranges); a setting not named keeps the Model tab's value. For A/B comparisons; a value that cannot be read is logged and the settings apply. |
| `NEURAL_FORGE_TOGGLE_KEY=F10` | layer | Toggle key by name (`F1`-`F12`, `Home`, `N`...) or Linux key code; overrides the GUI's. |
| `NEURAL_FORGE_HOTKEY_BACKEND=evdev\|x11` | layer | Forces one keyboard backend (default: evdev, then XInput2). |
| `NEURAL_FORGE_HOTKEY_GLOBAL=1` | layer | The toggle key acts even when another window has the focus. By default a press is ignored only while another window is known to have the focus; where that cannot be told (inside gamescope) the key acts. |
| `NEURAL_FORGE_MAX_MODEL_PIXELS=N` | layer | Largest raster the after-the-upscaler path gives the model (default 3840x2160 = 8294400). |
| `NEURAL_FORGE_LOG=/path` | layer | Log file (default: stderr), written by its own thread and rotated to `/path.1` at 64 MiB. |
| `NEURAL_FORGE_LOG_TIME=1` | layer | Testing aid: timestamp every log line. |
| `NEURAL_FORGE_SHM`, `NEURAL_FORGE_UID` | all | The shared-memory file, and the uid its directory is named after. Normally left alone. |
| `NEURAL_FORGE_PROBE_NGX=1` | layer | Testing aid: log the game's DLSS work as it reaches Vulkan ([PRE_UPSCALER_PROBE.md](PRE_UPSCALER_PROBE.md)). |
| `NEURAL_FORGE_PROBE_PARAMS=1`, `NEURAL_FORGE_PROBE_KERNEL=<name>` | layer | Testing aids: log every word of DLSS's input-kernel parameters, or of a named kernel's. |
| `NEURAL_FORGE_LOG_LAUNCH_QUEUES=1` | layer | Testing aid: log which queue each DLSS launch is submitted on. |
| `NEURAL_FORGE_CAPTURE_AT=N:MS[:FRAMES][,...]` | layer | Testing aid: capture the presented frame (or FRAMES consecutive frames) MS milliseconds after DLSS's frames resumed (after a gap) for the Nth time; 1 is the first frames. Several entries, separated by commas, are each taken once. |
| `NEURAL_FORGE_CACHE_DUMP_AT=N:MS[:K][,...]` | layer | Testing aid: dump K (default 16) consecutive holds before the upscaler for `scripts/cache_replay.py`, at the same moments `NEURAL_FORGE_CAPTURE_AT` names. `neural-forge-cli shmctl dump-cache [--holds K]` asks for one now. |
| `NEURAL_FORGE_WARMUP_SECS=N` | layer | Testing aid: seconds of steady rendering before the layer engages (default 5). |
| `NEURAL_FORGE_INLINE=off` | layer | Testing aid: never hold inside DLSS's command buffer. |
| `NEURAL_FORGE_NATIVE_CHAIN=0`, `NEURAL_FORGE_NATIVE_INLINE_CHAIN=1` | layer | Testing aids: barriers between the network's launches instead of counter chaining (before the upscaler), or counter chaining inside DLSS's buffer (where barriers are the default). |
| `NEURAL_FORGE_NATIVE_SKIP_GRAPH=1` | layer | Testing aid: record everything but the network's launches. |
| `NEURAL_FORGE_NATIVE_NO_HISTORY=1` | layer | Testing aid: every network frame without history, as after a reset. |
| `NEURAL_FORGE_PREUPSCALE_TRANSFER=aces\|linear` | layer | Testing aids: `aces` encodes the picture before the upscaler with RTX Remix's transfer (an ACES curve, and on the way back the model's change scaled by each pixel's own ratio, with Remix's highlight recovery); `linear` leaves out the sRGB curve (the network sees linear values, and the edit cache decodes the same way). `srgb` is the default. |
| `NEURAL_FORGE_PREUPSCALE_HIGHLIGHTS=off` | layer | Testing aid: before the upscaler (and on an HDR10 finished frame), apply the model's change to the brightest highlights too: no keeping of clipped values, no fade near the top, no highlight recovery with `aces`. |
| `NEURAL_FORGE_PREUPSCALE_PICK=N` | layer | Testing aid: take the Nth colour candidate where DLSS's input launch names several. |
| `NEURAL_FORGE_PREUPSCALE_DUMP_FRAMES=N`, `NEURAL_FORGE_PREUPSCALE_DUMP_HAZARD=ignore` | layer | Testing aids for `NEURAL_FORGE_PREUPSCALE=dump`: consecutive frames per dump (default 1); dump even where DLSS's buffer synchronises depth or motion vectors. |
| `NEURAL_FORGE_PROBE_DEPTH=1`, `NEURAL_FORGE_PROBE_RR=<a,b,...>\|1` | layer | Testing aids: log the depth choice and barriers of the first 120 holds; log the images of the launches whose kernel names match (first 4000). |
| `NEURAL_FORGE_PROBE_IMAGE=<kernel>[+],...` | layer | Testing aid: copy the images a named CUDA launch reads, before it (or after it with `+`), once, into `captures/probe-<ms>/`. |
| `NEURAL_FORGE_NATIVE_SHARED_ACTIVATIONS=0` | layer | Gives every network activation its own allocation again instead of sharing memory by lifetime (more video memory, same result). |
| `NEURAL_FORGE_POST_MOTION=flow\|zero\|none`, `NEURAL_FORGE_POST_HISTORY_REJECT=0.12` | layer | After the upscaler (the finished frame), the model's history: its previous output where the GPU's optical flow says the pixel was (`flow`, the default; `zero` on GPUs without `VK_NV_optical_flow`), at the same pixel (`zero`), or none (`none`: every frame on its own, which flickers in large soft patches on grainy or noisy frames). The history is not used where a pixel changed by more than the threshold (0 to 1, default 0.12) since the previous frame, and is dropped on a scene cut. With `flow` the game's device gets `VK_NV_optical_flow` and one optical-flow queue where the GPU offers them, whichever way the model runs in that game; `zero` and `none` leave them out. |
| `NEURAL_FORGE_RAISE_VULKAN=off` | layer | A game whose Vulkan instance asks for a version below 1.3 (Red Dead Redemption 2 asks for 1.1) gets one created for 1.3, which the model needs; `off` creates it as the game asks (the model then cannot run in that game). |
| `NEURAL_FORGE_SHARE_QUEUE=off` | layer | A game that takes every compute queue of the GPU (Red Dead Redemption 2) gets one of them shared with the model, every use of it synchronized; the model then runs before the upscaler only. `off`: no model on such a game. |
| `NEURAL_FORGE_FAIL_CREATE=N[@K]`, `NEURAL_FORGE_NATIVE_FAIL=chain`, `NEURAL_FORGE_NATIVE_FAIL=vram` | layer | Testing aids: let K network builds through, then fail the next N; report one counter-chain timeout; or have the video memory guard pause the network at its 600th and 900th frame, as if it did not fit. |
| `NEURAL_FORGE_GUI_OPEN=<tab>` | GUI | Testing aid: open a tab (`model`, `composition`, `debug`, `status`, `setup`, or `passes` for the per-pass dialog) at start, for screenshots. |
| `NEURAL_FORGE_BENCH=1`, `NEURAL_FORGE_REQUIRE_VULKAN=1`, `NEURAL_FORGE_TEST_DISPLAY=:N` | layer tests | Testing aids: run the hot-path GPU benchmark; fail instead of skipping when no Vulkan device exists; run the present-fence test on a real swapchain in an X server (an Xvfb's display). |
| `NEURAL_FORGE_CLI=/path` | `scripts/install.py`, `scripts/test_install.py` | The `neural-forge-cli` to use. |
| `NEURAL_FORGE_NATIVE_TOOLS=<dir>`, `NEURAL_FORGE_PYTHON=<python>` | build | The native build tools' folder (default `tools/native`), and the Python 3.10+ the build runs. |
| `NEURAL_FORGE_NATIVE_RIG`, `NEURAL_FORGE_NATIVE_MODEL`, `NEURAL_FORGE_NATIVE_HEADS`, `NEURAL_FORGE_NATIVE_PROFILE=1` | native tests | Testing aids: run the network tests on a dumped frame, with a model folder, saving or comparing output heads, and timing each kernel. |
| `NEURAL_FORGE_NR_NO_FUSE_PRE=1`, `_NO_FUSE_POOL=1`, `_NO_FUSE_UPRES=1`, `_NO_FUSE_POST=1`, `_NO_PTX_MLP=1`, `NEURAL_FORGE_NR_CHAIN=0`, `_VIT_CHAIN=0`, `_CHAIN_MASK=N`, `_DEFER_MAX=N`, `_PTX_GEMM=0`, `_PTX_GEMMT=0`, `_PTX_GEMMV=0`, `_PTX_QKV=0`, `_PTX_FFN=0`, `_PTX_ATTN=0`, `_PTX_BLOCK32=0`, `_SPLITK=N`, `_ATTN_STREAM=N` | network | Testing aids: turn off one of the network's fused kernels or PTX kernel families, or change how its launches are chained and split (`third_party/opendlss-nr/src/nr_graph.cpp`, `kernels.cpp`). A switch takes `1` or `0` and a number a whole number; anything else is reported once on the game's standard error and ignored. `NEURAL_FORGE_NR_VALIDATION=1` and `NEURAL_FORGE_NR_DEBUG=1` apply only to the network's own standalone Vulkan context (its tools), not inside a game. |
| `NEURAL_FORGE_SMOKE_SO=/path` | `scripts/smoke-test.sh` | Test this layer library instead of building the debug one. |

## CLI

The CLI is installed as `~/.local/share/neural-forge/bin/neural-forge-cli`; add that folder to your
`PATH`, or call it by that path, for the commands in this guide.

`neural-forge-cli <command>`: `init`, `status`, `doctor`, `report [--appid ID]`, `games`,
`enable GAME [--target-exe EXE[,EXE...]|auto] [--anti-cheat-ok]`, `disable GAME`, `config`, `import-binaries DIR`,
`extract-model DIR [--network full|small]`, `install --appdir DIR`, `uninstall [--purge]`, `profile list|save|load|delete NAME`,
`game-settings list|show GAME|set GAME NAME VALUE|clear GAME [NAME]` for a game's own settings, and
`shmctl status|set NAME VALUE|toggle NAME|capture [--frames N] [--view V]|capture --cancel|dump-cache [--holds K]|reset|lut FILE|none`
for the live settings. `doctor`, `report` and `shmctl status` only look: they create and change nothing.
`neural-forge-cli help [COMMAND]` (or `COMMAND --help`) has the details; an unknown or misspelt
argument prints the usage and exits with status 2.

`enable` switches on a game whose folder ships anti-cheat only with `--anti-cheat-ok`; `games` names
the anti-cheat each game ships, and `doctor` warns about a game that is on with it. What is seen: Easy
Anti-Cheat, BattlEye, EA Javelin, RICOCHET, XIGNCODE3, NetEase, HoYoverse, Denuvo Anti-Cheat, FACEIT,
EQU8, nProtect GameGuard, Anti-Cheat Expert (ACE) and Nexon Game Security, by the files and folders
they leave in the game's folder. Anti-cheat installed outside it (Riot Vanguard, FACEIT's client,
Valve's VAC) cannot be seen.

## Files

| Path | What |
|---|---|
| `~/.config/neural-forge/config.ini` | Paths (`binaries=`, `shm=`, `edit_lut=` for the look-up table) and every setting as `set_<name>=`. The layer applies the saved settings when it opens the channel. |
| `~/.config/neural-forge/profiles.ini` | Named profiles. |
| `~/.config/neural-forge/games.ini` | Each game's own settings: a `[<app id>]` or `[exe:<file.exe>]` section with only the `set_<name>=` lines that differ from `config.ini`'s. The layer applies the game's when it starts. |
| `~/.local/share/neural-forge/` | The installed layer and GUI, the extracted model (`model/`), an imported DLL (`binaries/`), captures (`captures/`). |
| `~/.local/state/neural-forge/layer.log` | The layer's session log: state changes and errors only, never per frame (kept to 256 KiB, the previous one as `layer.log.1`). Diagnose and the report read it. |
| `/tmp/neural-forge-$UID/shm.bin` | The shared memory the layer and the GUI talk through. |

## Uninstall

`neural-forge-cli uninstall` removes the files it installed (files you changed are kept). The
games you switched on keep Neural Forge's variables in their launch options, which do nothing without
the layer.
`neural-forge-cli uninstall --purge` removes everything Neural Forge wrote. First it takes its
variables out of every game's Steam launch options (your own options stay), in every Steam account,
and removes the files it keeps next to each account's `localconfig.vdf` (`.neural-forge.orig` and
`.neural-forge.bak`, its copies of that file, and `.neural-forge.lock`). Steam must be closed for
that: while it runs, or while a game's options cannot be changed safely, the purge removes nothing and
names each game with the launch options to set by hand instead. Then it removes the config,
everything under `~/.local/share/neural-forge` (the model, an imported DLL, captures),
`~/.local/state/neural-forge` and `/tmp/neural-forge-$UID`.

## More known issues

- **Each DLSS game needs a test run before it is claimed:** engines hand DLSS its inputs in
  different ways, and several broke an assumption the first time. See [Tested games](#tested-games).
- **The model after DLSS is tested in two games** (Crimson Desert and Hogwarts Legacy, with Ray
  Reconstruction). Where a game's DLSS output cannot be held, the model runs on the finished frame, at a
  much lower frame rate (36 against 120 fps shown in Crimson Desert).
- **HDR on the finished frame:** HDR10 is handled (above) and tested in Crimson Desert; float16
  (scRGB) swapchains are presented untouched, and HDR frames cannot be captured.
- **Comparing with NVIDIA's own DLSS 5 in a game that ships it:** Proton's dxvk-nvapi 0.9.3 or newer passes
  NVIDIA's driver override for it, so the launch options
  `DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE=on DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE_RENDER_PRESET_SELECTION=render_preset_a`
  (presets `a` to `d`) pick which of NVIDIA's presets the game runs. They do not change Neural Forge. Switch
  the game's own DLSS 5 off while Neural Forge runs, or the frame gets both.
- **Not carried over from upstream DLSS5VKLayer:** see
  [UPSTREAM_PARITY.md](UPSTREAM_PARITY.md).

## More troubleshooting

- **The model does not run.** The Status tab's Model placement line, or Diagnose, gives the reason:
  no model extracted, a missing device feature, or a failed build (retried on its own). The full
  layer log (`NEURAL_FORGE_LOG=/path` in the launch options) has the `[native]` lines.
- **Model placement says "DLSS input identified, not holding".** The layer found the frame the game hands
  to DLSS but could not hold it, so the model runs on the finished frame; the reason follows on the same
  line (`preupscale_state=5` and `preupscale_reason=` in `neural-forge-cli shmctl status`). "The model is switched
  off" (F11 or the app) and "the Placement setting runs no pass before the upscaler" are your own choices, not a
  problem. The layer log
  says each reason once: `[preupscale] no hold inside DLSS's buffer (identification N): <why>; colour input ...`
  when the hold inside DLSS's own command buffer refuses the frame, and `[preupscale] <why>; not holding` when
  the hold that splits the game's submission does. Attach **Save report**'s zip and that log to an issue.
- **The game crashes with Neural Forge on.** Run it with `NEURAL_FORGE_LOG=/path` (and `NEURAL_FORGE_LOG_TIME=1`)
  in the launch options and attach that log. If Neural Forge itself failed, its last line starts with
  `[panic]` and says where. Otherwise the last milestone lines say how far it got: `first hold recorded into
  DLSS's command buffer` (the model's work was put into the game's own DLSS work), `the first hold recorded into
  DLSS's command buffer ran` (that work ran on the GPU), or `first split hold written back`. A
  `hold inside DLSS's command buffer available on this device` line alone means nothing was held yet.
- **Steam overlay problems.** The layer leaves the overlay's own small swapchain alone, and only the
  executables in `NEURAL_FORGE_TARGET_EXE` (which switching a game on sets) can use Neural Forge.
- **"Installing the layer for Steam games failed: refusing to overwrite ..."** A file Neural Forge
  installed under `~/.local/share/neural-forge` was changed, or something else put a file there, so
  the AppImage leaves it alone and games keep the layer installed before. The message names the file
  and what to do: move that file aside and start the AppImage again; when the install record
  (`~/.local/share/neural-forge/installation.json`) is missing, run
  `~/.local/share/neural-forge/bin/neural-forge-cli uninstall` first.
- **32-bit games** are not supported (see [Requirements](../README.md#requirements)).
