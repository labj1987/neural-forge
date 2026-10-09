# DLSS kernel catalogue

2026-10-06. What DLSS's kernels are called, and what each word of their launch parameters holds,
for every DLSS version installed on the test machine. Observed from the layer's side of Vulkan with
`NEURAL_FORGE_PROBE_NGX=1`: `vkCreateCuFunctionNVX` gives the names, and the `[probe-ngx] layout`
lines (`preupscale.rs::probe_layout`) describe each 8-byte word of a launch's parameter buffer,
matched against the views the game registered through `VK_NVX_image_view_handle`. Nothing here
comes from reading NVIDIA's code.

## Runs

| Game (engine) | DLSS SR on disk | DLSS DLL family | How it was reached | Log |
|---|---|---|---|---|
| GTA V Enhanced (RAGE) | 310.4 | `hiluma_*` | `gta-bench.sh`, frame generation on | `~/nf-spike/gta/layout-gta-1` |
| Shadow Warrior 3 (UE4) | 2.2.11 | `cuda_engine_*` (no suffix) | main menu | `~/nf-spike/runs/layout-sw3` |
| God of War (own) | 2.3.4 | `cuda_engine_*` (no suffix) | main menu | `~/nf-spike/runs/layout-gow` |
| GTA San Andreas DE (UE4) | 2.3.4 | `cuda_engine_*` (no suffix) | main menu | `~/nf-spike/runs/layout-gtasa` |
| Marvel's Spider-Man Remastered (own) | 3.7.10 | `cuda_engine_*_rel_<flags>` | Continue (save loaded) | `~/nf-spike/runs/layout-spiderman` |
| Black Myth: Wukong Benchmark Tool (UE5) | 3.1.30 | `cuda_engine_*_rel_<flags>` | benchmark menu | `~/nf-spike/runs/layout-wukong` |
| Crimson Desert (own, Streamline 2.14.1) | 310.9.1 | `hiluma_*` created, **Ray Reconstruction `rr2_*` launched** | `cd-launch.sh` + `cd-play.sh` | `~/nf-spike/cd/layout-cd2` |

Each game used the DLL in its own folder: none of the logs shows Proton replacing it. Cyberpunk
2077 and Resident Evil Requiem were not installed. Labels in packed words come from the layer's
handle-to-image map. A 32-bit handle reused for another view can keep an older label, which is the
likely reason an output-size RG16F image is labelled `mvec` beside depth in some `mvlo` games.

## The input kernel, by family

The colour input is **the first view in the block** in every game. Handles are packed by DLSS
family, not by engine:

| Family | Versions seen | Input kernel | Block | Handle form | Views, in order |
|---|---|---|---|---|---|
| `hiluma_*` | 310.4 | `hiluma_engine_input_<depth>_<mv>_<range>_<model>_rel` | 248 bytes (31 words) | one 64-bit handle per word | 21 colour, 22 output-size RGBA16F, 23 motion vectors, 24 depth, 25-27 three 1x1 RGBA32F (exposure), 28 640x384 RGBA16F (network tile) |
| `cuda_engine_*` | 2.2.11, 2.3.4 | `cuda_engine_input_kernel` | 360 bytes (45 words) | two 32-bit handles per word, low half first | 36 [colour \| output-size RGBA16F], 37 [motion vectors \| depth], 38 [two more RG16F], 40 [mask \| 1x1 exposure] (God of War), 41/43 output-size images |
| `cuda_engine_*_rel_<flags>` | 3.7.10 | `cuda_engine_input_kernel_rel_hdr_mvdiff_mvlo` | 352 bytes (44 words) | two 32-bit per word | 35 [colour \| output-size], 37 [depth \| motion vectors], 38 motion vectors, 39-40 1x1 R16F exposure, 42 output-size |
| `cuda_engine_*_rel_<flags>` | 3.1.30 | `cuda_engine_input_kernel_rel_hdr_mvdiff_mvhi` | 408 bytes (51 words) | two 32-bit per word | 40 [colour \| output-size], 41 [motion vectors \| depth], 42 [two RG16F], 45 [two 1x1 R16F exposure], 49 output-size |

The scalar words, by their values (render 1485x836 to 2560x1440 unless noted). These are
readings, not specifications:

| Meaning (as observed) | `hiluma_*` 310.4 | `cuda_engine_*` 2.x (GTA SA) | 3.7.10 (Spider-Man, 1712x960) | 3.1.30 (Wukong, 1488x836) |
|---|---|---|---|---|
| Output size | | w0, w3 | w0, w3 | w0, w3 |
| Padded sizes (2560x1536, 1280x768) | | w1, w2 | w1, w2 | w1, w2 |
| 1 / output size | w15-17 | w4, w18 | w4, w19 | w4, w23 |
| Render size - 1 | w10, w12, w14 | w5 | w5 | w5 |
| Render size | | w16 | w17 | w21 |
| Output / render ratio | w1 | | | |
| Render / output ratio | w5, w6 | w14, w15, w17 | w15, w16, w18 | w16, w22 |
| Network tile width (640) | w8 | | | |

## Kernel names say how the game set DLSS up

A 310.x DLL creates every variant of its input and output kernels at start, then launches one. The
name spells out the game's creation flags:

- `hiluma_engine_{input,output}_{depthinv,depthreg}_{mvlo,mvhi}_{hdr,ldr}_{v1,v2}_rel`, plus
  output `..._max_v2_rel`: depth inverted or regular, motion vectors at render or output
  resolution, HDR or LDR colour, model generation.
- `cuda_engine_input_kernel_rel_{hdr,ldr}_{colvar,mvdiff}_{mvhi,mvlo}` and
  `cuda_engine_output_kernel_rel_{hdr,ldr}_gauss{3x3,5x5}` (3.x); the 2.x DLLs launch the bare
  `cuda_engine_input_kernel`/`cuda_engine_output_kernel`.

GTA V launches `hiluma_engine_input_depthinv_mvlo_hdr_v2_rel`; Spider-Man
`..._rel_hdr_mvdiff_mvlo`; Wukong `..._rel_hdr_mvdiff_mvhi`. Crimson Desert's created set includes
`hiluma_engine_output_depthreg_mvhi_ldr_v2_rel`, an **LDR** variant.

## Which family is which

| Kernels | Belongs to | Evidence |
|---|---|---|
| `hiluma_engine_*`, `cuda_engine_*`, `dltss_*`, `cuda_luma_convert_kernel`, `cuda_reduce_sum_kernel`, `cuda_*exposure*`, `cuda_upscale_sum_kernel`, `cuda_downsample_kernel` | DLSS Super Resolution | every SR game, menu or play |
| `main_kernel` (many buffer sizes), `k_conv_fp16_nhwc`, `k_pooling`, `k_upscale`, `k_element_wise`, `Kernel_*` (optical flow, warp, blend), **`k_initial_merge`, `custom_block*`, `k_central_block`, `custom_upsample*`** | DLSS Frame Generation | GTA V with frame generation on, which has no Ray Reconstruction, launches all of them |
| `rr2_*` (`rr2_enc0_kernel` ... `rr2_dec0_kernel_1`, `rr2_post_kernel`, `rr2_downsample_kernel_*`, `rr2_histogram_auto_exposure_basic_kernel`) | DLSS Ray Reconstruction | Crimson Desert in play with ray tracing; `rr2_enc0_kernel` names a render-size colour image, depth and motion vectors |

## What the layer could stop guessing

Candidates only; nothing in the layer was changed for this catalogue.

1. **Ray Reconstruction has a real signature: `rr2_`.** `preupscale.rs::Kernel::of` classes
   `custom_block*`, `k_central_block`, `k_initial_merge` and `custom_upsample*` as Ray
   Reconstruction, but those are Frame Generation's network. That misattribution is why gating by
   kernel name had to be switched off (`GATE_BY_KERNEL_NAME`, 2026-10-03: "Crimson Desert's SR
   launches custom_block*"). With the gate off, **Crimson Desert with Ray Reconstruction on is held
   at `rr2_enc0_kernel`'s colour image, that is, RR's input**, which the rule as first designed
   (PRE_UPSCALER_DESIGN.md, "DLSS Ray Reconstruction") says must never be held. Recognising RR by `rr2_` and re-enabling the gate would restore that rule. It would also
   change what Crimson Desert looks like with RR on (the model would stop running there), so it is
   the maintainer's call. The obvious alternative is to keep the hold and treat Crimson Desert's RR as
   validated by play.
2. **Handle packing by family.** `launch_kernel` tries every word whole and as two halves. The
   catalogue says which form a kernel uses: `hiluma_*` whole, `cuda_engine_*` halves. The current
   rule already gets both right; reading the form from the name would only remove the chance of a
   false half-match.
3. **"The first colour candidate in parameter order is the colour input"** (`input_launch`) is now
   observed fact for every family here, not a heuristic from one game.
4. **HDR or LDR from the kernel name.** The layer encodes the held frame as scene-linear HDR. An
   `_ldr_` input kernel means the game created DLSS without `IsHDR`, so the colour is already
   display-referred. No game held today launches an LDR input kernel. Crimson Desert creates LDR
   variants, so its SR path (with RR off) would be the first to check.
5. **Exposure.** The input kernel names its 1x1 exposure images: three RGBA32F in 310.4, R16F in
   2.x/3.x. `exposure_images` and `registered_exposure_input` find them by size and format; reading
   the input launch's own 1x1 views would remove the choice between several.

## Repeating it

Install the layer, then either run `gta-bench.sh` with `NEURAL_FORGE_PROBE_NGX=1`, or restart
Steam's `steam-relaunch` unit with that variable in its environment so every game launched through
Steam carries it (the layer still only runs where the game's launch options set
`NEURAL_FORGE_ENABLE=1`). Restart Steam without it afterwards.

```bash
grep -F '[probe-ngx] layout' launch.log
```
