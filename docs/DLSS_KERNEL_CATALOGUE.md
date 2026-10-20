# DLSS kernel catalogue

2026-10-06. What DLSS's kernels are called, and what each word of their launch parameters holds,
for every DLSS version installed on the test machine. Observed from the layer's side of Vulkan with
`NEURAL_FORGE_PROBE_NGX=1`: `vkCreateCuFunctionNVX` gives the names, and the `[probe-ngx] layout`
lines (`preupscale.rs::probe_layout`) describe each 8-byte word of a launch's parameter buffer,
matched against the views the game registered through `VK_NVX_image_view_handle`. Nothing here
comes from reading NVIDIA's code, except the last section, which reads the PTX text DLSS-NR's DLL carries.

## Runs

| Game (engine) | DLSS SR on disk | DLSS DLL family | How it was reached | Log |
|---|---|---|---|---|
| GTA V Enhanced (RAGE) | 310.4 | `hiluma_*` | `gta-bench.sh`, frame generation on | `~/nf-spike/gta/layout-gta-1` |
| Shadow Warrior 3 (UE4) | 2.2.11 | `cuda_engine_*` (no suffix) | main menu | `~/nf-spike/runs/layout-sw3` |
| God of War (own) | 2.3.4 | `cuda_engine_*` (no suffix) | main menu | `~/nf-spike/runs/layout-gow` |
| GTA San Andreas DE (UE4) | 2.3.4 | `cuda_engine_*` (no suffix) | main menu | `~/nf-spike/runs/layout-gtasa` |
| Marvel's Spider-Man Remastered (own) | 3.7.10 | `cuda_engine_*_rel_<flags>` | Continue (save loaded) | `~/nf-spike/runs/layout-spiderman` |
| Black Myth: Wukong Benchmark Tool (UE5) | 3.1.30 | `cuda_engine_*_rel_<flags>` | benchmark menu | `~/nf-spike/runs/layout-wukong` |
| Crimson Desert (own, Streamline 2.14.1) | 310.9.1 | `hiluma_*` created; **Ray Reconstruction `rr2_*` launched** with it on, **`rrlite_*` + `cuda_dldn_engine_*` (Super Resolution)** with it off | `cd-launch.sh` + `cd-play.sh`; `~/scratch/rr/cd-rr.sh` (2026-10-09) | `~/nf-spike/cd/layout-cd2`, `~/nf-spike/runs/rr-probe-on`, `rr-probe-off` |
| Hogwarts Legacy (UE4) | 310.0 | `hiluma_*` (Super Resolution) with Ray Reconstruction off; **Ray Reconstruction `cuda_dldn_engine_swin_*` + `cuda_dldn_engine_hkpn_*` launched** with it on | `~/scratch/rr/hl-rr.sh` (2026-10-09) | `~/nf-spike/runs/hl-learn`, `hl-rr-on`, `hl-new-rr-on` |

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
| `rr2_*` (`rr2_enc0_kernel` ... `rr2_dec0_kernel_1`, `rr2_post_kernel`, `rr2_downsample_kernel_*`, `rr2_histogram_auto_exposure_basic_kernel`) | DLSS Ray Reconstruction | Crimson Desert with its Ray Reconstruction option on; `rr2_enc0_kernel` names a render-size colour image, depth and motion vectors, and the guide buffers (albedo, normals, roughness, hit distance) |
| `cuda_dldn_engine_swin_enc*`, `cuda_dldn_engine_swin_dec*`, `cuda_dldn_engine_hkpn_output_kernel_transformer` | DLSS Ray Reconstruction (310.0) | Hogwarts Legacy with its Ray Reconstruction option on; `cuda_dldn_engine_swin_enc0_kernel` names the colour input with two `B10G11R11` albedo images, `R8G8B8A8_SNORM` normals and an `R32_SFLOAT` image |
| `rrlite_*` | DLSS Super Resolution (310.9.1) | Crimson Desert with its Ray Reconstruction option off; no guide buffers |
| `cuda_dldn_engine_{luma_convert,reduce_sum,auto_exposure_copy,multiscale_downsample}_kernel` | helpers of both | launched beside `rrlite_*` (Crimson Desert) and beside the `swin`/`hkpn` network (Hogwarts Legacy), so they tell nothing apart |

## What the layer could stop guessing

Candidates only; nothing in the layer was changed for this catalogue.

1. **Ray Reconstruction has a real signature: `rr2_`, or the denoiser engine's `swin`/`hkpn`
   network.** Done in 3.1.4 (PRE_UPSCALER_DESIGN.md, "DLSS Ray Reconstruction"): `Kernel::of`
   classes `rr2_*`, `cuda_dldn_engine_swin_*` and `cuda_dldn_engine_hkpn_*` as Ray Reconstruction and
   nothing else (the `custom_block*` family is Frame Generation's, the other `cuda_dldn_engine_*`
   kernels are shared helpers), and while it runs nothing is held and the model runs after the
   upscaler. Crimson Desert with Ray Reconstruction off launches `rrlite_*`, Hogwarts Legacy
   `hiluma_*`, both held as Super Resolution.
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

## The network's own kernels: build 310.9.16 against 310.8.0 (2026-10-19)

The sections above are DLSS Super Resolution's kernels seen from Vulkan. This one is DLSS-NR's network
kernels, read from the PTX text `nvngx_dlssnr.dll` carries, to see what NVIDIA changed in 310.9.16 (its
release notes claim half the GPU overhead of the September build and 20% less memory at 4K) and whether
OpenDLSS-NR's generators (`scripts/ptx/`, `third_party/opendlss-nr/UPSTREAM.md`) lack any of it.

**The DLLs.** 310.9.16.0 (266,424,944 bytes, SHA-256 `2BC19BC0…E28789F`, the build `extract-model` reads
both networks from) and 310.8.0.0 (165,840,496 bytes, SHA-256 `E16BCF15…6E1FC8E`, the build the 2.x
helper imported, still in its old data directory on the test machine). No other 310.x copy of DLSS-NR is
on the machine: the games carry DLSS SR, RR and FG DLLs only, and the Proton prefixes no NGX downloads.

**How it was read.** Each DLL's `.data` holds 15 CUDA fat binaries (header magic `0xBA55ED50`, length
from the header). Each one copied out on its own and given to `cuobjdump -xptx all` (CUDA binutils, RE_TOOLKIT.md)
yields one PTX module (`.version 9.4`, `.target sm_120`). Only the PTX text was read: no host code, no cubin,
no resource beyond the weights `extract-model` already reads. Entries were split at `.entry` and compared
statement by statement (multi-line statements joined, comments dropped) three ways: as text; after
renumbering registers and labels in order of first use; and as the multiset of (opcode, immediates), which
is equal when only the order of the statements and the registers they use moved. Every entry that failed all
three was read by hand.

| | 310.8.0 | 310.9.16 |
|---|---|---|
| PTX modules | 15 | 15 |
| PTX text | 37.1 MB | 17.5 MB |
| Entries | 231 | 120 |

Every 310.9.16 entry is in 310.8.0 under the same name. The 111 that went are the non-FP8 twins of the
network kernels (`cc_tinlayout_fused_swin_1h_32_1` beside `cc_tinlayout_fused_swin_1h_32_1_fp8`, and so
on for every Swin, split-Swin, ViT and decoder kernel): 310.9.16 keeps only the E4M3 path, which is most of
the PTX that went. Kernel families: `cc_tinlayout_fused_{pre_block,swin,post_block}_*` (window blocks at
32/64/128/256 channels, with `_ds` downsampling and `_upsample` variants), `cc_split_swin_16h_*` (512
channels), `cc_vit_*` and `cc_vit_1d_*` (the bottleneck ViT), `cc_dec_input_upsample_1024_512_*`, and the
helpers `cg2r_*`, `cuda_capture_*`, `cuda_clear_view_kernel`, `cuda_font_kernel`, `cc_cb_clear`. Suffixes
`_chained`, `_wait`, `_tilesync`, `_publish` are the counter-chained variants.

### What changed, by class

| Class | Entries | What |
|---|---|---|
| Identical | 8 | `cg2r_*`, `cuda_capture_*`, `cuda_clear_view_kernel`, `cuda_font_kernel` |
| Registers renumbered | 3 | `cc_cb_clear`, `cc_vit_1d_repack_{1d_to_2d,2d_to_1d}_fp8` |
| Reordered | 75 | the same statements in another order with other registers (scheduling): every `cc_split_swin_16h_*`, `cc_dec_input_upsample_*`, the 32-, 64- and 128-channel Swin kernels other than `_ds`, the post blocks, `cc_vit_{attention,projection,ffn_*}_*`, `cc_vit_1d_ffn_expand_fp8` |
| Same value, other addressing | 3 | `cc_vit_qkv_{,chained_,wait_}fp8`: one store address is built from a register that already holds `64 b + 32 (tid.y & 1)`, shifted by 2, instead of `2 b + (tid.y & 1)` shifted by 7 |
| Register cap only | 10 | `cc_tinlayout_fused_swin_8h_256_8_*` other than `_ds` (the 256-channel window kernels): `.maxnreg` 168 to 128, and some address arithmetic moved |
| Semantic | 21 | the four changes below |

So 112 of 120 entries differ as text, 34 beyond order and register numbering, and 31 in what they compute
or how they are launched. (The third-party notes this survey started from, uzbekunknown/dlss-nr-on-intel
`notes/phase72-preset-b.md`, Apache-2.0, ideas only, were summarised as 54 of 120 differing from "310.8"; their
notes were not read here, so which build and which test that count used is not known.)

### The semantic changes, and OpenDLSS-NR

| Change in 310.9.16 | Entries | OpenDLSS-NR's counterpart | Here |
|---|---|---|---|
| **Acquire after a chained wait, a full fence before publishing.** In 310.8 thread 0 spins on the producer's counter with `ld.relaxed.gpu`, then `bar.sync`, and the workgroup reads the producer's rows: nothing orders those reads after the producer's release. 310.9.16 adds `fence.acquire.gpu` after that `bar.sync` on every waiting path, and `membar.gl` before the `bar.sync` that precedes the publishing `st.release.gpu`. | 10: `cc_vit_1d_{qkv,projection,ffn_contract}_*`, `cc_vit_1d_ffn_expand_{chained,publish}_fp8`, `cc_vit_1d_attention_chained_fp8` | `swin.py` `sync_wait` / `sync_signal`, used by every chained kernel: warp 0 polls with `ld.acquire.gpu` then `bar.sync`; producers `fence.acq_rel.gpu`, `bar.sync`, then `red.release.gpu` | Already so, on both sides |
| **A padding store loop halved.** The downsampling kernels zero their output's 16-byte channel groups at pixels outside the image. 310.8 looped over twice as many groups as the output has (bounds 128, 256, 512, 1024 for the 64-, 128-, 256-, 512-channel outputs, 64 in the block-0 kernel; for 128 channels the index split `/16`, `/64` where 310.9.16 splits `/8`, `/32`): it wrote zeros past the output's own groups (where that lands depends on the strides NVIDIA's host passes; not followed). 310.9.16's bounds are half: 64, 128, 256, 512, and 32. | 9: `cc_tinlayout_fused_swin_{1h_32_1,2h_64_2,4h_128_4,8h_256_8}_ds{,_wait}_fp8`, `cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8` | None: the field is computed whole from reflection-padded features, activations are zeroed once when allocated (`nr_graph.cpp`), and the pools (`block32_e4m3` POOL) write only their own output | Not applicable |
| **The ViT attention's tail, without a barrier.** The 1D attention zeroes the padded tokens of its output (tokens rounded up to 32). 310.8: after a `bar.sync`, the workgroup with `ctaid.y == ceil(ceil(tokens / 128) / 2) - 1`. 310.9.16: no barrier, every workgroup with `2 ctaid.y + 2 >= ceil(tokens / 128)` (the `ceil(tokens / 128)` bound the third-party notes report), and `.maxnreg` 168 to 255. On a grid of `ceil(ceil(tokens / 128) / 2)` workgroups in y both tests pick the last one at every token count: the change removes a barrier. | 2: `cc_vit_1d_attention{,_chained}_fp8` | `global_normalize_e4m3` writes zero codes for the padded tokens and the attention kernels correct for the padding in their sums; no workgroup stores the tail or waits for it | Already so (no barrier to remove) |
| **The control mask against the auto mask** (block 0, the input lanes 11-15 it builds from launch parameters). 310.8: with a mask bound, lanes 13 and 14 (NVIDIA's skin and structure) are dropped and lane 12 is structure x mask.b. 310.9.16: a new integer parameter combines structure x mask.b with lanes 13 and 14 while the auto mask is on: 2 min, 3 max, any other non-zero value the mean, 0 not at all. A second new integer, above 15, moves lanes 13 and 14 up one (13 becomes 0). Lane 11 is tone x mask.g in both. | 2: `cc_tinlayout_fused_pre_block_swin_1h_32_1{,_ds}_fp8` | Not OpenDLSS-NR's: Neural Forge's own preprocess (`native_preprocess.comp`) builds these lanes | Mode 2 only, by choice (NATIVE_BACKEND.md, "The control mask"). The lane move is not reproduced: it is off at NGX's defaults (both networks are bit-exact against NGX at presets 0 and 1, NATIVE_BACKEND.md "Against NGX (3.5.2)"), and what turns it on is not in the PTX |

None of the four is a correctness change OpenDLSS-NR needs. NVIDIA's halved GPU time is not in these
kernels either: the same entries do the same work. What 310.9.16 changed for the GPU is the network NGX runs
by default, the small one at preset 0 (3.6 ms against the full network's 5.1 ms at 1486x836, NATIVE_BACKEND.md),
and the PTX it no longer carries.

### Register caps (the one launch change)

310.9.16 lowers the 256-channel window kernels' cap from 168 to 128 registers and raises the ViT attention's
to 255: launch tuning, the arithmetic is unchanged. The same question for OpenDLSS-NR's kernels, from the
driver itself (each module JIT-compiled through `cuModuleLoadData` with and without a cap, read with
`cuFuncGetAttribute` and `cuOccupancyMaxActiveBlocksPerMultiprocessor` at the launch's block size and dynamic
shared memory; nothing launched), RTX 5070 (sm_120, 64K registers and 100 KB shared memory per SM):

| Kernel | Registers (spill) | Workgroups per SM | With a cap |
|---|---|---|---|
| `block32_e4m3_f*` (128 threads, 24 KB) | 115-125 | 4 | 4 at 96 (shared memory bounds it) |
| `qkv_e4m3_K*` (capped 80 upstream) | 80 | 5 | 5 at 64, 56 bytes spilled |
| `mlp_e4m3_K256` | 104 | 4 | 4 at 80-96 |
| `gemmv_e4m3_*` (48-94 KB dynamic) | 96-255 | 1-2 | the same at 96-168, up to 192 bytes spilled |
| `gemm2_e4m3_K32_*` | 31-44 | 10-11 | 11 at 40 |
| `gemmt_e4m3_*` (384 threads, 24 KB) | 40-54 | 3-4 | 4 at 40, no spill |
| `ffn_e4m3_C128_R4{,_proj}` (512 threads, 45 KB) | 88, 84 | **1** | **2 at 64**, 48-56 bytes spilled |
| `ffn_e4m3_C64_R4`, `C256_R3/R4` | 73; 80, 64 (16-64 spilled) | 3; 1 | shared memory or the block size bounds them |

Three caps that change the count were timed with the rig's C-API test (`graph back to back (secondary)`,
40 runs, frame d1w, 1486x836, the binaries alternated on the idle GPU, every run bit-exact):

| Build | Full network, min / median | Small network, min / median |
|---|---|---|
| As upstream | 5.52 / 5.60 ms | 2.92 / 2.96 ms |
| `ffn_e4m3_C128_R4` and `_proj` capped at 64 | **5.33 / 5.41 ms** | **2.87 / 2.90 ms** |
| and every `gemmt_e4m3` at 40 | 5.33 / 5.40 ms | 2.86 / 2.90 ms |
| and `gemm2_e4m3_K32_*` at 40 | 5.32 / 5.41 ms | 2.86 / 2.90 ms |

The 128-channel FFN cap is worth 0.20 ms on the full network (3.5%) and 0.06 ms on the small one (2%), with
the same head; the other two are inside the noise and were not kept. `build.rs` generates the two FFN kernels
with the cap (`third_party/opendlss-nr/UPSTREAM.md` notes the exception to upstream's invocations).

Against NVIDIA's runtime at the same size (NATIVE_BACKEND.md, "Against NGX (3.5.2)", the helper's busy time,
which includes its own overhead, so an upper bound for NGX): small network NGX 3.6 ms, native 2.96 before and
2.90 ms after; full network NGX 5.1 ms, native 5.60 before and 5.41 ms after. The full network is still about
0.3 ms behind.

### Repeating it

In each DLL, find every `0xBA55ED50` in the file and copy that fat binary out on its own: its 16-byte header
is the magic (u32), a version (u16), the header size (u16, 16) and the payload size (u64). Give each copy to
`cuobjdump -xptx all` in its own directory, split the `.ptx` files at `.entry` and compare as described above.
The scripts that did this were scratch and are not kept.
