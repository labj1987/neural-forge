# 2.0 screen check (for Alex, about 15 minutes)

What changed: when a game uses DLSS Super Resolution, the model now runs on DLSS's input
(the game's HDR picture at render resolution, no HUD) every frame, before DLSS upscales it.
Everything else uses the old path. Measured in GTA V Enhanced, DLSS Balanced:
1440p 64-66 fps (old path 61), 4K 39 fps (old 29), with DLSS frame generation 53 real / 159
shown (old 29 / 86).

Setup: Steam launch option for GTA V Enhanced is now just `NEURAL_FORGE_ENABLE=1 %command%`
(remove the Smooth Motion part). DLSS Balanced.

1. Frame generation off first. Story Mode, daylight, a street with signs and trees. Press F11 a
   few times: effect off/on. Does "on" look better (skin, foliage, edges, signs), and are the
   colours right?
2. Turn the camera quickly left and right. Any second copy of edges or text, smearing, or
   shimmer that isn't there with F11 off?
3. Same at night with headlights and street lights: blown-out lights, colour fringes, flicker?
4. Frame generation on in GTA's settings, at 2x, 3x and 4x: does it look and feel right with
   F11 on? (GTA didn't always switch frame generation on in the tests; the fps counter shows
   whether it did.)
5. Optional A/B with the old path: add `NEURAL_FORGE_PREUPSCALE=off` before `%command%`,
   relaunch, look at the same spots. Remove it again afterwards.

Tell Claude: better / same / worse for 1-3, how 2x/3x/4x look, and anything odd. If anything
looks wrong, `NEURAL_FORGE_PREUPSCALE=off` in the launch option puts the 1.x path back until
it's fixed.
