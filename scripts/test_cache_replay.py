#!/usr/bin/env python3
"""Tests for scripts/cache_replay.py and scripts/cache_common.py on synthetic dumps.

Run with `python3 scripts/test_cache_replay.py`. Needs numpy; skips with a message without it."""

import contextlib
import dataclasses
import io
import json
import os
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # the tests say so and skip
    np = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if np is not None:
    import cache_common as cc
    import cache_replay as cr

NEEDS_NUMPY = "numpy is not installed (apt-get install -y python3-numpy); cache replay tests skipped"


def carried(dump, t, lag, p, rev=True):
    """Hold t's shown edits per mode, carried from hold t - lag along the layer's chain (the
    accumulated field's recurrence, cc.acc_step), with the frames used."""
    frames = cr.Frames(dump, rev, p, lag + 1)
    pos, on = cr.carried_position(frames, t, lag, p)
    shown, _ = cr.shown_edits(frames[t - lag], frames[t], pos, on, p)
    return shown, frames[t]


def rewrite_manifest(path, change):
    """Applies `change` to the dump's manifest (a dict) and writes it back."""
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    change(m)
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh)


def kept(shown, dst, p, mask):
    return 100 * (1 - cc.pair_metrics(shown, dst, p, mask)["pic"] / cc.base_metric(dst, mask))


@unittest.skipIf(np is None, NEEDS_NUMPY)
class CacheReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dirs, cls.info = {}, {}
        for kind in ("still", "pan", "depth-edge"):
            path = os.path.join(cls.tmp.name, kind)
            cls.info[kind] = cr.make_synthetic(path, kind)
            cls.dirs[kind] = path

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_loader_round_trip(self):
        holds, white, _ = cr.synthetic_holds("depth-edge")
        dump = cc.load_dump(self.dirs["depth-edge"])
        self.assertEqual((len(dump), dump.width, dump.height, dump.paper_white), (6, 96, 64, white))
        for k in (0, 3):
            got, want = dump.hold(k), holds[k]
            half = lambda x: x.astype(np.float16).astype(np.float32)  # noqa: E731
            np.testing.assert_array_equal(got.colour, half(want.colour))
            np.testing.assert_array_equal(got.answer, half(want.answer))
            np.testing.assert_array_equal(got.mvec, half(want.mvec))
            np.testing.assert_array_equal(got.depth, want.depth)
            np.testing.assert_array_equal(got.jitter, want.jitter)
            self.assertEqual(got.exposure, want.exposure)

    def test_loader_rejects_wrong_size(self):
        with tempfile.TemporaryDirectory() as d:
            holds, _, _ = cr.synthetic_holds("still", w=16, h=8, n=2)
            cc.write_dump(d, holds)
            with open(os.path.join(d, "hold-01", "mvec.rg16f"), "ab") as fh:
                fh.write(b"\0\0")
            with self.assertRaises(ValueError):
                cc.load_dump(d).hold(1)

    def test_reverse_z_inferred(self):
        for kind in ("still", "depth-edge"):
            dump = cc.load_dump(self.dirs[kind])
            rev, why = cr.resolve_reverse_z(dump, "auto")
            self.assertTrue(rev, f"{kind}: {why}")
        flipped = [1.0 - cc.load_dump(self.dirs["still"]).hold(0).depth]
        self.assertFalse(cc.infer_reverse_z(flipped)[0])

    def test_cleared_depth_alone_tells_nothing(self):
        # A loading screen's all-zero depth is no evidence (the layer's cache::DEPTH_GEOMETRY_MIN);
        # with one real frame beside it the real frame decides.
        blank = np.zeros((8, 16), np.float32)
        self.assertIsNone(cc.infer_reverse_z([blank])[0])
        real = cc.load_dump(self.dirs["depth-edge"]).hold(0).depth
        self.assertTrue(cc.infer_reverse_z([blank, real])[0])
        self.assertFalse(cc.infer_reverse_z([blank, 1.0 - real])[0])

    def test_null_fields_load_and_refresh(self):
        """A manifest's null jitter or exposure and a hold without vectors or depth load (no NaN, no
        TypeError), and the cadence refreshes on such holds as the layer does."""
        with tempfile.TemporaryDirectory() as d:
            holds, _, _ = cr.synthetic_holds("pan", w=32, h=16, n=6)
            cc.write_dump(d, holds)

            def change(m):
                m["frames"][1]["jitter"] = None
                m["frames"][3]["exposure"] = None
                m["frames"][5]["mvec_valid"] = False
            rewrite_manifest(d, change)
            dump = cc.load_dump(d)
            h1, h3, h5 = dump.hold(1), dump.hold(3), dump.hold(5)
            self.assertFalse(h1.jitter_known)
            self.assertTrue(np.isfinite(h1.jitter).all())
            self.assertEqual(h3.exposure, 0.0)
            self.assertFalse(h5.mvec_valid)
            self.assertFalse(h5.mvec.any())
            p = cc.Params()
            frames = cr.Frames(dump, True, p, 3)
            got = [(t, refresh) for t, _, refresh, _, _, _ in cr.run_cadence(dump, frames, cr.Cadence(interval=4), p)]
            # 0 refreshes; 1 has no jitter: refreshes and stores nothing, so 2 refreshes; 3 is left out
            # (unusable exposure); 4 refreshes after it; 5 has no vectors: refreshes.
            self.assertEqual(got, [(0, True), (1, True), (2, True), (4, True), (5, True)])
            r = cr.replay(dump, [1], p, True)["results"][1]["cache"]
            self.assertEqual(r["samples"], 0, "no hold here could be cached from the one before")

        with tempfile.TemporaryDirectory() as d:
            holds, _, _ = cr.synthetic_holds("depth-edge", w=32, h=16, n=3)
            cc.write_dump(d, holds)
            rewrite_manifest(d, lambda m: m["frames"][2].update(depth_valid=False))
            dump = cc.load_dump(d)
            self.assertIsNone(dump.hold(2).depth)
            frames = cr.Frames(dump, True, cc.Params(), 3)
            got = [refresh for _, _, refresh, _, _, _ in cr.run_cadence(dump, frames, cr.Cadence(interval=4), cc.Params())]
            self.assertEqual(got, [True, False, True], "another presence of depth than the refresh's refreshes")

    def test_export_carry_ends_where_the_carry_from_hold_0_ends(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
            holds, _, _ = cr.synthetic_holds("pan", w=32, h=16, n=6)
            cc.write_dump(d, holds)
            rewrite_manifest(d, lambda m: m["frames"][3].update(exposure=0.0))
            dump = cc.load_dump(d)
            self.assertEqual(cr.export_carry(dump, cc.Params(), True, out), 2)
            self.assertEqual(sorted(f for f in os.listdir(out) if f.startswith("cache-")), ["cache-01.f32", "cache-02.f32"])
            self.assertEqual(cr.export_carry(cc.load_dump(self.dirs["pan"]), cc.Params(), True, out), 5)

    def test_the_recurrence_accumulates_the_motion_back_to_the_refresh(self):
        """The shipped chain over several holds (acc_prev read): in the pan every hold moves 2.5 px right
        and 1 down, so three holds after the refresh the field is three steps, wherever the chain stays
        on screen, and NaN where it left."""
        dump = cc.load_dump(self.dirs["pan"])
        p = cc.Params()
        frames = cr.Frames(dump, True, p, 4)
        acc = None
        for k in (3, 4, 5):
            acc = cc.acc_step(acc, frames[k], frames[k - 1].jitter, p)
        h, w = dump.height, dump.width
        on = np.isfinite(acc).all(axis=-1)
        np.testing.assert_allclose(acc[on], np.broadcast_to([7.5, 3.0], acc[on].shape), atol=1e-5)
        self.assertTrue(on[: h - 4, : w - 8].all())
        self.assertFalse(on[-1, -1], "the bottom right corner's chain leaves the frame")
        pos, ok = cr.carried_position(frames, 5, 3, p)
        np.testing.assert_array_equal(ok, on)
        want = cc.pixel_centres(h, w) + np.array([7.5, 3.0], np.float32) + (frames[2].jitter - frames[5].jitter)
        np.testing.assert_allclose(pos[on], want[on], atol=1e-5)

    def test_still_keeps_everything(self):
        dump = cc.load_dump(self.dirs["still"])
        r = cr.replay(dump, [2], cc.Params(), True)["results"][2]
        self.assertGreater(r["cache"]["kept"], 99.9)
        self.assertLess(r["cache"]["err"], 1e-3)
        self.assertLess(r["none"]["kept"], 1e-6)

    def test_history_read_keeps_a_pixel_scale_edit(self):
        # A checkerboard edit read at a sub-pixel offset: the strongest tap's own edit comes through
        # whole, where a bilinear blend of the four taps keeps a fraction of it.
        dump = cc.load_dump(self.dirs["still"])
        p = cc.Params()
        f = cc.prepare(dump.hold(0), dump.paper_white, True, p)
        h, w = f.lin.shape
        ys, xs = np.mgrid[0:h, 0:w]
        chk = (((xs + ys) % 2) * 0.6 - 0.3).astype(np.float32)
        src = dataclasses.replace(f, edit=np.repeat(chk[..., None], 3, -1))
        pos = (cc.pixel_centres(h, w) + np.array([0.4, -0.3], np.float32)).astype(np.float32)
        read = cc.read_history(src, pos, f.lin, p)["edit"]
        inner = (slice(2, h - 2), slice(2, w - 2))
        np.testing.assert_array_equal(read[inner], src.edit[inner])
        blend = cc.sample_bilinear(src.edit, pos)
        self.assertLess(float(np.abs(blend[inner]).mean()), 0.3 * float(np.abs(src.edit[inner]).mean()))

    def test_pan_stays_aligned(self):
        dump = cc.load_dump(self.dirs["pan"])
        p = cc.Params()
        h, w = dump.height, dump.width
        for lag in (1, 3):
            shown, dst = carried(dump, 5, lag, p)
            mask = np.zeros((h, w), bool)
            mask[2:h - 2 - lag * 1 - 2, 2:w - 2 - 3 * lag - 2] = True  # away from where content enters
            for m in ("naive", "cache"):
                self.assertGreater(kept(shown[m], dst, p, mask), 99.0, f"lag {lag} {m}")
        # Without the change of jitter the carry lands half a pixel off: measurably worse.
        shown_nj, dst = carried(dump, 5, 1, cc.Params(jitter=False))
        shown_j, _ = carried(dump, 5, 1, p)
        mask = np.zeros((h, w), bool)
        mask[4:h - 6, 4:w - 8] = True
        self.assertLess(kept(shown_nj["naive"], dst, p, mask), kept(shown_j["naive"], dst, p, mask) - 1.0)

    def test_depth_edge_rejects_the_uncovered_background(self):
        dump = cc.load_dump(self.dirs["depth-edge"])
        info = self.info["depth-edge"]
        p = cc.Params()
        h, w = dump.height, dump.width
        t, lag = 5, 2
        shown, dst = carried(dump, t, lag, p)
        ys, xs = np.mgrid[0:h, 0:w] + 0.5
        r0, r1 = info["box_rows"]
        then, now = info["box_x"][t - lag], info["box_x"][t]
        # Covered by the box at t - lag, background at t (2 px margins off the edges).
        mask = (ys >= r0 + 2) & (ys < r1 - 2) & (xs >= then + 2) & (xs < now - 2)
        self.assertGreater(mask.sum(), 50)
        err = {m: float(np.abs((shown[m] - dst.edit) @ cc.LUMA)[mask].mean()) for m in ("naive", "cache")}
        self.assertGreater(err["naive"], 0.5, err)
        self.assertLess(err["cache"], 0.5 * err["naive"], err)
        whole = cr.replay(dump, [lag], p, True)["results"][lag]
        self.assertLess(whole["cache"]["worse"], whole["naive"]["worse"])

    def test_self_check_passes(self):
        for kind in ("still", "pan", "depth-edge"):
            dump = cc.load_dump(self.dirs[kind])
            out = io.StringIO()
            self.assertTrue(cr.self_check(dump, cc.Params(), True, out=out), out.getvalue())

    def test_lag_zero_is_identity_for_every_mode(self):
        dump = cc.load_dump(self.dirs["depth-edge"])
        r = cr.replay(dump, [0], cc.Params(max_luma_edit=0.0), True)["results"][0]
        for m in ("naive", "cache"):
            self.assertLess(r[m]["err"], 1e-6, m)
            self.assertGreater(r[m]["kept"], 99.999, m)

    def test_cadence_base_is_the_lag_replay(self):
        """At interval 2 every cached hold is a lag-1 carry from the refresh before it, so the cadence
        run's cached holds give the lag replay's lag-1 numbers on the same holds."""
        dump = cc.load_dump(self.dirs["depth-edge"])
        p = cc.Params()
        frames = cr.Frames(dump, True, p, 3)
        h, w = dump.height, dump.width
        for t, since, refresh, shown, dst, _ in cr.run_cadence(dump, frames, cr.Cadence(interval=2, guided=False), p):
            self.assertEqual(refresh, t % 2 == 0)
            self.assertEqual(since, t % 2)
            if refresh:
                np.testing.assert_array_equal(shown, dst.edit)
                continue
            acc = cc.acc_step(None, dst, frames[t - 1].jitter, p)
            pos, on = cc.chained_position(acc, dst, frames[t - 1], p)
            want = cc.carry(frames[t - 1], dst, pos, on, p)["cache"]
            np.testing.assert_allclose(shown, want, atol=1e-6)
        self.assertEqual((h, w), shown.shape[:2])

    def test_adaptive_cadence_refreshes_every_hold_once_still(self):
        """The regime: 3 still holds and every hold after them refreshes; a moving camera keeps the
        interval. The layer's cache::Regime mirrors this."""
        p = cc.Params()
        for kind, want in (("still", [True, False, True, True, True, True]), ("pan", [True, False] * 3)):
            dump = cc.load_dump(self.dirs[kind])
            frames = cr.Frames(dump, True, p, 3)
            got = [refresh for _, _, refresh, _, _, _ in cr.run_cadence(dump, frames, cr.Cadence(interval=2, adaptive=True), p)]
            self.assertEqual(got, want, kind)
        r = cr.Regime(cr.Cadence(interval=2, adaptive=True))
        for counts in [(60, 100)] * 2 + [(10, 100)] + [(60, 100)] * 2:
            r.observe(counts)
        self.assertFalse(r.still, "a moving hold restarts the count")
        r.observe((60, 100))
        self.assertEqual(r.interval(), 1)
        for _ in range(30):
            r.observe((10, 100))
            r.observe((60, 100))
        self.assertTrue(r.still, "a still hold restarts the count out of it")
        for _ in range(19):
            r.observe((10, 100))
        self.assertTrue(r.still)
        r.observe((10, 100))
        self.assertEqual(r.interval(), 2)

    def test_cadence_metrics_has_a_pulse_per_hold_since_the_refresh(self):
        dump = cc.load_dump(self.dirs["pan"])
        r = cr.cadence_metrics(dump, cr.Cadence(interval=3), cc.Params(), True)
        self.assertEqual(set(r["pulse"]), {"0", "1", "2"})

    def test_guided_filter_keeps_a_linear_edit(self):
        rng = np.random.default_rng(1)
        guide = rng.normal(size=(20, 30)).astype(np.float32)
        edit = np.stack([0.5 * guide + 0.1, -0.2 * guide, 0 * guide + 0.3], -1).astype(np.float32)
        out = cr.guided_fast(guide, edit, 1e-6)
        self.assertLess(float(np.abs(out - edit)[2:-2, 2:-2].max()), 1e-2)

    def test_command_line(self):
        out = os.path.join(self.tmp.name, "r.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cr.main([self.dirs["pan"], "--lags", "1", "2", "--json", out, "--quiet"]), 0)
            self.assertEqual(cr.main([self.dirs["still"], "--self-check"]), 0)
            self.assertEqual(cr.main([self.dirs["pan"], "--cadence", "2", "--json", out + ".cad"]), 0)
        with open(out, encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(set(doc["results"]), {"1", "2"})
        self.assertEqual(set(doc["results"]["1"]), set(cr.MODES))
        with open(out + ".cad", encoding="utf-8") as fh:
            self.assertEqual([r["label"] for r in json.load(fh)["runs"]], ["i2-base"])


if __name__ == "__main__":
    unittest.main()
