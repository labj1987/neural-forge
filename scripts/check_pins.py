#!/usr/bin/env python3
"""Reports the pinned versions Dependabot cannot see that are behind their upstream.

The pins: the native toolchain and OpenDLSS-NR in scripts/fetch-native-tools.sh, appimagetool and the
AppImage runtime in build-appimage.sh, and the Rust toolchain in rust-toolchain.toml. (GitHub Actions
are pinned by SHA and updated by Dependabot; Cargo crates by Dependabot.) Each is compared with
upstream's newest release, tag or commit through the GitHub API ($GITHUB_TOKEN raises its rate limit).

  check_pins.py               print a Markdown report; exit 0 when every pin is current, 10 when one
                              is behind, 1 when upstream could not be asked or named no version
                              to compare with (a pin that cannot be checked is never current)
  check_pins.py --self-test   check the pin parsing and version comparison offline

.github/workflows/pin-check.yml runs it weekly and keeps one issue up to date with the report.
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEHIND = 10


def read(rel):
    return (ROOT / rel).read_text()


def pins(fetch_text, build_text, toolchain_text):
    """[(name, pinned, kind, repo)]: kind is release, tag:<prefix> or commit."""
    out = []
    m = re.search(r"glslang/releases/download/([^/]+)/", fetch_text)
    out.append(("glslang", m.group(1), "release", "KhronosGroup/glslang"))
    m = re.search(r"Vulkan-Headers/archive/refs/tags/(v[0-9.]+)\.tar", fetch_text)
    out.append(("Vulkan-Headers", m.group(1), "tag:v1.", "KhronosGroup/Vulkan-Headers"))
    m = re.search(r"volk/archive/refs/tags/(vulkan-sdk-[0-9.]+)\.tar", fetch_text)
    out.append(("volk", m.group(1), "tag:vulkan-sdk-", "zeux/volk"))
    m = re.search(r"OpenDLSS-NR/archive/([0-9a-f]{40})\.tar", fetch_text)
    out.append(("OpenDLSS-NR", m.group(1), "commit", "maanHimself/OpenDLSS-NR"))
    m = re.search(r'^APPIMAGETOOL_VERSION="([^"]+)"', build_text, re.M)
    out.append(("appimagetool", m.group(1), "release", "AppImage/appimagetool"))
    m = re.search(r'^RUNTIME_VERSION="([^"]+)"', build_text, re.M)
    out.append(("AppImage type2-runtime", m.group(1), "release", "AppImage/type2-runtime"))
    m = re.search(r'^channel = "([^"]+)"', toolchain_text, re.M)
    out.append(("Rust toolchain", m.group(1), "release", "rust-lang/rust"))
    return out


def version_key(v):
    """A version as numbers for comparing (`v1.4.363`, `vulkan-sdk-1.4.357.0`, `16.6.0`, `20251108`)."""
    return tuple(int(n) for n in re.findall(r"\d+", v))


def newest(versions):
    """The highest of `versions` that is a plain release (no -rc, beta, continuous)."""
    plain = [v for v in versions if re.fullmatch(r"[A-Za-z-]*v?[0-9][0-9.]*", v)]
    return max(plain, key=version_key) if plain else None


def api(path):
    req = urllib.request.Request(f"https://api.github.com/{path}", headers={"Accept": "application/vnd.github+json", "User-Agent": "neural-forge-pin-check"})
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def latest(kind, repo):
    if kind == "commit":
        default = api(f"repos/{repo}")["default_branch"]
        return api(f"repos/{repo}/commits/{default}")["sha"]
    if kind == "release":
        releases = api(f"repos/{repo}/releases?per_page=100")
        return newest([r["tag_name"] for r in releases if not r["prerelease"] and not r["draft"]])
    prefix = kind.split(":", 1)[1]
    tags = api(f"repos/{repo}/tags?per_page=100")
    return newest([t["name"] for t in tags if t["name"].startswith(prefix)])


def state(kind, pinned, upstream):
    """behind, current, or unknown when upstream gave nothing to compare with."""
    if upstream is None:
        return "unknown"
    if kind == "commit":
        return "behind" if pinned != upstream else "current"
    return "behind" if version_key(upstream) > version_key(pinned) else "current"


def report(rows):
    lines = [
        "Pinned versions that Dependabot cannot see, compared with upstream by the weekly pin check",
        "(`scripts/check_pins.py`). Updating one: see the comment next to its pin; OpenDLSS-NR's patches must",
        "be rebased (`third_party/opendlss-nr/UPSTREAM.md`).",
        "",
        "| Pin | Pinned | Upstream | |",
        "|---|---|---|---|",
    ]
    shown = {"behind": "**behind**", "current": "current", "unknown": "**unknown**"}
    for name, pinned, upstream, st in rows:
        short = lambda v: v[:12] if v and re.fullmatch(r"[0-9a-f]{40}", v) else (v or "?")
        lines.append(f"| {name} | {short(pinned)} | {short(upstream)} | {shown[st]} |")
    return "\n".join(lines) + "\n"


def self_test():
    rows = pins(read("scripts/fetch-native-tools.sh"), read("build-appimage.sh"), read("rust-toolchain.toml"))
    assert [r[0] for r in rows] == ["glslang", "Vulkan-Headers", "volk", "OpenDLSS-NR", "appimagetool", "AppImage type2-runtime", "Rust toolchain"], rows
    assert all(r[1] for r in rows)
    assert version_key("vulkan-sdk-1.4.357.0") < version_key("vulkan-sdk-1.4.360.0")
    assert newest(["v1.4.363", "v1.4.370", "v1.4.99", "v1.5.0-rc1", "continuous"]) == "v1.4.370"
    assert newest(["continuous"]) is None
    assert state("release", "1.96.0", "1.97.0") == "behind" and state("release", "1.96.0", "1.96.0") == "current"
    assert state("commit", "a" * 40, "b" * 40) == "behind" and state("commit", "a" * 40, "a" * 40) == "current"
    # Upstream that names no version (only `continuous` or -rc tags) is not a current pin.
    assert state("commit", "a" * 40, None) == "unknown" and state("release", "1.96.0", None) == "unknown"
    text = report([("x", "1.0", "1.1", "behind"), ("y", "c" * 40, "c" * 40, "current"), ("z", "2.0", None, "unknown")])
    assert "| x | 1.0 | 1.1 | **behind** |" in text and "| y | cccccccccccc | cccccccccccc | current |" in text
    assert "| z | 2.0 | ? | **unknown** |" in text
    assert exit_status([("x", "1.0", "1.1", "behind")], []) == BEHIND
    assert exit_status([("y", "1.0", "1.0", "current")], []) == 0
    assert exit_status([("x", "1.0", "1.1", "behind"), ("z", "2.0", None, "unknown")], ["z: no version"]) == 1
    print("check_pins self-test ok")


def exit_status(rows, failed):
    """1 when any pin could not be checked (never reported as current), else BEHIND or 0."""
    if failed or any(r[3] == "unknown" for r in rows):
        return 1
    return BEHIND if any(r[3] == "behind" for r in rows) else 0


def main():
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    rows, failed = [], []
    for name, pinned, kind, repo in pins(read("scripts/fetch-native-tools.sh"), read("build-appimage.sh"), read("rust-toolchain.toml")):
        try:
            upstream = latest(kind, repo)
            if upstream is None:
                failed.append(f"{name}: upstream {repo} names no plain {kind.split(':')[0]} to compare with")
        except Exception as e:  # network, rate limit: reported, not guessed
            failed.append(f"{name}: {e}")
            upstream = None
        rows.append((name, pinned, upstream, state(kind, pinned, upstream)))
    print(report(rows), end="")
    if failed:
        print("\nCould not check upstream: " + "; ".join(failed))
    sys.exit(exit_status(rows, failed))


main()
