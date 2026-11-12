#!/usr/bin/env python3
"""Reports, and updates, the pinned versions Dependabot cannot see.

The pins: the native toolchain in scripts/fetch-native-tools.sh, cargo-deny in scripts/cargo-deny.sh,
appimagetool and the AppImage runtime in build-appimage.sh, and the Rust toolchain in
rust-toolchain.toml. (GitHub Actions are pinned by SHA and updated by Dependabot; Cargo crates by
Dependabot.) Each is compared with upstream's newest release or tag through the GitHub API
($GITHUB_TOKEN raises its rate limit). Vulkan-Headers and volk follow the Vulkan SDK releases
(vulkan-sdk-* tags), not Khronos's weekly spec-update tags: both are pinned to the same SDK, the newest
one both repositories have tagged, so the headers and the loader always come from one SDK.

  check_pins.py               print a Markdown report; exit 0 when every pin is current, 10 when one
                              is behind (or Vulkan-Headers and volk name different SDKs), 1 when
                              upstream could not be asked or named no version to compare with (a pin
                              that cannot be checked is never current)
  check_pins.py --bump        the same check, then move every pin that is behind to upstream's version:
                              the new archive is downloaded, its SHA-256 computed (and compared with
                              upstream's published .sha256 where there is one), and every place the pin
                              is named is rewritten. One line per bump on stdout. Exit 0 when something
                              was bumped, 3 when nothing was behind, 1 on any failure (nothing written)
  check_pins.py --self-test   check the parsing, comparison and rewriting offline

.github/workflows/pin-check.yml runs it weekly: it bumps what is behind, runs the full build and check
on the bump, and lands it on main when that passes; one issue carries the report.
"""
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEHIND = 10
NOTHING_TO_BUMP = 3

# Every file a pin is named in, by the short name the rewrites use.
FILES = {
    "fetch": "scripts/fetch-native-tools.sh",
    "build": "build-appimage.sh",
    "toolchain": "rust-toolchain.toml",
    "deny": "scripts/cargo-deny.sh",
    "attribution": "ATTRIBUTION.md",
}
SDK = "Vulkan SDK (Vulkan-Headers = volk)"


def read(rel):
    return (ROOT / rel).read_text()


def read_all():
    return {key: read(rel) for key, rel in FILES.items()}


def one(pattern, text, what, flags=0):
    m = re.search(pattern, text, flags)
    if not m:
        raise ValueError(f"{what}: pattern {pattern!r} not found")
    return m.group(1)


def pins(texts):
    """[(name, pinned, kind, repo)]: kind is release, tag:<prefix> or sdk (repo then names both repos)."""
    fetch, build, toolchain, deny = texts["fetch"], texts["build"], texts["toolchain"], texts["deny"]
    return [
        ("glslang", one(r"glslang/releases/download/([^/]+)/", fetch, "glslang"), "release", "KhronosGroup/glslang"),
        ("Vulkan-Headers", one(r"Vulkan-Headers/archive/refs/tags/(vulkan-sdk-[0-9.]+)\.tar", fetch, "Vulkan-Headers"), "sdk", "KhronosGroup/Vulkan-Headers zeux/volk"),
        ("volk", one(r"volk/archive/refs/tags/(vulkan-sdk-[0-9.]+)\.tar", fetch, "volk"), "sdk", "KhronosGroup/Vulkan-Headers zeux/volk"),
        ("cargo-deny", one(r'^CARGO_DENY_VERSION="([^"]+)"', deny, "cargo-deny", re.M), "release", "EmbarkStudios/cargo-deny"),
        ("appimagetool", one(r'^APPIMAGETOOL_VERSION="([^"]+)"', build, "appimagetool", re.M), "release", "AppImage/appimagetool"),
        ("AppImage type2-runtime", one(r'^RUNTIME_VERSION="([^"]+)"', build, "type2-runtime", re.M), "release", "AppImage/type2-runtime"),
        ("Rust toolchain", one(r'^channel = "([^"]+)"', toolchain, "Rust toolchain", re.M), "release", "rust-lang/rust"),
    ]


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


def tags(repo, prefix):
    """Every tag of `repo` starting with `prefix` (matching-refs lists them all, unlike /tags' first page)."""
    return [ref["ref"].removeprefix("refs/tags/") for ref in api(f"repos/{repo}/git/matching-refs/tags/{prefix}")]


def newest_common_sdk(per_repo):
    """The newest vulkan-sdk-* tag every repository has, so headers and loader come from one SDK."""
    common = set.intersection(*(set(t) for t in per_repo))
    return newest(sorted(common))


def latest(kind, repo, cache):
    if kind not in cache:
        if kind == "release":
            releases = api(f"repos/{repo}/releases?per_page=100")
            return newest([r["tag_name"] for r in releases if not r["prerelease"] and not r["draft"]])
        if kind == "sdk":
            cache[kind] = newest_common_sdk([tags(r, "vulkan-sdk-") for r in repo.split()])
        else:
            cache[kind] = newest(tags(repo, kind.split(":", 1)[1]))
    return cache[kind]


def state(kind, pinned, upstream):
    """behind, current, or unknown when upstream gave nothing to compare with."""
    if upstream is None:
        return "unknown"
    return "behind" if version_key(upstream) > version_key(pinned) else "current"


def sdk_row(rows):
    """The row that keeps Vulkan-Headers and volk on one SDK: current when both name the same tag."""
    headers = next(r[1] for r in rows if r[0] == "Vulkan-Headers")
    volk = next(r[1] for r in rows if r[0] == "volk")
    return (SDK, f"{headers} = {volk}", None, "current" if headers == volk else "mismatch")


def report(rows):
    lines = [
        "Pinned versions that Dependabot cannot see, compared with upstream by the weekly pin check",
        "(`scripts/check_pins.py`), which also bumps the ones that are behind and lands the bump on main",
        "once the full build and check passes on it.",
        "",
        "| Pin | Pinned | Upstream | |",
        "|---|---|---|---|",
    ]
    shown = {"behind": "**behind**", "current": "current", "unknown": "**unknown**", "mismatch": "**mismatch**"}
    for name, pinned, upstream, st in rows:
        upstream_shown = "" if name == SDK else (upstream or "?")
        lines.append(f"| {name} | {pinned} | {upstream_shown} | {shown[st]} |")
    return "\n".join(lines) + "\n"


def exit_status(rows, failed):
    """1 when any pin could not be checked (never reported as current), else BEHIND or 0."""
    if failed or any(r[3] == "unknown" for r in rows):
        return 1
    return BEHIND if any(r[3] in ("behind", "mismatch") for r in rows) else 0


# ---- Bumping -----------------------------------------------------------------------------------


def sha256_url(url):
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "neural-forge-pin-check"})
    with urllib.request.urlopen(req, timeout=300) as r:
        while chunk := r.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def text_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "neural-forge-pin-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def replace(texts, key, old, new, count=1):
    """Replace `old` by `new` in texts[key], which must hold it exactly `count` times."""
    found = texts[key].count(old)
    if found != count:
        raise ValueError(f"{FILES[key]}: expected {count} of {old!r}, found {found}")
    texts[key] = texts[key].replace(old, new)


def glslang_url(v):
    return f"https://github.com/KhronosGroup/glslang/releases/download/{v}/glslang-{v}-linux-x86_64-release.tar.gz"


def tag_url(repo, v):
    return f"https://github.com/{repo}/archive/refs/tags/{v}.tar.gz"


def deny_url(v):
    return f"https://github.com/EmbarkStudios/cargo-deny/releases/download/{v}/cargo-deny-{v}-x86_64-unknown-linux-musl.tar.gz"


def fetch_line(texts, name):
    """The `name|url|sha256|check` entry in fetch-native-tools.sh's PINS."""
    return one(rf'^\s*"({re.escape(name)}\|[^"]+)"', texts["fetch"], name, re.M).split("|")


def bump_fetch_pin(texts, name, new_url, sha):
    _, url, old_sha, _ = fetch_line(texts, name)
    replace(texts, "fetch", f"{url}|{old_sha}|", f"{new_url}|{sha}|")


def bump(texts, name, old, new, sha_of, sha_file):
    """Rewrite every place pin `name` is named from `old` to `new` in `texts` (in place)."""
    if name == "glslang":
        bump_fetch_pin(texts, name, glslang_url(new), sha_of(glslang_url(new)))
        replace(texts, "fetch", f"glslang {old} (", f"glslang {new} (")
    elif name in ("Vulkan-Headers", "volk"):
        repo = {"Vulkan-Headers": "KhronosGroup/Vulkan-Headers", "volk": "zeux/volk"}[name]
        bump_fetch_pin(texts, name, tag_url(repo, new), sha_of(tag_url(repo, new)))
        replace(texts, "fetch", f"{name} {old}", f"{name} {new}")
        replace(texts, "attribution", f"/{repo}) | {old} |", f"/{repo}) | {new} |")
    elif name == "cargo-deny":
        sha = sha_of(deny_url(new))
        published = sha_file(deny_url(new) + ".sha256").split()[0]
        if published != sha:
            raise ValueError(f"cargo-deny {new}: downloaded SHA-256 {sha} is not the published {published}")
        old_sha = one(r'^SHA="([0-9a-f]{64})"', texts["deny"], "cargo-deny SHA", re.M)
        replace(texts, "deny", f'CARGO_DENY_VERSION="{old}"', f'CARGO_DENY_VERSION="{new}"')
        replace(texts, "deny", f'SHA="{old_sha}"', f'SHA="{sha}"')
    elif name in ("appimagetool", "AppImage type2-runtime"):
        var, repo, asset = {
            "appimagetool": ("APPIMAGETOOL", "AppImage/appimagetool", "appimagetool-x86_64.AppImage"),
            "AppImage type2-runtime": ("RUNTIME", "AppImage/type2-runtime", "runtime-x86_64"),
        }[name]
        sha = sha_of(f"https://github.com/{repo}/releases/download/{new}/{asset}")
        old_sha = one(rf'^{var}_SHA256="([0-9a-f]{{64}})"', texts["build"], f"{name} SHA", re.M)
        replace(texts, "build", f'{var}_VERSION="{old}"', f'{var}_VERSION="{new}"')
        replace(texts, "build", f'{var}_SHA256="{old_sha}"', f'{var}_SHA256="{sha}"')
    elif name == "Rust toolchain":
        replace(texts, "toolchain", f'channel = "{old}"', f'channel = "{new}"')
    else:
        raise ValueError(f"no bump rule for {name}")


def bump_all(texts, rows, sha_of, sha_file):
    """Bump every behind (or SDK-mismatched) pin in `texts`; returns one summary line per bump.
    The SDK pair always moves together, to the newest SDK both repositories have."""
    lines = []
    mismatch = any(r[0] == SDK and r[3] == "mismatch" for r in rows)
    for name, pinned, upstream, st in rows:
        if name == SDK:
            continue
        sdk = name in ("Vulkan-Headers", "volk")
        if st == "behind" or (sdk and mismatch and upstream and upstream != pinned):
            bump(texts, name, pinned, upstream, sha_of, sha_file)
            lines.append(f"{name}: {pinned} -> {upstream}")
    after = {name: pinned for name, pinned, _, _ in pins(texts)}
    for name, pinned, upstream, st in rows:
        if name != SDK and (st == "behind" or (name in ("Vulkan-Headers", "volk") and mismatch)) and after[name] != upstream:
            raise ValueError(f"{name}: still reads {after[name]} after the bump to {upstream}")
    if after["Vulkan-Headers"] != after["volk"]:
        raise ValueError(f"Vulkan-Headers {after['Vulkan-Headers']} and volk {after['volk']} would name different SDKs")
    return lines


def check():
    """(rows with the SDK row appended, failures)."""
    rows, failed, cache = [], [], {}
    for name, pinned, kind, repo in pins(read_all()):
        try:
            upstream = latest(kind, repo, cache)
            if upstream is None:
                failed.append(f"{name}: upstream {repo} names no plain {kind.split(':')[0]} to compare with")
        except Exception as e:  # network, rate limit: reported, not guessed
            failed.append(f"{name}: {e}")
            upstream = None
        rows.append((name, pinned, upstream, state(kind, pinned, upstream)))
    rows.append(sdk_row(rows))
    return rows, failed


# ---- Self-test ---------------------------------------------------------------------------------

FIXTURE = {
    "fetch": """#   glslang 16.5.0 (Linux x86_64 release binary), Vulkan-Headers vulkan-sdk-1.4.357.0 and volk vulkan-sdk-1.4.357.0.
PINS=(
  "glslang|https://github.com/KhronosGroup/glslang/releases/download/16.5.0/glslang-16.5.0-linux-x86_64-release.tar.gz|""" + "a" * 64 + """|bin/glslang"
  "Vulkan-Headers|https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.357.0.tar.gz|""" + "b" * 64 + """|include/vulkan/vulkan_core.h"
  "volk|https://github.com/zeux/volk/archive/refs/tags/vulkan-sdk-1.4.357.0.tar.gz|""" + "c" * 64 + """|volk.c"
)
""",
    "build": 'APPIMAGETOOL_VERSION="1.9.0"\nAPPIMAGETOOL_SHA256="' + "d" * 64 + '"\nRUNTIME_VERSION="20251108"\nRUNTIME_SHA256="' + "e" * 64 + '"\n',
    "toolchain": '[toolchain]\nchannel = "1.98.0"\n',
    "deny": 'CARGO_DENY_VERSION="0.19.9"\nSHA="' + "f" * 64 + '"\n',
    "attribution": "| [volk](https://github.com/zeux/volk) | vulkan-sdk-1.4.357.0 | MIT | x |\n"
    "| [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | vulkan-sdk-1.4.357.0 | Apache-2.0 / MIT | x |\n",
}


def fake_sha(url):
    return hashlib.sha256(url.encode()).hexdigest()


def self_test():
    rows = pins(read_all())
    assert [r[0] for r in rows] == ["glslang", "Vulkan-Headers", "volk", "cargo-deny", "appimagetool", "AppImage type2-runtime", "Rust toolchain"], rows
    assert all(r[1] for r in rows)
    assert rows[1][1] == rows[2][1], "the repository's Vulkan-Headers and volk name different SDKs"
    assert version_key("vulkan-sdk-1.4.357.0") < version_key("vulkan-sdk-1.4.360.0")
    assert newest(["v1.4.363", "v1.4.370", "v1.4.99", "v1.5.0-rc1", "continuous"]) == "v1.4.370"
    assert newest(["continuous"]) is None
    assert newest_common_sdk([["vulkan-sdk-1.4.357.0", "vulkan-sdk-1.4.363.0", "vulkan-sdk-1.4.370.0"], ["vulkan-sdk-1.4.357.0", "vulkan-sdk-1.4.363.0"]]) == "vulkan-sdk-1.4.363.0"
    assert newest_common_sdk([["vulkan-sdk-1.4.370.0"], ["vulkan-sdk-1.4.363.0"]]) is None
    assert state("release", "1.99.1", "1.100.0") == "behind" and state("release", "1.100.0", "1.99.1") == "current"   # 1.100 is newer than 1.99, not older
    assert state("release", "1.96.0", "1.97.0") == "behind" and state("release", "1.96.0", "1.96.0") == "current"
    # Upstream that names no version (only `continuous` or -rc tags) is not a current pin.
    assert state("sdk", "vulkan-sdk-1.4.363.0", None) == "unknown" and state("release", "1.96.0", None) == "unknown"
    text = report([("x", "1.0", "1.1", "behind"), ("y", "3.0", "3.0", "current"), ("z", "2.0", None, "unknown"), (SDK, "a = b", None, "mismatch")])
    assert "| x | 1.0 | 1.1 | **behind** |" in text and "| y | 3.0 | 3.0 | current |" in text
    assert "| z | 2.0 | ? | **unknown** |" in text and f"| {SDK} | a = b |  | **mismatch** |" in text
    assert exit_status([("x", "1.0", "1.1", "behind")], []) == BEHIND
    assert exit_status([("y", "1.0", "1.0", "current")], []) == 0
    assert exit_status([("y", "1.0", "1.0", "current"), (SDK, "a = b", None, "mismatch")], []) == BEHIND
    assert exit_status([("x", "1.0", "1.1", "behind"), ("z", "2.0", None, "unknown")], ["z: no version"]) == 1
    sdk_rows = [("Vulkan-Headers", "vulkan-sdk-1.4.363.0", None, "current"), ("volk", "vulkan-sdk-1.4.357.0", None, "current")]
    assert sdk_row(sdk_rows)[3] == "mismatch" and sdk_row(sdk_rows[:1] + [("volk", "vulkan-sdk-1.4.363.0", None, "current")])[3] == "current"

    # Bumping, on fixture text: every pin one step back.
    texts = dict(FIXTURE)
    up = {"glslang": "16.6.0", "Vulkan-Headers": "vulkan-sdk-1.4.363.0", "volk": "vulkan-sdk-1.4.363.0", "cargo-deny": "0.20.2", "appimagetool": "1.9.1", "AppImage type2-runtime": "20251108", "Rust toolchain": "1.99.1"}
    rows = [(n, p, up[n], state(k, p, up[n])) for n, p, k, _ in pins(texts)]
    rows.append(sdk_row(rows))
    lines = bump_all(texts, rows, fake_sha, lambda url: fake_sha(url.removesuffix(".sha256")) + "  file\n")
    assert lines == ["glslang: 16.5.0 -> 16.6.0", "Vulkan-Headers: vulkan-sdk-1.4.357.0 -> vulkan-sdk-1.4.363.0", "volk: vulkan-sdk-1.4.357.0 -> vulkan-sdk-1.4.363.0", "cargo-deny: 0.19.9 -> 0.20.2", "appimagetool: 1.9.0 -> 1.9.1", "Rust toolchain: 1.98.0 -> 1.99.1"], lines
    assert {n: p for n, p, _, _ in pins(texts)} == up
    assert f"download/16.6.0/glslang-16.6.0-linux-x86_64-release.tar.gz|{fake_sha(glslang_url('16.6.0'))}|bin/glslang" in texts["fetch"]
    assert "glslang 16.6.0 (" in texts["fetch"] and "Vulkan-Headers vulkan-sdk-1.4.363.0 and volk vulkan-sdk-1.4.363.0." in texts["fetch"]
    assert f'SHA="{fake_sha(deny_url("0.20.2"))}"' in texts["deny"]
    assert f'APPIMAGETOOL_SHA256="{fake_sha("https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage")}"' in texts["build"]
    assert 'RUNTIME_SHA256="' + "e" * 64 in texts["build"]   # current: untouched
    assert texts["attribution"].count("vulkan-sdk-1.4.363.0") == 2 and "1.4.357" not in texts["attribution"]
    assert "a" * 64 not in texts["fetch"] and "b" * 64 not in texts["fetch"] and "c" * 64 not in texts["fetch"]
    # Nothing behind: nothing bumped.
    current = [(n, up[n], up[n], "current") for n in up] + [(SDK, "x = x", None, "current")]
    assert bump_all(dict(texts), current, fake_sha, None) == []
    # A mismatched pair is brought to the common SDK even when neither is behind it.
    texts2 = dict(FIXTURE)
    texts2["fetch"] = texts2["fetch"].replace("volk/archive/refs/tags/vulkan-sdk-1.4.357.0", "volk/archive/refs/tags/vulkan-sdk-1.4.363.0").replace("volk vulkan-sdk-1.4.357.0", "volk vulkan-sdk-1.4.363.0")
    texts2["attribution"] = texts2["attribution"].replace("volk) | vulkan-sdk-1.4.357.0", "volk) | vulkan-sdk-1.4.363.0")
    rows2 = [("Vulkan-Headers", "vulkan-sdk-1.4.357.0", "vulkan-sdk-1.4.363.0", "behind"), ("volk", "vulkan-sdk-1.4.363.0", "vulkan-sdk-1.4.363.0", "current")]
    rows2.append(sdk_row(rows2))
    assert bump_all(texts2, rows2, fake_sha, None) == ["Vulkan-Headers: vulkan-sdk-1.4.357.0 -> vulkan-sdk-1.4.363.0"]
    # A published checksum that disagrees with the download stops the bump.
    try:
        bump_all(dict(FIXTURE), [r for r in rows if r[0] == "cargo-deny"] + [(SDK, "x = x", None, "current")], fake_sha, lambda url: "0" * 64)
    except ValueError as e:
        assert "not the published" in str(e)
    else:
        raise AssertionError("a wrong published cargo-deny checksum was accepted")
    # A pin whose text has drifted from what the rewrite expects fails loudly instead of half-bumping.
    drifted = dict(FIXTURE, attribution="")
    try:
        bump_all(drifted, [r for r in rows if r[0] == "Vulkan-Headers"] + [(SDK, "x = x", None, "current")], fake_sha, None)
    except ValueError as e:
        assert "ATTRIBUTION.md" in str(e)
    else:
        raise AssertionError("a bump with a missing site was accepted")
    print("check_pins self-test ok")


def main():
    args = sys.argv[1:]
    if args == ["--self-test"]:
        return self_test()
    if args not in ([], ["--bump"]):
        sys.exit(f"usage: {sys.argv[0]} [--bump | --self-test]")
    rows, failed = check()
    if not args:
        print(report(rows), end="")
        if failed:
            print("\nCould not check upstream: " + "; ".join(failed))
        sys.exit(exit_status(rows, failed))
    status = exit_status(rows, failed)
    if status == 1:
        sys.exit("cannot bump: upstream could not be checked: " + "; ".join(failed or ["a pin has no upstream version"]))
    if status == 0:
        print("every pin is current; nothing to bump")
        sys.exit(NOTHING_TO_BUMP)
    texts = read_all()
    try:
        lines = bump_all(texts, rows, sha256_url, text_url)
    except Exception as e:  # nothing is written unless every bump succeeded
        sys.exit(f"bump failed, nothing written: {e}")
    for key, rel in FILES.items():
        if texts[key] != read(rel):
            (ROOT / rel).write_text(texts[key])
    print("\n".join(lines))


main()
