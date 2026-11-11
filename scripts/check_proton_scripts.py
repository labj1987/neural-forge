#!/usr/bin/env python3
"""Does Neural Forge's patch of Proton's `proton` script still apply to the newest upstream scripts?

A Neural Forge Proton copy gets two DLL overrides after the `NVIDIA_WINE_DLL_DIR` block of its source's `proton` script
(crates/supervisor/src/proton_tool.rs, `patch_proton_script`; docs/PROTON_NGX.md). The patch is refused unless that block
is there exactly once. Valve's script still carries a TODO to replace the block with an `NGXCore\\NGXPath` registry key:
when that lands (or CachyOS's fork changes it) the anchor breaks and no copy can be made from that Proton. This fetches
the scripts of

  Valve's Proton Experimental   the newest experimental_<major>.<minor> branch, and bleeding-edge
  proton-cachyos                the newest cachyos_<major>.<minor>_<date>/main release branch, and bleeding-edge

through the GitHub API ($GITHUB_TOKEN raises its rate limit) and runs the patch on each
(`cargo run -p neural-forge-supervisor --example check_proton_script`).

  check_proton_scripts.py              print a Markdown report; exit 0 when the patch applies to every script, 10 when it
                                       does not to one, 1 when upstream could not be asked or the patch could not be run
                                       (nothing is known then)
  check_proton_scripts.py --self-test  check the branch choice and the report offline

.github/workflows/proton-script-check.yml runs it weekly; one issue carries the report.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BROKEN = 10
REPOS = {"ValveSoftware/Proton": "Valve's Proton Experimental", "CachyOS/proton-cachyos": "proton-cachyos"}


def newest(names, pattern):
    """The branch among `names` matching `pattern` (its groups are ints, compared as a tuple) with the highest groups."""
    found = []
    for name in names:
        m = re.fullmatch(pattern, name)
        if m:
            found.append((tuple(int(g) for g in m.groups()), name))
    return max(found)[1] if found else None


def refs_for(repo, branches):
    """The branches of `repo` whose `proton` script is checked."""
    if repo == "ValveSoftware/Proton":
        picks = [newest(branches, r"experimental_(\d+)\.(\d+)")]
    else:
        picks = [newest(branches, r"cachyos_(\d+)\.(\d+)_(\d{8})/main")]
    picks.append("bleeding-edge" if "bleeding-edge" in branches else None)
    return [p for p in picks if p]


def api(path, raw=False):
    req = urllib.request.Request(f"https://api.github.com/{path}")
    req.add_header("Accept", "application/vnd.github.raw" if raw else "application/vnd.github+json")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def branches_of(repo):
    names, page = [], 1
    while True:
        chunk = json.loads(api(f"repos/{repo}/branches?per_page=100&page={page}"))
        names += [b["name"] for b in chunk]
        if len(chunk) < 100:
            return names
        page += 1


def report(results):
    """`results`: (repo, ref, outcome line) with 'ok ...' or 'fail ...: why'. The Markdown report, and whether all are ok."""
    bad = [r for r in results if not r[2].startswith("ok")]
    lines = []
    if bad:
        lines += [
            "Neural Forge's patch of Proton's `proton` script no longer applies to the newest upstream script(s), so no copy can be made from that Proton once it ships:",
            "",
        ]
    else:
        lines += ["Neural Forge's patch of Proton's `proton` script applies to the newest upstream scripts:", ""]
    for repo, ref, outcome in results:
        mark = "ok" if outcome.startswith("ok") else outcome.split(" ", 2)[0]
        why = "" if mark == "ok" else " " + outcome.split(": ", 1)[-1]
        lines.append(f"- {REPOS[repo]} (`{repo}` @ `{ref}`): {mark}{why}")
    if bad:
        lines += [
            "",
            "Valve's script has a TODO to replace the `NVIDIA_WINE_DLL_DIR` block with an `NGXCore\\NGXPath` registry key; when it lands the anchor in "
            "`proton_tool::patch_proton_script` has to move to it (docs/PROTON_NGX.md).",
        ]
    return "\n".join(lines) + "\n", not bad


def run_patch(files):
    """The example's output lines for `files` (paths in), keyed by path."""
    cmd = os.environ.get("NF_CHECK_PROTON_SCRIPT")
    cmd = [cmd] if cmd else ["cargo", "run", "-q", "-p", "neural-forge-supervisor", "--example", "check_proton_script", "--"]
    out = subprocess.run(cmd + [str(f) for f in files], cwd=ROOT, capture_output=True, text=True)
    if out.returncode not in (0, BROKEN):
        raise RuntimeError(f"the patch check could not run (exit {out.returncode}): {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout.splitlines()


def main():
    results, files = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for repo in REPOS:
            for ref in refs_for(repo, branches_of(repo)):
                path = Path(tmp) / f"{len(files)}.proton"
                path.write_bytes(api(f"repos/{repo}/contents/proton?ref={urllib.parse.quote(ref, safe='')}", raw=True))
                files.append((repo, ref, path))
        lines = run_patch([f[2] for f in files])
    by_path = {l.split(" ", 2)[1].rstrip(":"): l for l in lines}
    for repo, ref, path in files:
        line = by_path.get(str(path))
        if line is None:
            raise RuntimeError(f"no answer for {repo} @ {ref}")
        results.append((repo, ref, line.replace(str(path), "proton")))
    text, ok = report(results)
    print(text, end="")
    return 0 if ok else BROKEN


def self_test():
    v = ["experimental_10.0", "experimental_9.0", "experimental_11.0", "experimental_5.13-dxvk-d3d11-funny-refcount", "bleeding-edge", "proton_11.0"]
    assert refs_for("ValveSoftware/Proton", v) == ["experimental_11.0", "bleeding-edge"], refs_for("ValveSoftware/Proton", v)
    c = ["cachyos_10.0_20251120/main", "cachyos_11.0_20261014/main", "cachyos_11.0_20261014/main_native", "cachyos_11.0_20261001/main", "cachyos_9.0_20250117/main-runtime", "cachyos_main"]
    assert refs_for("CachyOS/proton-cachyos", c) == ["cachyos_11.0_20261014/main"], refs_for("CachyOS/proton-cachyos", c)
    assert refs_for("CachyOS/proton-cachyos", c + ["bleeding-edge"])[-1] == "bleeding-edge"
    assert newest(["experimental_9.0", "experimental_10.0"], r"experimental_(\d+)\.(\d+)") == "experimental_10.0", "10 is newer than 9"
    assert newest([], r"x(\d+)") is None
    good = [("ValveSoftware/Proton", "experimental_11.0", "ok proton"), ("CachyOS/proton-cachyos", "bleeding-edge", "ok proton")]
    text, ok = report(good)
    assert ok and "applies" in text and "no longer" not in text, text
    bad = good + [("CachyOS/proton-cachyos", "cachyos_11.0_20261014/main", "fail proton: the proton script's NVIDIA_WINE_DLL_DIR block was found 0 times, not once")]
    text, ok = report(bad)
    assert not ok and "no longer applies" in text and "found 0 times" in text and "NGXPath" in text, text
    print("check_proton_scripts.py: self-test ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) == 1:
        try:
            sys.exit(main())
        except Exception as e:  # upstream could not be asked, or the check could not run: nothing is known
            print(f"could not check: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        sys.exit(f"usage: {sys.argv[0]} [--self-test]")
