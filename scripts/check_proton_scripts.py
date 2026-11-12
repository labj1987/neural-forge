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
                                       does not to one (or a repo has no release branch the check knows), 2 when upstream
                                       could not be reached (a network failure: retry later, nothing is known), 1 when the
                                       check itself failed (the patch example did not build or run, the API answered in a
                                       shape it does not know, any other error): that must fail the run
  check_proton_scripts.py --decide STATE ISSUE_STATE OLD_BODY NEW_BODY
                                       print what to do with the standing issue (create, edit, reopen, close; nothing
                                       for none): STATE is current or broken, ISSUE_STATE none, OPEN or CLOSED, the
                                       bodies are files. An issue closed by hand is not reopened while the report is
                                       the one it was closed with (the report's hash is kept in the body)
  check_proton_scripts.py --self-test  check the branch choice, the report, the exit codes and the issue decisions offline

.github/workflows/proton-script-check.yml runs it weekly; one issue carries the report.
"""
import contextlib
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BROKEN = 10
NETWORK = 2
TOOL = 1
MARKER = re.compile(r"<!-- report-sha256: ([0-9a-f]{64}) -->")
REPOS = {"ValveSoftware/Proton": "Valve's Proton Experimental", "CachyOS/proton-cachyos": "proton-cachyos"}


def newest(names, pattern):
    """The branch among `names` matching `pattern` (its groups are ints, compared as a tuple) with the highest groups."""
    found = []
    for name in names:
        m = re.fullmatch(pattern, name)
        if m:
            found.append((tuple(int(g) for g in m.groups()), name))
    return max(found)[1] if found else None


RELEASE_BRANCH = {"ValveSoftware/Proton": r"experimental_(\d+)\.(\d+)", "CachyOS/proton-cachyos": r"cachyos_(\d+)\.(\d+)_(\d{8})/main"}


def refs_for(repo, branches):
    """The branches of `repo` whose `proton` script is checked. The release branch comes first and is None when no branch
    matches its pattern (upstream renamed them: the check no longer knows what to look at, which is a failure)."""
    return [newest(branches, RELEASE_BRANCH[repo])] + (["bleeding-edge"] if "bleeding-edge" in branches else [])


def api(path, raw=False):
    req = urllib.request.Request(f"https://api.github.com/{path}")
    req.add_header("Accept", "application/vnd.github.raw" if raw else "application/vnd.github+json")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


class ToolError(Exception):
    """The check itself failed (not the network): the run must fail."""


def branches_of(repo):
    names, page = [], 1
    while True:
        chunk = json.loads(api(f"repos/{repo}/branches?per_page=100&page={page}"))
        if not isinstance(chunk, list) or not all(isinstance(b, dict) and isinstance(b.get("name"), str) for b in chunk):
            raise ToolError(f"the branch list of {repo} has a shape this check does not know: {str(chunk)[:200]}")
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
    text = "\n".join(lines) + "\n"
    return text + f"\n<!-- report-sha256: {hashlib.sha256(text.encode()).hexdigest()} -->\n", not bad


def marker(body):
    """The report hash kept in an issue body, or None."""
    m = MARKER.search(body or "")
    return m.group(1) if m else None


def decide(state, issue_state, old_body, new_body):
    """What to do with the standing issue: a list of create, edit, reopen, close. At most one notification per change
    of the report: an issue closed by hand stays closed while the report is the one it was closed with."""
    same = marker(old_body) is not None and marker(old_body) == marker(new_body)
    if state == "broken":
        if issue_state == "none":
            return ["create"]
        if same:
            return []
        return ["edit"] if issue_state == "OPEN" else ["edit", "reopen"]
    if state == "current" and issue_state == "OPEN":
        return ["edit", "close"]
    return []


def run_patch(files):
    """The example's output lines for `files` (paths in), keyed by path."""
    cmd = os.environ.get("NF_CHECK_PROTON_SCRIPT")
    cmd = [cmd] if cmd else ["cargo", "run", "-q", "-p", "neural-forge-supervisor", "--example", "check_proton_script", "--"]
    out = subprocess.run(cmd + [str(f) for f in files], cwd=ROOT, capture_output=True, text=True)
    if out.returncode not in (0, BROKEN):
        raise ToolError(f"the patch check could not run (exit {out.returncode}): {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout.splitlines()


def main():
    results, files = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for repo in REPOS:
            for ref in refs_for(repo, branches_of(repo)):
                if ref is None:
                    results.append((repo, "-", f"fail proton: no release branch matches {RELEASE_BRANCH[repo]} any more"))
                    continue
                path = Path(tmp) / f"{len(files)}.proton"
                path.write_bytes(api(f"repos/{repo}/contents/proton?ref={urllib.parse.quote(ref, safe='')}", raw=True))
                files.append((repo, ref, path))
        lines = run_patch([f[2] for f in files])
    by_path = {l.split(" ", 2)[1].rstrip(":"): l for l in lines}
    for repo, ref, path in files:
        line = by_path.get(str(path))
        if line is None:
            raise ToolError(f"no answer for {repo} @ {ref}")
        results.append((repo, ref, line.replace(str(path), "proton")))
    text, ok = report(results)
    print(text, end="")
    return 0 if ok else BROKEN


def is_network(e):
    """Whether `e` is a failure to reach upstream (retry later) rather than one of the check itself."""
    if isinstance(e, urllib.error.HTTPError):
        rate_limited = e.code == 403 and (e.headers or {}).get("X-RateLimit-Remaining") == "0"
        return e.code >= 500 or e.code == 429 or rate_limited
    return isinstance(e, (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError))


def run():
    """main() with its failures turned into exit codes."""
    try:
        return main()
    except Exception as e:
        if is_network(e):
            print(f"could not reach upstream: {e}", file=sys.stderr)
            return NETWORK
        print(f"the check failed: {type(e).__name__}: {e}", file=sys.stderr)
        return TOOL


def exit_code_tests():
    """run()'s exit codes with upstream and the patch example stubbed."""
    global api, run_patch
    real_api, real_patch = api, run_patch
    hdr = {"X-RateLimit-Remaining": "0"}
    for e, want in [
        (urllib.error.URLError("no route"), NETWORK),
        (socket.timeout("timed out"), NETWORK),
        (ConnectionResetError("reset"), NETWORK),
        (urllib.error.HTTPError("u", 502, "bad gateway", {}, None), NETWORK),
        (urllib.error.HTTPError("u", 403, "rate limited", hdr, None), NETWORK),
        (urllib.error.HTTPError("u", 404, "not found", {}, None), TOOL),
        (urllib.error.HTTPError("u", 401, "bad credentials", {}, None), TOOL),
        (KeyError("name"), TOOL),
    ]:
        def failing(*_a, _e=e, **_k):
            raise _e

        api = failing
        try:
            got = run()
        finally:
            api = real_api
        assert got == want, (e, got, want)
    try:
        api = lambda path, raw=False: b'{"message": "moved"}'
        assert run() == TOOL, "an API answer of the wrong shape fails the run"
        api = lambda path, raw=False: b"[]" if not raw else b"script"
        run_patch = lambda files: [f"ok {f}" for f in files]
        assert run() == BROKEN, "no release branch is a failure, not 'applies'"
        api = lambda path, raw=False: json.dumps([{"name": "experimental_11.0"}, {"name": "cachyos_11.0_20261014/main"}]).encode() if not raw else b"script"
        assert run() == 0
        run_patch = real_patch
        os.environ["NF_CHECK_PROTON_SCRIPT"] = "false"
        assert run() == TOOL, "the patch example failing to run fails the run"
    finally:
        api, run_patch = real_api, real_patch
        os.environ.pop("NF_CHECK_PROTON_SCRIPT", None)


def self_test():
    v = ["experimental_10.0", "experimental_9.0", "experimental_11.0", "experimental_5.13-dxvk-d3d11-funny-refcount", "bleeding-edge", "proton_11.0"]
    assert refs_for("ValveSoftware/Proton", v) == ["experimental_11.0", "bleeding-edge"], refs_for("ValveSoftware/Proton", v)
    c = ["cachyos_10.0_20251120/main", "cachyos_11.0_20261014/main", "cachyos_11.0_20261014/main_native", "cachyos_11.0_20261001/main", "cachyos_9.0_20250117/main-runtime", "cachyos_main"]
    assert refs_for("CachyOS/proton-cachyos", c) == ["cachyos_11.0_20261014/main"], refs_for("CachyOS/proton-cachyos", c)
    assert refs_for("ValveSoftware/Proton", ["proton_11.0", "bleeding-edge"]) == [None, "bleeding-edge"], "a missing release branch is kept as None"
    assert refs_for("CachyOS/proton-cachyos", c + ["bleeding-edge"])[-1] == "bleeding-edge"
    assert newest(["experimental_9.0", "experimental_10.0"], r"experimental_(\d+)\.(\d+)") == "experimental_10.0", "10 is newer than 9"
    assert newest([], r"x(\d+)") is None
    good = [("ValveSoftware/Proton", "experimental_11.0", "ok proton"), ("CachyOS/proton-cachyos", "bleeding-edge", "ok proton")]
    text, ok = report(good)
    assert ok and "applies" in text and "no longer" not in text, text
    bad = good + [("CachyOS/proton-cachyos", "cachyos_11.0_20261014/main", "fail proton: the proton script's NVIDIA_WINE_DLL_DIR block was found 0 times, not once")]
    text, ok = report(bad)
    assert not ok and "no longer applies" in text and "found 0 times" in text and "NGXPath" in text, text
    assert marker(text) and marker(text) != marker(report(good)[0])
    # Issue decisions: one notification per change of the report, never a reopen of a hand-closed issue with the same one.
    other = report(bad[:1] + [("CachyOS/proton-cachyos", "-", "fail proton: no release branch matches x any more")])[0]
    assert decide("broken", "none", "", text) == ["create"]
    assert decide("broken", "OPEN", text, text) == []
    assert decide("broken", "OPEN", other, text) == ["edit"]
    assert decide("broken", "CLOSED", text, text) == [], "closed by hand with the same report: left closed"
    assert decide("broken", "CLOSED", other, text) == ["edit", "reopen"]
    assert decide("broken", "CLOSED", "a body without a hash", text) == ["edit", "reopen"]
    assert decide("current", "OPEN", text, report(good)[0]) == ["edit", "close"]
    assert decide("current", "CLOSED", text, report(good)[0]) == [] and decide("current", "none", "", report(good)[0]) == []
    # Exit codes: a network failure is 2 (retry, quiet), a failure of the check itself 1, a missing release branch 10.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exit_code_tests()
    print("check_proton_scripts.py: self-test ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) == 6 and sys.argv[1] == "--decide":
        print(" ".join(decide(sys.argv[2], sys.argv[3], Path(sys.argv[4]).read_text(), Path(sys.argv[5]).read_text())))
    elif len(sys.argv) == 1:
        sys.exit(run())
    else:
        sys.exit(f"usage: {sys.argv[0]} [--self-test | --decide STATE ISSUE_STATE OLD_BODY NEW_BODY]")
