#!/usr/bin/env python3
"""Ties the committed SPIR-V binaries to the GLSL sources they were built from.

`crates/layer/shaders/*.spv` are committed binaries (embedded
with `include_bytes!`);
nothing builds them, so an edited `.comp` could silently ship a stale `.spv`.
Each folder's `shaders.sha256` records the SHA-256 of each source and of the binary compiled from it.

  check_shaders.py            verify: every source hash and binary hash matches the
                              manifest, no .spv is left without its source, and each
                              binary passes spirv-val (required: install spirv-tools).
  check_shaders.py --update   recompile every .comp with the pinned glslang
                              (scripts/fetch-native-tools.sh; never one on PATH), validate
                              with spirv-val, remove .spv files whose source is gone, and
                              rewrite the manifest.

Edit a shader, run `--update`, commit the .comp, .spv and manifest together. Verification
is a pure hash comparison on purpose: SPIR-V bytes differ between glslang releases, so a
CI-side recompile-and-compare would fail for reasons that are not a stale binary.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = [ROOT / "crates/layer/shaders"]
# Set per folder by the loop at the bottom; the functions below read them.
SHADERS = DIRS[0]
MANIFEST = SHADERS / "shaders.sha256"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return sorted(SHADERS.glob("*.comp"))


def pinned_glslang() -> Path:
    """The glslang fetch-native-tools.sh pins: the committed binaries are its output, and another
    release writes different bytes."""
    tools = Path(os.environ.get("NEURAL_FORGE_NATIVE_TOOLS", "").strip() or ROOT / "tools/native")
    return tools / "glslang/bin/glslang"


def spirv_val() -> str:
    tool = shutil.which("spirv-val")
    if tool is None:
        sys.exit("spirv-val is required (apt install spirv-tools): the binaries are validated, not only compared")
    return tool


def validate(spv: Path) -> bool:
    tool = spirv_val()
    result = subprocess.run([tool, str(spv)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  {spv.name} failed spirv-val:\n{result.stdout}{result.stderr}")
    return result.returncode == 0


def update() -> int:
    glslang = pinned_glslang()
    if not glslang.is_file():
        print(f"{glslang} is missing: run scripts/fetch-native-tools.sh (--update uses only the pinned glslang)", file=sys.stderr)
        return 2
    spirv_val()
    lines = []
    for comp in sources():
        spv = comp.with_suffix(".spv")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / spv.name
            result = subprocess.run([str(glslang), "-V", str(comp), "-o", str(out)], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"{comp.name} does not compile:\n{result.stdout}{result.stderr}", file=sys.stderr)
                return 1
            if not validate(out):
                return 1
            spv.write_bytes(out.read_bytes())
        lines.append(f"{comp.name} {sha(comp)} {spv.name} {sha(spv)}")
        print(f"rebuilt {spv.name}")
    for orphan in sorted(set(SHADERS.glob("*.spv")) - {c.with_suffix(".spv") for c in sources()}):
        orphan.unlink()
        print(f"removed {orphan.name}: its source is gone")
    MANIFEST.write_text("\n".join(lines) + "\n")
    return 0


def verify() -> int:
    if not MANIFEST.is_file():
        print(f"missing {MANIFEST}; run scripts/check_shaders.py --update", file=sys.stderr)
        return 1
    recorded = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            comp, comp_hash, spv, spv_hash = line.split()
            recorded[comp] = (comp_hash, spv, spv_hash)
    failures = []
    for comp in sources():
        entry = recorded.pop(comp.name, None)
        if entry is None:
            failures.append(f"{comp.name}: not in shaders.sha256")
            continue
        comp_hash, spv_name, spv_hash = entry
        spv = SHADERS / spv_name
        if sha(comp) != comp_hash:
            failures.append(f"{comp.name}: source changed since its .spv was built -- run scripts/check_shaders.py --update")
        elif not spv.is_file() or sha(spv) != spv_hash:
            failures.append(f"{spv_name}: binary does not match the manifest -- run scripts/check_shaders.py --update")
        elif not validate(spv):
            failures.append(f"{spv_name}: fails spirv-val")
    failures += [f"{name}: listed in shaders.sha256 but the source is gone" for name in recorded]
    listed = {entry[1] for entry in recorded.values()} | {c.with_suffix(".spv").name for c in sources()}
    failures += [f"{spv.name}: a binary without its source -- run scripts/check_shaders.py --update" for spv in sorted(SHADERS.glob("*.spv")) if spv.name not in listed]
    for failure in failures:
        print(failure, file=sys.stderr)
    if not failures:
        print("Shaders: OK")
    return 1 if failures else 0


if __name__ == "__main__":
    status = 0
    for SHADERS in DIRS:
        MANIFEST = SHADERS / "shaders.sha256"
        status |= update() if "--update" in sys.argv[1:] else verify()
    sys.exit(status)
