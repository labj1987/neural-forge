#!/usr/bin/env python3
"""Ties the committed SPIR-V binaries to the GLSL sources they were built from.

`crates/layer/shaders/*.spv` and `crates/helper/shaders/*.spv` are committed binaries (embedded
with `include_bytes!`);
nothing builds them, so an edited `.comp` could silently ship a stale `.spv`.
Each folder's `shaders.sha256` records the SHA-256 of each source and of the binary compiled from it.

  check_shaders.py            verify: every source hash and binary hash matches the
                              manifest, and each binary passes spirv-val when available.
  check_shaders.py --update   recompile every .comp with `glslangValidator -V`, validate
                              with spirv-val, and rewrite the manifest.

Edit a shader, run `--update`, commit the .comp, .spv and manifest together. Verification
is a pure hash comparison on purpose: SPIR-V bytes differ between glslang releases, so a
CI-side recompile-and-compare would fail for reasons that are not a stale binary.
"""
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = [ROOT / "crates/layer/shaders", ROOT / "crates/helper/shaders"]
# Set per folder by the loop at the bottom; the functions below read them.
SHADERS = DIRS[0]
MANIFEST = SHADERS / "shaders.sha256"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return sorted(SHADERS.glob("*.comp"))


def validate(spv: Path) -> bool:
    tool = shutil.which("spirv-val")
    if tool is None:
        print(f"  (spirv-val not installed; skipping validation of {spv.name})")
        return True
    result = subprocess.run([tool, str(spv)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  {spv.name} failed spirv-val:\n{result.stdout}{result.stderr}")
    return result.returncode == 0


def update() -> int:
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        print("glslangValidator is required for --update", file=sys.stderr)
        return 2
    lines = []
    for comp in sources():
        spv = comp.with_suffix(".spv")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / spv.name
            subprocess.run([glslang, "-V", str(comp), "-o", str(out)], check=True, capture_output=True)
            if not validate(out):
                return 1
            spv.write_bytes(out.read_bytes())
        lines.append(f"{comp.name} {sha(comp)} {spv.name} {sha(spv)}")
        print(f"rebuilt {spv.name}")
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
