#!/usr/bin/env python3
"""Check activation, desktop, packaging and private API namespace boundaries.

Every check runs (none is an `assert`, so `python3 -O` checks the same), and each failure is reported
with what was expected; the script exits 1 after listing them all.
"""
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
failures = []


def check(ok, message):
    if not ok:
        failures.append(message)


def text(rel):
    return (root / rel).read_text()


manifest = json.loads(text('data/neural_forge_layer.json'))['layer']
check(manifest.get('enable_environment') == {'NEURAL_FORGE_ENABLE': '1'},
      f"data/neural_forge_layer.json: enable_environment is {manifest.get('enable_environment')!r}, not NEURAL_FORGE_ENABLE=1")
check(manifest.get('disable_environment') == {'NEURAL_FORGE_DISABLE': '1'},
      f"data/neural_forge_layer.json: disable_environment is {manifest.get('disable_environment')!r}, not NEURAL_FORGE_DISABLE=1")
check(manifest.get('name') == 'VK_LAYER_neuralforge_neural',
      f"data/neural_forge_layer.json: the layer is named {manifest.get('name')!r}, not VK_LAYER_neuralforge_neural")
check(str(manifest.get('name')) in text('crates/layer/src/lib.rs'),
      f"crates/layer/src/lib.rs does not name the manifest's layer {manifest.get('name')!r}")
check(manifest.get('library_path') == './libneural_forge_layer.so',
      f"data/neural_forge_layer.json: library_path is {manifest.get('library_path')!r}, not ./libneural_forge_layer.so")
identity = 'io.github.labj1987.NeuralForge'
meta = ET.parse(root / f'data/{identity}.appdata.xml').getroot()
for tag, want in (('id', identity), ('launchable', f'{identity}.desktop')):
    found = meta.find(tag)
    got = found.text if found is not None else None
    check(got == want, f"data/{identity}.appdata.xml: <{tag}> is {got!r}, not {want!r}")
check('\nExec=neural-forge\n' in text(f'data/{identity}.desktop'), f"data/{identity}.desktop has no Exec=neural-forge line")
# The pre-0.1.77 spelling (`NEURALFORGE_*`, `neuralforge` paths) is gone everywhere: Rust sources, scripts,
# data files and the packaging script. Only the two frozen identifiers (and one history file name) keep the old spelling.
# The history file keeps its name (it records the project before the rename).
FROZEN = ('VK_LAYER_neuralforge_neural', 'io.github.labj1987.NeuralForge', 'development-before-neuralforge.md')
scanned = list((root / 'crates').rglob('*.rs'))
for folder in ('scripts', 'data'):
    scanned += [p for p in (root / folder).rglob('*') if p.is_file() and p.suffix != '.spv']
scanned.append(root / 'build-appimage.sh')
for path in scanned:
    if path == Path(__file__).resolve():
        continue
    try:
        code = path.read_text()
    except UnicodeDecodeError:
        continue  # binary data
    for frozen in FROZEN:
        code = code.replace(frozen, '')
    rel = path.relative_to(root)
    old = re.search(r'neuralforge|NEURALFORGE', code)
    check(not old, f"{rel}: the old spelling {old.group(0) if old else ''!r} (use neural-forge / NEURAL_FORGE_)")
    if path.suffix == '.rs':
        env = re.search(r'(?:var|var_os|set_var)\("(?:DLSSNR_|VKLayer_DLSS5)\w*', code)
        check(not env, f"{rel}: reads an old or upstream environment variable ({env.group(0) if env else ''})")
        check('NEURALFORGE.Color' not in code, f"{rel}: the old NEURALFORGE.Color name")
check(text('README.md').startswith('# Neural Forge\n'), "README.md does not start with the '# Neural Forge' heading")
check('`dlssnr` is a from-scratch' not in text('ATTRIBUTION.md'), "ATTRIBUTION.md still describes the project as dlssnr")
check('|labj1987|neural-forge|latest|' in text('build-appimage.sh'), "build-appimage.sh's update information does not name labj1987/neural-forge")
check('https://github.com/labj1987/neural-forge' in text('README.md'), "README.md does not link https://github.com/labj1987/neural-forge")
if failures:
    for message in failures:
        print(f"error: {message}", file=sys.stderr)
    print(f"Namespace contract: {len(failures)} failures", file=sys.stderr)
    sys.exit(1)
print('Namespace contract: OK')
