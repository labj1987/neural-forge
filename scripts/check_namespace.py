#!/usr/bin/env python3
"""Check activation, desktop, packaging and private API namespace boundaries."""
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'data/neural_forge_layer.json').read_text())['layer']
assert manifest['enable_environment'] == {'NEURAL_FORGE_ENABLE': '1'}
assert manifest['disable_environment'] == {'NEURAL_FORGE_DISABLE': '1'}
assert manifest['name'] == 'VK_LAYER_neuralforge_neural'
assert manifest['name'] in (root / 'crates/layer/src/lib.rs').read_text()
assert manifest['library_path'] == './libneural_forge_layer.so'
identity = 'io.github.labj1987.NeuralForge'
meta = ET.parse(root / f'data/{identity}.appdata.xml').getroot()
assert meta.find('id').text == identity
assert meta.find('launchable').text == f'{identity}.desktop'
assert '\nExec=neural-forge\n' in (root / f'data/{identity}.desktop').read_text()
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
    assert not re.search(r'neuralforge|NEURALFORGE', code), path
    if path.suffix == '.rs':
        assert not re.search(r'(?:var|var_os|set_var)\("(?:DLSSNR_|VKLayer_DLSS5)', code), path
        assert 'NEURALFORGE.Color' not in code, path
assert (root / 'README.md').read_text().startswith('# Neural Forge\n')
assert '`dlssnr` is a from-scratch' not in (root / 'ATTRIBUTION.md').read_text()
assert '|labj1987|neural-forge|latest|' in (root / 'build-appimage.sh').read_text()
assert 'https://github.com/labj1987/neural-forge' in (root / 'README.md').read_text()
print('Namespace contract: OK')
