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
# The pre-0.1.77 spelling (`NEURALFORGE_*`, `neuralforge` paths) is gone everywhere.
for path in (root / 'crates').rglob('*.rs'):
    code = path.read_text()
    assert 'NEURALFORGE_' not in code, path
    assert not re.search(r'(?<![A-Za-z_])neuralforge[-_/]', code.replace('VK_LAYER_neuralforge_neural', '')), path
    assert not re.search(r'(?:var|var_os|set_var)\("(?:DLSSNR_|VKLayer_DLSS5)', code), path
    assert 'nvngx_neuralforge' not in code, path
    assert 'NEURALFORGE.Color' not in code, path
assert (root / 'README.md').read_text().startswith('# Neural Forge\n')
assert '`dlssnr` is a from-scratch' not in (root / 'ATTRIBUTION.md').read_text()
assert '|labj1987|neural-forge|latest|' in (root / 'build-appimage.sh').read_text()
assert 'https://github.com/labj1987/neural-forge' in (root / 'README.md').read_text()
print('Namespace contract: OK')
