#!/usr/bin/env python3
"""Check activation, desktop, packaging and private API namespace boundaries."""
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'data/VK_LAYER_neuralforge_neural.json').read_text())['layer']
assert manifest['enable_environment'] == {'NEURALFORGE_ENABLE': '1'}
assert manifest['disable_environment'] == {'NEURALFORGE_DISABLE': '1'}
assert manifest['name'] in (root / 'crates/layer/src/lib.rs').read_text()
assert manifest['library_path'] == './libneuralforge_layer.so'
identity = 'io.github.labj1987.NeuralForge'
meta = ET.parse(root / f'data/{identity}.appdata.xml').getroot()
assert meta.find('id').text == identity
assert meta.find('launchable').text == f'{identity}.desktop'
assert '\nExec=neural-forge\n' in (root / f'data/{identity}.desktop').read_text()
for path in (root / 'crates').rglob('*.rs'):
    code = path.read_text()
    assert not re.search(r'(?:var|var_os|set_var)\("(?:DLSSNR_|VKLayer_DLSS5)', code), path
    assert 'nvngx_neuralforge' not in code, path
    assert 'NEURALFORGE.Color' not in code, path
assert (root / 'README.md').read_text().startswith('# Neural Forge\n')
assert '`dlssnr` is a from-scratch' not in (root / 'ATTRIBUTION.md').read_text()
assert '|labj1987|neural-forge|latest|' in (root / 'build-appimage.sh').read_text()
assert 'https://github.com/labj1987/neural-forge' in (root / 'README.md').read_text()
print('Namespace contract: OK')
