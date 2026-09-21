#!/usr/bin/env python3
"""Exercise install/uninstall and provenance gates entirely in temporary paths."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('install.py')
CLI = Path(os.environ.get('NEURALFORGE_CLI', Path(__file__).resolve().parent.parent / 'target/debug/neural-forge-cli'))
class InstallTests(unittest.TestCase):
    def test_coexistence_and_owned_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            app = root / 'AppDir'
            identity = 'io.github.labj1987.NeuralForge'
            contents = {
                'bin/neural-forge': 'gui', 'bin/neural-forge-cli': 'cli',
                'lib/neuralforge/libneuralforge_layer.so': 'layer',
                'lib/neuralforge/helper/neural-forge-helper.exe': 'helper',
                f'share/applications/{identity}.desktop': '[Desktop Entry]\nExec=neural-forge\n',
                'share/icons/hicolor/scalable/apps/neural-forge.svg': '<svg/>',
                f'share/metainfo/{identity}.appdata.xml': '<component/>',
                'share/vulkan/implicit_layer.d/VK_LAYER_neuralforge_neural.json': json.dumps({'layer': {'name': 'VK_LAYER_neuralforge_neural'}}),
            }
            for name, text in contents.items():
                p = app / 'usr' / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            upstream = data / 'vulkan/implicit_layer.d/VkLayer_DLSS5.json'
            upstream.parent.mkdir(parents=True)
            upstream.write_text('upstream sentinel')
            def run(*args, ok=True):
                result = subprocess.run(['python3', str(SCRIPT), '--cli', str(CLI), *args], env={**os.environ, 'XDG_DATA_HOME': str(data)}, capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, ok, result.stderr)
            run('install', '--appdir', str(app))
            run('install', '--appdir', str(app))
            binary = data / 'neuralforge/bin/neural-forge'
            with binary.open('rb') as running_image:
                (app / 'usr/bin/neural-forge').write_text('updated gui')
                run('install', '--appdir', str(app))
                self.assertEqual(running_image.read(), b'gui')
                self.assertEqual(binary.read_text(), 'updated gui')
            binary.write_text('user changed')
            run('install', '--appdir', str(app), ok=False)
            run('uninstall')
            self.assertEqual(binary.read_text(), 'user changed')
            self.assertEqual(upstream.read_text(), 'upstream sentinel')
            self.assertFalse((data / 'neuralforge/bin/neural-forge-cli').exists())
            legacy = root / 'legacy.json'
            legacy.write_text(json.dumps({'layer': {'name': 'VK_LAYER_NV_dlssnr'}}))
            run('archive-legacy-manifest', '--legacy-manifest', str(legacy), ok=False)
            self.assertTrue(legacy.exists())
            legacy.write_text(json.dumps({'layer': {
                'name': 'VK_LAYER_dlssnr_neural', 'library_path': '/old/libdlssnr_layer.so',
                'enable_environment': {'VKLayer_DLSS5': '1'}, 'disable_environment': {'DLSSNR_DISABLE': '1'},
            }}))
            run('archive-legacy-manifest', '--legacy-manifest', str(legacy))
            self.assertFalse(legacy.exists())
            self.assertTrue((data / 'neuralforge/legacy/VK_LAYER_dlssnr_neural.json.disabled').exists())

if __name__ == '__main__': unittest.main()
