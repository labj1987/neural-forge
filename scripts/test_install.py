#!/usr/bin/env python3
"""Exercise install/uninstall and provenance gates entirely in temporary paths."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('install.py')
CLI = Path(os.environ.get('NEURAL_FORGE_CLI', Path(__file__).resolve().parent.parent / 'target/debug/neural-forge-cli'))
def scratch_env(root, data):
    # Every XDG home lives in the scratch dir, so the CLI never sees the real ones. The unique uid names a
    # runtime dir (and pid file) that cannot belong to a real helper.
    return {**os.environ, 'XDG_DATA_HOME': str(data), 'XDG_CONFIG_HOME': str(root / 'config'),
            'XDG_STATE_HOME': str(root / 'state'), 'NEURAL_FORGE_UID': f'installtest-{os.getpid()}'}
class InstallTests(unittest.TestCase):
    def test_coexistence_and_owned_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            app = root / 'AppDir'
            identity = 'io.github.labj1987.NeuralForge'
            contents = {
                'bin/neural-forge': 'gui', 'bin/neural-forge-cli': 'cli',
                'lib/neural-forge/libneural_forge_layer.so': 'layer',
                'lib/neural-forge/helper/neural-forge-helper.exe': 'helper',
                f'share/applications/{identity}.desktop': '[Desktop Entry]\nExec=neural-forge\n',
                'share/icons/hicolor/scalable/apps/neural-forge.svg': '<svg/>',
                f'share/metainfo/{identity}.appdata.xml': '<component/>',
                'share/vulkan/implicit_layer.d/neural_forge_layer.json': json.dumps({'layer': {'name': 'VK_LAYER_neuralforge_neural'}}),
            }
            for name, text in contents.items():
                p = app / 'usr' / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            upstream = data / 'vulkan/implicit_layer.d/VkLayer_DLSS5.json'
            upstream.parent.mkdir(parents=True)
            upstream.write_text('upstream sentinel')
            def run(*args, ok=True):
                result = subprocess.run(['python3', str(SCRIPT), '--cli', str(CLI), *args], env=scratch_env(root, data), capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, ok, result.stderr)
            run('install', '--appdir', str(app))
            run('install', '--appdir', str(app))
            binary = data / 'neural-forge/bin/neural-forge'
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
            self.assertFalse((data / 'neural-forge/bin/neural-forge-cli').exists())

    def test_purge_removes_everything_ours_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            app = root / 'AppDir'
            identity = 'io.github.labj1987.NeuralForge'
            for name, text in {
                'bin/neural-forge': 'gui', 'bin/neural-forge-cli': 'cli',
                'lib/neural-forge/libneural_forge_layer.so': 'layer',
                'lib/neural-forge/helper/neural-forge-helper.exe': 'helper',
                f'share/applications/{identity}.desktop': '[Desktop Entry]\nExec=neural-forge\n',
                'share/icons/hicolor/scalable/apps/neural-forge.svg': '<svg/>',
                f'share/metainfo/{identity}.appdata.xml': '<component/>',
                'share/vulkan/implicit_layer.d/neural_forge_layer.json': json.dumps({'layer': {'name': 'VK_LAYER_neuralforge_neural'}}),
            }.items():
                p = app / 'usr' / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            upstream = data / 'vulkan/implicit_layer.d/VkLayer_DLSS5.json'
            upstream.parent.mkdir(parents=True)
            upstream.write_text('upstream sentinel')
            env = scratch_env(root, data)
            runtime = Path(f"/tmp/neural-forge-{env['NEURAL_FORGE_UID']}")
            def cli(*args):
                result = subprocess.run([str(CLI), *args], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            cli('install', '--appdir', str(app))
            (data / 'neural-forge/binaries').mkdir(parents=True, exist_ok=True)
            (data / 'neural-forge/binaries/nvngx_dlssnr.dll').write_text('dll')
            (root / 'config/neural-forge').mkdir(parents=True, exist_ok=True)
            (root / 'config/neural-forge/config.ini').write_text('set_intensity=1\n')
            (root / 'state/neural-forge').mkdir(parents=True, exist_ok=True)
            (root / 'state/neural-forge/helper.log').write_text('log')
            runtime.mkdir(mode=0o700, exist_ok=True)
            (runtime / 'shm.bin').write_text('shm')
            cli('uninstall', '--purge')
            for gone in [data / 'neural-forge', root / 'config/neural-forge', root / 'state/neural-forge', runtime,
                         data / f'applications/{identity}.desktop', data / 'vulkan/implicit_layer.d/neural_forge_layer.json']:
                self.assertFalse(gone.exists(), gone)
            self.assertEqual(upstream.read_text(), 'upstream sentinel')
            self.assertFalse((data / 'applications').exists(), 'an emptied directory we created should go too')

if __name__ == '__main__': unittest.main()
