#!/usr/bin/env python3
"""Exercise install/uninstall and provenance gates entirely in temporary paths."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('install.py')
CLI = Path(os.environ.get('NEURAL_FORGE_CLI', Path(__file__).resolve().parent.parent / 'target/debug/neural-forge-cli'))
def scratch_env(root, data):
    # Every XDG home lives in the scratch dir: the CLI migrates old `neuralforge`
    # directories at startup and must never see the real ones. The unique uid names a
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
            self.assertTrue((data / 'neural-forge/legacy/VK_LAYER_dlssnr_neural.json.disabled').exists())

    def test_upgrade_from_0_1_76_layout(self):
        # What a 0.1.76 install left behind: neuralforge dirs, lib/neuralforge/libneuralforge_layer.so,
        # VK_LAYER_neuralforge_neural.json, all recorded in installation.json.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, app = root / 'data', root / 'AppDir'
            identity = 'io.github.labj1987.NeuralForge'
            new = {
                'bin/neural-forge': 'gui', 'bin/neural-forge-cli': 'cli',
                'lib/neural-forge/libneural_forge_layer.so': 'layer',
                'lib/neural-forge/helper/neural-forge-helper.exe': 'helper',
                f'share/applications/{identity}.desktop': '[Desktop Entry]\nExec=neural-forge\n',
                'share/icons/hicolor/scalable/apps/neural-forge.svg': '<svg/>',
                f'share/metainfo/{identity}.appdata.xml': '<component/>',
                'share/vulkan/implicit_layer.d/neural_forge_layer.json': json.dumps({'layer': {'name': 'VK_LAYER_neuralforge_neural'}}),
            }
            for name, text in new.items():
                p = app / 'usr' / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            old_root = data / 'neuralforge'
            lib = old_root / 'lib/neuralforge/libneuralforge_layer.so'
            manifest = data / 'vulkan/implicit_layer.d/VK_LAYER_neuralforge_neural.json'
            old_files = {
                old_root / 'bin/neural-forge': 'old gui', old_root / 'bin/neural-forge-cli': 'old cli',
                lib: 'old layer', old_root / 'lib/neuralforge/helper/neural-forge-helper.exe': 'old helper',
                manifest: json.dumps({'layer': {'name': 'VK_LAYER_neuralforge_neural', 'library_path': str(lib)}}),
            }
            record = {}
            for path, text in old_files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                record[str(path)] = hashlib.sha256(text.encode()).hexdigest()
            (old_root / 'installation.json').write_text(json.dumps(record))
            (old_root / 'prefix/pfx').mkdir(parents=True)
            (old_root / 'prefix/pfx/user.reg').write_text('registry')
            inode = (old_root / 'prefix').stat().st_ino
            cfg = root / 'config/neuralforge'
            cfg.mkdir(parents=True)
            (cfg / 'config.ini').write_text(f'binaries={old_root}/binaries\n')
            (root / 'state/neuralforge').mkdir(parents=True)
            (root / 'state/neuralforge/helper.log').write_text('log')
            result = subprocess.run(['python3', str(SCRIPT), '--cli', str(CLI), 'install', '--appdir', str(app)],
                                    env=scratch_env(root, data), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(old_root.exists())
            self.assertFalse((root / 'config/neuralforge').exists() or (root / 'state/neuralforge').exists())
            new_root = data / 'neural-forge'
            self.assertEqual((new_root / 'prefix').stat().st_ino, inode)  # renamed, not copied
            self.assertEqual((new_root / 'prefix/pfx/user.reg').read_text(), 'registry')
            self.assertEqual((root / 'config/neural-forge/config.ini').read_text(), f'binaries={new_root}/binaries\n')
            self.assertEqual((root / 'state/neural-forge/helper.log').read_text(), 'log')
            self.assertEqual((new_root / 'bin/neural-forge').read_text(), 'gui')
            self.assertEqual((new_root / 'lib/neural-forge/libneural_forge_layer.so').read_text(), 'layer')
            self.assertFalse((new_root / 'lib/neuralforge').exists())
            self.assertFalse(manifest.exists())
            self.assertTrue((data / 'vulkan/implicit_layer.d/neural_forge_layer.json').is_file())
            layers = sorted(p.name for p in (data / 'vulkan/implicit_layer.d').iterdir())
            self.assertEqual(layers, ['neural_forge_layer.json'])
            recorded = json.loads((new_root / 'installation.json').read_text())
            self.assertTrue(all(Path(p).exists() and 'neuralforge' not in p for p in recorded), recorded)
            self.assertIn('migrated', result.stderr)

if __name__ == '__main__': unittest.main()
