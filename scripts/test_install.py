#!/usr/bin/env python3
"""Exercise install/uninstall and provenance gates entirely in temporary paths."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('install.py')
ROOT = Path(__file__).resolve().parent.parent
# The debug CLI where cargo builds it from the repo root (a relative CARGO_TARGET_DIR is relative to that).
CLI = Path(os.environ.get('NEURAL_FORGE_CLI', ROOT / os.environ.get('CARGO_TARGET_DIR', 'target') / 'debug/neural-forge-cli'))
def scratch_env(root, data):
    # Every XDG home lives in the scratch dir, so the CLI never sees the real ones. The unique uid names a
    # runtime dir that cannot belong to a real session.
    # HOME too: purge looks for Steam (and its launch options) under it.
    return {**os.environ, 'HOME': str(root / 'home'), 'XDG_DATA_HOME': str(data), 'XDG_CONFIG_HOME': str(root / 'config'),
            'XDG_STATE_HOME': str(root / 'state'), 'NEURAL_FORGE_UID': f'installtest-{os.getpid()}'}
def steam_running():
    # What the CLI checks before it edits Steam's launch options: a `steam` or `steamwebhelper` process.
    for pid in Path('/proc').iterdir():
        if pid.name.isdigit():
            try:
                if (pid / 'comm').read_text().rstrip('\n') in ('steam', 'steamwebhelper'):
                    return True
            except OSError:
                pass
    return False
class InstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The tests drive the real CLI: build the debug one when it is the default and missing.
        if 'NEURAL_FORGE_CLI' not in os.environ and not CLI.is_file():
            subprocess.run(['cargo', 'build', '--locked', '-p', 'neural-forge-cli'], cwd=ROOT, check=True)

    def test_coexistence_and_owned_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            app = root / 'AppDir'
            identity = 'io.github.labj1987.NeuralForge'
            contents = {
                'bin/neural-forge': 'gui', 'bin/neural-forge-cli': 'cli',
                'lib/neural-forge/libneural_forge_layer.so': 'layer',
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
            # Outside the scratch dir: gone after the test whichever way it ends (with Steam running the purge
            # is refused and leaves it).
            self.addCleanup(shutil.rmtree, runtime, ignore_errors=True)
            def cli(*args):
                result = subprocess.run([str(CLI), *args], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            cli('install', '--appdir', str(app))
            (data / 'neural-forge/binaries').mkdir(parents=True, exist_ok=True)
            (data / 'neural-forge/binaries/nvngx_dlssnr.dll').write_text('dll')
            (root / 'config/neural-forge').mkdir(parents=True, exist_ok=True)
            (root / 'config/neural-forge/config.ini').write_text('set_intensity=1\n')
            (root / 'state/neural-forge').mkdir(parents=True, exist_ok=True)
            (root / 'state/neural-forge/state.txt').write_text('state')
            runtime.mkdir(mode=0o700, exist_ok=True)
            (runtime / 'shm.bin').write_text('shm')
            # What the install really put in place (and nothing for menu integration: that is the
            # AppImage integrator's job).
            installed = [data / 'neural-forge/bin/neural-forge', data / 'neural-forge/bin/neural-forge-cli',
                         data / 'neural-forge/lib/neural-forge/libneural_forge_layer.so',
                         data / 'vulkan/implicit_layer.d/neural_forge_layer.json', data / 'neural-forge/installation.json']
            for path in installed:
                self.assertTrue(path.is_file(), path)
            self.assertFalse((data / 'applications').exists(), 'the installer writes no menu entry')
            # A Steam account with a game switched on, and the copies of its localconfig.vdf an edit left.
            steam_config = root / 'home/.local/share/Steam/userdata/1/config'
            (root / 'home/.local/share/Steam/steamapps').mkdir(parents=True)
            steam_config.mkdir(parents=True)
            localconfig = steam_config / 'localconfig.vdf'
            text = ('"UserLocalConfigStore"\n{\n\t"Software"\n\t{\n\t\t"Valve"\n\t\t{\n\t\t\t"Steam"\n\t\t\t{\n\t\t\t\t"apps"\n'
                    '\t\t\t\t{\n\t\t\t\t\t"3240220"\n\t\t\t\t\t{\n\t\t\t\t\t\t"LaunchOptions"\t\t"NEURAL_FORGE_ENABLE=1 '
                    'NEURAL_FORGE_TARGET_EXE=G.exe gamemoderun %command%"\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n}\n')
            localconfig.write_text(text)
            copies = [steam_config / f'localconfig.vdf.neural-forge.{suffix}' for suffix in ('orig', 'bak', 'lock')]
            for copy in copies:
                copy.write_text(text)
            # A lost record: purge still takes the manifest that would load the deleted library.
            (data / 'neural-forge/installation.json').unlink()
            if steam_running():
                # This machine's Steam is running (the purge reads the real /proc): refused, nothing removed.
                result = subprocess.run([str(CLI), 'uninstall', '--purge'], env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('app 3240220: Steam is running', result.stderr)
                self.assertIn('nothing was removed', result.stderr)
                self.assertEqual(localconfig.read_text(), text)
                for kept in installed[:-1] + copies:
                    self.assertTrue(kept.exists(), kept)
                return
            cli('uninstall', '--purge')
            for gone in installed + copies + [data / 'neural-forge', root / 'config/neural-forge', root / 'state/neural-forge', runtime]:
                self.assertFalse(gone.exists(), gone)
            self.assertNotIn('NEURAL_FORGE', localconfig.read_text())
            self.assertIn('"gamemoderun %command%"', localconfig.read_text())
            self.assertEqual(upstream.read_text(), 'upstream sentinel')

if __name__ == '__main__': unittest.main()
