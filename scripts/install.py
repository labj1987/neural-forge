#!/usr/bin/env python3
"""Install an extracted NeuralForge AppDir; remove only unchanged tracked files.
No system package, upstream path, config, runtime file or Wine prefix is removed.

install/uninstall delegate to `neural-forge-cli` (the single implementation, in
crates/supervisor/src/install.rs); archive-legacy-manifest is handled here.
"""
import argparse
import json
import os
from pathlib import Path

APP_ID = 'io.github.labj1987.NeuralForge'
LAYER = 'VK_LAYER_neuralforge_neural'  # the Vulkan layer name is frozen

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['install', 'uninstall', 'archive-legacy-manifest'])
    parser.add_argument('--appdir', type=Path)
    parser.add_argument('--legacy-manifest', type=Path)
    parser.add_argument('--cli', type=Path, help='neural-forge-cli to delegate install/uninstall to')
    args = parser.parse_args()
    data = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share')))
    root = data / 'neural-forge'
    record = root / 'installation.json'
    if args.command == 'archive-legacy-manifest':
        # This exact identity belongs to this Rust repository; upstream's NV
        # layer identity is different. Never infer ownership from directory names.
        src = args.legacy_manifest
        if src is None or src.is_symlink():
            parser.error('provide a regular --legacy-manifest file')
        layer = json.loads(src.read_text()).get('layer', {})
        if (layer.get('name') != 'VK_LAYER_dlssnr_neural'
                or Path(layer.get('library_path', '')).name != 'libdlssnr_layer.so'
                or layer.get('enable_environment') != {'VKLayer_DLSS5': '1'}
                or layer.get('disable_environment') != {'DLSSNR_DISABLE': '1'}):
            parser.error('manifest does not match this repository’s legacy identity')
        archive = root / 'legacy' / 'VK_LAYER_dlssnr_neural.json.disabled'
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open('xb') as output:
            output.write(src.read_bytes())
        src.unlink()
        print(f'Archived {src} to {archive}; config, runtime and libraries untouched')
        return
    # `install` and `uninstall` are implemented once, in Rust
    # (crates/supervisor/src/install.rs), and reached through `neural-forge-cli`. This
    # script used to carry a second copy of the same on-disk format; CI only exercised
    # that copy, so the one users actually run (the GUI's Setup tab) could drift.
    cli = find_cli(args)
    if args.command == 'install' and args.appdir is None:
        parser.error('install requires --appdir')
    command = [str(cli), args.command] + (['--appdir', str(args.appdir)] if args.command == 'install' else [])
    os.execv(str(cli), command)

def find_cli(args):
    """The neural-forge-cli to delegate to: --cli, $NEURAL_FORGE_CLI, the AppDir being
    installed, an already-installed copy, then a local cargo build."""
    repo = Path(__file__).resolve().parent.parent
    data = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share')))
    candidates = [args.cli, os.environ.get('NEURAL_FORGE_CLI') and Path(os.environ['NEURAL_FORGE_CLI'])]
    if args.appdir is not None:
        candidates.append(args.appdir / 'usr/bin/neural-forge-cli')
    candidates += [data / 'neural-forge/bin/neural-forge-cli', repo / 'target/release/neural-forge-cli', repo / 'target/debug/neural-forge-cli']
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate)
    raise SystemExit('neural-forge-cli not found; build it (cargo build -p neural-forge-cli) or pass --cli')

if __name__ == '__main__': main()
