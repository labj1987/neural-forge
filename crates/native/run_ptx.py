# Runs one OpenDLSS-NR PTX generator under `python3 -I`. Isolated mode leaves the script's own directory off
# sys.path, and the generators import ptxgen and swin from it, so this puts it back. Bytecode is not written,
# so the build leaves no __pycache__ in third_party/.
#   python3 -I run_ptx.py <generator.py> [args...]
import os
import runpy
import sys

sys.dont_write_bytecode = True
script = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(script))
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
