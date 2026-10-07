# Reverse-engineering toolkit survey

Surveyed 2026-10-06: agent-driven (MCP) reverse-engineering tools for native binaries, starting
from [darbra/awesome-ai-reverse](https://github.com/darbra/awesome-ai-reverse) (its binary/native
section) plus tools already identified. What is installed on the test machine, where, and how to
check it is in [RUNNING_AND_MEASURING.md](RUNNING_AND_MEASURING.md), section 10.

**Scope.** The decompilers here are not used on NVIDIA's code; interface metadata and observed
behaviour are. The full rule is in [CLAUDE.md](../CLAUDE.md), "Working with NVIDIA's binaries".
[DLSSNR_PARAMETERS.md](DLSSNR_PARAMETERS.md) shows the approach: the feature's parameters come from
the helper's own read log, public headers and open-source consumers.

**Rule for installing.** Beyond the three installed first, a tool is installed only if it is free,
headless, and does something those three cannot.

| Tool | Attaches to | Headless or GUI | Licence | Installed | Why or why not |
|---|---|---|---|---|---|
| [morluto/rea](https://github.com/morluto/rea) | Ghidra, Hopper or IDA for native code; its own analysers for JavaScript/Electron, .NET, APK and web; process and UI capture | Headless (the Hopper demo runs on a private Xvfb display) | MIT | **Yes**, 4.1.0 | Overview and evidence: `inspect-artifact`, `analyze`, `search`, `evidence-export`. Its Ghidra path rejects Windows DLLs; the Hopper demo cannot save and stops after 30 minutes. |
| [mrphrazer/ghidra-headless-mcp](https://github.com/mrphrazer/ghidra-headless-mcp) | Ghidra through pyghidra | Headless | GPL-2.0 | **Yes**, 0.1.0 with Ghidra 12.1.4 | The main decompiler: 212 tools (decompile, function list, strings, references, renaming, types, patching with undo). Has a fake backend for testing and a CLI (`ghidra_cli`) with a persistent server. |
| CUDA `cuobjdump` / `nvdisasm` | CUDA fat binaries and cubins | CLI | NVIDIA CUDA Toolkit EULA | **Yes**, 13.4.92 | Lists and disassembles GPU code in binaries whose licence allows it (your own CUDA builds, open-source projects). Installed from NVIDIA's redistributable tarballs, not Ubuntu's `nvidia-cuda-toolkit` (too old for `sm_120`, and it pulls driver libraries over the runfile driver). |
| [LaurieWired/GhidraMCP](https://github.com/LaurieWired/GhidraMCP) | A Ghidra GUI plugin plus a Python MCP bridge | GUI (the Ghidra CodeBrowser must be open) | Apache-2.0 | No | Needs the Ghidra GUI running; ghidra-headless-mcp covers the same and more without it. Last pushed 2025-06. |
| [radareorg/radare2-mcp](https://github.com/radareorg/radare2-mcp) (r2mcp) | radare2, optionally r2ghidra | Headless | MIT | No | The fallback if pyghidra had failed to build; it built. Its one edge (fast disassembly without a long auto-analysis) is covered by opening in Ghidra with `--update-analysis false`. |
| [mrexodia/ida-pro-mcp](https://github.com/mrexodia/ida-pro-mcp) | IDA Pro plugin | GUI (IDA) | MIT (IDA is paid) | No | Paid disassembler. REA can use it through `REA_IDA_MCP_CONFIG` if IDA is ever bought. |
| [blacktop/ida-mcp-rs](https://github.com/blacktop/ida-mcp-rs) | IDA Pro via idalib | Headless | MIT (IDA is paid) | No | Paid disassembler. |
| [P4nda0s/IDA-NO-MCP](https://github.com/P4nda0s/IDA-NO-MCP) | IDA Pro | GUI (IDA) | none stated | No | Paid disassembler, no licence. |
| [fosdickio/binary_ninja_mcp](https://github.com/fosdickio/binary_ninja_mcp) | Binary Ninja plugin plus bridge | GUI (Binary Ninja) | GPL-3.0 (Binary Ninja is paid) | No | Paid disassembler. |
| [buzzer-re/Rikugan](https://github.com/buzzer-re/Rikugan) | An agent inside IDA Pro or Binary Ninja | GUI | MIT | No | Needs a paid disassembler. |
| [president-xd/revula](https://github.com/president-xd/revula) | Wraps Capstone, radare2, objdump, Ghidra, RetDec, GDB, LLDB, Frida, YARA, capa | Headless | GPL-3.0 | No | A malware-triage aggregator (entropy, YARA, ATT&CK mapping). Its decompiler is the same headless Ghidra already installed; nothing here needs the rest. |
| [dnakov/frida-mcp](https://github.com/dnakov/frida-mcp) | Frida (dynamic instrumentation of a running process) | Headless | MIT | No, noted for later | The one capability the installed tools lack: watching a live process. Relevant later for observing the helper's own NGX calls under Wine (arguments, results, timing) without changing the code. Not needed today; last pushed 2025-05. |
| [zhaoxuya520/reverse-skill](https://github.com/zhaoxuya520/reverse-skill) | Prompt and skill pack, no engine | n/a | MIT | No | Routing prompts for other tools; no analysis of its own. |
| [angusdevgo/Seep-Reverse-Lab](https://github.com/angusdevgo/Seep-Reverse-Lab) | Multi-platform workbench aimed at client-side authorisation audits | Mixed | GPL-3.0 | No | Built around auditing authorisation checks, which is outside this project's scope. |

## What would change the picture

- **Frida, if the helper's own calls need watching.** The helper already logs every NGX call it
  makes (`[ngx]` and `[params]` lines, RUNNING_AND_MEASURING.md section 6), so Frida would only add
  value for a call the helper cannot log itself, such as timing inside a Vulkan entry point under
  Wine.
- **IDA, if ever bought.** REA and ida-mcp-rs would both pick it up without new plumbing.
