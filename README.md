# nimbus-cli

An agentic coding CLI powered by [NVIDIA NIM](https://build.nvidia.com) models.
Point it at a folder, describe a change in plain English, and it reads and edits
your real files on disk — across multiple files, with diffs and undo.

```
./nimbus ~/your/project
nimbus› add a --json flag to proxy.py and update the README
```

No build step. No Electron. One Python file + the `openai` SDK.

## Quick start

```bash
git clone https://github.com/igniterDJ/nimbus-cli
cd nimbus-cli
python3 -m venv .venu && .venu/bin/pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-...
./nimbus ~/your/project
```

Or drop your key in a `.env` file (gitignored):
```
NVIDIA_API_KEY=nvapi-...
NIMBUS_MODEL=qwen/qwen3.5-122b-a10b
```

## What it does

- **Reads and edits multiple files** in one request — explores, reads, then applies precise search/replace edits with a diff preview
- **Runs shell commands** — build, test, lint; verify its own work
- **Two modes**, switchable live:
  - `confirm` (default) — shows a colored diff and asks `[y]es / [n]o / [a]lways` before every file write or command
  - `auto` — applies edits and runs commands without asking
- **Session undo** — every file is snapshotted before first edit; `/undo` reverts everything, `/diff` shows the full session diff
- **`@file` mentions** — attach a file's content to your request: `explain @src/config.py`
- **Retry/backoff** on rate limits and transient API errors
- **Context trimming** — long sessions stay within the model's context window
- **Project grounding** — file tree and README/AGENTS.md injected into the system prompt

## Usage

```bash
./nimbus [folder] [--auto] [--model MODEL] [-p "one-shot prompt"]
```

| Flag | Description |
|---|---|
| `folder` | Project folder to work in (default: current dir) |
| `--auto` | Start in autonomous mode |
| `--model` | NVIDIA NIM model to use |
| `--base-url` | Override the API base URL |
| `--api-key` | Pass the key directly (prefer `.env`) |
| `-p "..."` | Run a single request non-interactively, then exit |

## In-session commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/auto` | Switch to autonomous mode |
| `/confirm` | Switch back to confirm mode |
| `/mode` | Show current mode |
| `/model [name]` | Show or change the model |
| `/open <path>` | Switch the working folder |
| `/pwd` | Show working folder |
| `/files` | Print the file tree |
| `/diff` | Show every change made this session |
| `/undo` | Revert all of this session's file changes |
| `/clear` | Clear conversation history |
| `/exit` | Quit |

## Models

The default is `meta/llama-3.3-70b-instruct`. Any NVIDIA NIM model that
supports function calling works. Set a persistent default with `NIMBUS_MODEL`
in your `.env` or environment.

Good options on NIM:
- `qwen/qwen3.5-122b-a10b`
- `qwen/qwen2.5-coder-32b-instruct`
- `meta/llama-3.3-70b-instruct`
- `nvidia/llama-3.1-nemotron-70b-instruct`

nimbus handles both the **native OpenAI `tool_calls`** format (Llama) and
the **XML/Hermes text format** that Qwen models emit — so both families work.

## Architecture

```
nimbus.py    the agent loop — tools, diffs, confirms, retries
nimbus       launcher script (uses .venu if present, else python3)
```

The agent has 7 built-in tools: `list_directory`, `find_files`, `read_file`,
`write_file`, `replace_in_file`, `run_command`, `search`.

## Requirements

- Python 3.8+
- `openai >= 1.30` (`pip install -r requirements.txt`)
- An NVIDIA API key — [build.nvidia.com](https://build.nvidia.com)

## License

MIT
