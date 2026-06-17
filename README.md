# nimbus-cli

An agentic coding CLI powered by [NVIDIA NIM](https://build.nvidia.com) models.
Point it at a folder, describe a change in plain English, and it reads and edits
your real files on disk — across multiple files, with diffs and undo.

```
nimbus
nimbus› add a --json flag to proxy.py and update the README
```

No build step. No Electron. One Python file + the `openai` SDK.

## Quick start

**1. Clone and install**
```bash
git clone https://github.com/igniterDJ/nimbus-cli
cd nimbus-cli
python3 -m venv .venu && .venu/bin/pip install -r requirements.txt
```

**2. Create a `.env` file with your key and model**
```bash
cat > .env << 'EOF'
NVIDIA_API_KEY=nvapi-...
NIMBUS_MODEL=qwen/qwen3.5-122b-a10b
EOF
```
Get your API key at [build.nvidia.com](https://build.nvidia.com).
Full model list is there too — click any model and copy its API name.

> **Note:** `.env` is listed in `.gitignore` and is **never committed** — it holds
> your secret API key. Cloning the repo does **not** give you a `.env`; you must
> create your own as shown above. Likewise, anything you put in `.env`
> (`NIMBUS_MODEL`, `NIMBUS_MODEL_POOL`, `NIMBUS_BASE_URL`, …) stays local to your
> machine and is not uploaded to git. To share config without secrets, commit an
> `.env.example` instead.

**3. Install globally (optional)**

```bash
./nimbus --install
```

This symlinks the launcher to `~/.local/bin/nimbus`. If `~/.local/bin` is not on your PATH, the command will warn you and print the exact `export PATH` line to add to your shell config.

**4. Run it**

```bash
# From the repo directory (before or without global install)
./nimbus ~/your/project

# From any project directory (after global install)
cd ~/your/project
nimbus
```

That's it. The key and model are picked up from `.env` automatically on every run — no `export` needed.

## What it does

- **Reads and edits multiple files** in one request — explores, reads, then applies precise search/replace edits with a diff preview
- **Runs shell commands** — build, test, lint; verify its own work
- **Two modes**, switchable live:
  - `confirm` (default) — shows a colored diff and asks `[y]es / [n]o / [a]lways` before every file write or command
  - `auto` — applies edits and runs commands without asking
- **Plan mode** — read-only investigation; nimbus explores and presents a numbered plan, then `/build` to execute it
- **Session persistence** — every turn is auto-saved to `~/.nimbus/sessions/`; resume any past session with `--resume`/`--continue` or `/resume`
- **Project memory** — a `NIMBUS.md` file in your project is injected as highest-priority context; the `remember` tool appends durable notes to it
- **Permissions** — allow/deny command rules persisted in `.nimbus/settings.json`; dangerous commands (`rm -rf /`, `sudo`, `git push --force`, …) always require confirmation, even in auto mode
- **Web tools** — `web_fetch` pulls a page as readable text; `web_search` returns top results (Tavily/Brave if a key is set, else DuckDuckGo)
- **MCP support** — connect [Model Context Protocol](https://modelcontextprotocol.io) servers to add external tools (see [MCP](#mcp))
- **Repo map** — a symbol-level outline of your codebase (`/map`) injected into context for grounding
- **Session undo** — every file is snapshotted before first edit; `/undo` reverts everything, `/diff` shows the full session diff
- **`@file` mentions** — attach a file's content to your request: `explain @src/config.py`
- **Retry/backoff** on rate limits and transient API errors
- **Context trimming** — long sessions stay within the model's context window (`/compact` to force it; `/cost` to inspect usage)

## Usage

```bash
nimbus [folder] [--auto] [--plan] [--model MODEL] [-p "one-shot prompt"] [--resume [ID]] [--continue]
```

| Flag | Description |
|---|---|
| `folder` | Project folder to work in (default: current dir) |
| `--auto` | Start in autonomous mode (apply edits & run commands without asking) |
| `--plan` | Start in plan mode (read-only investigation; `/build` to execute) |
| `--model` | NVIDIA NIM model to use (default: `$NIMBUS_MODEL` or `deepseek-ai/deepseek-v4-flash`) |
| `--base-url` | Override the API base URL (`$NIMBUS_BASE_URL` or NVIDIA) |
| `--api-key` | Pass the key directly (prefer `.env`) |
| `-p "..."` | Run a single request non-interactively, then exit |
| `--resume [ID]` | Resume a session by ID, or the most recent for this folder |
| `--continue` | Resume the most recent session for this folder |
| `--install` | Install `nimbus` as a global command (symlink to `~/.local/bin`) |
| `--uninstall` | Remove the global `nimbus` command |

## In-session commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/auto` | Switch to autonomous mode |
| `/confirm` | Switch back to confirm mode |
| `/mode` | Show current mode |
| `/plan` | Enter plan mode (read-only investigation) |
| `/build` | Leave plan mode and execute |
| `/model [name]` | Show or change the model (interactive picker with no arg) |
| `/fast` `/code` `/max` | Switch model tier (small / balanced / largest) & jump to its top model |
| `/pool` | List the model tiers and show the active one (alias: `/pools`) |
| `/nextmodel` | Rotate to the next model within the active tier (alias: `/next`) |
| `/open <path>` | Switch the working folder |
| `/pwd` | Show working folder |
| `/files` | Print the file tree |
| `/map [--refresh]` | Show the repo symbol map |
| `/memory` | Print the project's `NIMBUS.md` |
| `/sessions` | List saved sessions for this folder |
| `/resume <id>` | Resume a saved session |
| `/new` | Start a fresh session |
| `/diff` | Show every change made this session |
| `/undo` | Revert all of this session's file changes |
| `/permissions` | Show current allow/deny rules |
| `/allow <pattern>` | Add a command allow rule (persisted) |
| `/deny <pattern>` | Add a command deny rule (persisted) |
| `/mcp` | Show connected MCP servers and their tools |
| `/cost` / `/tokens` | Show token usage and context size |
| `/compact` | Summarize and shrink the conversation history |
| `/clear` | Clear conversation history |
| `/exit` | Quit |

## Sessions

Every turn is saved to `~/.nimbus/sessions/` automatically. Pick up where you left off:

```bash
nimbus ~/your/project --continue      # most recent session for this folder
nimbus ~/your/project --resume         # same as --continue
nimbus ~/your/project --resume <id>    # a specific session by ID
```

Inside the REPL, `/sessions` lists them and `/resume <id>` loads one.

## Project memory — NIMBUS.md

Drop a `NIMBUS.md` file in your project root with conventions, commands, or facts
you want nimbus to always know. It's injected as the highest-priority project
context. nimbus can also write to it itself via the `remember` tool — ask it to
"remember that we run tests with `pytest -q`" and it appends the note.

## Permissions

Command execution is governed by allow/deny rules loaded from
`~/.nimbus/settings.json` (global) and `<project>/.nimbus/settings.json` (project):

```json
{
  "permissions": {
    "allow_commands": ["pytest*", "git status"],
    "deny_commands": ["curl*"]
  }
}
```

Manage them live with `/permissions`, `/allow <pattern>`, and `/deny <pattern>`
(changes persist to the project's `.nimbus/settings.json`). Dangerous commands —
`rm -rf /`, `sudo`, `mkfs`, `dd if=`, fork bombs, `chmod -R 777`, `git push --force` —
always prompt for confirmation, even in auto mode.

## Web search

`web_fetch` works out of the box. `web_search` uses, in order of preference:

- **Tavily** — set `TAVILY_API_KEY`
- **Brave** — set `BRAVE_API_KEY`
- **DuckDuckGo** — no key needed (default fallback)

## MCP

nimbus can connect to [Model Context Protocol](https://modelcontextprotocol.io)
stdio servers to expose their tools to the model. Install the optional package
and declare servers under `mcpServers` in `~/.nimbus/settings.json` or
`<project>/.nimbus/settings.json`:

```bash
.venu/bin/pip install mcp
```

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    }
  }
}
```

Use `/mcp` to see connected servers and the tools they contribute.

## Models

The default is `deepseek-ai/deepseek-v4-flash`. Any NVIDIA NIM model that
supports function calling works. Set a persistent default with `NIMBUS_MODEL`
in your `.env` or environment, or switch live with `/model`.

**Model tiers & rotation.** nimbus ships with three curated model tiers baked
into the code (`MODEL_POOLS` in `nimbus.py`), so model switching works out of the
box on a fresh clone — no `.env` needed:

| Command | Tier | For |
| --- | --- | --- |
| `/fast` | small / low-latency | quick edits & chat |
| `/code` | mid-size daily drivers | balanced coding & agents (**default**) |
| `/max`  | largest / most capable | hard reasoning & long-horizon work |

`/fast`, `/code`, `/max` switch the active tier and jump to its top model. Within
a tier, step to the next model with `/nextmodel` (alias `/next`) — handy for
comparing models or escaping a slow one; rotation wraps around. nimbus also walks
the active tier automatically when a model is rate-limited or degraded. Use
`/pool` to list the tiers and see which is active. nimbus starts in `/code`.

To use your own flat pool instead, set `NIMBUS_MODEL_POOL` in `.env` to a
comma-separated list of model IDs (best/fastest first). It **overrides the tiers**
with a single custom pool that `/nextmodel` rotates:

```bash
NIMBUS_MODEL=moonshotai/kimi-k2.6
NIMBUS_MODEL_POOL=moonshotai/kimi-k2.6,deepseek-ai/deepseek-v4-flash,qwen/qwen3.5-122b-a10b
```

Good options on NIM:
- `deepseek-ai/deepseek-v4-flash` — default, fast coding & agents, 1M context
- `qwen/qwen3.5-122b-a10b` — 122B (10B active), tool calling, coding
- `openai/gpt-oss-120b` — reasoning, coding (strong for code analysis)
- `nvidia/nemotron-3-super-120b-a12b` — agentic, tool calling
- `openai/gpt-oss-20b` — efficient reasoning MoE

nimbus handles both the **native OpenAI `tool_calls`** format (Llama) and
the **XML/Hermes text format** that Qwen and other models emit — so both
families work.

## Architecture

```
nimbus.py        the agent loop — tools, diffs, confirms, retries, sessions, permissions
mcp_client.py    MCP stdio client — discovers and proxies external MCP server tools
nimbus           launcher script (uses .venu if present, else python3)
NIMBUS.md        optional per-project memory injected into the system prompt
```

The agent has 11 built-in tools: `list_directory`, `find_files`, `read_file`,
`write_file`, `replace_in_file`, `run_command`, `search`, `repo_map`,
`web_fetch`, `web_search`, `remember` — plus any tools contributed by connected
MCP servers.

## Tests

A stdlib `unittest` suite covers the pure helpers and the Agent's file-editing
and safety logic (path-escape refusal, permission rules, plan-mode read-only,
whitespace-tolerant edits, undo, the repetition guard, and tool-call parsing):

```bash
.venu/bin/python -m unittest discover -s tests -v
```

These run fully offline — no API key or network needed.

## Requirements

- Python 3.8+
- `openai >= 1.30`, `rich >= 13`, `prompt_toolkit >= 3.0` (`pip install -r requirements.txt`)
- Optional: `mcp >= 1.0` for MCP server support
- An NVIDIA API key — [build.nvidia.com](https://build.nvidia.com)

## License

MIT — see [LICENSE](LICENSE).
