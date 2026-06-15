# nimbus Upgrade Plan — implementation prompts for Sonnet

This file turns 12 requested features into **6 self-contained prompts**. Run them
**in order**, one per session. In each new session say, e.g.:

> Implement **Prompt 1** from `upgradeprompt.md`. Follow every decision in it,
> then run the acceptance tests at the end of that prompt and report results.

All hard design decisions are already made below — your job (Sonnet) is to
implement them, test them against the live NVIDIA API, and not regress existing
behavior. Do **not** re-litigate the decisions; if something is genuinely
impossible, note it and pick the closest option, then continue.

---

## Orientation: current architecture (read first, every session)

Everything lives in one file: **`nimbus.py`** (~700 lines), launched by the
`./nimbus` bash script (which runs it with this repo's `.venu` Python). Key
symbols you will touch:

- **`class Agent`** — `__init__(self, root, model, client, auto)`; holds
  `self.root` (Path), `self.model` (str), `self.client` (OpenAI SDK),
  `self.auto` (bool), `self.messages` (list[dict], OpenAI chat format),
  `self.backups` (dict[str, str|None] for `/undo`).
- **`Agent._create(self)`** — makes the chat completion with retry/backoff and a
  `Spinner`; **non-streaming**; returns the SDK response object.
- **`Agent.run_turn(self, user_input)`** — the agent loop: append user msg, loop
  up to `MAX_ITERS`: call `_create()`, read `resp.choices[0].message`, handle
  **native** `tool_calls` AND **text-format** tool calls (Qwen/Hermes XML) via
  `parse_text_tool_calls()` / `strip_tool_calls()`, dispatch tools, append
  results, repeat until no tool calls → final answer.
- **`Agent.tools(self)`** — returns the OpenAI tool schema list. Tools:
  `list_directory, find_files, read_file, write_file, replace_in_file,
  run_command, search`. Dispatched by **`Agent._dispatch(name, args)`** to
  `_tool_*` methods.
- **`Agent._confirm(self, summary, diff=None)`** — the approval gate. Returns
  `True` immediately if `self.auto`; otherwise prints diff and asks
  `[y]es / [n]o / [a]lways`. `a` flips `self.auto = True` for the session.
- **`Agent._system_prompt()` / `_project_context()`** — builds the system message;
  `_project_context` injects a 2-level file tree + an excerpt of the first
  existing file in `CONTEXT_FILES = ("AGENTS.md","CLAUDE.md","README.md","README.rst")`.
- **`Agent._trim_history()`** — current crude context control: drops oldest
  non-system messages once history > `MAX_CONTEXT_CHARS`, never leaving an
  orphaned `role:"tool"` message.
- **`Agent._expand_mentions(text)`** — expands `@path` into attached file content.
- **`Agent.undo()` / `session_diff()`** — `/undo` and `/diff`.
- **`repl(agent)` / `_handle_command(agent, line)`** — the REPL and slash commands.
- **`main()`** — argparse: `folder`, `--model`, `--auto`, `--base-url`,
  `--api-key`, `-p/--prompt`. Resolves model/base-url AFTER `load_dotenv()` so
  `NIMBUS_MODEL`/`NIMBUS_BASE_URL` (env or `.env`) take effect; explicit CLI wins.
- Constants block at top: `DEFAULT_MODEL="meta/llama-3.3-70b-instruct"`,
  `DEFAULT_BASE_URL`, `MAX_ITERS`, `MAX_TOKENS=8192`, `MAX_READ_BYTES`,
  `MAX_CMD_OUTPUT`, `CMD_TIMEOUT`, `MAX_CONTEXT_CHARS`, `RETRY_DELAYS`,
  `IGNORE_DIRS`, `CONTEXT_FILES`.

The NVIDIA API is **OpenAI-compatible** (`base_url=https://integrate.api.nvidia.com/v1`).
**Two tool-call dialects must keep working everywhere you touch the loop:**
1. **Native** `tool_calls` (e.g. `meta/llama-3.3-70b-instruct`).
2. **Text/XML** tool calls emitted in `content` (e.g. `qwen/qwen3.5-122b-a10b`).

---

## Global conventions & non-negotiables (apply to every prompt)

1. **Don't regress.** These must still pass after your change: multi-file
   read+edit; confirm mode diff+`[y/n/a]`; auto mode; `/undo` + `/diff`; `.env`
   key discovery (root → cwd → script dir); retry/backoff; the text-format
   (Qwen) AND native (Llama) tool-call paths.
2. **Test both dialects.** Any change to `run_turn`/`_create`/tools must be
   tested with BOTH `--model qwen/qwen3.5-122b-a10b` AND
   `--model meta/llama-3.3-70b-instruct`.
3. **Key/setup.** The NVIDIA key is already in this repo's gitignored `.env`
   (`NVIDIA_API_KEY=...`, `NIMBUS_MODEL=qwen/qwen3.5-122b-a10b`). Run tests with
   `.venu/bin/python nimbus.py ...`. Standard test pattern:
   ```bash
   P=/tmp/nvtest; rm -rf $P; mkdir -p $P
   .venu/bin/python nimbus.py $P --auto --model <model> -p "<task>" </dev/null 2>&1
   ```
   For commands that may exceed 2 min, run them in the background and poll.
4. **Single file by default.** Keep everything in `nimbus.py`. The ONLY allowed
   extra module is `mcp_client.py` in Prompt 6. Keep the `./nimbus` launcher working.
5. **New dependencies** go in `requirements.txt` with a comment, and must be
   present in `.venu` (install with `.venu/bin/pip install ...`). Prefer stdlib;
   the only new deps allowed are `rich` (Prompt 1) and `mcp` (Prompt 6, optional).
6. **Config & state dirs** (introduced across prompts) — use exactly these:
   - Global: `~/.nimbus/` (mode `0700`). Files: `settings.json`, `sessions/`.
   - Project: `<root>/.nimbus/settings.json` (project overrides global). Add
     `.nimbus/` to the project's `.gitignore` if a `.gitignore` exists.
7. **Always** run `.venu/bin/python -m py_compile nimbus.py` and update the
   `nimbus` section of `README.md` + the in-REPL `/help` text for any new
   command/flag.
8. **Style.** Match the existing code: ANSI color helpers (`info/warn/err`,
   `GREEN/DIM/...`), small focused methods, `from __future__ import annotations`,
   type hints, no heavy frameworks. Keep `_TTY` checks so non-tty output stays clean.

---

# Prompt 1 — Streaming output + Markdown/syntax rendering

**Goal:** model output appears token-by-token, and the final prose renders as
Markdown with syntax-highlighted code blocks.

### Decisions
- **Library:** add `rich>=13` to `requirements.txt` and install into `.venu`.
  Use `rich` ONLY for rendering model prose and the live stream; keep the
  existing ANSI `info/warn/err` status lines as-is. Create one module-level
  `from rich.console import Console; console = Console()`.
- **Streaming:** change the completion call to `stream=True` and add
  `stream_options={"include_usage": True}` (needed by Prompt 2; harmless now).
  Keep all retry/backoff logic, but only retry on errors raised **before/at**
  stream creation; once tokens flow, surface mid-stream errors normally.
- **Accumulate deltas** from chunks (`chunk.choices[0].delta`):
  - `delta.content` → append to a `content` buffer.
  - `delta.tool_calls` → accumulate by `index` into a dict
    `{index: {"id":..., "name":..., "arguments": "<concatenated str>"}}`; the
    `id`/`name` arrive on the first fragment, `arguments` stream in pieces.
  - `delta.reasoning_content` or `delta.reasoning` (some NIM models) → stream to
    the screen dimmed in real time, but DO NOT store in `self.messages`.
  - The final usage-only chunk has empty `choices`; guard with
    `if not chunk.choices: capture chunk.usage; continue`.
- **Refactor:** replace `_create()` with `_stream()` that returns a small result
  object/tuple: `(content: str, tool_calls: list[dict], usage)`. `tool_calls`
  items are in the same shape `run_turn` already builds:
  `{"id","type":"function","function":{"name","arguments"}}`.
  `run_turn` then no longer reads `resp.choices[0].message`; it consumes this.
- **Live Markdown rendering of prose:** while streaming the assistant's textual
  answer, use `rich.live.Live` with `rich.markdown.Markdown(buffer)` updating at
  `refresh_per_second=8`. On completion, leave the rendered Markdown in place.
- **Suppress raw tool-call XML in the live view:** if the streaming `content`
  buffer contains `"<tool_call>"` or `"<function="`, stop Markdown-rendering it
  and instead show a dim status line like `· preparing tool call…`. (Qwen streams
  its tool calls as content; users must not see raw XML rendered as Markdown.)
- **Spinner:** show the existing `Spinner` only until the first token/þchunk
  arrives, then stop it and start streaming.
- **Non-tty / `-p` mode:** if `not _TTY`, skip `rich.Live`; just `print` content
  as it streams (plain). Markdown rendering is tty-only.
- After the stream ends, **text-format tool-call parsing is unchanged**: run
  `parse_text_tool_calls(content)` when there were no native tool calls.

### Integration points
- `Agent._create` → `Agent._stream`. Update the single call site in `run_turn`.
- Keep `MAX_TOKENS`, `temperature=0.2`, `tools`, `tool_choice="auto"`.

### Acceptance tests
1. `py_compile` clean; `.venu/bin/pip show rich` succeeds.
2. Interactive (tty) run with `meta/llama-3.3-70b-instruct`: ask "explain
   quicksort with a Python code block" → tokens stream; final answer shows a
   highlighted code block.
3. `qwen/qwen3.5-122b-a10b`: a file-editing task (create+run a script) still
   works AND no raw `<tool_call>`/`<function=` XML is shown to the user.
4. `-p` non-interactive (piped, non-tty): output still printed, no crash, no rich
   control codes leaking.
5. Multi-file edit + `/undo` still work.

---

# Prompt 2 — Cost/token tracking + History compaction

**Goal:** show token usage per turn and per session; replace crude trimming with
real summary-based compaction. (Depends on Prompt 1's `include_usage`.)

### Decisions — token tracking
- NVIDIA NIM does not publish a simple uniform per-token price, so **track
  tokens, not dollars, by default.** Optional cost: if env
  `NIMBUS_PRICE_IN` and `NIMBUS_PRICE_OUT` are set (USD per 1M tokens), also show
  an estimated `$` figure; otherwise omit `$`.
- Add `self.usage = {"prompt":0,"completion":0,"total":0,"requests":0}` in
  `__init__`. After each stream, add the captured `usage` (prompt_tokens,
  completion_tokens, total_tokens). If a model returns no usage, estimate
  completion via `len(content)//4` and prompt via `sum(len(content))//4`.
- **Per-turn line** (dim, after the turn completes):
  `↑{prompt} ↓{completion} tok · turn │ {total} session` (+ ` │ ~${cost}` if priced).
- **Context gauge:** add `NIMBUS_CONTEXT_LIMIT` (default `128000`) constant/env.
  Track an estimate `self.context_tokens` = `sum(len(str(m.content))//4)` over
  `self.messages` (plus ~200/tool_call). Show it in `/cost` and use it for
  compaction.
- **Command `/cost`** (alias `/tokens`): print session totals, request count,
  and `context: ~X / LIMIT tokens (Y%)`.

### Decisions — history compaction (replaces `_trim_history`)
- Trigger when `context_tokens > 0.75 * NIMBUS_CONTEXT_LIMIT` (auto), or via
  `/compact` (manual, force).
- **Compaction algorithm** (preserve tool-call integrity — operate on whole
  rounds, where a "round" runs from one `role:"user"` (non-tool-result) message
  up to just before the next such user message):
  1. Always keep `messages[0]` (system).
  2. Keep the **most recent rounds** that fit in ~30% of the limit verbatim.
  3. Summarize **everything older** in ONE extra model call (use `self.model`,
     `stream=False`, low temperature) with a prompt like: *"Summarize this
     conversation for continuity. Capture: the user's goals, files
     created/edited and how, key decisions, commands run and outcomes, and any
     unfinished work. Be concise (<400 words). Output plain text."*
  4. Replace the summarized messages with a single
     `{"role":"user","content":"[Earlier conversation summary]\n<summary>"}`
     inserted right after the system message.
  5. Recompute `context_tokens`. Print a dim notice:
     `· compacted N older messages into a summary`.
- Keep the old "never leave an orphan tool message" safety as a final guard.
- The summarization call's tokens count toward `self.usage` too.

### Acceptance tests
1. After any turn, the dim token line appears with non-zero numbers; `/cost`
   shows session totals + context %.
2. Drive a long multi-turn session (or synthetically pad `self.messages`) past
   75% of a deliberately low `NIMBUS_CONTEXT_LIMIT` (e.g. export
   `NIMBUS_CONTEXT_LIMIT=4000`) and confirm auto-compaction fires, the summary
   message is present, no orphan `role:"tool"` messages remain (assert in a unit
   check), and the next turn still works.
3. `/compact` forces compaction on demand.

---

# Prompt 3 — Session persistence / resume + Persistent memory

**Goal:** sessions are saved and resumable; the project gets an `NIMBUS.md`
memory file the agent reads and can append to.

### Decisions — session persistence
- **Storage:** `~/.nimbus/sessions/` (dir `0700`). One JSON file per session:
  `{ "version":1, "id", "root", "model", "title", "created", "updated",
     "messages":[...], "usage":{...} }`. `title` = first user message, trimmed to
  60 chars. Use `time.strftime("%Y%m%dT%H%M%S")` + `uuid.uuid4().hex[:6]` for the
  id/filename (normal program — `time`/`uuid` are fine here).
- **Index:** `~/.nimbus/sessions/index.json` = list of
  `{id, root, title, model, created, updated, message_count}`; update on save.
- **Autosave:** write the session file after every turn and on exit. Create the
  session record lazily on the first user turn.
- **CLI flags** (in `main()`):
  - `--resume [ID]`: with an ID, load that session; with no ID, load the **most
    recent session whose `root` matches the current folder**.
  - `--continue`: alias for `--resume` (no ID) — most recent for this folder.
  - If `--resume` finds nothing, print a notice and start fresh.
- **REPL commands:** `/sessions` lists this folder's sessions (id, title,
  updated, msgs); `/resume <id>` loads one (replacing current `self.messages` and
  `self.usage`); `/new` starts a fresh session.
- **On resume:** restore `self.messages` and `self.usage`. Do **NOT** restore
  `self.backups` (files may have changed on disk) — `/undo` only covers edits
  made after resuming. Print: `resumed session <id> (<n> messages)`.
- Re-derive the system prompt (`messages[0]`) fresh from the CURRENT project
  context on resume (project may have changed), but keep the rest verbatim.

### Decisions — persistent memory
- **File:** `NIMBUS.md` in the project root (peer of CLAUDE.md/QWEN.md/AGENTS.md).
- **Read:** make `NIMBUS.md` the **highest-priority** context file and load it
  **in full** (cap ~4000 chars) into the system prompt under a heading
  `--- Project memory (NIMBUS.md) ---`. Update `CONTEXT_FILES` handling so
  NIMBUS.md is always included (in full) in addition to the existing README/AGENTS
  excerpt.
- **Write:** add a tool **`remember`** with arg `note: string`. It appends
  `- <note>` under a `## nimbus memory` section in `NIMBUS.md` (creating the file
  / section if missing). Subject to the same `_confirm` gate as a write (show the
  appended line). Returns confirmation.
- System-prompt guidance: tell the model to call `remember` for **durable,
  reusable** facts (build/test commands, conventions, gotchas, where things
  live) — NOT transient chat.
- **Command `/memory`:** print current `NIMBUS.md`.

### Acceptance tests
1. Run a task, exit, then `--continue` in the same folder → prior messages
   restored; ask a follow-up that depends on earlier context and confirm it
   "remembers".
2. `--resume <id>` from `/sessions` works; `/new` starts clean.
3. Ask the agent to "remember that tests run with `python3 test_math.py`" →
   `NIMBUS.md` gets the bullet; restart and confirm it's loaded into context
   (ask "how do I run the tests here?" with no other hint).
4. Index stays valid JSON; session files are under `~/.nimbus/sessions/`.

---

# Prompt 4 — Persisted permissions + Plan mode

**Goal:** allow/deny rules that survive restarts and gate commands/writes; a
read-only Plan mode.

### Decisions — settings & permissions
- **Settings files:** `~/.nimbus/settings.json` (global) and
  `<root>/.nimbus/settings.json` (project). **Project overrides global**; merge
  lists (project + global, with deny taking precedence over allow). Create files
  on demand. Add `.nimbus/` to project `.gitignore` if present.
- **Schema** (explicit lists, easy to reason about):
  ```json
  {
    "permissions": {
      "allow_commands": ["git status*", "git diff*", "ls*", "cat *", "python3 *", "npm test*"],
      "deny_commands":  ["rm -rf *", "sudo *", "git push*", ":(){*", "dd *", "mkfs*"],
      "allow_writes":   ["*"],
      "deny_writes":    [".env", "*.key", ".git/*"]
    }
  }
  ```
  Matching is `fnmatch` (case-sensitive) on the full command string / the
  project-relative path.
- **Gate logic** (in `_confirm`, or a new `_permitted(kind, target)` consulted by
  the tools BEFORE `_confirm`):
  - `kind="command"`: if any `deny_commands` matches → auto-REJECT with a red
    note (never runs, even in auto). Else if any `allow_commands` matches →
    auto-APPROVE (even in confirm mode, no prompt). Else fall through to normal
    confirm/auto behavior.
  - `kind="write"` (write_file/replace_in_file/remember): same precedence with
    `deny_writes`/`allow_writes` against the relative path.
- **Dangerous-command guard (independent of rules):** keep a built-in regex list
  (`rm\s+-rf\s+/`, `sudo`, `mkfs`, `dd\s+if=`, `:\(\)\s*\{`, `>\s*/dev/sd`,
  `chmod\s+-R\s+777`, `git\s+push\s+--force`). A match ALWAYS forces an explicit
  confirmation with a red warning, **overriding auto mode** (but still blocked
  outright if it also matches `deny_commands`).
- **Persisting from the prompt:** when the user answers `a` (always) at a confirm
  prompt, ask a follow-up: `persist as: [s]ession / [p]roject / [g]lobal / [n]o`.
  - `s` = current behavior (`self.auto=True` for session).
  - `p`/`g` = derive a rule and write it. For commands, propose
    `"<first token> *"` (e.g. `npm *`) and show it for confirmation before
    saving; for writes, propose the relative path or its glob. Save to the chosen
    settings file's correct allow list.
- **Commands:** `/permissions` prints effective merged rules; `/allow <pattern>`
  and `/deny <pattern>` add a command rule to the **project** settings file.

### Decisions — plan mode
- Add `self.plan: bool` (default False). It is independent of `self.auto`; plan
  mode is **read-only and overrides everything**.
- **Enter:** `--plan` CLI flag, or `/plan` command. **Exit:** `/build`, `/auto`,
  or `/confirm` (any acting mode). Show mode in the banner/`/mode` as one of
  `PLAN` / `CONFIRM` / `AUTO`.
- **Behavior:** in plan mode, the mutating tools `write_file`, `replace_in_file`,
  `run_command`, `remember` return
  `"BLOCKED: PLAN MODE is read-only. Investigate with read-only tools and present
  a concrete numbered plan instead."` (do not execute). Read-only tools
  (`list_directory, find_files, read_file, search`, web tools, MCP read tools)
  work normally.
- Append to the system prompt when in plan mode: *"You are in PLAN MODE.
  Investigate using read-only tools only, then output a concrete, numbered,
  step-by-step implementation plan. Do not attempt to modify files or run
  commands."* (Re-build messages[0] when mode changes, like `/open` does.)
- The plan stays in history; after `/build` the user can say "go" and the model
  executes it normally.

### Acceptance tests
1. Add `deny_commands: ["rm -rf *"]` (project) → ask the agent to `rm -rf` a temp
   subdir in auto mode → it is BLOCKED, not run. Add `allow_commands:
   ["ls*"]` → `ls` runs in confirm mode with no prompt.
2. Dangerous guard: even in `--auto`, a `sudo`/`rm -rf /`-style command forces a
   red confirmation.
3. `a` → `p` at a prompt writes a rule to `<root>/.nimbus/settings.json`; restart
   and confirm the rule auto-approves.
4. `--plan`: ask it to "refactor X" → it produces a numbered plan and makes NO
   edits (verify files unchanged). `/build` then "go" → it executes.

---

# Prompt 5 — Line-range reads + Repo map / code graph

**Goal:** navigate large files by line range; give the model a compact symbol map
of the repo.

### Decisions — line-range reads
- Extend the `read_file` tool schema with optional `offset` (1-based start line)
  and `limit` (max lines). Default = whole file (current capped behavior).
- When `offset`/`limit` given, return only that slice, with **correct original
  line numbers** in the `N\t` prefix.
- For a whole-file read of a file with **> 2000 lines**, return the first 2000
  lines plus a trailing note: `... file has <N> lines; call read_file with offset/limit to read more.`
- Keep `MAX_READ_BYTES` as the byte cap.

### Decisions — repo map / code graph
- **Function `build_repo_map(root) -> str`** (module-level). Walk non-ignored
  files (respect `IGNORE_DIRS`), cap at **400 files** and skip files > 500 KB.
  For each file, extract top-level definitions:
  - **Python (`.py`):** use stdlib `ast` — module-level `def`/`async def`/`class`,
    and one level of methods inside classes. Render signatures
    (`def name(args)`, `class Name`).
  - **Other languages** (`.js,.ts,.tsx,.jsx,.go,.rs,.java,.c,.cpp,.h,.rb,.php`):
    lightweight regex extraction of common definition starts:
    `function name`, `class Name`, `def name`, `func name`, `type Name`,
    `struct Name`, `const X =`, `export (default )?(function|class|const) ...`.
    Best-effort; skip on no matches.
  - Output a compact indented tree:
    ```
    proxy.py
      class Proxy
        def do_GET(self)
        def do_POST(self)
    nimbus.py
      class Agent
        def run_turn(self, user_input)
    ```
- **Tool `repo_map`** (no args, optional `path` to scope a subtree): returns the
  full map (cap ~20000 chars).
- **System-prompt injection:** in `_project_context`, ADD a truncated repo map
  (cap ~6000 chars; include the most top-level files first). Keep the existing
  file tree too. Cache the map in `self._repo_map` (compute once per session;
  recompute on `/open` and on `/map --refresh`).
- **Command `/map`** prints the map; `/map --refresh` recomputes.

### Acceptance tests
1. `read_file` with `offset=10, limit=5` returns exactly those 5 lines with
   line numbers 10–14; whole-file read of a >2000-line file shows the truncation
   note.
2. `build_repo_map` unit check on THIS repo lists `class Agent` and its methods
   from `nimbus.py` and `class Proxy` from `proxy.py`.
3. `/map` prints the tree; system prompt contains a (truncated) map (inspect by
   asking the model "what are the main classes in this project?" and getting a
   correct answer without it reading files).
4. Both model dialects still complete an edit task.

---

# Prompt 6 — Web search/fetch + MCP support

**Goal:** let the agent fetch web pages, run web searches, and use external MCP
tool servers. (Largest prompt — implement web tools first, then MCP.)

### Decisions — web tools (stdlib only)
- **Tool `web_fetch`** (`url: string`): fetch http/https with `urllib.request`,
  a browser-like `User-Agent`, 20s timeout. Reject non-http(s). Cap download at
  2 MB (check `Content-Length`, and hard-stop reading at the cap). Convert HTML
  to readable text with a stdlib `html.parser.HTMLParser` subclass that drops
  `script`/`style` and collapses whitespace; return up to ~10000 chars. Non-HTML
  (json/text) returned as-is (capped).
- **Tool `web_search`** (`query: string`): provider auto-detection by env:
  1. `TAVILY_API_KEY` → POST `https://api.tavily.com/search` (best for LLMs).
  2. else `BRAVE_API_KEY` → Brave Search API.
  3. else fallback: GET `https://html.duckduckgo.com/html/?q=...` and scrape the
     top results (title + url + snippet) with the HTMLParser. Best-effort; if it
     fails, return a clear message telling the user to set `TAVILY_API_KEY`.
  Return the top **5** results as `title — url\n  snippet`.
- **Gating:** web tools make network calls (outward-facing). In confirm mode,
  `_confirm` them showing the URL/query; in auto mode allow. Add optional
  `allow_web`/`deny_web` lists to the permissions schema (Prompt 4) matched by
  fnmatch on the URL/query; default allow.
- Document the optional `TAVILY_API_KEY` in README.

### Decisions — MCP (Model Context Protocol)
- **Scope:** support **stdio** MCP servers (the common case). HTTP/SSE is a
  stretch goal — only if straightforward; otherwise note it as unsupported.
- **Dependency:** `mcp` Python SDK (`.venu/bin/pip install mcp`); add to
  `requirements.txt` as an **optional** dep with a comment. If `mcp` is not
  importable OR no servers are configured, skip ALL MCP logic silently.
- **Config:** read `mcpServers` from the merged settings files (Prompt 4),
  using the **same schema as Claude Desktop / Claude Code** so users can paste
  existing configs:
  ```json
  { "mcpServers": {
      "filesystem": {"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","<path>"]},
      "git": {"command":"uvx","args":["mcp-server-git"]}
  }}
  ```
- **Put MCP code in `mcp_client.py`** (the one allowed extra module), imported
  lazily by `nimbus.py`. The MCP SDK is async; the launcher/agent is sync, so:
  - Start a dedicated **asyncio event loop in a background daemon thread**.
  - Provide sync wrappers using `asyncio.run_coroutine_threadsafe(coro, loop)` +
    `.result(timeout=...)` for: connecting servers, `list_tools()`, `call_tool()`.
  - A `class McpManager` that: on `connect_all(servers)` spawns each stdio server
    via the SDK's stdio client, opens a session, lists tools, and stores them;
    exposes `tool_schemas()` (OpenAI format) and `call(full_name, args) -> str`;
    and `close()` on exit. A server that fails to start logs a `warn` and is
    skipped (don't crash).
- **Tool naming / routing:** expose MCP tools to the model as
  `mcp__<server>__<tool>`. Convert each MCP tool's JSON input schema to the
  OpenAI `tools` entry. In `Agent.tools()`, append `mcp_manager.tool_schemas()`.
  In `_dispatch`, if `name.startswith("mcp__")`, route to `mcp_manager.call(...)`
  and return its text content (join text parts; note non-text content).
- **Gating:** MCP tool calls go through `_confirm` in confirm mode (show server +
  tool + args summary); auto mode allows. (Optionally honor permission rules.)
- **Lifecycle:** connect after agent init in `main()` (and after `/open`?, keep
  it simple: connect once at startup). Close on exit (REPL end and `-p` end).
- **Command `/mcp`:** list connected servers and their tool names; show errors
  for any that failed to start.

### Acceptance tests
1. `web_fetch` on a known stable URL (e.g. `https://example.com`) returns its
   visible text; `web_search "site:python.org pathlib"` returns ~5 results
   (with whatever provider is available — test the DuckDuckGo fallback if no key).
2. Web tools are confirmed in confirm mode, allowed in auto.
3. With NO `mcp` installed / no `mcpServers`: nimbus runs exactly as before
   (verify nothing breaks, no errors).
4. With `mcp` installed and a filesystem MCP server configured pointing at a temp
   dir: `/mcp` lists it; ask the agent to use the MCP filesystem tool to list/read
   a file there and confirm it works; a failing server config logs a warning and
   nimbus still starts.
5. Both model dialects still complete a normal edit task with MCP tools present
   in the schema.

---

## Final acceptance (after all 6 prompts)

Run this end-to-end smoke in a fresh temp git repo and confirm all pass:
- streaming + Markdown render of a final answer with a code block;
- a multi-file edit verified by running a command, with per-turn token line;
- `/cost`, `/compact`, `/sessions`, `/resume`, `/memory`, `/permissions`,
  `/plan`→`/build`, `/map`, `/mcp` all respond;
- `--continue` restores a prior session;
- a denied command is blocked; an allowed command runs without a prompt;
- `web_fetch` works; MCP (if configured) works; absence of MCP doesn't break anything;
- the original guarantees still hold: confirm/auto, `/undo`, `/diff`, multi-file
  edits, `.env` key discovery, retry, AND both the Qwen (text) and Llama (native)
  tool-call dialects.

Update `README.md` and `/help` to document every new flag, command, tool, and
env var (`NIMBUS_CONTEXT_LIMIT`, `NIMBUS_PRICE_IN/OUT`, `TAVILY_API_KEY`,
`BRAVE_API_KEY`). Keep `requirements.txt` accurate (`rich`, optional `mcp`).
