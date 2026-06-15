#!/usr/bin/env python3
"""nimbus — a minimal, NVIDIA-powered agentic coding CLI.

Open a folder, describe a change in plain English, and nimbus reads and edits
the *real* files on disk (and can run shell commands to build/test) using
NVIDIA NIM models over the OpenAI-compatible API at integrate.api.nvidia.com.

It is the agentic sibling of this repo's `proxy.py` chat console: same
NVIDIA-first, no-framework spirit, but it actually changes your code.

Two modes, switchable live:
  - confirm  : show a diff and ask before every file write / shell command
  - auto     : apply edits and run commands without asking (lean on /undo or git)

Run:
    export NVIDIA_API_KEY=nvapi-...
    python3 nimbus.py [folder]        # defaults to the current directory
    python3 nimbus.py [folder] --auto # start in autonomous mode

Type /help inside the session for commands.
"""
from __future__ import annotations

import argparse
import difflib
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit(
        "The 'openai' package is required.\n"
        "  pip install openai   (or use this repo's .venu: .venu/bin/python nimbus.py)"
    )

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"  # current, widely available, tool-calling
# Other good NVIDIA NIM models that support function calling — switch with /model:
#   qwen/qwen2.5-coder-32b-instruct
#   qwen/qwen3.5-122b-a10b
#   nvidia/llama-3.1-nemotron-70b-instruct
#   deepseek-ai/deepseek-v3
# Set a persistent default with:  export NIMBUS_MODEL=...  (or NIMBUS_MODEL in .env)

MAX_ITERS = 60            # tool-call rounds per user turn before we stop
MAX_TOKENS = 8192         # completion cap (prevents truncated tool calls)
MAX_READ_BYTES = 200_000
MAX_MENTION_BYTES = 60_000  # cap per @file injected into a prompt
MAX_CMD_OUTPUT = 16_000   # chars of command output fed back to the model
CMD_TIMEOUT = 240         # seconds per shell command
MAX_CONTEXT_CHARS = 160_000  # trim old turns once history grows past this
RETRY_DELAYS = (1, 2, 4, 8, 16)  # backoff schedule for transient API errors

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venu", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".idea", ".vscode",
    ".obsidian", ".ipynb_checkpoints",
}
CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md", "README.md", "README.rst")

# ----------------------------------------------------------------------------- colors
_TTY = sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _TTY else ""


RESET = _c("\033[0m")
BOLD = _c("\033[1m")
DIM = _c("\033[2m")
RED = _c("\033[31m")
GREEN = _c("\033[32m")
YELLOW = _c("\033[33m")
BLUE = _c("\033[34m")
CYAN = _c("\033[36m")
MAGENTA = _c("\033[35m")


def info(msg: str) -> None:
    print(f"{CYAN}{msg}{RESET}")


def warn(msg: str) -> None:
    print(f"{YELLOW}{msg}{RESET}")


def err(msg: str) -> None:
    print(f"{RED}{msg}{RESET}")


class Spinner:
    """A tiny background spinner shown while we wait on the API."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "thinking"):
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if _TTY:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        i = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{DIM}{self.FRAMES[i % len(self.FRAMES)]} {self.label}…{RESET}")
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.1)

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
        if _TTY:
            sys.stdout.write("\r" + " " * (len(self.label) + 12) + "\r")
            sys.stdout.flush()


# --------------------------------------------------------------- text-format tool calls
_TC_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FN_BLOCK = re.compile(r"<function=.*?</function>", re.DOTALL)
_FN_NAME = re.compile(r"<function=([^>\s]+)")
_PARAM = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)
_BOOL_KEYS = {"replace_all"}
_INT_KEYS = {"depth"}


def parse_text_tool_calls(content: str):
    """Parse tool calls a model emitted as TEXT instead of via the native
    tool_calls field (Qwen / Hermes style). Handles the XML form
    (<function=name><parameter=key>val</parameter>) and a JSON form
    (<tool_call>{"name":..., "arguments":{...}}</tool_call>).

    Returns a list of (name, args_dict). Empty if none found.
    """
    if not content or ("<tool_call>" not in content and "<function=" not in content):
        return []
    blocks = _TC_BLOCK.findall(content) or _FN_BLOCK.findall(content)
    calls = []
    for block in blocks:
        stripped = block.strip()
        if stripped.startswith("{"):  # JSON form
            try:
                obj = json.loads(stripped)
                args = obj.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append((obj["name"], args if isinstance(args, dict) else {}))
                continue
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        name_m = _FN_NAME.search(block)
        if not name_m:
            continue
        args = {}
        for key, raw in _PARAM.findall(block):
            val = raw.strip("\n")
            if key in _BOOL_KEYS:
                args[key] = val.strip().lower() in ("true", "1", "yes")
            elif key in _INT_KEYS and val.strip().lstrip("-").isdigit():
                args[key] = int(val.strip())
            else:
                args[key] = val
        calls.append((name_m.group(1), args))
    return calls


def strip_tool_calls(content: str) -> str:
    """Remove tool-call XML from content so leftover prose can be shown cleanly."""
    out = _TC_BLOCK.sub("", content)
    out = _FN_BLOCK.sub("", out)
    return out.strip()


# ----------------------------------------------------------------------------- env / key
def load_dotenv(root: Path) -> None:
    """Load KEY=VALUE pairs from .env files, without overriding the real env.

    Searches the opened project folder, the current directory, and the
    directory nimbus.py lives in — so your key is found whether it sits next
    to the tool or inside the project you're editing. The real environment
    always wins (setdefault), and earlier files win over later ones.
    """
    script_dir = Path(__file__).resolve().parent
    seen: set[Path] = set()
    for base in (root, Path.cwd(), script_dir):
        env_path = (base / ".env").resolve()
        if env_path in seen or not env_path.is_file():
            continue
        seen.add(env_path)
        for line in env_path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def resolve_api_key(root: Path, cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        return key
    warn("No NVIDIA_API_KEY found (checked env and .env).")
    info("Get one at https://build.nvidia.com — it looks like nvapi-...")
    try:
        key = getpass.getpass("Paste your NVIDIA API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nNo key provided.")
    if not key:
        sys.exit("No key provided.")
    if _yes_no("Save it to ./.env (gitignored) so you don't have to paste it again?"):
        with (root / ".env").open("a") as f:
            f.write(f"\nNVIDIA_API_KEY={key}\n")
        info("Saved to .env")
    return key


def _yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        ans = input(f"{prompt} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not ans:
        return default_yes
    return ans in ("y", "yes")


# ----------------------------------------------------------------------------- the agent
class Agent:
    def __init__(self, root: Path, model: str, client: OpenAI, auto: bool):
        self.root = root
        self.model = model
        self.client = client
        self.auto = auto
        self.backups: dict[str, str | None] = {}  # path -> original text (None = new file)
        self.messages: list[dict] = [{"role": "system", "content": self._system_prompt()}]

    # ---- system prompt (with project grounding)
    def _system_prompt(self) -> str:
        base = (
            "You are nimbus, a precise agentic coding assistant working directly on the "
            f"user's files inside the project root: {self.root}\n\n"
            "You can read, create, and edit multiple files and run shell commands using the "
            "provided tools. Guidelines:\n"
            "- ALWAYS read a file with read_file before editing it; never guess its contents.\n"
            "- Prefer replace_in_file for edits (give enough surrounding context that old_string "
            "is unique). Use write_file only to create new files or fully rewrite small ones.\n"
            "- Use find_files/search to locate code across the project before editing.\n"
            "- Make the smallest change that satisfies the request. Match the surrounding code "
            "style. Do not reformat unrelated code.\n"
            "- Work step by step: explore, read what you need, edit, then if useful run a command "
            "to verify (build/lint/tests).\n"
            "- Paths are relative to the project root unless absolute.\n"
            "- When done, give a brief plain-text summary of what you changed and why. Do not "
            "dump entire files back to the user.\n"
        )
        return base + self._project_context()

    def _project_context(self) -> str:
        parts = ["\n--- Project layout (top levels) ---\n"]
        parts.append(self._tool_list_directory(".", 2)[:2500])
        for name in CONTEXT_FILES:
            p = self.root / name
            if p.is_file():
                try:
                    snippet = p.read_text(errors="replace")[:1500]
                except Exception:
                    break
                parts.append(f"\n\n--- {name} (excerpt) ---\n{snippet}")
                break
        return "".join(parts)

    # ---- path safety
    def _resolve(self, path: str) -> Path:
        p = Path(path)
        p = (self.root / p).resolve() if not p.is_absolute() else p.resolve()
        return p

    def _inside_root(self, p: Path) -> bool:
        try:
            p.relative_to(self.root)
            return True
        except ValueError:
            return False

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    # ---- tool schemas (OpenAI function-calling format)
    def tools(self) -> list[dict]:
        def fn(name, desc, props, required):
            return {
                "type": "function",
                "function": {
                    "name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            }

        s = {"type": "string"}
        return [
            fn("list_directory", "List files and folders (tree) under a path in the project.",
               {"path": {**s, "description": "Relative path; default project root."},
                "depth": {"type": "integer", "description": "How many levels deep (default 2)."}},
               []),
            fn("find_files", "Find files by glob pattern, e.g. '**/*.py' or 'src/**/*.js'.",
               {"pattern": {**s, "description": "Glob pattern, relative to project root."}},
               ["pattern"]),
            fn("read_file", "Read a text file. Returns its contents with line numbers.",
               {"path": {**s, "description": "File path relative to project root."}},
               ["path"]),
            fn("write_file", "Create a new file or fully overwrite an existing one.",
               {"path": s, "content": {**s, "description": "Full new file contents."}},
               ["path", "content"]),
            fn("replace_in_file",
               "Replace an exact substring in a file. old_string must match exactly and be "
               "unique unless replace_all is true.",
               {"path": s,
                "old_string": {**s, "description": "Exact text to find (include context)."},
                "new_string": {**s, "description": "Replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence."}},
               ["path", "old_string", "new_string"]),
            fn("run_command", "Run a shell command in the project root. Returns combined output.",
               {"command": {**s, "description": "The shell command to run."}},
               ["command"]),
            fn("search", "Search file contents for a regex/text pattern (like grep).",
               {"pattern": s, "path": {**s, "description": "Where to search; default root."}},
               ["pattern"]),
        ]

    # ---- confirmation gate
    def _confirm(self, summary: str, diff: str | None = None) -> bool:
        if self.auto:
            return True
        print()
        print(f"{BOLD}{summary}{RESET}")
        if diff:
            print(diff)
        while True:
            try:
                ans = input(f"{YELLOW}Apply? [y]es / [n]o / [a]lways (switch to auto): {RESET}").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return False
            if ans in ("y", "yes", ""):
                return True
            if ans in ("n", "no"):
                return False
            if ans in ("a", "always"):
                self.auto = True
                warn("Switched to AUTO mode for the rest of this session (/confirm to revert).")
                return True

    @staticmethod
    def _diff(old: str, new: str, path: str) -> str:
        lines = difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}",
        )
        out = []
        for ln in lines:
            if ln.startswith("+") and not ln.startswith("+++"):
                out.append(f"{GREEN}{ln.rstrip()}{RESET}")
            elif ln.startswith("-") and not ln.startswith("---"):
                out.append(f"{RED}{ln.rstrip()}{RESET}")
            elif ln.startswith("@@"):
                out.append(f"{CYAN}{ln.rstrip()}{RESET}")
            else:
                out.append(ln.rstrip())
        return "\n".join(out) if out else f"{DIM}(no changes){RESET}"

    def _backup(self, p: Path) -> None:
        """Snapshot a file's original content the first time we touch it."""
        key = str(p)
        if key not in self.backups:
            self.backups[key] = p.read_text(errors="replace") if p.is_file() else None

    # ---- tool implementations
    def _tool_list_directory(self, path: str = ".", depth: int = 2) -> str:
        base = self._resolve(path)
        if not base.exists():
            return f"ERROR: path not found: {path}"
        if not self._inside_root(base):
            return "ERROR: path is outside the project root."
        lines: list[str] = []

        def walk(d: Path, prefix: str, level: int):
            if level > depth:
                return
            try:
                entries = sorted(d.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            except PermissionError:
                return
            for e in entries:
                if e.name in IGNORE_DIRS:
                    continue
                lines.append(f"{prefix}{e.name}{'/' if e.is_dir() else ''}")
                if e.is_dir():
                    walk(e, prefix + "  ", level + 1)

        lines.append(f"{self._rel(base) or '.'}/")
        walk(base, "  ", 1)
        return "\n".join(lines)[:MAX_READ_BYTES]

    def _tool_find_files(self, pattern: str) -> str:
        try:
            matches = []
            for p in self.root.glob(pattern):
                if p.is_file() and not any(part in IGNORE_DIRS for part in p.parts):
                    matches.append(self._rel(p))
                    if len(matches) >= 500:
                        break
        except Exception as e:
            return f"ERROR: bad glob pattern: {e}"
        return "\n".join(sorted(matches)) if matches else "(no files matched)"

    def _tool_read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            return f"ERROR: file not found: {path}"
        try:
            data = p.read_bytes()
        except Exception as e:
            return f"ERROR: cannot read {path}: {e}"
        truncated = len(data) > MAX_READ_BYTES
        if truncated:
            data = data[:MAX_READ_BYTES]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return f"ERROR: {path} is not a UTF-8 text file (binary?)."
        numbered = "\n".join(f"{i + 1}\t{ln}" for i, ln in enumerate(text.splitlines()))
        if truncated:
            numbered += f"\n... [truncated at {MAX_READ_BYTES} bytes]"
        return numbered

    def _tool_write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if not self._inside_root(p):
            return "ERROR: refusing to write outside the project root."
        old = p.read_text(errors="replace") if p.is_file() else ""
        verb = "Edit" if p.is_file() else "Create"
        diff = self._diff(old, content, self._rel(p))
        if not self._confirm(f"{verb} file: {self._rel(p)}", diff):
            return "SKIPPED: user declined the write."
        try:
            self._backup(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        except Exception as e:
            return f"ERROR: cannot write {path}: {e}"
        info(f"  ✓ wrote {self._rel(p)} ({len(content)} bytes)")
        return f"OK: wrote {self._rel(p)} ({len(content)} bytes)."

    def _tool_replace_in_file(self, path: str, old_string: str, new_string: str,
                              replace_all: bool = False) -> str:
        p = self._resolve(path)
        if not self._inside_root(p):
            return "ERROR: refusing to edit outside the project root."
        if not p.is_file():
            return f"ERROR: file not found: {path}"
        text = p.read_text(errors="replace")
        count = text.count(old_string)
        if count == 0:
            return ("ERROR: old_string not found. Read the file again and copy the exact text "
                    "including whitespace.")
        if count > 1 and not replace_all:
            return (f"ERROR: old_string matches {count} times; it must be unique. Add more "
                    "surrounding context, or set replace_all=true.")
        new_text = (text.replace(old_string, new_string) if replace_all
                    else text.replace(old_string, new_string, 1))
        diff = self._diff(text, new_text, self._rel(p))
        n = count if replace_all else 1
        if not self._confirm(f"Edit file: {self._rel(p)} ({n} replacement{'s' if n > 1 else ''})", diff):
            return "SKIPPED: user declined the edit."
        try:
            self._backup(p)
            p.write_text(new_text)
        except Exception as e:
            return f"ERROR: cannot write {path}: {e}"
        info(f"  ✓ edited {self._rel(p)} ({n} replacement{'s' if n > 1 else ''})")
        return f"OK: edited {self._rel(p)} ({n} replacement(s))."

    def _tool_run_command(self, command: str) -> str:
        if not self._confirm(f"Run command:  {MAGENTA}{command}{RESET}"):
            return "SKIPPED: user declined to run the command."
        info(f"  $ {command}")
        try:
            proc = subprocess.run(
                command, shell=True, cwd=self.root, timeout=CMD_TIMEOUT,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {CMD_TIMEOUT}s."
        except Exception as e:
            return f"ERROR: failed to run command: {e}"
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > MAX_CMD_OUTPUT:
            out = out[:MAX_CMD_OUTPUT // 2] + "\n...[truncated]...\n" + out[-MAX_CMD_OUTPUT // 2:]
        if out.strip():
            print(DIM + out.rstrip()[:4000] + RESET)
        return f"exit code: {proc.returncode}\n--- output ---\n{out}"

    def _tool_search(self, pattern: str, path: str = ".") -> str:
        base = self._resolve(path)
        if not self._inside_root(base):
            return "ERROR: path is outside the project root."
        if shutil.which("rg"):
            try:
                proc = subprocess.run(
                    ["rg", "-n", "--no-heading", "-S", pattern, str(base)],
                    capture_output=True, text=True, timeout=60,
                )
                return (proc.stdout or "(no matches)")[:MAX_CMD_OUTPUT]
            except Exception:
                pass
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"ERROR: bad regex: {e}"
        hits: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for fn in filenames:
                fp = Path(dirpath) / fn
                try:
                    for i, line in enumerate(fp.read_text(errors="ignore").splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{self._rel(fp)}:{i}: {line.strip()[:200]}")
                            if len(hits) >= 200:
                                return "\n".join(hits) + "\n...[capped at 200 matches]"
                except Exception:
                    continue
        return "\n".join(hits) if hits else "(no matches)"

    def _dispatch(self, name: str, args: dict) -> str:
        handlers = {
            "list_directory": self._tool_list_directory,
            "find_files": self._tool_find_files,
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "replace_in_file": self._tool_replace_in_file,
            "run_command": self._tool_run_command,
            "search": self._tool_search,
        }
        h = handlers.get(name)
        if not h:
            return f"ERROR: unknown tool {name}"
        try:
            return h(**args)
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"
        except Exception as e:
            return f"ERROR: {name} failed: {e}"

    @staticmethod
    def _announce(name: str, args: dict) -> None:
        if name in ("list_directory", "read_file", "search", "find_files"):
            a = args.get("path") or args.get("pattern") or ""
            print(f"{DIM}· {name} {a}{RESET}")

    # ---- context-window management
    def _trim_history(self) -> None:
        def size() -> int:
            return sum(len(str(m.get("content") or "")) + 200 * len(m.get("tool_calls", []))
                       for m in self.messages)

        while size() > MAX_CONTEXT_CHARS and len(self.messages) > 6:
            del self.messages[1]  # drop oldest non-system message
            # don't leave a leading orphan tool result (would break the API contract)
            while len(self.messages) > 1 and self.messages[1].get("role") == "tool":
                del self.messages[1]

    # ---- API call with retry/backoff
    def _create(self):
        kwargs = dict(
            model=self.model, messages=self.messages, tools=self.tools(),
            tool_choice="auto", temperature=0.2, max_tokens=MAX_TOKENS,
        )
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                with Spinner():
                    return self.client.chat.completions.create(**kwargs)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                s = str(e).lower()
                transient = any(k in s for k in (
                    "429", "rate limit", "500", "502", "503", "504", "timeout",
                    "timed out", "connection", "temporarily", "overloaded", "unavailable",
                ))
                if attempt < len(RETRY_DELAYS) and transient:
                    wait = RETRY_DELAYS[attempt]
                    warn(f"(transient API error — retrying in {wait}s: {str(e)[:90]})")
                    time.sleep(wait)
                    continue
                raise

    # ---- @file mention expansion
    def _expand_mentions(self, text: str) -> str:
        extras = []
        for tok in re.findall(r"@([^\s]+)", text):
            p = self._resolve(tok.rstrip(".,;:?!)\"'"))
            if p.is_file() and self._inside_root(p):
                try:
                    body = p.read_text(errors="replace")[:MAX_MENTION_BYTES]
                except Exception:
                    continue
                extras.append(f"--- {self._rel(p)} ---\n{body}")
        if extras:
            return text + "\n\n[Files the user attached with @]\n" + "\n\n".join(extras)
        return text

    # ---- the turn loop
    def run_turn(self, user_input: str) -> None:
        self.messages.append({"role": "user", "content": self._expand_mentions(user_input)})
        for _ in range(MAX_ITERS):
            self._trim_history()
            try:
                resp = self._create()
            except KeyboardInterrupt:
                warn("\n(interrupted — returning to prompt)")
                return
            except Exception as e:
                err(f"API error: {e}")
                return

            msg = resp.choices[0].message
            content = msg.content or ""
            native_calls = msg.tool_calls or []
            text_calls = [] if native_calls else parse_text_tool_calls(content)

            entry: dict = {"role": "assistant", "content": content}
            if native_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in native_calls
                ]
            self.messages.append(entry)

            if not native_calls and not text_calls:
                if content.strip():
                    print(f"\n{content.strip()}\n")
                return  # final answer, turn complete

            prose = content.strip() if native_calls else strip_tool_calls(content)
            if prose:
                print(f"\n{prose}\n")

            if native_calls:
                for tc in native_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    self._announce(name, args)
                    result = self._dispatch(name, args)
                    self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                          "content": str(result)})
            else:
                # Text-format path: report results back as a user message, since
                # there are no tool_call ids for role:"tool" messages to reference.
                results = []
                for name, args in text_calls:
                    self._announce(name, args)
                    result = self._dispatch(name, args)
                    results.append(f"[{name}] -> {result}")
                self.messages.append({
                    "role": "user",
                    "content": "Tool results (continue, or give your final summary):\n\n"
                               + "\n\n".join(results),
                })
        warn(f"(stopped after {MAX_ITERS} tool rounds — ask me to continue if needed)")

    # ---- session undo / diff
    def undo(self) -> None:
        if not self.backups:
            info("Nothing to undo — nimbus hasn't changed any files this session.")
            return
        n = len(self.backups)
        if not self.auto and not _yes_no(f"Revert {n} file(s) to their pre-session state?", default_yes=False):
            return
        reverted = 0
        for key, original in list(self.backups.items()):
            p = Path(key)
            try:
                if original is None:
                    if p.is_file():
                        p.unlink()
                else:
                    p.write_text(original)
                reverted += 1
            except Exception as e:
                err(f"  failed to revert {self._rel(p)}: {e}")
        info(f"Reverted {reverted} file(s).")
        self.backups.clear()

    def session_diff(self) -> str:
        if not self.backups:
            return f"{DIM}(no changes made this session){RESET}"
        out = []
        for key, original in self.backups.items():
            p = Path(key)
            cur = p.read_text(errors="replace") if p.is_file() else ""
            out.append(self._diff(original or "", cur, self._rel(p)))
        return "\n".join(out)


# ----------------------------------------------------------------------------- git safety
def git_safety_check(root: Path) -> None:
    is_repo = (root / ".git").exists() or subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=root,
        capture_output=True, text=True,
    ).stdout.strip() == "true"
    if not is_repo:
        warn("⚠ Not a git repo — your safety net is nimbus's own /undo (reverts this session). "
             "git is still recommended for real projects.")
        return
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        warn("⚠ You have uncommitted changes. Commit/stash first so you can cleanly "
             "review nimbus's edits with `git diff`.")


# ----------------------------------------------------------------------------- REPL
BANNER = f"""{BOLD}{GREEN}nimbus{RESET} — NVIDIA-powered agentic coding CLI
{DIM}reads & edits your files directly. /help for commands, /exit to quit.{RESET}"""

HELP = f"""{BOLD}Commands{RESET}
  /help            show this help
  /auto            switch to AUTONOMOUS mode (apply edits & run commands without asking)
  /confirm         switch to CONFIRM mode (ask before every change — the default)
  /mode            show the current mode
  /model [name]    show or change the NVIDIA model
  /open <path>     switch the working folder
  /pwd             show the working folder
  /files           print the file tree
  /diff            show every change nimbus made this session
  /undo            revert all of this session's file changes
  /clear           clear the conversation history (keep the folder & settings)
  /exit, /quit     leave

Use {BOLD}@path/to/file{RESET} in a request to attach that file's contents.
Anything else is a request — e.g. "add a --json flag to proxy.py and update the README"."""


def repl(agent: Agent) -> None:
    print(BANNER)
    info(f"folder: {agent.root}")
    info(f"model:  {agent.model}")
    print(f"mode:   {BOLD}{'AUTO (no confirmations)' if agent.auto else 'CONFIRM (asks first)'}{RESET}\n")
    while True:
        try:
            line = input(f"{BOLD}{BLUE}nimbus›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            continue
        if line.startswith("/"):
            if _handle_command(agent, line):
                return
            continue
        try:
            agent.run_turn(line)
        except KeyboardInterrupt:
            warn("\n(interrupted)")


def _handle_command(agent: Agent, line: str) -> bool:
    """Returns True if the session should exit."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    if cmd in ("/exit", "/quit"):
        print("bye")
        return True
    if cmd == "/help":
        print(HELP)
    elif cmd == "/auto":
        agent.auto = True
        warn("AUTO mode: edits and commands apply without confirmation.")
    elif cmd == "/confirm":
        agent.auto = False
        info("CONFIRM mode: I'll show a diff and ask before every change.")
    elif cmd == "/mode":
        print(f"mode: {BOLD}{'AUTO' if agent.auto else 'CONFIRM'}{RESET}")
    elif cmd == "/model":
        if arg:
            agent.model = arg
            info(f"model set to {arg}")
        else:
            print(f"model: {agent.model}")
    elif cmd == "/open":
        if not arg:
            err("usage: /open <path>")
        else:
            new = Path(arg).expanduser().resolve()
            if not new.is_dir():
                err(f"not a directory: {new}")
            else:
                agent.root = new
                agent.backups.clear()
                agent.messages[0] = {"role": "system", "content": agent._system_prompt()}
                info(f"folder: {new}")
                git_safety_check(new)
    elif cmd == "/pwd":
        print(agent.root)
    elif cmd == "/files":
        print(agent._tool_list_directory(".", 2))
    elif cmd == "/diff":
        print(agent.session_diff())
    elif cmd == "/undo":
        agent.undo()
    elif cmd == "/clear":
        agent.messages = [{"role": "system", "content": agent._system_prompt()}]
        info("conversation cleared.")
    else:
        err(f"unknown command: {cmd} (try /help)")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="nimbus",
        description="NVIDIA-powered agentic coding CLI — reads & edits your files directly.",
    )
    ap.add_argument("folder", nargs="?", default=".", help="project folder to work in (default: .)")
    ap.add_argument("--model", default=None,
                    help=f"NVIDIA NIM model (default: $NIMBUS_MODEL or {DEFAULT_MODEL})")
    ap.add_argument("--auto", action="store_true",
                    help="start in autonomous mode (apply edits & run commands without asking)")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base URL (default: $NIMBUS_BASE_URL or NVIDIA)")
    ap.add_argument("--api-key", default=None, help="NVIDIA API key (else env/.env/prompt)")
    ap.add_argument("-p", "--prompt", default=None,
                    help="run a single request non-interactively, then exit")
    args = ap.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    load_dotenv(root)
    # Resolve after .env is loaded so NIMBUS_MODEL / NIMBUS_BASE_URL take effect;
    # an explicit CLI flag still wins.
    model = args.model or os.environ.get("NIMBUS_MODEL") or DEFAULT_MODEL
    base_url = args.base_url or os.environ.get("NIMBUS_BASE_URL") or DEFAULT_BASE_URL
    api_key = resolve_api_key(root, args.api_key)
    client = OpenAI(base_url=base_url, api_key=api_key)
    agent = Agent(root, model, client, auto=args.auto)

    git_safety_check(root)

    if args.prompt:
        if not args.auto:
            warn("Note: -p runs non-interactively; use --auto to skip per-change prompts.")
        agent.run_turn(args.prompt)
        return

    repl(agent)


if __name__ == "__main__":
    main()
