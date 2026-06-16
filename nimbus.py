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
import ast
import difflib
import getpass
import json
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

try:
    import readline as _readline
    _readline.parse_and_bind("tab: complete")
except ImportError:
    _readline = None  # Windows fallback — history still works via input() on some terminals

try:
    from mcp_client import McpManager as _McpManager
    _MCP_AVAILABLE = True
except ImportError:
    _McpManager = None
    _MCP_AVAILABLE = False

try:
    from openai import OpenAI
except ImportError:
    sys.exit(
        "The 'openai' package is required.\n"
        "  pip install openai   (or use this repo's .venu: .venu/bin/python nimbus.py)"
    )

try:
    from rich.console import Console
    from rich.markdown import Markdown
    _RICH = True
except ImportError:
    _RICH = False

# Module-level Rich console (used only for model prose rendering; existing ANSI
# info/warn/err status lines are kept as-is).
if _RICH:
    console = Console()

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"  # current, widely available, tool-calling

# NVIDIA NIM models with tool/function-calling support — shown in /model picker
KNOWN_MODELS = [
    ("nvidia/nemotron-3-nano-30b-a3b",          "Nemotron Nano 30B (3B active) — fastest, tool calling"),
    ("meta/llama-3.3-70b-instruct",             "Llama 3.3 70B — default, reliable tool calling"),
    ("deepseek-ai/deepseek-v4-flash",           "DeepSeek V4 Flash — fast coding & agents, 1M ctx"),
    ("nvidia/nemotron-3-super-120b-a12b",       "Nemotron Super 120B (12B active) — agentic, tool calling"),
    ("openai/gpt-oss-20b",                      "GPT-OSS 20B MoE — efficient reasoning"),
    ("qwen/qwen3-next-80b-a3b-instruct",        "Qwen3-Next 80B (3B active) — ultra-fast, long context"),
    ("qwen/qwen3.5-122b-a10b",                  "Qwen 3.5 122B (10B active) — tool calling, coding"),
    ("z-ai/glm-5.1",                            "GLM 5.1 — agentic workflows, coding"),
    ("mistralai/mistral-small-4-119b-2603",     "Mistral Small 4 119B MoE — coding, 256k context"),
    ("nvidia/llama-3.3-nemotron-super-49b-v1",  "Nemotron Super 49B — tool calling, reasoning"),
    ("stepfun-ai/step-3.5-flash",               "Step 3.5 Flash 200B sparse MoE — fast, agentic"),
    ("deepseek-ai/deepseek-v4-pro",             "DeepSeek V4 Pro — 1M context, coding"),
    ("moonshotai/kimi-k2.6",                    "Kimi K2.6 1T MoE — agentic, long-horizon coding"),
    ("openai/gpt-oss-120b",                     "GPT-OSS 120B MoE — reasoning, coding"),
    ("meta/llama-3.1-8b-instruct",              "Llama 3.1 8B — smallest, fastest dense model"),
]
# Set a persistent default via NIMBUS_MODEL in .env
# For rate-limit rotation, set NIMBUS_MODEL_POOL as a comma-separated list of model IDs

MAX_ITERS = 60            # tool-call rounds per user turn before we stop
MAX_TOKENS = 8192         # completion cap (prevents truncated tool calls)
MAX_READ_BYTES = 200_000
MAX_MENTION_BYTES = 60_000  # cap per @file injected into a prompt
MAX_CMD_OUTPUT = 16_000   # chars of command output fed back to the model
CMD_TIMEOUT = 240         # seconds per shell command
MAX_CONTEXT_CHARS = 160_000  # trim old turns once history grows past this
NIMBUS_CONTEXT_LIMIT = int(os.environ.get("NIMBUS_CONTEXT_LIMIT", "128000"))
RETRY_DELAYS = (1, 2, 4, 8, 16)  # backoff schedule for transient API errors

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venu", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".idea", ".vscode",
    ".obsidian", ".ipynb_checkpoints",
}
CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md", "README.md", "README.rst")

_REPO_MAP_EXTS = frozenset(".js .ts .tsx .jsx .go .rs .java .c .cpp .h .rb .php".split())
_OTHER_DEF_RE = re.compile(
    r"^\s*(?:"
    r"(?:export\s+(?:default\s+)?)?(?:function|class)\s+(\w+)"
    r"|def\s+(\w+)"
    r"|func\s+(\w+)"
    r"|type\s+(\w+)\s*(?:struct|interface|=)"
    r"|struct\s+(\w+)"
    r"|const\s+([A-Z_]\w*)\s*="
    r"|export\s+(?:default\s+)?(?:const|let|var)\s+(\w+)\s*="
    r")",
    re.MULTILINE,
)


DANGEROUS_COMMANDS = [
    r'rm\s+-rf\s+/', r'sudo', r'mkfs', r'dd\s+if=',
    r':\(\)\s*\{', r'>\s*/dev/sd', r'chmod\s+-R\s+777', r'git\s+push\s+--force',
]

def load_nimbus_settings(root: Path) -> dict:
    import json as _json
    defaults = {'allow_commands': [], 'deny_commands': [], 'allow_writes': ['*'], 'deny_writes': ['.env', '*.key', '.git/*']}
    merged = {k: list(v) for k, v in defaults.items()}
    for p in [Path.home() / '.nimbus' / 'settings.json', root / '.nimbus' / 'settings.json']:
        if p.exists():
            try:
                data = _json.loads(p.read_text())
                perms = data.get('permissions', {})
                for k in ('allow_commands', 'deny_commands', 'allow_writes', 'deny_writes'):
                    if k in perms:
                        merged[k] = list(perms[k])
            except Exception:
                pass
    return merged


_SESSIONS_DIR = Path.home() / ".nimbus" / "sessions"


def save_session(agent) -> None:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    sid = agent.session_id
    if not sid:
        return
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    data = {
        "version": 1, "id": sid, "root": str(agent.root), "model": agent.model,
        "title": agent.session_title or "", "created": agent._session_created or now,
        "updated": now, "messages": agent.messages, "usage": agent.usage,
    }
    p = _SESSIONS_DIR / f"{sid}.json"
    p.write_text(json.dumps(data, indent=2, default=str))
    idx_path = _SESSIONS_DIR / "index.json"
    try:
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else []
    except Exception:
        idx = []
    entry = {"id": sid, "root": str(agent.root), "title": data["title"],
             "model": agent.model, "created": data["created"],
             "updated": now, "message_count": len(agent.messages)}
    idx = [e for e in idx if e.get("id") != sid]
    idx.insert(0, entry)
    idx_path.write_text(json.dumps(idx, indent=2))


def load_session(session_id: str) -> dict | None:
    p = _SESSIONS_DIR / f"{session_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def list_sessions(root: Path) -> list:
    idx_path = _SESSIONS_DIR / "index.json"
    if not idx_path.exists():
        return []
    try:
        idx = json.loads(idx_path.read_text())
        return [e for e in idx if e.get("root") == str(root)]
    except Exception:
        return []


def build_repo_map(root: Path) -> str:
    """Walk the project and return a compact symbol tree (capped at 20000 chars).

    Python files are parsed with ast; other languages use a best-effort regex.
    """
    lines: list[str] = []
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
        for fname in sorted(filenames):
            if file_count >= 400:
                break
            fp = Path(dirpath) / fname
            suffix = fp.suffix.lower()
            if suffix != ".py" and suffix not in _REPO_MAP_EXTS:
                continue
            try:
                if fp.stat().st_size > 500_000:
                    continue
            except OSError:
                continue
            file_count += 1
            try:
                rel = str(fp.relative_to(root))
            except ValueError:
                rel = str(fp)

            symbols: list[str] = []
            if suffix == ".py":
                try:
                    tree = ast.parse(fp.read_text(errors="replace"), filename=rel)
                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            try:
                                sig = ast.unparse(node.args)
                            except Exception:
                                sig = "..."
                            symbols.append(f"  def {node.name}({sig})")
                        elif isinstance(node, ast.ClassDef):
                            symbols.append(f"  class {node.name}")
                            for child in ast.iter_child_nodes(node):
                                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    try:
                                        sig = ast.unparse(child.args)
                                    except Exception:
                                        sig = "..."
                                    symbols.append(f"    def {child.name}({sig})")
                except Exception:
                    pass
            else:
                try:
                    source = fp.read_text(errors="replace")
                    for m in _OTHER_DEF_RE.finditer(source):
                        name = next((g for g in m.groups() if g), None)
                        if name:
                            symbols.append(f"  {name}")
                        if len(symbols) >= 30:
                            break
                except Exception:
                    pass

            if symbols:
                lines.append(rel)
                lines.extend(symbols)
        if file_count >= 400:
            break

    result = "\n".join(lines)
    return result[:20000] if len(result) > 20000 else result


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
_INT_KEYS = {"depth", "offset", "limit"}


_BARE_JSON_TC = re.compile(
    r'\{"name"\s*:\s*"([^"]+)"\s*,\s*"(?:parameters|arguments)"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\})',
    re.DOTALL,
)
# Only recognize these as valid bare-JSON tool calls to avoid false positives in prose.
_KNOWN_TOOLS = frozenset({
    "list_directory", "find_files", "read_file", "write_file", "replace_in_file",
    "run_command", "search", "repo_map", "web_fetch", "web_search", "remember",
})

def parse_text_tool_calls(content: str):
    """Parse tool calls a model emitted as TEXT instead of via the native
    tool_calls field (Qwen / Hermes style). Handles the XML form
    (<function=name><parameter=key>val</parameter>), a JSON form
    (<tool_call>{"name":..., "arguments":{...}}</tool_call>), and a bare JSON
    form ({"name":"tool","parameters":{...}} without any wrapper tag).

    Returns a list of (name, args_dict). Empty if none found.
    """
    if not content:
        return []
    if "<tool_call>" not in content and "<function=" not in content:
        # Try bare-JSON form — only for known nimbus tool names to avoid false positives.
        calls = []
        for m in _BARE_JSON_TC.finditer(content):
            name = m.group(1)
            if name not in _KNOWN_TOOLS:
                continue
            try:
                args = json.loads(m.group(2))
                if isinstance(args, dict):
                    for k in list(args):
                        if k in _INT_KEYS and isinstance(args[k], str) and args[k].lstrip("-").isdigit():
                            args[k] = int(args[k])
                    calls.append((name, args))
            except (json.JSONDecodeError, TypeError):
                pass
        return calls
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
        self.plan = False  # PLAN MODE flag — must be set BEFORE _system_prompt() is called
        self.mcp_manager = None
        self.backups: dict[str, str | None] = {}  # path -> original text (None = new file)
        self._repo_map: str | None = None
        self._read_cache: dict[str, tuple[str, bool]] = {}  # path -> (raw_text, byte_truncated)
        self._turn_reads: dict[str, int] = {}  # path -> read count this turn (reset each turn)
        self.messages: list[dict] = [{"role": "system", "content": self._system_prompt()}]
        self.usage: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0, "requests": 0}
        self.context_tokens: int = 0
        self.permissions = load_nimbus_settings(self.root)
        self.session_id: str | None = None
        self.session_title: str | None = None
        self._session_created: str | None = None

    # ---- system prompt (with project grounding)
    def _system_prompt(self) -> str:
        # Append plan mode instruction if in plan mode
        # (will be added by _plan_system_prompt_addition())
        base = (
            "You are nimbus, an agentic coding assistant with FULL READ AND WRITE ACCESS to the "
            f"user's files inside the project root: {self.root}\n"
            f"You are running as model: {self.model}\n\n"
            "You MUST use the provided tools to read and edit files directly on disk. "
            "Never claim you cannot edit files — you have write_file and replace_in_file tools "
            "that perform real edits. Never just show code suggestions; always apply changes "
            "using the tools.\n\n"
            "CRITICAL: When asked to 'implement Prompt N from <spec>.md':\n"
            "  1. Use search or read_file with offset/limit to find the prompt section in the spec.\n"
            "  2. Read nimbus.py (the code file) to understand where to add the new features.\n"
            "  3. Edit nimbus.py (NOT the spec file) with replace_in_file or write_file.\n"
            "  4. Run 'python3 -m py_compile nimbus.py' to verify there are no syntax errors.\n"
            "Spec files (*.md) are READ-ONLY documentation — NEVER edit them when implementing features.\n"
            "Do NOT describe the implementation — USE THE TOOLS to write actual code to disk.\n\n"
            "Guidelines:\n"
            "- ALWAYS read a file with read_file before editing it; never guess its contents.\n"
            "- Prefer replace_in_file for edits (give enough surrounding context that old_string "
            "is unique). Use write_file only to create new files or fully rewrite small ones.\n"
            "- Use find_files/search to locate code across the project before editing.\n"
            "- For large files (>100 lines), use search to find the relevant section first, "
            "then read_file with offset=<line_number> limit=<count> to read just that section. "
            "- Do NOT re-read the whole file repeatedly — the section outline returned on repeated "
            "reads shows you the line numbers to jump to.\n"
            "- Make the smallest change that satisfies the request. Match the surrounding code "
            "style. Do not reformat unrelated code.\n"
            "- Work step by step: explore, read what you need, edit, then if useful run a command "
            "to verify (build/lint/tests).\n"
            "- Paths are relative to the project root unless absolute.\n"
            "- When done, give a brief plain-text summary of what you changed and why. Do not "
            "dump entire files back to the user.\n"
            "- For greetings or simple conversational messages, respond naturally without using tools.\n"
        )
        if self.plan:
            base += self._plan_system_prompt_addition()
        return base + self._project_context()

    def _plan_system_prompt_addition(self) -> str:
        return ("\n\nYou are in PLAN MODE. Investigate using read-only tools only, "
                "then output a concrete, numbered, step-by-step implementation plan. "
                "Do not attempt to modify files or run commands.")

    def _permitted(self, kind: str, target: str) -> str | None:
        perms = self.permissions
        if kind == 'command':
            for pat in perms.get('deny_commands', []):
                if fnmatch.fnmatch(target, pat):
                    return f'BLOCKED by deny_commands rule: {pat!r} — command not run.'
            for pat in perms.get('allow_commands', []):
                if fnmatch.fnmatch(target, pat):
                    return None
        elif kind == 'write':
            basename = Path(target).name
            for pat in perms.get('deny_writes', []):
                if fnmatch.fnmatch(target, pat) or fnmatch.fnmatch(basename, pat):
                    return f'BLOCKED by deny_writes rule: {pat!r} — file not written.'
            for pat in perms.get('allow_writes', []):
                if fnmatch.fnmatch(target, pat) or fnmatch.fnmatch(basename, pat):
                    return None
            # If allow_writes is non-empty and nothing matched, block
            if perms.get('allow_writes'):
                return f'BLOCKED: not in allow_writes — file not written.'
        return None

    def _project_context(self) -> str:
        parts = ["\n--- Project layout (top levels) ---\n"]
        parts.append(self._tool_list_directory(".", 2)[:2500])
        # NIMBUS.md is highest-priority context (loaded in full, up to 4000 chars)
        nimbus_md = self.root / "NIMBUS.md"
        if nimbus_md.is_file():
            try:
                snippet = nimbus_md.read_text(errors="replace")[:4000]
                parts.append(f"\n\n--- Project memory (NIMBUS.md) ---\n{snippet}")
            except Exception:
                pass
        for name in CONTEXT_FILES:
            p = self.root / name
            if p.is_file():
                try:
                    snippet = p.read_text(errors="replace")[:1500]
                except Exception:
                    continue
                parts.append(f"\n\n--- {name} (excerpt) ---\n{snippet}")
                break
        if self._repo_map is None:
            self._repo_map = build_repo_map(self.root)
        if self._repo_map:
            parts.append(f"\n\n--- Repo map (symbols) ---\n{self._repo_map[:6000]}")
        return "".join(parts)

    # ---- token tracking helpers
    def _update_usage(self, usage) -> None:
        """Merge usage from a stream response into self.usage."""
        if usage is None:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        tt = getattr(usage, "total_tokens", 0) or 0
        self.usage["prompt"] += pt
        self.usage["completion"] += ct
        self.usage["total"] += tt
        self.usage["requests"] += 1

    def _estimate_context_tokens(self) -> int:
        """Rough token estimate over self.messages (~4 chars/token + 200/tool_call)."""
        total = 0
        for m in self.messages:
            total += len(str(m.get("content") or "")) // 4
            total += 200 * len(m.get("tool_calls", []))
        self.context_tokens = total
        return total

    def _print_turn_usage(self, turn_usage) -> None:
        """Print dim token line after a turn completes."""
        pt = getattr(turn_usage, "prompt_tokens", 0) or 0
        ct = getattr(turn_usage, "completion_tokens", 0) or 0
        if pt == 0 and ct == 0:
            # Estimate if model returned no usage
            ct = self.usage["completion"]  # already accumulated
        st = self.usage["total"]
        line = f"{DIM}↑{pt} ↓{ct} tok · turn │ {st} session"
        price_in = os.environ.get("NIMBUS_PRICE_IN")
        price_out = os.environ.get("NIMBUS_PRICE_OUT")
        if price_in and price_out:
            try:
                cost = (self.usage["prompt"] * float(price_in)
                        + self.usage["completion"] * float(price_out)) / 1_000_000
                line += f" │ ~${cost:.4f}"
            except (ValueError, TypeError):
                pass
        line += RESET
        print(line)

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
        built = [
            fn("list_directory", "List files and folders (tree) under a path in the project.",
               {"path": {**s, "description": "Relative path; default project root."},
                "depth": {"type": "integer", "description": "How many levels deep (default 2)."}},
               []),
            fn("find_files", "Find files by glob pattern, e.g. '**/*.py' or 'src/**/*.js'.",
               {"pattern": {**s, "description": "Glob pattern, relative to project root."}},
               ["pattern"]),
            fn("read_file", "Read a text file. Returns its contents with line numbers.",
               {"path": {**s, "description": "File path relative to project root."},
                "offset": {"type": "integer", "description": "1-based start line (default: 1, whole file)."},
                "limit": {"type": "integer", "description": "Max lines to return (default: all)."}},
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
            fn("repo_map", "Get a compact map of all files and top-level symbols (classes, functions) in the project.",
               {"path": {**s, "description": "Subtree to scope (default: project root)."}},
               []),
            fn("web_fetch", "Fetch a web page and return readable text content.", {"url": {**s, "description": "HTTP or HTTPS URL to fetch."}}, ["url"]),
            fn("web_search", "Search the web. Returns top 5 results as title, URL, snippet.", {"query": {**s, "description": "Search query string."}}, ["query"]),
            fn("remember", "Append a durable note to NIMBUS.md (project memory). Use for facts, conventions, commands to remember across sessions.", {"note": {**s, "description": "The note to remember."}}, ["note"]),
        ]
        if self.mcp_manager:
            return built + self.mcp_manager.tool_schemas()
        return built

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

    def _tool_read_file(self, path: str, offset: int = 1, limit: int | None = None) -> str:
        p = self._resolve(path)
        if not p.is_file():
            return (f"ERROR: file not found: {path}\n"
                    f"Use list_directory or find_files to locate the file first.")
        cache_key = str(p.resolve())

        # On 2nd+ whole-file read of the same file this turn, return a section outline
        # instead of re-sending the full content — gives the model line numbers to use
        # with offset/limit rather than forcing it to re-read everything.
        self._turn_reads[cache_key] = self._turn_reads.get(cache_key, 0) + 1
        if self._turn_reads[cache_key] > 1 and offset == 1 and limit is None and cache_key in self._read_cache:
            cached_lines = self._read_cache[cache_key][0].splitlines()
            total = len(cached_lines)
            headings = [(i + 1, ln) for i, ln in enumerate(cached_lines) if ln.startswith("#")]
            if headings:
                outline = "\n".join(f"  Line {n}: {ln}" for n, ln in headings)
                return (f"[Already read this turn — {total} lines total. Section outline:]\n"
                        f"{outline}\n"
                        f"[Use read_file with offset=<N> limit=<M> to read a specific section.]")
            else:
                return (f"[Already read this turn — {total} lines. "
                        f"Use read_file with offset/limit to read a specific section, or use search.]")

        if cache_key in self._read_cache:
            text, byte_truncated = self._read_cache[cache_key]
        else:
            try:
                data = p.read_bytes()
            except Exception as e:
                return f"ERROR: cannot read {path}: {e}"
            byte_truncated = len(data) > MAX_READ_BYTES
            if byte_truncated:
                data = data[:MAX_READ_BYTES]
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                return f"ERROR: {path} is not a UTF-8 text file (binary?)."
            self._read_cache[cache_key] = (text, byte_truncated)

        all_lines = text.splitlines()
        total = len(all_lines)
        whole_file = (offset == 1 and limit is None)

        if whole_file and total > 2000:
            slice_lines = all_lines[:2000]
            numbered = "\n".join(f"{i + 1}\t{ln}" for i, ln in enumerate(slice_lines))
            numbered += f"\n... file has {total} lines; call read_file with offset/limit to read more."
            return numbered

        start = max(0, offset - 1)
        end = min(start + limit, total) if limit is not None else total
        slice_lines = all_lines[start:end]
        numbered = "\n".join(f"{start + i + 1}\t{ln}" for i, ln in enumerate(slice_lines))
        if whole_file and byte_truncated:
            numbered += f"\n... [truncated at {MAX_READ_BYTES} bytes]"

        # For whole-file reads of large files with headings, prepend a section map
        # so the model can jump to the right section immediately.
        if whole_file and total > 80:
            headings = [(i + 1, ln) for i, ln in enumerate(all_lines) if ln.startswith("#")]
            if len(headings) >= 2:
                outline = "\n".join(f"  Line {n}: {ln}" for n, ln in headings)
                numbered = (f"[Section outline — use read_file with offset/limit to jump to a section:]\n"
                            f"{outline}\n\n") + numbered
        return numbered

    def _tool_write_file(self, path: str, content: str) -> str:
        if self.plan:
            return "BLOCKED: PLAN MODE is read-only. Investigate with read-only tools and present a concrete numbered plan instead."

        p = self._resolve(path)
        if not self._inside_root(p):
            return "ERROR: refusing to write outside the project root."
        block = self._permitted('write', self._rel(p) or path)
        if block:
            return block
        old = p.read_text(errors="replace") if p.is_file() else ""
        # Guard against accidentally overwriting a large file with a tiny stub.
        if old and len(content) < len(old) * 0.1 and len(old) > 500:
            return (
                f"ERROR: refusing to overwrite {self._rel(p)} ({len(old)} bytes) "
                f"with only {len(content)} bytes — this looks like a partial stub.\n"
                "To add code to an existing file, use replace_in_file instead of write_file.\n"
                "Use write_file only to create NEW files or fully rewrite SMALL ones."
            )
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
        self._read_cache.pop(str(p.resolve()), None)
        self._turn_reads.pop(str(p.resolve()), None)
        info(f"  ✓ wrote {self._rel(p)} ({len(content)} bytes)")
        return f"OK: wrote {self._rel(p)} ({len(content)} bytes)."

    def _tool_replace_in_file(self, path: str, old_string: str, new_string: str,
                              replace_all: bool = False) -> str:
        if self.plan:
            return "BLOCKED: PLAN MODE is read-only. Investigate with read-only tools and present a concrete numbered plan instead."

        p = self._resolve(path)
        if not self._inside_root(p):
            return "ERROR: refusing to edit outside the project root."
        block = self._permitted('write', self._rel(p) or path)
        if block:
            return block
        if p.suffix.lower() in (".md", ".rst") and not self._confirm(
                f"Edit documentation file: {self._rel(p)} — are you sure? "
                "Spec/README files should generally not be edited when implementing features; "
                "the implementation goes in .py files."):
            return ("SKIPPED: refused to edit documentation file. "
                    "To implement features, edit .py source files instead.")
        if not p.is_file():
            return f"ERROR: file not found: {path}"
        text = p.read_text(errors="replace")
        count = text.count(old_string)
        if count == 0:
            # Help the model self-correct without re-reading: find first line of old_string in file
            first_line = old_string.strip().splitlines()[0].strip() if old_string.strip() else ""
            hint = ""
            if first_line:
                lines = text.splitlines()
                matches = [i + 1 for i, ln in enumerate(lines) if first_line in ln]
                if matches:
                    ctx_lines = []
                    for ln_no in matches[:2]:
                        start = max(0, ln_no - 3)
                        end = min(len(lines), ln_no + 3)
                        ctx_lines.append(f"[Line {ln_no} context:]")
                        ctx_lines.extend(f"{start + i + 1}\t{lines[start + i]}"
                                         for i in range(end - start))
                    hint = ("\nFirst line of your old_string was found at: " +
                            ", ".join(str(n) for n in matches[:2]) +
                            "\n" + "\n".join(ctx_lines) +
                            "\nCompare carefully — whitespace/indentation may differ.")
                else:
                    hint = f"\nFirst line of your old_string ({first_line!r}) was not found anywhere in the file."
            return (f"ERROR: old_string not found in {path}.{hint}\n"
                    "Use read_file with offset/limit to get the exact current text, then retry.")
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
        self._read_cache.pop(str(p.resolve()), None)
        self._turn_reads.pop(str(p.resolve()), None)
        info(f"  ✓ edited {self._rel(p)} ({n} replacement{'s' if n > 1 else ''})")
        return f"OK: edited {self._rel(p)} ({n} replacement(s))."

    def _tool_run_command(self, command: str) -> str:
        if self.plan:
            return "BLOCKED: PLAN MODE is read-only. Investigate with read-only tools and present a concrete numbered plan instead."

        block = self._permitted('command', command)
        if block:
            return block
        is_dangerous = any(re.search(p, command) for p in DANGEROUS_COMMANDS)
        was_auto = self.auto
        if is_dangerous:
            self.auto = False
        try:
            confirmed = self._confirm(f"Run command:  {MAGENTA}{command}{RESET}" + (f" {BOLD}[DANGEROUS — forced confirm]{RESET}" if is_dangerous else ""))
        finally:
            self.auto = was_auto
        if not confirmed:
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
    def _tool_remember(self, note: str) -> str:
        nimbus_md = self.root / "NIMBUS.md"
        block = self._permitted('write', 'NIMBUS.md')
        if block:
            return block
        if not self._confirm(f"Append to NIMBUS.md: {note!r}"):
            return "SKIPPED: user declined."
        try:
            if nimbus_md.is_file():
                content = nimbus_md.read_text(errors="replace")
            else:
                content = "# NIMBUS.md — Project memory\n\n"
            if "## nimbus memory" not in content:
                content += "\n## nimbus memory\n"
            content += f"- {note}\n"
            nimbus_md.write_text(content)
            self._read_cache.pop(str(nimbus_md.resolve()), None)
            self._turn_reads.pop(str(nimbus_md.resolve()), None)
            return f"OK: noted in NIMBUS.md"
        except Exception as e:
            return f"ERROR: could not write NIMBUS.md: {e}"

    def _tool_web_fetch(self, url: str) -> str:
        if not url.lower().startswith(('http://', 'https://')):
            return "ERROR: only http(s) URLs are allowed."
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'nimbus'})
            with urllib.request.urlopen(req, timeout=20) as response:
                max_bytes = 2 * 1024 * 1024  # 2 MiB
                content_type = response.headers.get('Content-Type', '')
                raw = response.read(max_bytes)
                charset = 'utf-8'
                m = re.search(r'charset=([^\s;]+)', content_type, re.I)
                if m:
                    charset = m.group(1)
                try:
                    text = raw.decode(charset, errors='replace')
                except Exception:
                    text = raw.decode('utf-8', errors='replace')
                if 'text/html' in content_type.lower():
                    class _HTMLTextExtractor(HTMLParser):
                        def __init__(self):
                            super().__init__()
                            self.parts = []
                            self.skip = False
                        def handle_starttag(self, tag, attrs):
                            if tag in ('script', 'style'):
                                self.skip = True
                        def handle_endtag(self, tag):
                            if tag in ('script', 'style'):
                                self.skip = False
                        def handle_data(self, data):
                            if not self.skip:
                                cleaned = re.sub(r'\s+', ' ', data)
                                if cleaned.strip():
                                    self.parts.append(cleaned.strip() + ' ')
                        def get_text(self):
                            return ''.join(self.parts).strip()
                    parser = _HTMLTextExtractor()
                    parser.feed(text)
                    text = parser.get_text()
                return text[:10000]
        except Exception as e:
            return f"ERROR: failed to fetch {url}: {e}"

    def _tool_web_search(self, query: str) -> str:
        import json as _json
        tavily_key = os.environ.get('TAVILY_API_KEY', '')
        brave_key = os.environ.get('BRAVE_API_KEY', '')
        results = []
        try:
            if tavily_key:
                data = _json.dumps({'api_key': tavily_key, 'query': query, 'max_results': 5}).encode()
                req = urllib.request.Request('https://api.tavily.com/search',
                    data=data, headers={'Content-Type': 'application/json', 'User-Agent': 'nimbus'})
                with urllib.request.urlopen(req, timeout=20) as r:
                    resp = _json.loads(r.read(1024 * 1024))
                for item in resp.get('results', [])[:5]:
                    results.append(f"{item.get('title', '')} — {item.get('url', '')}\n  {item.get('content', '')[:200]}")
            elif brave_key:
                enc = urllib.parse.quote_plus(query)
                req = urllib.request.Request(
                    f'https://api.search.brave.com/res/v1/web/search?q={enc}&count=5',
                    headers={'Accept': 'application/json', 'Accept-Encoding': 'identity',
                             'X-Subscription-Token': brave_key, 'User-Agent': 'nimbus'})
                with urllib.request.urlopen(req, timeout=20) as r:
                    resp = _json.loads(r.read(1024 * 1024))
                for item in resp.get('web', {}).get('results', [])[:5]:
                    results.append(f"{item.get('title', '')} — {item.get('url', '')}\n  {item.get('description', '')[:200]}")
            else:
                enc = urllib.parse.quote_plus(query)
                req = urllib.request.Request(
                    f'https://html.duckduckgo.com/html/?q={enc}',
                    headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0'})
                with urllib.request.urlopen(req, timeout=20) as r:
                    html = r.read(1024 * 1024).decode('utf-8', errors='replace')
                class _DDGParser(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.results = []
                        self._in_title = False
                        self._in_snippet = False
                        self._cur = {}
                    def handle_starttag(self, tag, attrs):
                        d = dict(attrs)
                        cls = d.get('class', '')
                        if tag == 'a' and 'result__a' in cls:
                            self._in_title = True
                            self._cur = {'url': d.get('href', ''), 'title': ''}
                        elif 'result__snippet' in cls:
                            self._in_snippet = True
                            self._cur.setdefault('snippet', '')
                    def handle_endtag(self, tag):
                        if self._in_title:
                            self._in_title = False
                        if self._in_snippet:
                            self._in_snippet = False
                            if self._cur.get('title'):
                                self.results.append(dict(self._cur))
                                self._cur = {}
                    def handle_data(self, data):
                        if self._in_title:
                            self._cur['title'] = self._cur.get('title', '') + data
                        elif self._in_snippet:
                            self._cur['snippet'] = self._cur.get('snippet', '') + data
                p = _DDGParser()
                p.feed(html)
                for item in p.results[:5]:
                    results.append(f"{item.get('title', '').strip()} — {item.get('url', '')}\n  {item.get('snippet', '').strip()[:200]}")
                if not results:
                    return 'No results found. Set TAVILY_API_KEY for reliable web search.'
        except Exception as e:
            return f'ERROR: web search failed: {e}. Set TAVILY_API_KEY for reliable results.'
        return '\n'.join(results) if results else 'No results found.'

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
            # Smart-case: case-insensitive when pattern is all lowercase (mirrors rg -S)
            flags = re.IGNORECASE if pattern == pattern.lower() else 0
            rx = re.compile(pattern, flags)
        except re.error as e:
            return f"ERROR: bad regex: {e}"
        hits: list[str] = []

        def _scan_file(fp: Path) -> bool:
            """Scan one file; append matches. Returns True if the 200-match cap was hit."""
            try:
                for i, line in enumerate(fp.read_text(errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{self._rel(fp)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= 200:
                            return True
            except Exception:
                pass
            return False

        if base.is_file():
            # os.walk yields nothing for a file path, so scan it directly.
            _scan_file(base)
        else:
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
                if any(_scan_file(Path(dirpath) / fn) for fn in filenames):
                    break
        if len(hits) >= 200:
            return "\n".join(hits) + "\n...[capped at 200 matches]"
        return "\n".join(hits) if hits else "(no matches)"

    def _tool_repo_map(self, path: str = ".") -> str:
        base = self._resolve(path)
        if not self._inside_root(base):
            return "ERROR: path is outside the project root."
        result = build_repo_map(base)
        return (result[:20000] if len(result) > 20000 else result) or "(no files with symbols found)"

    def _dispatch(self, name: str, args: dict) -> str:
        handlers = {
            "list_directory": self._tool_list_directory,
            "find_files": self._tool_find_files,
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "replace_in_file": self._tool_replace_in_file,
            "run_command": self._tool_run_command,
            "search": self._tool_search,
            "repo_map": self._tool_repo_map,
            "web_fetch": self._tool_web_fetch,
            "web_search": self._tool_web_search,
            "remember": self._tool_remember,
        }
        if name.startswith("mcp__") and self.mcp_manager:
            return self.mcp_manager.call(name, args)
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

    # ---- context-window management (summary-based compaction)
    def _compact_history(self, force: bool = False) -> None:
        """Compact older conversation rounds into a summary when context is large.

        Triggered automatically when context_tokens > 75% of NIMBUS_CONTEXT_LIMIT,
        or manually via /compact (force=True).
        """
        self._estimate_context_tokens()
        threshold = 0.75 * NIMBUS_CONTEXT_LIMIT
        if not force and self.context_tokens <= threshold:
            return

        # Build rounds: each round starts at a non-tool-result user message
        rounds: list[list[int]] = []  # each is a list of message indices
        current_round: list[int] = []

        for i, m in enumerate(self.messages):
            if m.get("role") == "user" and m.get("content", "").startswith(
                    "Tool results (continue, or give your final summary):"):
                # This is a synthetic user message from text-format tool results;
                # it belongs to the current round, not a new one.
                current_round.append(i)
                continue
            if m.get("role") == "user" and current_round:
                rounds.append(current_round)
                current_round = [i]
            else:
                current_round.append(i)
        if current_round:
            rounds.append(current_round)

        if len(rounds) <= 1:
            return  # nothing to compact

        # Keep most recent rounds that fit in ~30% of limit (by char count)
        budget_chars = int(0.30 * NIMBUS_CONTEXT_LIMIT) * 4  # rough chars
        kept_rounds: list[int] = []
        used_chars = 0
        for r in reversed(rounds):
            r_chars = sum(len(str(self.messages[i].get("content") or "")) for i in r)
            if used_chars + r_chars > budget_chars and kept_rounds:
                break
            kept_rounds.insert(0, r)
            used_chars += r_chars

        if not kept_rounds:
            kept_rounds = [rounds[-1]]  # always keep at least the latest round

        # Summarize everything older
        old_indices = []
        for r in rounds:
            if r is not kept_rounds[0]:
                old_indices.extend(r)
            else:
                break

        if not old_indices:
            return

        # Build the text to summarize
        old_msgs = [self.messages[i] for i in old_indices]
        summary_text = "\n\n".join(
            f"[{m.get('role','?')}] {str(m.get('content',''))[:2000]}" for m in old_msgs
        )[:12000]

        summary_prompt = (
            "Summarize this conversation for continuity. Capture: the user's goals, "
            "files created/edited and how, key decisions, commands run and outcomes, "
            "and any unfinished work. Be concise (<400 words). Output plain text.\n\n"
            + summary_text
        )

        # Make a one-off summarization call (non-streaming, low temperature)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.0,
                max_tokens=1024,
                stream=False,
            )
            summary = resp.choices[0].message.content or "(summary unavailable)"
            # Count summarization call tokens
            if hasattr(resp, "usage") and resp.usage:
                self._update_usage(resp.usage)
            else:
                self.usage["completion"] += len(summary) // 4
                self.usage["requests"] += 1
        except Exception as e:
            warn(f"(compaction summarization failed: {e}; keeping older messages)")
            return

        # Replace old messages with a single summary user message
        summary_msg = {
            "role": "user",
            "content": f"[Earlier conversation summary]\n{summary}",
        }

        # Rebuild messages: system + summary + kept rounds
        new_messages = [self.messages[0]]  # system prompt
        new_messages.append(summary_msg)
        kept_start = kept_rounds[0][0]
        for i in range(kept_start, len(self.messages)):
            new_messages.append(self.messages[i])

        n_removed = len(self.messages) - len(new_messages) + 1  # +1 for summary msg
        self.messages = new_messages
        self._estimate_context_tokens()
        print(f"{DIM}· compacted {n_removed} older messages into a summary{RESET}")

    # ---- model rotation (rate-limit fallback)
    def _rotate_model(self) -> bool:
        """Switch to the next model in NIMBUS_MODEL_POOL. Returns True if rotated."""
        pool_str = os.environ.get("NIMBUS_MODEL_POOL", "")
        if not pool_str:
            return False
        pool = [m.strip() for m in pool_str.split(",") if m.strip()]
        if len(pool) < 2:
            return False
        try:
            idx = pool.index(self.model)
        except ValueError:
            idx = -1
        next_model = pool[(idx + 1) % len(pool)]
        if next_model == self.model:
            return False
        old = self.model
        self.model = next_model
        warn(f"(rate-limited on {old} — switching to {self.model})")
        return True

    # ---- API call with retry/backoff → streaming
    def _stream(self) -> tuple[str, list[dict], object]:
        """Stream a completion and return (content, tool_calls, usage).

        tool_calls items are in the same shape run_turn already builds:
        {"id", "type":"function", "function":{"name","arguments"}}.
        """
        kwargs = dict(
            model=self.model, messages=self.messages, tools=self.tools(),
            tool_choice="auto", temperature=0.2, max_tokens=MAX_TOKENS,
            stream=True, stream_options={"include_usage": True},
        )
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                with Spinner():
                    stream = self.client.chat.completions.create(**kwargs)
                break  # stream created successfully
            except KeyboardInterrupt:
                raise
            except Exception as e:
                s = str(e).lower()
                # Context too full: reduce max_tokens to fit and retry immediately
                if "too large" in s and "max_tokens" in s:
                    m = re.search(r"(\d+) > (\d+) - (\d+)", str(e))
                    if m:
                        available = int(m.group(2)) - int(m.group(3)) - 200
                        if available >= 256 and kwargs["max_tokens"] > available:
                            kwargs["max_tokens"] = available
                            warn(f"(context nearly full — retrying with max_tokens={available})")
                            continue
                transient = any(k in s for k in (
                    "429", "rate limit", "500", "502", "503", "504", "timeout",
                    "timed out", "connection", "temporarily", "overloaded", "unavailable",
                    "degraded",
                ))
                if attempt < len(RETRY_DELAYS) and transient:
                    # On rate-limit or degraded model, rotate to next model before sleeping
                    if ("429" in s or "rate limit" in s or "degraded" in s) and self._rotate_model():
                        kwargs["model"] = self.model
                        continue  # retry immediately with new model
                    wait = RETRY_DELAYS[attempt]
                    warn(f"(transient API error — retrying in {wait}s: {str(e)[:90]})")
                    time.sleep(wait)
                    continue
                raise

        # ---- accumulate deltas from stream chunks ----
        # Rich path: nothing is written during streaming; run_turn renders one clean
        # Markdown block after the full response arrives.
        # non-Rich / non-TTY: stream chars directly; run_turn must NOT re-render.
        content_buf = ""
        tc_accum: dict[int, dict] = {}  # index → {id, name, arguments}
        usage = None

        try:
            for chunk in stream:
                # Final usage-only chunk
                if not chunk.choices:
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta

                # --- content delta ---
                if delta.content:
                    content_buf += delta.content

                    if "<tool_call>" in content_buf or "<function=" in content_buf:
                        # Suppress raw tool-call XML (text-format models)
                        if _TTY:
                            sys.stdout.write(f"\r{DIM}· preparing tool call…{RESET}\r")
                            sys.stdout.flush()
                    elif not (_TTY and _RICH):
                        # non-Rich or piped: stream content as it arrives
                        sys.stdout.write(delta.content)
                        sys.stdout.flush()
                    # Rich TTY: stay silent; content rendered once after stream ends

                # --- tool_calls delta ---
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_accum:
                            tc_accum[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tc_accum[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc_accum[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc_accum[idx]["arguments"] += tc_delta.function.arguments

                # --- reasoning delta (some NIM models) ---
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if _TTY:
                        sys.stdout.write(f"{DIM}{delta.reasoning_content}{RESET}")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(delta.reasoning_content)
                        sys.stdout.flush()
                if hasattr(delta, "reasoning") and delta.reasoning:
                    if _TTY:
                        sys.stdout.write(f"{DIM}{delta.reasoning}{RESET}")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(delta.reasoning)
                        sys.stdout.flush()

        except KeyboardInterrupt:
            pass  # surface gracefully

        # Newline after streamed chars (non-Rich paths wrote directly to stdout)
        if content_buf and not (_TTY and _RICH):
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Build tool_calls list in index order
        tool_calls_list = []
        for idx in sorted(tc_accum):
            entry = tc_accum[idx]
            tool_calls_list.append({
                "id": entry["id"],
                "type": "function",
                "function": {"name": entry["name"], "arguments": entry["arguments"]},
            })

        return content_buf, tool_calls_list, usage

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
        if self.session_id is None:
            self.session_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
            self._session_created = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.session_title = user_input.strip()[:60]
        self._turn_reads.clear()
        self.messages.append({"role": "user", "content": self._expand_mentions(user_input)})
        turn_usage = None
        _last_sig: str | None = None
        _repeat: int = 0
        for _ in range(MAX_ITERS):
            self._compact_history()
            try:
                content, native_calls, usage = self._stream()
                turn_usage = usage
            except KeyboardInterrupt:
                warn("\n(interrupted — returning to prompt)")
                save_session(self)
                return
            except Exception as e:
                err(f"API error: {e}")
                save_session(self)
                return

            # Update session usage
            if usage:
                self._update_usage(usage)

            # Text-format tool-call parsing is unchanged
            text_calls = [] if native_calls else parse_text_tool_calls(content)

            entry: dict = {"role": "assistant", "content": content}
            if native_calls:
                entry["tool_calls"] = native_calls
            self.messages.append(entry)

            if not native_calls and not text_calls:
                # Final answer — Rich renders once here; non-Rich already streamed to stdout.
                if content.strip():
                    if _TTY and _RICH:
                        console.print(Markdown(content.strip()))
                    # else: content already on screen from streaming; don't re-print
                # Print turn usage line
                self._print_turn_usage(turn_usage)
                save_session(self)
                return  # turn complete

            prose = content.strip() if native_calls else strip_tool_calls(content)
            if prose:
                if _TTY and _RICH:
                    # Rich: nothing was written during streaming, render prose now
                    console.print(Markdown(prose))
                elif native_calls:
                    pass  # non-Rich: prose already streamed to stdout, skip
                else:
                    print(f"\n{prose}\n")  # text-format calls: stripped prose not yet shown

            if native_calls:
                for tc in native_calls:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    sig = f"{name}:{json.dumps(args, sort_keys=True)}"
                    if sig == _last_sig:
                        _repeat += 1
                    else:
                        _last_sig = sig
                        _repeat = 1
                    _read_only = {"list_directory", "find_files", "read_file", "search"}
                    _loop_limit = 6 if name in _read_only else 3
                    if _repeat >= _loop_limit:
                        warn(f"(loop: {name} called with same args {_repeat}× — aborting turn)")
                        save_session(self)
                        return
                    self._announce(name, args)
                    result = self._dispatch(name, args)
                    self.messages.append({"role": "tool", "tool_call_id": tc["id"],
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
        save_session(self)

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
  /model [name]    list & pick from known models (or set by name directly)
  /open <path>     switch the working folder
  /pwd             show the working folder
  /files           print the file tree
  /map             print the repo symbol map
  /map --refresh   recompute the map (e.g. after adding files)
  /diff            show every change nimbus made this session
  /undo            revert all of this session's file changes
  /clear           clear the conversation history (keep the folder & settings)
  /cost, /tokens   show token usage (session totals and context window %)
  /compact         force history compaction (summarize older turns)
  /exit, /quit     leave

  {BOLD}Session{RESET}
  /sessions        list saved sessions for this folder
  /resume <id>     restore a saved session by ID
  /new             start a fresh session (clear history)
  /memory          print project memory (NIMBUS.md)

  {BOLD}Plan mode{RESET}
  /plan            enter PLAN mode (read-only; model investigates and plans)
  /build           exit plan mode and execute the plan

  {BOLD}Permissions{RESET}
  /permissions     show effective allow/deny rules
  /allow <pattern> add an allow_commands rule to project settings
  /deny <pattern>  add a deny_commands rule to project settings

  {BOLD}MCP{RESET}
  /mcp             list connected MCP servers and their tools

Use {BOLD}@path/to/file{RESET} in a request to attach that file's contents.
Anything else is a request — e.g. "add a --json flag to proxy.py and update the README"."""


def repl(agent: Agent) -> None:
    # Load/save persistent readline history so up/down arrows recall previous sessions.
    _hist = Path.home() / ".nimbus_history"
    if _readline:
        try:
            _readline.read_history_file(_hist)
        except FileNotFoundError:
            pass
        _readline.set_history_length(500)
        import atexit
        atexit.register(lambda: _readline.write_history_file(_hist))

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
            agent.messages[0] = {"role": "system", "content": agent._system_prompt()}
            info(f"model set to {arg}")
        else:
            print(f"\n{BOLD}Current model:{RESET} {agent.model}\n")
            print(f"{BOLD}Available models:{RESET}")
            for i, (name, desc) in enumerate(KNOWN_MODELS, 1):
                marker = f"{GREEN}*{RESET}" if name == agent.model else " "
                print(f"  {marker} {BOLD}{i}{RESET}. {name}")
                print(f"       {DIM}{desc}{RESET}")
            print(f"\n  {DIM}Enter a number to switch, or /model <name> to use any NIM model.{RESET}\n")
            try:
                choice = input("  choice (or Enter to keep current): ").strip()
            except (EOFError, KeyboardInterrupt):
                choice = ""
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(KNOWN_MODELS):
                    agent.model = KNOWN_MODELS[idx][0]
                    agent.messages[0] = {"role": "system", "content": agent._system_prompt()}
                    info(f"model set to {agent.model}")
                else:
                    err(f"invalid choice: {choice}")
            elif choice:
                agent.model = choice
                agent.messages[0] = {"role": "system", "content": agent._system_prompt()}
                info(f"model set to {agent.model}")
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
                agent._repo_map = None
                agent.messages[0] = {"role": "system", "content": agent._system_prompt()}
                info(f"folder: {new}")
                git_safety_check(new)
    elif cmd == "/pwd":
        print(agent.root)
    elif cmd == "/files":
        print(agent._tool_list_directory(".", 2))
    elif cmd == "/map":
        if arg == "--refresh":
            agent._repo_map = None
            agent.messages[0] = {"role": "system", "content": agent._system_prompt()}
            info("Repo map refreshed.")
        else:
            if agent._repo_map is None:
                agent._repo_map = build_repo_map(agent.root)
            print(agent._repo_map or "(no files with symbols found)")
    elif cmd == "/sessions":
        sessions = list_sessions(agent.root)
        if not sessions:
            print("  (no sessions for this folder)")
        else:
            for s in sessions[:20]:
                print(f"  {s['id']}  {s.get('updated','')[:16]}  {s.get('message_count',0)} msgs  {s.get('title','')[:50]}")
    elif cmd == "/resume":
        if not arg:
            err("usage: /resume <session-id>")
        else:
            data = load_session(arg)
            if data is None:
                err(f"session not found: {arg}")
            else:
                agent.messages = data["messages"]
                agent.messages[0] = {"role": "system", "content": agent._system_prompt()}
                agent.usage = data.get("usage", agent.usage)
                agent.session_id = data["id"]
                agent.session_title = data.get("title", "")
                agent._session_created = data.get("created")
                info(f"resumed session {data['id']} ({len(agent.messages)} messages)")
    elif cmd == "/new":
        agent.messages = [{"role": "system", "content": agent._system_prompt()}]
        agent.session_id = None
        agent.session_title = None
        agent._session_created = None
        info("started new session.")
    elif cmd == "/memory":
        nimbus_md = agent.root / "NIMBUS.md"
        if nimbus_md.is_file():
            print(nimbus_md.read_text(errors="replace"))
        else:
            print("  (no NIMBUS.md in this project)")
    elif cmd == "/diff":
        print(agent.session_diff())
    elif cmd == "/undo":
        agent.undo()
    elif cmd == "/clear":
        agent.messages = [{"role": "system", "content": agent._system_prompt()}]
        info("conversation cleared.")
    elif cmd == "/plan":
        agent.plan = True
        agent.messages[0] = {"role": "system", "content": agent._system_prompt()}
        info("PLAN MODE: read-only. Use read-only tools to investigate, then present a numbered plan. /build to execute.")
    elif cmd == "/build":
        agent.plan = False
        agent.messages[0] = {"role": "system", "content": agent._system_prompt()}
        info("CONFIRM mode: executing plan.")
    elif cmd in ("/cost", "/tokens"):
        agent._estimate_context_tokens()
        total = agent.usage["total"]
        reqs = agent.usage["requests"]
        ctx = agent.context_tokens
        limit = NIMBUS_CONTEXT_LIMIT
        pct = (ctx / limit * 100) if limit > 0 else 0
        line = f"{BOLD}Session usage:{RESET} {total} tokens in {reqs} request(s)"
        line += f"\n{DIM}context: ~{ctx} / {limit} tokens ({pct:.1f}%){RESET}"
        price_in = os.environ.get("NIMBUS_PRICE_IN")
        price_out = os.environ.get("NIMBUS_PRICE_OUT")
        if price_in and price_out:
            try:
                cost = (agent.usage["prompt"] * float(price_in)
                        + agent.usage["completion"] * float(price_out)) / 1_000_000
                line += f"\n{DIM}estimated cost: ~${cost:.4f}{RESET}"
            except (ValueError, TypeError):
                pass
        print(line)
    elif cmd == "/compact":
        agent._compact_history(force=True)
    elif cmd == "/permissions":
        perms = agent.permissions
        for k, v in perms.items():
            print(f"  {k}: {v if v else '(none)'}")
    elif cmd == "/allow":
        if not arg:
            err("usage: /allow <pattern>")
        else:
            agent.permissions.setdefault('allow_commands', []).append(arg)
            settings_path = agent.root / '.nimbus' / 'settings.json'
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            import json as _j
            data = json.loads(settings_path.read_text()) if settings_path.exists() else {}
            data.setdefault('permissions', {}).setdefault('allow_commands', []).append(arg)
            settings_path.write_text(_j.dumps(data, indent=2))
            info(f"Added allow rule: {arg!r}")
    elif cmd == "/deny":
        if not arg:
            err("usage: /deny <pattern>")
        else:
            agent.permissions.setdefault('deny_commands', []).append(arg)
            settings_path = agent.root / '.nimbus' / 'settings.json'
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            import json as _j
            data = _j.loads(settings_path.read_text()) if settings_path.exists() else {}
            data.setdefault('permissions', {}).setdefault('deny_commands', []).append(arg)
            settings_path.write_text(_j.dumps(data, indent=2))
            info(f"Added deny rule: {arg!r}")
    elif cmd == "/mcp":
        if agent.mcp_manager:
            print(agent.mcp_manager.status())
        else:
            print("  MCP: not available (mcp package not installed or no mcpServers configured)")
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
    ap.add_argument("--plan", action="store_true",
                    help="start in plan mode (read-only investigation; /build to execute)")
    ap.add_argument("--resume", nargs="?", const="__latest__", default=None,
                    help="resume a session by ID, or most recent for this folder")
    ap.add_argument("--continue", dest="resume_latest", action="store_true",
                    help="resume most recent session for this folder (alias for --resume)")
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
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    agent = Agent(root, model, client, auto=args.auto)
    if args.plan:
        agent.plan = True
        agent.messages[0] = {"role": "system", "content": agent._system_prompt()}

    # Session resume
    resume_id = args.resume
    if not resume_id and args.resume_latest:
        resume_id = "__latest__"
    if resume_id:
        if resume_id == "__latest__":
            sessions = list_sessions(root)
            resume_id = sessions[0]["id"] if sessions else None
            if not resume_id:
                info("No previous sessions found for this folder — starting fresh.")
        if resume_id:
            data = load_session(resume_id)
            if data:
                agent.messages = data["messages"]
                agent.messages[0] = {"role": "system", "content": agent._system_prompt()}
                agent.usage = data.get("usage", agent.usage)
                agent.session_id = data["id"]
                agent.session_title = data.get("title", "")
                agent._session_created = data.get("created")
                info(f"resumed session {data['id']} ({len(agent.messages)} messages)")
            else:
                info(f"Session {resume_id!r} not found — starting fresh.")

    git_safety_check(root)

    if _MCP_AVAILABLE and _McpManager:
        mcp_servers = {}
        for settings_path in [Path.home() / ".nimbus" / "settings.json",
                           root / ".nimbus" / "settings.json"]:
            if settings_path.exists():
                try:
                    import json as _json
                    cfg = _json.loads(settings_path.read_text())
                    mcp_servers.update(cfg.get("mcpServers", {}))
                except Exception:
                    pass
        if mcp_servers:
            agent.mcp_manager = _McpManager()
            agent.mcp_manager.connect_all(mcp_servers)

    if args.prompt:
        if not args.auto:
            warn("Note: -p runs non-interactively; use --auto to skip per-change prompts.")
        agent.run_turn(args.prompt)
        if agent.mcp_manager:
            agent.mcp_manager.close()
        return

    repl(agent)
    if agent.mcp_manager:
        agent.mcp_manager.close()


if __name__ == "__main__":
    main()
