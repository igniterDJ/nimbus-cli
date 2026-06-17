# nimbus Bug Report & Resolution

Originally generated from a code review of `nimbus.py` and `mcp_client.py`, then
re-verified against the current code and resolved. Each item below carries its
current **status**.

| # | Severity | Status |
|---|----------|--------|
| 1 | Critical | ✅ Fixed (spinner now wraps the whole stream-consumption loop) |
| 2 | Medium | ✅ Hardened (defensive `stream = None` + guard) |
| 3 | Medium | ✅ Fixed (per-turn token delta + content-length estimate) |
| 4 | Low-Med | ✅ Fixed (summary includes tool calls; None-content safe) |
| 5 | Low | ⚙️ By design (kept as a safety guard; see notes) |
| 6 | Low | ✅ Fixed (`NIMBUS.md` excluded from the doc-edit guard) |
| 7 | Low | ✅ Fixed (empty response is now surfaced, not silent) |
| 8 | Cosmetic | ✅ Fixed (tool-call status line is cleared) |
| 9 | Low | ✅ Fixed (`_streamed_content` initialized in `__init__`) |
| 10 | Low | ✅ Fixed (only trust `rg` exit 0/1; else fall back) |

Regression tests for the behavioural fixes (3, 4, 6) live in
`tests/test_nimbus.py`. Run the full suite offline with:

```bash
.venu/bin/python -m unittest discover -s tests -v
```

---

## BUG 1 (Critical): Spinner exits before stream is consumed — ✅ Fixed

The spinner used to cover only stream *creation*, not *consumption*, so the user
saw activity for ~200ms and then nothing while tokens streamed.

`_stream()` now builds a `spinner_ctx` (a real `Spinner` in Rich-TTY mode, a
`_NullSpinner` otherwise) and wraps the entire chunk-consumption loop in it. In
non-Rich / piped mode the streamed text is itself the progress indicator, so the
null spinner avoids fighting it for the line.

## BUG 2 (Medium): `stream` could be undefined on a retry edge case — ✅ Hardened

With `for attempt in range(len(RETRY_DELAYS) + 1)` and a final `raise`, the loop
always either binds `stream` or raises — so it was already safe. To keep it safe
under future edits to `RETRY_DELAYS` or the loop, `stream` is now initialized to
`None` before the loop and guarded (`raise RuntimeError` if still `None`) before
consumption.

## BUG 3 (Medium): Per-turn token line showed session totals — ✅ Fixed

When the model returned no usage, the fallback printed the cumulative session
completion as if it were this turn's. `run_turn` now snapshots session usage at
turn start and `_turn_token_delta()` reports `session_total − snapshot` (the
whole turn, across every streamed call). If the model reported no usage at all,
it estimates completion tokens from the final response length (~4 chars/token)
instead of leaking the session total.

## BUG 4 (Low-Med): Compaction summary missed tool-call info — ✅ Fixed

Assistant messages that drive native `tool_calls` often have empty prose, so the
summarizer never saw what tools ran. `_render_history_for_summary()` (extracted
and unit-tested) now appends `[called tools: name(args), …]` for each message
and tolerates `None` content — important because `_compact_history()` runs
outside `run_turn`'s try/except, so a crash there would take down the program.

## BUG 5 (Low): Write guard blocks small-file overwrites — ⚙️ By design

`write_file` still refuses to overwrite a >500-byte file with content under 10%
of its size, and points the model at `replace_in_file`. This is a deliberate
guard against a model truncating a real file with a stub; the documented escape
hatch (`replace_in_file`, or deleting the file first) is sufficient in practice.
Kept intentionally.

## BUG 6 (Low): `.md`/`.rst` edit guard was overly broad — ✅ Fixed

The guard now excludes `NIMBUS.md` (the project's own memory file, which is
legitimately editable) while still asking for confirmation on other docs.

## BUG 7 (Low): Empty stream silently returned an empty result — ✅ Fixed

A turn that yields no prose, no tool calls, and no usage (an API/model glitch)
used to return silently — a one-shot run would exit with no output. `run_turn`
now tracks whether the turn produced anything and warns when it produced nothing.

## BUG 8 (Cosmetic): Tool-call status line not cleared — ✅ Fixed

The transient `· preparing tool call…` line is now wiped (`\r` + padding + `\r`)
once the stream ends, so no stray characters survive into the next render.

## BUG 9 (Low): `_streamed_content` side-effect was fragile — ✅ Fixed

`self._streamed_content` is initialized to `False` in `__init__`, so it is never
undefined even if `_stream()` raises before assigning it.

## BUG 10 (Low): `rg` failure silently fell through — ✅ Fixed

`search` now inspects `rg`'s exit code: it trusts the output only on `0` (matches)
or `1` (no matches). On exit `2` (a real error) it falls through to the Python
regex engine instead of misreporting `(no matches)`.
