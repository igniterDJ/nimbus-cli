"""Test suite for nimbus.

Covers the pure helper functions and the file-editing / safety logic of the
Agent's tools — the parts that can be exercised without hitting the network or
a live model. Run from the repo root:

    python3 -m unittest discover -s tests -v

(Requires the runtime deps in requirements.txt — openai, rich — since nimbus.py
imports them at module load.)
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nimbus  # noqa: E402


# --------------------------------------------------------------------------- helpers
class RepetitionGuardTests(unittest.TestCase):
    def test_repeated_lines_flagged(self):
        loop = "Also test that the edit command works.\n\n" * 30
        self.assertTrue(nimbus._looks_degenerate(loop))

    def test_glued_substring_flagged(self):
        self.assertTrue(nimbus._looks_degenerate("foobar" * 100))

    def test_normal_prose_not_flagged(self):
        prose = ("All 8 tests pass. The rename method was added to core.py and "
                 "wired into the CLI; the README documents the new flag.")
        self.assertFalse(nimbus._looks_degenerate(prose))

    def test_varied_code_not_flagged(self):
        code = "def f():\n    return 1\n\ndef g():\n    return 2\n\ndef h():\n    return 3\n"
        self.assertFalse(nimbus._looks_degenerate(code))

    def test_a_few_repeats_not_flagged(self):
        self.assertFalse(nimbus._looks_degenerate("done.\ndone.\ndone."))


class WhitespaceFlexibleMatchTests(unittest.TestCase):
    def test_operator_spacing_difference(self):
        text = "def main(argv: int | None=None) -> int:\n    return 0\n"
        old = "def main(argv: int | None = None) -> int:"
        spans = nimbus._ws_flexible_spans(text, old)
        self.assertEqual(len(spans), 1)
        s, e = spans[0]
        self.assertEqual(text[s:e], "def main(argv: int | None=None) -> int:")

    def test_indentation_difference(self):
        text = "class A:\n    def foo(self):\n        return 1\n"
        old = "  def foo(self):\n    return 1"  # 2-space indent vs file's 4
        spans = nimbus._ws_flexible_spans(text, old)
        self.assertEqual(len(spans), 1)
        self.assertIn("def foo(self):", text[spans[0][0]:spans[0][1]])

    def test_short_needle_refused(self):
        # Too little signal to fuzzy-match safely.
        self.assertEqual(nimbus._ws_flexible_spans("x = 1\ny = 2\n", "y = 2"), [])

    def test_no_false_match(self):
        text = "alpha = 1\nbeta = 2\n"
        self.assertEqual(nimbus._ws_flexible_spans(text, "this content is absent entirely"), [])

    def test_multiple_candidates_reported(self):
        text = "value_here = 1\n# ...\nvalue_here = 1\n"
        spans = nimbus._ws_flexible_spans(text, "value_here = 1")
        self.assertEqual(len(spans), 2)


class DdgUrlTests(unittest.TestCase):
    def test_decodes_redirect(self):
        href = ("//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F"
                "library%2Fargparse.html&rut=abc")
        self.assertEqual(nimbus._ddg_clean_url(href),
                         "https://docs.python.org/3/library/argparse.html")

    def test_passthrough_clean_url(self):
        self.assertEqual(nimbus._ddg_clean_url("https://example.com/x"),
                         "https://example.com/x")

    def test_nested_query_params(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com%2Fp%3Fx%3D1%26y%3D2"
        self.assertEqual(nimbus._ddg_clean_url(href), "https://a.com/p?x=1&y=2")


class TextToolCallParsingTests(unittest.TestCase):
    def test_xml_function_form(self):
        content = ('<function=write_file><parameter=path>a.py</parameter>'
                   '<parameter=content>print(1)</parameter></function>')
        calls = nimbus.parse_text_tool_calls(content)
        self.assertEqual(calls, [("write_file", {"path": "a.py", "content": "print(1)"})])

    def test_json_tool_call_form(self):
        content = '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'
        self.assertEqual(nimbus.parse_text_tool_calls(content),
                         [("read_file", {"path": "x.py"})])

    def test_bare_json_known_tool(self):
        content = 'Sure.\n{"name": "list_directory", "parameters": {"path": "."}}'
        self.assertEqual(nimbus.parse_text_tool_calls(content),
                         [("list_directory", {"path": "."})])

    def test_bare_json_unknown_tool_ignored(self):
        content = '{"name": "not_a_real_tool", "parameters": {}}'
        self.assertEqual(nimbus.parse_text_tool_calls(content), [])

    def test_int_coercion(self):
        content = ('<function=read_file><parameter=path>x.py</parameter>'
                   '<parameter=offset>10</parameter></function>')
        name, args = nimbus.parse_text_tool_calls(content)[0]
        self.assertEqual(args["offset"], 10)
        self.assertIsInstance(args["offset"], int)

    def test_no_tool_calls(self):
        self.assertEqual(nimbus.parse_text_tool_calls("just some prose"), [])

    def test_strip_tool_calls(self):
        content = "before <tool_call>{\"name\":\"x\"}</tool_call> after"
        self.assertNotIn("tool_call", nimbus.strip_tool_calls(content))


class DangerousCommandTests(unittest.TestCase):
    import re as _re

    def _is_dangerous(self, cmd):
        import re
        return any(re.search(p, cmd) for p in nimbus.DANGEROUS_COMMANDS)

    def test_flags_destructive(self):
        for cmd in ["rm -rf /", "sudo rm x", "git push --force origin main",
                    "chmod -R 777 .", "dd if=/dev/zero of=/dev/sda"]:
            self.assertTrue(self._is_dangerous(cmd), cmd)

    def test_allows_safe(self):
        for cmd in ["ls -la", "python3 -m pytest", "git status", "rm build/tmp.o"]:
            self.assertFalse(self._is_dangerous(cmd), cmd)


# --------------------------------------------------------------------------- Agent tools
class AgentToolTests(unittest.TestCase):
    def setUp(self):
        # Silence nimbus's console output during tests.
        self._saved = {n: getattr(nimbus, n) for n in ("info", "warn", "err")}
        for n in self._saved:
            setattr(nimbus, n, lambda *a, **k: None)
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp).resolve()
        self.agent = nimbus.Agent(root=self.root, model="test", client=None, auto=True)

    def tearDown(self):
        for n, fn in self._saved.items():
            setattr(nimbus, n, fn)
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- write_file
    def test_write_creates_file(self):
        res = self.agent._tool_write_file("hello.py", "print('hi')\n")
        self.assertTrue(res.startswith("OK"))
        self.assertEqual((self.root / "hello.py").read_text(), "print('hi')\n")

    def test_write_refuses_outside_root(self):
        res = self.agent._tool_write_file("../escape.py", "x")
        self.assertIn("outside the project root", res)
        self.assertFalse((self.root.parent / "escape.py").exists())

    def test_write_blocked_by_deny_writes(self):
        # default deny_writes includes .env
        res = self.agent._tool_write_file(".env", "SECRET=1")
        self.assertIn("BLOCKED", res)
        self.assertFalse((self.root / ".env").exists())

    def test_write_tiny_stub_guard(self):
        big = "x = 1\n" * 200  # > 500 bytes
        (self.root / "big.py").write_text(big)
        res = self.agent._tool_write_file("big.py", "x")
        self.assertIn("refusing to overwrite", res)
        self.assertEqual((self.root / "big.py").read_text(), big)  # unchanged

    # ---- read_file
    def test_read_file(self):
        (self.root / "r.txt").write_text("line1\nline2\n")
        out = self.agent._tool_read_file("r.txt")
        self.assertIn("line1", out)
        self.assertIn("line2", out)

    # ---- replace_in_file
    def test_replace_exact(self):
        (self.root / "c.py").write_text("a = 1\nb = 2\n")
        res = self.agent._tool_replace_in_file("c.py", "b = 2", "b = 3")
        self.assertTrue(res.startswith("OK"))
        self.assertEqual((self.root / "c.py").read_text(), "a = 1\nb = 3\n")

    def test_replace_whitespace_fallback(self):
        # File has no spaces around '=' and 4-space indent; old_string differs.
        (self.root / "c.py").write_text(
            "class A:\n    def main(argv: int | None=None) -> int:\n        return 0\n")
        old = "  def main(argv: int | None = None) -> int:\n    return 0"
        new = "  def main(argv: int | None = None) -> int:\n    return 1"
        res = self.agent._tool_replace_in_file("c.py", old, new)
        self.assertTrue(res.startswith("OK"))
        self.assertIn("whitespace", res)
        self.assertIn("return 1", (self.root / "c.py").read_text())

    def test_replace_not_found(self):
        (self.root / "c.py").write_text("a = 1\n")
        res = self.agent._tool_replace_in_file("c.py", "nonexistent content here xyz", "z")
        self.assertIn("not found", res)

    def test_replace_non_unique(self):
        (self.root / "c.py").write_text("dup = 1\ndup = 1\n")
        res = self.agent._tool_replace_in_file("c.py", "dup = 1", "dup = 2")
        self.assertIn("must be unique", res)

    def test_replace_all(self):
        (self.root / "c.py").write_text("dup = 1\ndup = 1\n")
        res = self.agent._tool_replace_in_file("c.py", "dup = 1", "dup = 2", replace_all=True)
        self.assertTrue(res.startswith("OK"))
        self.assertEqual((self.root / "c.py").read_text(), "dup = 2\ndup = 2\n")

    # ---- permissions
    def test_permitted_deny_command(self):
        self.agent.permissions = {"allow_commands": [], "deny_commands": ["curl*"],
                                  "allow_writes": ["*"], "deny_writes": []}
        self.assertIsNotNone(self.agent._permitted("command", "curl http://x"))
        self.assertIsNone(self.agent._permitted("command", "ls"))

    def test_permitted_write_deny(self):
        self.agent.permissions = {"allow_commands": [], "deny_commands": [],
                                  "allow_writes": ["*"], "deny_writes": ["*.key"]}
        self.assertIsNotNone(self.agent._permitted("write", "secrets.key"))
        self.assertIsNone(self.agent._permitted("write", "main.py"))

    # ---- plan mode is read-only
    def test_plan_mode_blocks_writes(self):
        self.agent.plan = True
        res = self.agent._tool_write_file("x.py", "data")
        self.assertIn("PLAN MODE", res)
        self.assertFalse((self.root / "x.py").exists())

    def test_plan_mode_blocks_replace(self):
        (self.root / "c.py").write_text("a = 1\n")
        self.agent.plan = True
        res = self.agent._tool_replace_in_file("c.py", "a = 1", "a = 2")
        self.assertIn("PLAN MODE", res)
        self.assertEqual((self.root / "c.py").read_text(), "a = 1\n")

    # ---- backup + undo round trip
    def test_undo_reverts_edit(self):
        target = self.root / "c.py"
        target.write_text("original\n")
        self.agent._tool_replace_in_file("c.py", "original", "modified")
        self.assertEqual(target.read_text(), "modified\n")
        self.agent.undo()
        self.assertEqual(target.read_text(), "original\n")

    def test_undo_removes_created_file(self):
        self.agent._tool_write_file("new.py", "created\n")
        self.assertTrue((self.root / "new.py").is_file())
        self.agent.undo()
        self.assertFalse((self.root / "new.py").exists())

    # ---- listing / finding
    def test_list_directory(self):
        (self.root / "a.py").write_text("")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "b.py").write_text("")
        out = self.agent._tool_list_directory(".")
        self.assertIn("a.py", out)
        self.assertIn("sub", out)

    def test_find_files(self):
        (self.root / "a.py").write_text("")
        (self.root / "b.txt").write_text("")
        out = self.agent._tool_find_files("*.py")
        self.assertIn("a.py", out)
        self.assertNotIn("b.txt", out)


    # ---- per-turn token accounting (regression: BUG 3)
    def test_turn_delta_uses_session_increment(self):
        # Simulate one streamed call that reported usage.
        self.agent.usage = {"prompt": 0, "completion": 0, "total": 0, "requests": 0}
        snap_p, snap_c = self.agent.usage["prompt"], self.agent.usage["completion"]
        self.agent.usage["prompt"] += 1200
        self.agent.usage["completion"] += 340
        pt, ct = self.agent._turn_token_delta(snap_p, snap_c, "the answer")
        self.assertEqual((pt, ct), (1200, 340))

    def test_turn_delta_estimates_when_no_usage(self):
        # Model reported no usage at all -> delta is 0; estimate from content len.
        self.agent.usage = {"prompt": 5000, "completion": 800, "total": 5800, "requests": 3}
        snap_p, snap_c = 5000, 800  # nothing accumulated this turn
        content = "x" * 400
        pt, ct = self.agent._turn_token_delta(snap_p, snap_c, content)
        self.assertEqual(pt, 0)
        self.assertEqual(ct, 100)  # 400 chars // 4 — NOT the 800 session total

    def test_turn_delta_zero_when_empty(self):
        self.agent.usage = {"prompt": 10, "completion": 10, "total": 20, "requests": 1}
        self.assertEqual(self.agent._turn_token_delta(10, 10, "   "), (0, 0))

    # ---- compaction summary (regression: BUG 4 + None-content safety)
    def test_summary_includes_tool_calls(self):
        msgs = [
            {"role": "user", "content": "add a flag"},
            {"role": "assistant", "content": None,  # native tool call, empty prose
             "tool_calls": [{"function": {"name": "write_file",
                                          "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "content": "OK: wrote a.py"},
        ]
        text = nimbus.Agent._render_history_for_summary(msgs)
        self.assertIn("write_file", text)       # the action is captured
        self.assertIn("a.py", text)
        self.assertIn("add a flag", text)

    def test_summary_tolerates_none_content(self):
        # Must not raise AttributeError on None content (compaction runs outside
        # run_turn's try/except, so a crash here would take down the program).
        msgs = [{"role": "assistant", "content": None}]
        self.assertIsInstance(nimbus.Agent._render_history_for_summary(msgs), str)

    # ---- documentation-edit guard (regression: BUG 6)
    def test_md_guard_skips_nimbus_md(self):
        calls = []
        self.agent._confirm = lambda summary, diff=None: (calls.append(summary), True)[1]
        (self.root / "NIMBUS.md").write_text("a = 1\n")
        res = self.agent._tool_replace_in_file("NIMBUS.md", "a = 1", "a = 2")
        self.assertTrue(res.startswith("OK"))
        # No "documentation file" warning prompt for NIMBUS.md.
        self.assertFalse(any("documentation file" in c for c in calls))

    def test_md_guard_warns_on_readme(self):
        calls = []
        self.agent._confirm = lambda summary, diff=None: (calls.append(summary), True)[1]
        (self.root / "README.md").write_text("a = 1\n")
        self.agent._tool_replace_in_file("README.md", "a = 1", "a = 2")
        self.assertTrue(any("documentation file" in c for c in calls))


if __name__ == "__main__":
    unittest.main()
