"""Tests for examples/review.py against a fake client: nothing here calls the API."""
import importlib.util
import io
import json
import os
import signal
import sys
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, strategies as st

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("review", os.path.join(HERE, "..", "examples", "review.py"))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

JOB_KEY = "sk-ant-api03-" + "Q" * 40 + "-jobkeyAA"   # stands in for the key exec sets
FABLE_ATTEMPTS = ["claude-fable-5-2", "claude-opus-4-8", "claude-opus-5"]


def text(t):
    return SimpleNamespace(type="text", text=t)


def message(content=None, stop="end_turn", model="claude-fable-5-2", usage=None, details=None):
    return SimpleNamespace(
        model=model, stop_reason=stop, stop_details=details,
        content=content if content is not None else [text("VERDICT: APPROVE\n\nNo findings.")],
        usage=usage if usage is not None else {"input_tokens": 1000, "output_tokens": 500},
    )


class FakeStream:
    def __init__(self, msg, raises=None, during=None, partial=None):
        self.msg, self.raises, self.during, self.request_id = msg, raises, during, "req_fake123"
        self.partial = partial

    @property
    def current_message_snapshot(self):   # like the SDK: asserts until an event has arrived
        assert self.partial is not None
        return self.partial

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        if self.during:
            self.during()
        if self.raises is not None:
            raise self.raises
        return self.msg


class FakeClient:
    """raises: an exception when the stream opens; mid_stream: one while it streams."""

    def __init__(self, msg=None, counted=1000, raises=None, mid_stream=None, during=None, partial=None):
        self.calls, self.retries = [], []
        outer = self

        class Messages:
            def count_tokens(self, **kw):
                outer.calls.append(("count", kw))
                n = counted(kw["model"]) if callable(counted) else counted
                return SimpleNamespace(input_tokens=n)

        class BetaMessages:
            def stream(self, **kw):
                outer.calls.append(("stream", kw))
                if raises is not None:
                    raise raises
                return FakeStream(msg if msg is not None else message(), mid_stream, during, partial)

        class Unretried:      # what with_options(max_retries=0) returns
            beta = SimpleNamespace(messages=BetaMessages())

        class RetryingBeta:   # the client's own retrying path must not carry the paid call
            def stream(self, **kw):
                raise AssertionError("the paid call went through the retrying client")

        self.messages = Messages()
        self.beta = SimpleNamespace(messages=RetryingBeta())
        self._unretried = Unretried()

    def with_options(self, **kw):
        self.retries.append(kw.get("max_retries"))
        return self._unretried

    def kinds(self):
        return [k for k, _ in self.calls]

    def counted_models(self):
        return [kw["model"] for k, kw in self.calls if k == "count"]


@pytest.fixture
def work(tmp_path):
    (tmp_path / "brief.md").write_text("Check the parser for off-by-one errors.\n")
    (tmp_path / "a.py").write_text("def f(x):\n    return x[1:]\n")
    (tmp_path / "change.diff").write_text("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n")
    return tmp_path


def go(work, client, *extra, env=None, stdin_bytes=b"", out=None, brief=None):
    args = review.parser().parse_args(
        ["--brief", brief or str(work / "brief.md"), "--out", out or str(work / "out.md"), *extra])
    err, stdout = io.StringIO(), io.StringIO()
    env = {"ANTHROPIC_API_KEY": JOB_KEY, "CLAUDE_CREDITS_ACCOUNT": "me@example.org"} if env is None else env
    code = review.run(args, client, stdin=SimpleNamespace(buffer=io.BytesIO(stdin_bytes)), env=env,
                      err=err, out=stdout, clock=iter([0.0, 12.0]).__next__)
    go.stdout = stdout.getvalue()
    return code, err.getvalue()


def outputs(work):
    md = (work / "out.md").read_text()
    rec = json.loads((work / "out.md.json").read_text())
    return md, rec


def test_writes_the_review_with_tokens_and_cost(work):
    client = FakeClient()
    code, err = go(work, client, str(work / "a.py"), "--diff", str(work / "change.diff"))
    assert code == 0 and client.kinds() == ["count", "count", "count", "stream"]
    assert client.counted_models() == FABLE_ATTEMPTS and client.retries == [0]
    md, rec = outputs(work)
    assert "VERDICT: APPROVE" in md and "Written by: claude-fable-5-2" in md and "WARNING" not in md
    assert rec["cost_usd"] == pytest.approx((1000 * 8 + 500 * 40) / 1e6)  # $0.028
    assert "$0.0280" in md and rec["request_id"] == "req_fake123" and not rec["fell_back"]
    # The account stays in the local record, out of the review text people may paste.
    assert rec["account"] == "me@example.org" and "me@example.org" not in md
    assert [i["kind"] for i in rec["inputs"]] == ["brief", "file", "diff"]
    assert b"\r\n" not in (work / "out.md").read_bytes()


def test_request_shape(work):
    client = FakeClient()
    go(work, client, str(work / "a.py"), "--diff", str(work / "change.diff"))
    kw = dict(client.calls[-1][1])
    assert kw["model"] == "claude-fable-5-2" and kw["output_config"] == {"effort": "high"}
    assert "thinking" not in kw  # Fable 5.2 thinks by default; explicit settings can 400
    # Named targets, not "default", so the cap prices exactly the models that can run.
    assert kw["betas"] == [review.FALLBACK_BETA]
    assert kw["fallbacks"] == [{"model": "claude-opus-4-8"}, {"model": "claude-opus-5"}]
    content = kw["messages"][0]["content"]
    assert [c["type"] for c in content] == ["document", "document", "text"]  # materials, then brief
    assert content[0]["title"] == f"file: {work / 'a.py'}"
    assert content[0]["source"]["data"].splitlines()[1] == "2      return x[1:]"
    assert content[1]["source"]["data"].startswith("--- a/a.py")  # diffs aren't numbered
    assert content[2]["text"].startswith("Brief:") and "off-by-one" in content[2]["text"]
    # The credit refuses requests carrying these identity lines.
    assert "Claude Code" not in kw["system"] and "Agent SDK" not in kw["system"]
    # Asking the model to write out its reasoning invites reasoning_extraction refusals.
    assert "reasoning" not in kw["system"].lower()
    for _, count_kw in client.calls[:-1]:
        assert count_kw["messages"] == kw["messages"] and count_kw["system"] == kw["system"]


def test_no_fallback_and_haiku_send_no_fallback_params(work):
    for extra in (["--no-fallback"], ["--model", "claude-haiku-5-5"]):
        client = FakeClient()
        go(work, client, str(work / "a.py"), "--overwrite", *extra)
        kw = client.calls[-1][1]
        assert "fallbacks" not in kw and "betas" not in kw and client.kinds() == ["count", "stream"]


def test_haiku_never_gets_fallback_params():
    brief = review.Material("brief", "b", "check it")
    params = review.build_request(brief, [review.Material("file", "f", "x")], "claude-haiku-5-5",
                                  "low", 100, fallback=True, line_numbers=True)
    assert "fallbacks" not in params and "betas" not in params


def test_pinned_worst_case():
    counts = dict.fromkeys(FABLE_ATTEMPTS, 1000)
    # 1,300 input (the margin) on each; Fable $8/$40, Opus 4.8 and Opus 5 $5/$25.
    assert review.worst_case_cost(counts, 32_000) == pytest.approx(
        (1300 * 8 + 32_000 * 40 + 2 * (1300 * 5 + 32_000 * 25)) / 1e6)  # $2.9034
    assert review.worst_case_cost({"claude-fable-5-2": 1000}, 32_000) == pytest.approx(
        (1300 * 8 + 32_000 * 40) / 1e6)


def test_each_attempt_priced_on_its_own_count(work):
    heavier = {"claude-fable-5-2": 1000, "claude-opus-4-8": 1000, "claude-opus-5": 400_000}
    client = FakeClient(counted=heavier.get)
    with pytest.raises(review.Refused, match="above --cap"):
        go(work, client, str(work / "a.py"), "--cap", "4")
    assert "stream" not in client.kinds()


def test_cap_refuses_before_sending(work):
    client = FakeClient(counted=500_000)
    with pytest.raises(review.Refused, match="above --cap"):
        go(work, client, str(work / "a.py"), "--cap", "1")
    assert "stream" not in client.kinds()
    assert not (work / "out.md").exists() and not (work / "out.md.json").exists()


@pytest.mark.parametrize("cap", ["0", "-1", "nan", "inf"])
def test_bad_caps_are_refused(work, cap):
    client = FakeClient()
    with pytest.raises(review.Refused, match="--cap"):
        go(work, client, str(work / "a.py"), "--cap", cap)
    assert client.calls == []


def test_dry_run_sends_nothing(work):
    client = FakeClient()
    code, err = go(work, client, str(work / "a.py"), "--dry-run")
    assert code == 0 and "stream" not in client.kinds() and "worst case $" in err
    assert not (work / "out.md").exists()


def test_context_window_uses_the_margin(work):
    client = FakeClient(counted=900_000)   # 990,200 with the margin, plus 32,000 output
    with pytest.raises(review.Refused, match="context window"):
        go(work, client, str(work / "a.py"), "--cap", "1000")
    assert "stream" not in client.kinds()


@pytest.mark.parametrize("planted, flag, refused", [
    (JOB_KEY, [], True),
    (JOB_KEY, ["--allow-key-like"], True),           # the job's own key is never sent
    ("sk-ant-admin01-" + "z" * 30, [], True),
    ("sk-ant-admin01-" + "z" * 30, ["--allow-key-like"], False),
    ("-----BEGIN OPENSSH PRIVATE KEY-----", [], True),
    ("-----BEGIN PGP PRIVATE KEY BLOCK-----", [], True),
    ("ghp_" + "a" * 36, [], True),
    ("AKIA" + "ABCDEFGHIJKLMNOP", [], True),
    ("ASIA" + "ABCDEFGHIJKLMNOP", [], True),
    ("xoxb-" + "1234567890-abcdef", [], True),
    ("sk-proj-" + "b" * 30, [], True),
    ("sk_live_" + "c" * 24, [], True),
    ("AIza" + "d" * 35, [], True),
    ("npm_" + "e" * 36, [], True),
    ("sk-" + "f" * 48, [], True),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N", [], True),
    ("mentions sk-ant- prefixes in prose", [], False),
])
def test_secret_guard_stops_before_any_call(work, planted, flag, refused):
    (work / "a.py").write_text(f"x = '{planted}'\n")
    client = FakeClient()
    if refused:
        with pytest.raises(review.Refused):
            go(work, client, str(work / "a.py"), *flag)
        assert client.calls == []
    else:
        assert go(work, client, str(work / "a.py"), *flag)[0] == 0


@pytest.mark.parametrize("where", ["brief", "diff", "stdin"])
def test_secret_guard_covers_every_input(work, where):
    planted = f"token = '{JOB_KEY}'\n"
    client = FakeClient()
    with pytest.raises(review.Refused, match="the API key this job runs with"):
        if where == "brief":
            (work / "brief.md").write_text(planted)
            go(work, client, str(work / "a.py"))
        elif where == "diff":
            (work / "change.diff").write_text(planted)
            go(work, client, "--diff", str(work / "change.diff"))
        else:
            go(work, client, "--diff", "-", stdin_bytes=planted.encode())
    assert client.calls == []


def test_job_key_with_trailing_newline_is_still_caught(work):
    (work / "a.py").write_text(f"x = '{JOB_KEY}'\n")
    client = FakeClient()
    with pytest.raises(review.Refused, match="the API key this job runs with"):
        go(work, client, str(work / "a.py"), "--allow-key-like",
           env={"ANTHROPIC_API_KEY": JOB_KEY + "\n"})
    assert client.calls == []


def test_key_never_reaches_outputs(work):
    client = FakeClient()
    code, err = go(work, client, str(work / "a.py"))
    md, _ = outputs(work)
    blob = md + (work / "out.md.json").read_text() + err + json.dumps(client.calls, default=str)
    assert JOB_KEY not in blob and JOB_KEY[:20] not in blob


def test_refusal_writes_no_review(work):
    msg = message(stop="refusal", content=[text("partial words")],
                  details={"category": "cyber", "explanation": None})
    code, _ = go(work, FakeClient(msg), str(work / "a.py"))
    md, rec = outputs(work)
    assert code == 3 and "DECLINED (category: cyber)" in md and "partial words" not in md
    assert rec["exit_code"] == 3 and rec["stop_reason"] == "refusal"


def test_both_models_declining_writes_no_review(work):
    usage = {"input_tokens": 1000, "output_tokens": 30, "iterations": [
        {"type": "message", "model": "claude-fable-5-2", "input_tokens": 1000, "output_tokens": 20},
        {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1000, "output_tokens": 30},
    ]}
    content = [text("fable partial"), SimpleNamespace(type="fallback"), text("opus partial")]
    code, _ = go(work, FakeClient(message(stop="refusal", model="claude-opus-4-8", usage=usage,
                                          content=content)), str(work / "a.py"))
    md, _ = outputs(work)
    assert code == 3 and "the fallback declined too" in md
    assert "fable partial" not in md and "opus partial" not in md


@pytest.mark.parametrize("stop, content, status", [
    ("max_tokens", None, "CUT OFF"),
    ("model_context_window_exceeded", None, "STOPPED EARLY"),
    ("pause_turn", None, "STOPPED EARLY"),
    ("end_turn", [text("   \n")], "NO REVIEW TEXT"),
    ("end_turn", [], "NO REVIEW TEXT"),
    ("max_tokens", [], "before any review text"),
])
def test_only_a_finished_review_counts_as_complete(work, stop, content, status):
    code, _ = go(work, FakeClient(message(stop=stop, content=content)), str(work / "a.py"))
    md, rec = outputs(work)
    assert code == 4 and status in md and rec["status"] != "complete"


def test_fallback_is_priced_per_attempt_and_drops_declined_text(work):
    usage = {"input_tokens": 1000, "output_tokens": 400, "iterations": [
        {"type": "message", "model": "claude-fable-5-2", "input_tokens": 1000, "output_tokens": 50},
        {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1000, "output_tokens": 400},
    ]}
    content = [text("half a sentence from"), SimpleNamespace(type="fallback"), text("VERDICT: APPROVE\nfine")]
    # A stream's top-level model can still name the requested model after a fallback.
    msg = message(model="claude-fable-5-2", usage=usage, content=content)
    code, _ = go(work, FakeClient(msg), str(work / "a.py"))
    md, rec = outputs(work)
    assert code == 0 and rec["fell_back"] and rec["served_model"] == "claude-opus-4-8"
    assert "Written by: claude-opus-4-8 (a fallback" in md
    assert rec["cost_usd"] == pytest.approx((1000 * 8 + 50 * 40 + 1000 * 5 + 400 * 25) / 1e6)
    assert "half a sentence" not in md and "partial text is left out" in md


def test_fallback_then_cut_off(work):
    usage = {"input_tokens": 1, "output_tokens": 1, "iterations": [
        {"type": "message", "model": "claude-fable-5-2", "input_tokens": 1000, "output_tokens": 0},
        {"type": "fallback_message", "model": "claude-opus-5", "input_tokens": 1000, "output_tokens": 900},
    ]}
    content = [SimpleNamespace(type="fallback"), text("VERDICT: REQUEST_CHANGES\n1. P1 ...")]
    code, _ = go(work, FakeClient(message(stop="max_tokens", usage=usage, content=content)), str(work / "a.py"))
    md, rec = outputs(work)
    assert code == 4 and "CUT OFF" in md and rec["served_model"] == "claude-opus-5"


def test_fallback_without_itemized_usage_reports_unknowns(work):
    content = [text("declined bit"), SimpleNamespace(type="fallback"), text("VERDICT: APPROVE")]
    msg = message(model="claude-fable-5-2", content=content)   # no usage.iterations
    code, _ = go(work, FakeClient(msg), str(work / "a.py"))
    md, rec = outputs(work)
    assert rec["fell_back"] and rec["cost_usd"] is None and rec["served_model"] is None
    assert "Written by: unknown" in md and "Cost at list price: unknown" in md
    assert rec["attempts"][-1]["model"] is None and "unreported attempt (unreported)" in md


def test_itemized_fallback_without_a_model_name(work):
    usage = {"input_tokens": 1, "output_tokens": 1, "iterations": [
        {"type": "message", "input_tokens": 1000, "output_tokens": 300}]}
    content = [SimpleNamespace(type="fallback"), text("VERDICT: APPROVE")]
    go(work, FakeClient(message(usage=usage, content=content)), str(work / "a.py"))
    md, rec = outputs(work)
    assert rec["fell_back"] and rec["served_model"] is None and "Written by: unknown" in md


def test_dated_snapshot_is_not_a_fallback(work):
    msg = message(model="claude-fable-5-2-20261101")
    code, _ = go(work, FakeClient(msg), str(work / "a.py"))
    md, rec = outputs(work)
    assert code == 0 and not rec["fell_back"] and rec["cost_usd"] == pytest.approx(0.028)


def test_a_model_outside_the_priced_set_is_flagged(work):
    usage = {"input_tokens": 1, "output_tokens": 1, "iterations": [
        {"type": "message", "model": "claude-fable-5-2", "input_tokens": 1000, "output_tokens": 0},
        {"type": "fallback_message", "model": "claude-something-new", "input_tokens": 1000, "output_tokens": 10},
    ]}
    content = [SimpleNamespace(type="fallback"), text("VERDICT: APPROVE")]
    code, err = go(work, FakeClient(message(usage=usage, content=content)), str(work / "a.py"))
    md, rec = outputs(work)
    assert rec["unpriced_models_ran"] == ["claude-something-new"] and rec["cost_usd"] is None
    assert "WARNING: a model outside the priced set ran" in md and "WARNING" in err


def test_a_bill_above_the_worst_case_is_flagged(work):
    msg = message(usage={"input_tokens": 1000, "output_tokens": 999_999})
    go(work, FakeClient(msg), str(work / "a.py"), "--no-fallback", "--max-tokens", "100")
    md, rec = outputs(work)
    assert rec["over_worst_case"] and "is above the worst case" in md


def test_refuses_to_overwrite(work):
    (work / "out.md").write_text("earlier review")
    client = FakeClient()
    with pytest.raises(review.Refused, match="exists"):
        go(work, client, str(work / "a.py"))
    assert client.calls == [] and (work / "out.md").read_text() == "earlier review"


def test_refuses_when_only_the_record_exists(work):
    (work / "out.md.json").write_text("{}")
    with pytest.raises(review.Refused, match="exists"):
        go(work, FakeClient(), str(work / "a.py"))


def test_unwritable_out_is_refused_before_paying(work):
    client = FakeClient()
    with pytest.raises(review.Refused, match="doesn't exist"):
        go(work, client, str(work / "a.py"), out=str(work / "missing" / "out.md"))
    (work / "adir").mkdir()
    with pytest.raises(review.Refused, match="is a folder"):
        go(work, client, str(work / "a.py"), "--overwrite", out=str(work / "adir"))
    assert client.calls == []


def test_the_record_failing_after_the_review_is_written(work, monkeypatch):
    real = review._atomic_write
    def json_fails(path, text):
        if path.endswith(".json"):
            raise OSError(28, "No space left on device")
        real(path, text)
    monkeypatch.setattr(review, "_atomic_write", json_fails)
    code, err = go(work, FakeClient(), str(work / "a.py"))
    assert code == 6 and '"cost_usd": 0.028' in go.stdout


def test_any_error_after_paying_still_saves_the_text(work, monkeypatch):
    def broken(*a):
        raise TypeError("an unexpected response shape")
    monkeypatch.setattr(review, "usage_rows", broken)
    code, err = go(work, FakeClient(), str(work / "a.py"))
    assert code == 6 and "TypeError" in err and "VERDICT: APPROVE" in go.stdout
    assert '"request_id": "req_fake123"' in go.stdout and '"output_tokens": 500' in go.stdout


def test_the_rescue_writes_utf8_whatever_stdout_is(work, monkeypatch):
    monkeypatch.setattr(review, "_atomic_write", lambda p, t: (_ for _ in ()).throw(PermissionError(13, "locked")))
    raw = io.BytesIO()
    cp1252 = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    args = review.parser().parse_args(["--brief", str(work / "brief.md"), "--out", str(work / "out.md"),
                                       str(work / "a.py")])
    msg = message(content=[text("VERDICT: APPROVE \u2192 fine \u2264 1")])
    code = review.run(args, FakeClient(msg), env={"ANTHROPIC_API_KEY": JOB_KEY}, err=io.StringIO(),
                      out=cp1252, clock=iter([0.0, 1.0]).__next__)
    assert code == 6 and "\u2192 fine \u2264".encode("utf-8") in raw.getvalue()


def test_an_out_file_that_appears_during_the_request(work):
    client = FakeClient(during=lambda: (work / "out.md").write_text("another run's review"))
    code, err = go(work, client, str(work / "a.py"))
    assert code == 6 and (work / "out.md").read_text() == "another run's review"
    assert "VERDICT: APPROVE" in go.stdout


def test_a_read_only_target_is_refused_before_paying(work):
    target = work / "out.md"
    target.write_text("old")
    target.chmod(0o444)
    client = FakeClient()
    try:
        with pytest.raises(review.Refused, match="can't replace"):
            go(work, client, str(work / "a.py"), "--overwrite")
    finally:
        target.chmod(0o644)
    assert client.calls == []


def test_a_write_failure_after_paying_prints_the_review(work, monkeypatch):
    def broken(path, text):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(review, "_atomic_write", broken)
    code, err = go(work, FakeClient(), str(work / "a.py"))
    assert code == 6 and "paid (request id req_fake123), but couldn't write" in err
    assert "VERDICT: APPROVE" in go.stdout and '"cost_usd": 0.028' in go.stdout


def test_atomic_write_leaves_no_temp_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "r.md"
    def fail(src, dst):
        raise PermissionError(13, "locked")
    monkeypatch.setattr(review.os, "replace", fail)
    with pytest.raises(PermissionError):
        review._atomic_write(str(target), "x")
    assert os.listdir(tmp_path) == []


def test_inputs(work):
    (work / "bin").write_bytes(b"\x00\x01")
    (work / "latin").write_bytes("caf\xe9".encode("latin-1"))
    (work / "blank").write_text("  \n")
    for bad, why in (("bin", "binary"), ("latin", "UTF-8"), ("blank", "empty"), ("missing", "can't read")):
        with pytest.raises(review.Refused, match=why):
            go(work, FakeClient(), str(work / bad))
    with pytest.raises(review.Refused, match="nothing to review"):
        go(work, FakeClient())
    with pytest.raises(review.Refused, match="only one input can read stdin"):
        go(work, FakeClient(), "--diff", "-", brief="-")
    with pytest.raises(review.Refused, match="only one input can read stdin"):
        go(work, FakeClient(), "-", "--diff", "-")
    code, _ = go(work, FakeClient(), "--diff", "-", stdin_bytes=b"--- a\n+++ b\n")
    assert code == 0 and outputs(work)[1]["inputs"][1]["name"] == "(stdin)"


def test_utf16_and_crlf_inputs(work):
    (work / "ps.diff").write_bytes("﻿--- a\r\n+++ b\r\n".encode("utf-16-le"))
    (work / "crlf.py").write_bytes(b"one\r\ntwo\r\n")
    client = FakeClient()
    assert go(work, client, str(work / "crlf.py"), "--diff", str(work / "ps.diff"))[0] == 0
    content = client.calls[-1][1]["messages"][0]["content"]
    assert content[0]["source"]["data"] == "1  one\n2  two"
    assert content[1]["source"]["data"].startswith("--- a")


def test_line_numbers_follow_newlines_only(work):
    (work / "ff.py").write_text("one\f\ntwo\u2028still two\nthree\n")
    client = FakeClient()
    go(work, client, str(work / "ff.py"))
    data = client.calls[-1][1]["messages"][0]["content"][0]["source"]["data"]
    assert data.split("\n") == ["1  one\f", "2  two\u2028still two", "3  three"]


def test_price_lookup():
    assert review.prices_for("claude-opus-5-5-20261101", 10) == review.PRICES["claude-opus-5-5"]
    assert review.prices_for("claude-opus-5", 10) == (5.00, 25.00)
    assert review.prices_for("claude-haiku-5-5", 200_000) == review.HAIKU_LONG_PRICES
    for unknown in ("claude-opus-55", "claude-opus-5-1", None):
        with pytest.raises(KeyError):
            review.prices_for(unknown, 10)


def test_input_margin_and_cache_prices():
    assert review.billable_input_bound(1000) == 1300 and review.billable_input_bound(1) == 202
    row = review.attempt_cost("claude-fable-5-2", {"input_tokens": 100, "output_tokens": 10,
                                                   "cache_creation_input_tokens": 1000,
                                                   "cache_read_input_tokens": 10_000})
    assert row["cost_usd"] == pytest.approx((100 * 8 + 1000 * 8 * 1.25 + 10_000 * 0.20 + 10 * 40) / 1e6)
    reads = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 1_000_000}
    for model, price in (("claude-fable-5-2", 0.20), ("claude-opus-5-5", 0.20),
                         ("claude-sonnet-5-5", 0.10), ("claude-opus-5", 0.50),
                         ("claude-opus-4-8", 0.50), ("claude-sonnet-5", 0.20),
                         ("claude-haiku-5-5", 0.05), ("claude-fable-5-2-20261101", 0.20)):
        assert review.attempt_cost(model, reads)["cost_usd"] == pytest.approx(price), model


def test_api_error_messages(monkeypatch):
    monkeypatch.setenv("CLAUDE_CREDITS_ACCOUNT", "me@example.org")
    assert "claude-credits mark me@example.org exhausted" in review.api_error_message(
        400, "Your credit balance is too low to access the Anthropic API.")
    assert "nothing was billed" in review.api_error_message(529, "Overloaded")
    assert "(500)" in review.api_error_message(500, "boom")


class FakeSDK:
    class APIError(Exception):
        pass

    class APIStatusError(APIError):
        def __init__(self, status_code, text):
            super().__init__(text)
            self.status_code = status_code

    class APIConnectionError(APIError):
        pass

    def __init__(self, client):
        self._client = client

    def Anthropic(self, **kw):
        return self._client


class LostConnection(Exception):   # stands in for the HTTP library's read errors
    pass


@pytest.mark.parametrize("client_kw, code, says", [
    ({"raises": FakeSDK.APIStatusError(400, "Your credit balance is too low")}, 5, "used up"),
    ({"raises": FakeSDK.APIStatusError(529, "Overloaded")}, 5, "nothing was billed"),
    ({"raises": FakeSDK.APIConnectionError("down")}, 5, "before streaming began"),
    # Once the stream is open, any failure may already be billed, whatever its class.
    ({"mid_stream": LostConnection("connection reset")}, 5, "may still be billed"),
    ({"mid_stream": FakeSDK.APIStatusError(529, "overloaded_error event")}, 5, "may still be billed"),
    ({"mid_stream": KeyboardInterrupt()}, 130, "interrupted"),
])
def test_main_maps_errors_to_exit_codes(work, monkeypatch, capsys, client_kw, code, says):
    monkeypatch.setenv("ANTHROPIC_API_KEY", JOB_KEY)
    argv = ["--brief", str(work / "brief.md"), "--out", str(work / "out.md"), str(work / "a.py")]
    rc = review.main(argv, sdk=FakeSDK(FakeClient(**client_kw)))
    assert rc == code and says in capsys.readouterr().err
    assert not (work / "out.md").exists()


def test_main_refusals_exit_2(work, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", JOB_KEY)
    for extra in (["--max-tokens", "0"], ["--cap", "nan"]):
        argv = ["--brief", str(work / "brief.md"), "--out", str(work / "out.md"), str(work / "a.py"), *extra]
        client = FakeClient()
        assert review.main(argv, sdk=FakeSDK(client)) == 2
        assert client.calls == []


@settings(max_examples=300, deadline=None)
@given(
    model=st.sampled_from(review.MODELS),
    fallback=st.booleans(),
    max_tokens=st.integers(1, 128_000),
    data=st.data(),
)
def test_the_cap_bounds_every_bill(model, fallback, max_tokens, data):
    """Whatever attempts run (the requested model, then any of the named fallbacks, each
    at most once), with billed input within each model's counting margin and output
    within max_tokens, the bill never exceeds the worst case the cap is checked against."""
    allowed = [model] + (list(review.FALLBACK_TARGETS[model]) if fallback else [])
    counts = {m: data.draw(st.integers(1, 900_000)) for m in allowed}
    worst = review.worst_case_cost(counts, max_tokens)
    ran = data.draw(st.lists(st.sampled_from(allowed), unique=True, min_size=1))
    bill = 0.0
    for m in ran:
        u = {"input_tokens": data.draw(st.integers(0, review.billable_input_bound(counts[m]))),
             "output_tokens": data.draw(st.integers(0, max_tokens))}
        bill += review.attempt_cost(m, u)["cost_usd"]
    assert bill <= worst + 1e-12


# Round 3 (independent review): names, partial output, the cap's input margin.

def test_the_job_key_in_a_file_name_is_refused_without_printing_it(work):
    named = work / f"notes-{JOB_KEY}.txt"
    named.write_text("plain text\n")
    client = FakeClient()
    with pytest.raises(review.Refused, match="the name of file input 2") as e:
        go(work, client, str(named), "--allow-key-like")
    assert JOB_KEY not in str(e.value) and client.calls == []


def test_a_key_like_file_name_is_refused_unless_allowed(work):
    dummy = "sk-ant-api03-" + "Z" * 40
    named = work / f"{dummy}.txt"
    named.write_text("plain text\n")
    with pytest.raises(review.Refused, match="looks like an Anthropic") as e:
        go(work, FakeClient(), str(named))
    assert dummy not in str(e.value)
    assert go(work, FakeClient(), str(named), "--allow-key-like")[0] == 0


def test_a_key_in_the_brief_name_is_refused(work):
    named = work / f"{JOB_KEY}.md"
    named.write_text("Review it.\n")
    with pytest.raises(review.Refused, match="the name of brief input 1"):
        go(work, FakeClient(), str(work / "a.py"), brief=str(named))


def partial_message():
    return message(content=[text("VERDICT: REQUEST_CHANGES\n\n1. P1 half a find")], stop=None,
                   usage={"input_tokens": 1000, "output_tokens": 20})


@pytest.mark.parametrize("exc, code", [(LostConnection("connection reset"), 5), (KeyboardInterrupt(), 130)])
def test_what_streamed_is_saved_when_the_stream_breaks(work, exc, code):
    rc, err = go(work, FakeClient(mid_stream=exc, partial=partial_message()), str(work / "a.py"))
    md, rec = outputs(work)
    assert rc == code and "INCOMPLETE" in md and "half a find" in md and "the bill may be higher" in md
    assert rec["usage_incomplete"] is True and rec["exit_code"] == code
    assert rec["request_id"] == "req_fake123" and rec["attempts"][0]["output_tokens"] == 20


def test_ctrl_c_before_any_output_names_the_request(work):
    args = review.parser().parse_args(["--brief", str(work / "brief.md"), "--out", str(work / "out.md"),
                                       str(work / "a.py")])
    err = io.StringIO()
    with pytest.raises(KeyboardInterrupt):
        review.run(args, FakeClient(mid_stream=KeyboardInterrupt()), env={"ANTHROPIC_API_KEY": JOB_KEY},
                   err=err, out=io.StringIO(), clock=iter([0.0, 1.0]).__next__)
    assert "req_fake123" in err.getvalue() and not (work / "out.md").exists()


def test_a_stream_failure_before_output_names_the_request(work, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", JOB_KEY)
    argv = ["--brief", str(work / "brief.md"), "--out", str(work / "out.md"), str(work / "a.py")]
    assert review.main(argv, sdk=FakeSDK(FakeClient(mid_stream=LostConnection("reset")))) == 5
    assert "request id req_fake123" in capsys.readouterr().err


def test_ctrl_c_while_saving_still_hands_over_the_review(work, monkeypatch):
    def interrupted(*a):
        raise KeyboardInterrupt()
    monkeypatch.setattr(review, "usage_rows", interrupted)
    code, err = go(work, FakeClient(), str(work / "a.py"))
    assert code == 130 and "VERDICT: APPROVE" in go.stdout and "req_fake123" in go.stdout


def test_the_bill_is_flagged_once_input_passes_the_counting_margin(work):
    counted = 1000
    bound = review.billable_input_bound(counted)
    for billed, flagged in ((bound, False), (bound + 1, True)):
        msg = message(usage={"input_tokens": billed, "output_tokens": 32_000})
        go(work, FakeClient(msg, counted=counted), str(work / "a.py"), "--no-fallback", "--overwrite")
        md, rec = outputs(work)
        assert rec["over_worst_case"] is flagged and ("WARNING" in md) is flagged


# Round 4 (in-session check): signals, an early end, final counts, names before reading.

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


@posix_only
@pytest.mark.parametrize("sig", ["SIGTERM", "SIGHUP"])
def test_a_signal_mid_stream_saves_what_arrived_and_restores_handlers(work, sig):
    signum = getattr(signal, sig)

    def sentinel(*a):   # the caller's own handler, which the run must put back
        raise AssertionError("the caller's handler ran during the request")

    original = signal.signal(signum, sentinel)
    try:
        client = FakeClient(partial=partial_message(), during=lambda: os.kill(os.getpid(), signum))
        rc, err = go(work, client, str(work / "a.py"))
        assert signal.getsignal(signum) is sentinel
    finally:
        signal.signal(signum, original)
    md, rec = outputs(work)
    assert rc == 128 + signum and f"stopped by signal {signum}" in md and "half a find" in md
    assert rec["usage_incomplete"] is True and rec["exit_code"] == 128 + signum


@posix_only
def test_a_signal_before_any_output_exits_with_its_code(work, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", JOB_KEY)
    argv = ["--brief", str(work / "brief.md"), "--out", str(work / "out.md"), str(work / "a.py")]
    client = FakeClient(during=lambda: os.kill(os.getpid(), signal.SIGTERM))
    assert review.main(argv, sdk=FakeSDK(client)) == 143
    err = capsys.readouterr().err
    assert "stopped by signal 15" in err and "request id req_fake123" in err
    assert not (work / "out.md").exists()


def test_the_request_id_is_printed_as_soon_as_the_stream_opens(work):
    code, err = go(work, FakeClient(), str(work / "a.py"))
    assert code == 0 and err.index("review: request id req_fake123") < err.index("complete")


def test_a_stream_that_ends_before_its_final_event_is_incomplete(work):
    rc, err = go(work, FakeClient(partial_message()), str(work / "a.py"))
    md, rec = outputs(work)
    assert rc == 5 and "INCOMPLETE: the stream ended before Anthropic's final event" in md
    assert rec["usage_incomplete"] is True and "half a find" in md


def test_a_failure_after_the_final_counts_keeps_the_real_outcome(work):
    done = message(usage={"input_tokens": 1000, "output_tokens": 500})
    rc, err = go(work, FakeClient(mid_stream=LostConnection("reset"), partial=done), str(work / "a.py"))
    md, rec = outputs(work)
    assert rc == 0 and rec["usage_incomplete"] is False and "the token counts are final" in md
    declined = message(stop="refusal", content=[text("partial words")], details={"category": "cyber"})
    rc, err = go(work, FakeClient(mid_stream=KeyboardInterrupt(), partial=declined), str(work / "a.py"),
                 "--overwrite")
    md, rec = outputs(work)
    assert rc == 3 and "DECLINED" in md and "partial words" not in md


def test_a_key_in_a_missing_file_name_is_refused_before_reading(work):
    with pytest.raises(review.Refused, match="the name of file input 2") as e:
        go(work, FakeClient(), str(work / f"{JOB_KEY}.txt"), "--allow-key-like")
    assert JOB_KEY not in str(e.value) and "can't read" not in str(e.value)


def test_the_job_key_split_across_lines_is_still_caught(work):
    (work / "a.py").write_text(f"key, wrapped by an editor:\n{JOB_KEY[:15]}\n  {JOB_KEY[15:]}\n")
    with pytest.raises(review.Refused, match="the API key this job runs with"):
        go(work, FakeClient(), str(work / "a.py"), "--allow-key-like")


def test_an_auth_token_is_screened_like_the_api_key(work):
    token = "oauth-" + "T" * 40
    (work / "a.py").write_text(f"t = '{token}'\n")
    with pytest.raises(review.Refused, match="the API key this job runs with"):
        go(work, FakeClient(), str(work / "a.py"), "--allow-key-like", env={"ANTHROPIC_AUTH_TOKEN": token})


def test_when_stdout_fails_too_the_request_id_still_reaches_stderr(work, monkeypatch):
    monkeypatch.setattr(review, "_atomic_write", lambda p, t: (_ for _ in ()).throw(PermissionError(13, "locked")))

    class DeadOut:
        def write(self, *a):
            raise OSError(5, "stdout gone")

        def flush(self):
            raise OSError(5, "stdout gone")

    args = review.parser().parse_args(["--brief", str(work / "brief.md"), "--out", str(work / "out.md"),
                                       str(work / "a.py")])
    err = io.StringIO()
    code = review.run(args, FakeClient(), env={"ANTHROPIC_API_KEY": JOB_KEY}, err=err, out=DeadOut(),
                      clock=iter([0.0, 1.0]).__next__)
    assert code == 6 and "stdout failed too" in err.getvalue() and "req_fake123" in err.getvalue()
