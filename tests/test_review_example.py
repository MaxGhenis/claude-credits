"""Tests for examples/review.py against a fake client: nothing here calls the API."""
import importlib.util
import io
import json
import os
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, strategies as st

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("review", os.path.join(HERE, "..", "examples", "review.py"))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

JOB_KEY = "sk-ant-api03-" + "Q" * 40 + "-jobkeyAA"   # stands in for the key exec sets


def text(t):
    return SimpleNamespace(type="text", text=t)


def message(content=None, stop="end_turn", model="claude-fable-5-2", usage=None, details=None):
    return SimpleNamespace(
        model=model, stop_reason=stop, stop_details=details,
        content=content if content is not None else [text("VERDICT: APPROVE\n\nNo findings.")],
        usage=usage if usage is not None else {"input_tokens": 1000, "output_tokens": 500},
    )


class FakeStream:
    def __init__(self, msg):
        self.msg, self.request_id = msg, "req_fake123"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, msg=None, counted=1000):
        self.calls = []
        outer = self

        class Messages:
            def count_tokens(self, **kw):
                outer.calls.append(("count", kw))
                return SimpleNamespace(input_tokens=counted)

        class BetaMessages:
            def stream(self, **kw):
                outer.calls.append(("stream", kw))
                return FakeStream(msg if msg is not None else message())

        self.messages = Messages()
        self.beta = SimpleNamespace(messages=BetaMessages())

    def kinds(self):
        return [k for k, _ in self.calls]


@pytest.fixture
def work(tmp_path):
    (tmp_path / "brief.md").write_text("Check the parser for off-by-one errors.\n")
    (tmp_path / "a.py").write_text("def f(x):\n    return x[1:]\n")
    (tmp_path / "change.diff").write_text("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n")
    return tmp_path


def go(work, client, *extra, env=None, stdin_bytes=b""):
    args = review.parser().parse_args(
        ["--brief", str(work / "brief.md"), "--out", str(work / "out.md"), *extra])
    err = io.StringIO()
    env = {"ANTHROPIC_API_KEY": JOB_KEY, "CLAUDE_CREDITS_ACCOUNT": "me@example.org"} if env is None else env
    code = review.run(args, client, stdin=SimpleNamespace(buffer=io.BytesIO(stdin_bytes)), env=env,
                      err=err, clock=iter([0.0, 12.0]).__next__)
    return code, err.getvalue()


def outputs(work):
    md = (work / "out.md").read_text()
    rec = json.loads((work / "out.md.json").read_text())
    return md, rec


def test_writes_the_review_with_tokens_and_cost(work):
    client = FakeClient()
    code, err = go(work, client, str(work / "a.py"), "--diff", str(work / "change.diff"))
    assert code == 0 and client.kinds() == ["count", "stream"]
    md, rec = outputs(work)
    assert "VERDICT: APPROVE" in md and "Written by: claude-fable-5-2" in md
    assert rec["cost_usd"] == pytest.approx((1000 * 8 + 500 * 40) / 1e6)  # $0.028
    assert "$0.0280" in md and rec["request_id"] == "req_fake123"
    assert rec["account"] == "me@example.org" and "me@example.org" in md
    assert [i["kind"] for i in rec["inputs"]] == ["brief", "file", "diff"]


def test_request_shape(work):
    client = FakeClient()
    go(work, client, str(work / "a.py"), "--diff", str(work / "change.diff"))
    kw = dict(client.calls[1][1])
    assert kw["model"] == "claude-fable-5-2" and kw["output_config"] == {"effort": "high"}
    assert "thinking" not in kw  # Fable 5.2 thinks by default; explicit settings can 400
    assert kw["betas"] == [review.FALLBACK_BETA] and kw["fallbacks"] == "default"
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
    count_kw = client.calls[0][1]
    assert count_kw["messages"] == kw["messages"] and count_kw["system"] == kw["system"]


def test_no_fallback_and_haiku_send_no_fallback_params(work):
    for extra in (["--no-fallback"], ["--model", "claude-haiku-5-5"]):
        client = FakeClient()
        go(work, client, str(work / "a.py"), "--overwrite", *extra)
        kw = client.calls[1][1]
        assert "fallbacks" not in kw and "betas" not in kw


def test_cap_refuses_before_sending(work):
    client = FakeClient(counted=900_000)
    with pytest.raises(review.Refused, match="above --cap"):
        go(work, client, str(work / "a.py"), "--cap", "1")
    assert client.kinds() == ["count"]
    assert not (work / "out.md").exists() and not (work / "out.md.json").exists()


def test_dry_run_sends_nothing(work):
    client = FakeClient()
    code, err = go(work, client, str(work / "a.py"), "--dry-run")
    assert code == 0 and client.kinds() == ["count"] and "worst case $" in err
    assert not (work / "out.md").exists()


def test_context_window(work):
    client = FakeClient(counted=990_000)
    with pytest.raises(review.Refused, match="context window"):
        go(work, client, str(work / "a.py"), "--cap", "1000")
    assert client.kinds() == ["count"]


@pytest.mark.parametrize("planted, flag, refused", [
    (JOB_KEY, [], True),
    (JOB_KEY, ["--allow-key-like"], True),           # the job's own key is never sent
    ("sk-ant-admin01-" + "z" * 30, [], True),
    ("sk-ant-admin01-" + "z" * 30, ["--allow-key-like"], False),
    ("-----BEGIN OPENSSH PRIVATE KEY-----", [], True),
    ("ghp_" + "a" * 36, [], True),
    ("AKIA" + "ABCDEFGHIJKLMNOP", [], True),
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


def test_cut_off_review_is_marked(work):
    code, _ = go(work, FakeClient(message(stop="max_tokens")), str(work / "a.py"))
    md, _ = outputs(work)
    assert code == 4 and "CUT OFF" in md and "VERDICT: APPROVE" in md


def test_fallback_is_priced_per_attempt_and_drops_declined_text(work):
    usage = {"input_tokens": 1000, "output_tokens": 400, "iterations": [
        {"type": "message", "model": "claude-fable-5-2", "input_tokens": 1000, "output_tokens": 50},
        {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1000, "output_tokens": 400},
    ]}
    content = [text("half a sentence from"), SimpleNamespace(type="fallback"), text("VERDICT: APPROVE\nfine")]
    msg = message(model="claude-opus-4-8", usage=usage, content=content)
    code, _ = go(work, FakeClient(msg), str(work / "a.py"))
    md, rec = outputs(work)
    assert code == 0 and rec["fell_back"] and rec["served_model"] == "claude-opus-4-8"
    assert rec["cost_usd"] == pytest.approx((1000 * 8 + 50 * 40 + 1000 * 5 + 400 * 25) / 1e6)
    assert "half a sentence" not in md and "partial text is left out" in md


def test_unknown_served_model_reports_unknown_cost(work):
    msg = message(model="claude-something-new", usage={"input_tokens": 10, "output_tokens": 10})
    go(work, FakeClient(msg), str(work / "a.py"))
    md, rec = outputs(work)
    assert rec["cost_usd"] is None and "unknown" in md


def test_refuses_to_overwrite(work):
    (work / "out.md").write_text("earlier review")
    client = FakeClient()
    with pytest.raises(review.Refused, match="exists"):
        go(work, client, str(work / "a.py"))
    assert client.calls == [] and (work / "out.md").read_text() == "earlier review"


def test_inputs(work):
    (work / "bin").write_bytes(b"\x00\x01")
    (work / "latin").write_bytes("caf\xe9".encode("latin-1"))
    (work / "blank").write_text("  \n")
    for bad, why in (("bin", "binary"), ("latin", "UTF-8"), ("blank", "empty"), ("missing", "can't read")):
        with pytest.raises(review.Refused, match=why):
            go(work, FakeClient(), str(work / bad))
    with pytest.raises(review.Refused, match="nothing to review"):
        go(work, FakeClient())
    code, _ = go(work, FakeClient(), "--diff", "-", stdin_bytes=b"--- a\n+++ b\n")
    assert code == 0 and outputs(work)[1]["inputs"][1]["name"] == "(stdin)"


def test_price_lookup_prefers_longest_prefix():
    assert review.prices_for("claude-opus-5-5-20261101", 10) == review.PRICES["claude-opus-5-5"]
    assert review.prices_for("claude-opus-5", 10) == (5.00, 25.00)
    assert review.prices_for("claude-haiku-5-5", 200_000) == review.HAIKU_LONG_PRICES
    with pytest.raises(KeyError):
        review.prices_for("claude-opus-55", 10)


def test_input_margin_and_cache_prices():
    assert review.billable_input_bound(1000) == 1300 and review.billable_input_bound(1) == 202
    row = review.attempt_cost("claude-fable-5-2", {"input_tokens": 100, "output_tokens": 10,
                                                   "cache_creation_input_tokens": 1000,
                                                   "cache_read_input_tokens": 10_000})
    assert row["cost_usd"] == pytest.approx((100 * 8 + 1000 * 8 * 1.25 + 10_000 * 8 * 0.10 + 10 * 40) / 1e6)


def test_haiku_never_gets_fallback_params():
    brief = review.Material("brief", "b", "check it")
    params = review.build_request(brief, [review.Material("file", "f", "x")], "claude-haiku-5-5",
                                  "low", 100, fallback=True, line_numbers=True)
    assert "fallbacks" not in params and "betas" not in params


def test_credit_error_message(monkeypatch):
    monkeypatch.setenv("CLAUDE_CREDITS_ACCOUNT", "me@example.org")
    assert "claude-credits mark me@example.org exhausted" in review.api_error_message(
        400, "Your credit balance is too low to access the Anthropic API.")
    assert "(529)" in review.api_error_message(529, "Overloaded")


@settings(max_examples=300, deadline=None)
@given(
    model=st.sampled_from(review.MODELS),
    fallback=st.booleans(),
    counted=st.integers(1, 900_000),
    max_tokens=st.integers(1, 128_000),
    data=st.data(),
)
def test_the_cap_bounds_every_bill(model, fallback, counted, max_tokens, data):
    """Whatever attempts run (the requested model, then any permitted fallbacks, each at
    most once), with billed input within the counting margin and output within
    max_tokens, the bill never exceeds the worst case the cap is checked against."""
    worst = review.worst_case_cost(model, counted, max_tokens, fallback)
    allowed = [model] + (list(review.FALLBACK_TARGETS[model]) if fallback else [])
    ran = data.draw(st.lists(st.sampled_from(allowed), unique=True, min_size=1))
    bound = review.billable_input_bound(counted)
    bill = 0.0
    for m in ran:
        u = {"input_tokens": data.draw(st.integers(0, bound)), "output_tokens": data.draw(st.integers(0, max_tokens))}
        bill += review.attempt_cost(m, u)["cost_usd"]
    assert bill <= worst + 1e-12
