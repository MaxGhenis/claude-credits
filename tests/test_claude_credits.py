"""Tests for claude-credits.

Pure logic gets Hypothesis properties: spending order, renewal dates, key classification.
Commands run against a fake API, through the real `keyring` code path with an in-memory
backend, and through the file store on disk. Set CLAUDE_CREDITS_REAL_KEYRING=1 to also
round-trip a dummy key through your actual OS keychain (it's deleted afterwards).

Run: uv run --with pytest --with hypothesis --with keyring pytest -q
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import os
import secrets
import stat
import string
import sys
from pathlib import Path

import keyring
import keyring.backend
import keyring.errors
import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import claude_credits as cc  # noqa: E402

KEY_ALPHABET = string.ascii_letters + string.digits + "-_"
REF = dt.date(2026, 10, 22)


def fake_key(kind: str = "usr", n: int = 95) -> str:
    return f"sk-ant-{kind}-" + "".join(secrets.choice(KEY_ALPHABET) for _ in range(n))


keys = st.text(alphabet=KEY_ALPHABET, min_size=40, max_size=120).map(lambda s: "sk-ant-usr-" + s)
dates = st.dates(min_value=dt.date(2025, 1, 1), max_value=dt.date(2030, 12, 31))


class MemoryKeyring(keyring.backend.KeyringBackend):
    priority = 1

    def __init__(self):
        super().__init__()
        self.items = {}

    def get_password(self, service, username):
        return self.items.get((service, username))

    def set_password(self, service, username, password):
        self.items[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) not in self.items:
            raise keyring.errors.PasswordDeleteError("not found")
        del self.items[(service, username)]


# ---------- keys ----------

@given(keys)
def test_console_keys_classify_as_usr(key):
    assert cc.classify(key) == "usr"


@given(st.sampled_from(["oat01", "ort01", "admin01"]), st.text(alphabet=KEY_ALPHABET, min_size=40, max_size=100))
def test_oauth_and_admin_tokens_are_never_accepted(prefix, body):
    with pytest.raises(cc.Refused):
        cc.classify(f"sk-ant-{prefix}-{body}")


@given(st.text(max_size=200))
def test_classify_accepts_only_the_key_shape(s):
    try:
        kind = cc.classify(s)
    except cc.Refused:
        return
    assert cc.KEY_RE.match(s) and kind in ("usr", "svc", "api")


@given(keys)
def test_fingerprint_is_short_stable_and_reveals_no_key_text(key):
    fp = cc.fingerprint(key)
    assert fp == cc.fingerprint(key) and len(fp) == len("sha256:") + 16
    body = key[len("sk-ant-usr-"):]
    assert all(body[i:i + 8] not in fp for i in range(len(body) - 7))


# ---------- renewal dates ----------

@given(st.integers(1, 31), dates)
def test_renewal_on_or_after_is_the_first_renewal(day, ref):
    r = cc.renewal_on_or_after(day, ref)
    assert r >= ref and (r - ref).days <= 31
    is_renewal = lambda d: d.day == min(day, calendar.monthrange(d.year, d.month)[1])
    assert is_renewal(r)
    assert not any(is_renewal(ref + dt.timedelta(n)) for n in range((r - ref).days))


@given(st.integers(1, 31), dates)
def test_days_left_is_between_1_and_31(day, ref):
    assert 1 <= cc.days_left({"renews_day": day}, ref) <= 31


def test_days_left_examples():
    assert cc.days_left({"renews_day": 22}, REF) == 31
    assert cc.days_left({"renews_day": 17}, REF) == 26
    assert cc.days_left({}, REF) is None
    assert cc.renewal_on_or_after(31, dt.date(2026, 2, 3)) == dt.date(2026, 2, 28)


# ---------- spending order ----------

entry = st.fixed_dictionaries({
    "status": st.sampled_from(["ok", "unknown", "invalid", "exhausted"]),
    "renews_day": st.one_of(st.none(), st.integers(1, 31)),
    "last_used": st.one_of(st.none(), st.sampled_from(["2026-10-01T00:00:00Z", "2026-10-21T00:00:00Z"])),
    "status_until": st.one_of(st.none(), st.sampled_from(["2026-10-01", "2026-11-30"])),
})
key_sets = st.dictionaries(st.from_regex(r"[a-z]{1,8}", fullmatch=True), entry, max_size=10)


@given(key_sets)
def test_order_offers_each_live_key_once_and_no_dead_one(ks):
    out = cc.order(ks, REF)
    live = {n for n, e in ks.items() if cc.effective_status(e, REF) in ("ok", "unknown")}
    assert sorted(out) == sorted(live) and len(out) == len(set(out))


@given(key_sets)
def test_order_spends_soonest_expiry_first_then_ok_before_unknown(ks):
    out = cc.order(ks, REF)
    rank = [(cc.days_left(ks[n], REF) is None, cc.days_left(ks[n], REF) or 0,
             cc.effective_status(ks[n], REF) != "ok") for n in out]
    assert rank == sorted(rank)


@given(key_sets, st.randoms())
def test_order_ignores_insertion_order(ks, rnd):
    items = list(ks.items())
    rnd.shuffle(items)
    assert cc.order(dict(items), REF) == cc.order(ks, REF)


def test_exhaustion_lapses_when_until_passes():
    a = {"status": "exhausted", "status_until": "2026-10-22"}
    assert cc.effective_status(a, REF) == "unknown"
    assert cc.effective_status({**a, "status_until": "2026-10-23"}, REF) == "exhausted"
    assert cc.effective_status({"status": "exhausted"}, REF) == "unknown"   # no end date: re-check


# ---------- commands ----------

class FakeAPI:
    def __init__(self):
        self.orgs, self.exhausted, self.calls = {}, set(), []

    def http(self, method, path, key, body=None):
        self.calls.append((method, path))
        if key not in self.orgs:
            return 401, {}, {"error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        h = {"anthropic-organization-id": self.orgs[key], "anthropic-ratelimit-requests-limit": "1000"}
        if path.startswith("/v1/models"):
            return 200, h, {"data": [{"id": "claude-haiku-5-5"}, {"id": "claude-opus-5-5"}]}
        if key in self.exhausted:
            return 400, h, {"error": {"type": "invalid_request_error",
                                      "message": "Your credit balance is too low to access the Anthropic API."}}
        return 200, h, {"content": [{"type": "text", "text": "o"}]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    api, kr = FakeAPI(), MemoryKeyring()
    monkeypatch.setenv("CLAUDE_CREDITS_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr(cc, "http", api.http)
    monkeypatch.setattr(cc, "today", lambda: REF)

    def no_real_exec(*a):  # a test that forgets to stub exec must fail, not replace pytest
        raise AssertionError("os.execve called for real in a test")
    monkeypatch.setattr(cc.os, "execve", no_real_exec)
    monkeypatch.setattr(cc.subprocess, "call", no_real_exec)
    keyring.set_keyring(kr)
    pasted = {"v": ""}
    monkeypatch.setattr(cc.getpass, "getpass", lambda prompt="": pasted["v"])
    monkeypatch.setattr(cc.sys.stdin, "isatty", lambda: True, raising=False)
    return api, kr, pasted, tmp_path


def add(env, name="default", org="org-1", key=None, *extra):
    api, kr, pasted, _ = env
    key = key or fake_key()
    api.orgs[key] = org
    pasted["v"] = key
    return cc.main(["add", name, *extra]), key


def registry_text(tmp) -> str:
    return (tmp / "cfg" / "registry.json").read_text()


def test_add_puts_the_key_in_the_keychain_and_only_metadata_in_the_registry(env, capsys):
    api, kr, _, tmp = env
    rc, key = add(env)
    assert rc == 0 and kr.items[("claude-credits", "default")] == key
    text = registry_text(tmp)
    assert key not in text and key[11:31] not in text
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(tmp / "cfg" / "registry.json").st_mode) == 0o600
    out = capsys.readouterr()
    assert key not in out.out + out.err
    e = json.loads(text)["keys"]["default"]
    assert (e["status"], e["org_id"], e["store"], e["kind"]) == ("ok", "org-1", "keyring", "usr")


@settings(max_examples=25, deadline=None)
@given(keys)
def test_no_key_text_reaches_the_registry(tmp_path_factory, key):
    tmp = tmp_path_factory.mktemp("prop")
    api = FakeAPI()
    api.orgs[key] = "org-p"
    keyring.set_keyring(MemoryKeyring())
    old_home, old_http = os.environ.get("CLAUDE_CREDITS_HOME"), cc.http
    os.environ["CLAUDE_CREDITS_HOME"], cc.http = str(tmp), api.http
    try:
        cc.STORES["keyring"].put("default", key)
        with cc.registry_txn() as reg:
            reg["keys"]["default"] = {"store": "keyring", "fingerprint": cc.fingerprint(key)}
        assert cc.main(["check"]) == 0
        text = (tmp / "registry.json").read_text()
        body = key[len("sk-ant-usr-"):]
        assert all(body[i:i + 12] not in text for i in range(len(body) - 11))
    finally:
        cc.http = old_http
        if old_home is None:
            os.environ.pop("CLAUDE_CREDITS_HOME", None)
        else:
            os.environ["CLAUDE_CREDITS_HOME"] = old_home


def test_add_reads_stdin_too(env, monkeypatch):
    api, kr, _, _ = env
    key = fake_key()
    api.orgs[key] = "org-s"
    import io
    monkeypatch.setattr(cc.sys, "stdin", io.StringIO(key + "\n"))
    assert cc.main(["add", "ci", "--stdin"]) == 0 and kr.items[("claude-credits", "ci")] == key


def test_add_refuses_a_rejected_key_and_stores_nothing(env, capsys):
    api, kr, pasted, tmp = env
    pasted["v"] = fake_key()
    assert cc.main(["add"]) == 2
    assert kr.items == {} and not (tmp / "cfg" / "registry.json").exists()
    assert "rejected" in capsys.readouterr().err


@pytest.mark.parametrize("bad, why", [
    ("sk-ant-oat01-" + "a" * 90, "setup-token"),
    ("sk-ant-admin01-" + "a" * 90, "Admin API key"),
    ("hello", "doesn't look like"),
])
def test_add_refuses_tokens_that_are_not_api_keys(env, capsys, bad, why):
    api, kr, pasted, _ = env
    pasted["v"] = bad
    assert cc.main(["add"]) == 2 and kr.items == {}
    assert why in capsys.readouterr().err


def test_add_refuses_legacy_keys_without_force(env):
    rc, _ = add(env, "old", "org-l", fake_key("api03"))
    assert rc == 2
    rc, _ = add(env, "old", "org-l", fake_key("api03"), "--force")  # first attempt stored nothing
    assert rc == 0


def test_add_refuses_a_second_key_in_the_same_org(env, capsys):
    assert add(env, "a", "org-1")[0] == 0
    rc, _ = add(env, "b", "org-1")
    assert rc == 2 and "already stored as 'a'" in capsys.readouterr().err


def test_add_refuses_the_same_key_twice(env):
    rc, key = add(env, "a", "org-1")
    assert add(env, "b", "org-1", key, "--force")[0] == 2


def test_add_offline_stores_without_calling_the_api(env):
    api, kr, _, tmp = env
    rc, key = add(env, "air", "org-x", None, "--offline")
    assert rc == 0 and api.calls == [] and kr.items[("claude-credits", "air")] == key
    assert json.loads(registry_text(tmp))["keys"]["air"]["status"] == "unknown"


def test_bad_names_are_refused(env):
    assert add(env, "../etc")[0] == 2


def test_file_store_is_0600_and_holds_the_key(env):
    api, kr, _, tmp = env
    rc, key = add(env, "srv", "org-f", None, "--file-store")
    assert rc == 0 and kr.items == {}
    keys_file = tmp / "cfg" / "keys.json"
    assert json.loads(keys_file.read_text()) == {"srv": key}
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(keys_file).st_mode) == 0o600
        os.chmod(keys_file, 0o644)
        assert cc.main(["check", "srv"]) == 2  # refuses a key file others can read


def test_moving_a_key_between_stores_leaves_no_copy(env):
    api, kr, _, tmp = env
    rc, key = add(env, "k", "org-m", None, "--file-store")
    assert add(env, "k", "org-m", key, "--replace")[0] == 0          # same name, now keyring
    assert kr.items[("claude-credits", "k")] == key
    assert json.loads((tmp / "cfg" / "keys.json").read_text()) == {}


def test_check_records_exhaustion_until_renewal(env):
    api, *_ , tmp = env
    _, key = add(env, "default", "org-1", None, "--renews-day", "17")
    api.exhausted.add(key)
    assert cc.main(["check"]) == 1
    e = json.loads(registry_text(tmp))["keys"]["default"]
    assert e["status"] == "exhausted" and dt.date.fromisoformat(e["status_until"]).day == 17


def test_pick_verify_skips_an_exhausted_key(env, capsys):
    api = env[0]
    # the exhausted key expires first (Oct 25 vs Oct 28), so it is tested and must be skipped
    _, k1 = add(env, "soon", "org-1", None, "--renews-day", "25")
    _, k2 = add(env, "later", "org-2", None, "--renews-day", "28")
    api.exhausted.add(k1)
    capsys.readouterr()
    assert cc.main(["pick", "--verify"]) == 0
    assert capsys.readouterr().out.strip() == "later"


def test_pick_refuses_when_nothing_is_usable(env):
    _, k1 = add(env)
    env[0].exhausted.add(k1)
    assert cc.main(["pick", "--verify"]) == 2


def test_exec_passes_the_key_only_through_the_environment(env, monkeypatch, capsys):
    _, key = add(env)
    seen = {}
    monkeypatch.setattr(cc.os, "execve", lambda f, a, e: seen.update(argv=a, env=e))
    monkeypatch.setattr(cc.subprocess, "call", lambda a, env: seen.update(argv=a, env=env) or 0)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-should-go")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "should-go-too")
    capsys.readouterr()
    cc.main(["exec", "--", "python3", "job.py"])
    assert os.path.basename(seen["argv"][0]).startswith("python3") and seen["argv"][1:] == ["job.py"]
    assert seen["env"]["ANTHROPIC_API_KEY"] == key and seen["env"]["CLAUDE_CREDITS_NAME"] == "default"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in seen["env"] and "ANTHROPIC_AUTH_TOKEN" not in seen["env"]
    out = capsys.readouterr()
    assert key not in out.out + out.err


def test_exec_warns_before_running_claude(env, monkeypatch, capsys):
    add(env)
    monkeypatch.setattr(cc.os, "execve", lambda f, a, e: None)
    monkeypatch.setattr(cc.subprocess, "call", lambda a, env: 0)
    cc.main(["exec", "--", "claude", "-p", "hi"])
    assert "refused every `claude -p`" in capsys.readouterr().err


def test_exec_without_a_command_refuses(env):
    assert cc.main(["exec", "--"]) == 2


def test_remove_needs_yes_and_deletes_from_the_store(env):
    api, kr, _, _ = env
    add(env)
    assert cc.main(["remove", "default"]) == 2 and kr.items
    assert cc.main(["remove", "default", "--yes"]) == 0 and kr.items == {}
    assert cc.main(["remove", "default", "--yes"]) == 2


def test_list_shows_no_secrets(env, capsys):
    _, key = add(env, "default", "org-1", None, "--renews-day", "17")
    capsys.readouterr()
    assert cc.main(["list"]) == 0 and cc.main(["list", "--json"]) == 0
    out = capsys.readouterr().out
    assert key not in out and "default" in out


def test_probe_reads_rate_limits_and_names_the_org(env):
    api = env[0]
    key = fake_key()
    api.orgs[key] = "org-7"
    r = cc.probe(key)
    assert r["status"] == "ok" and r["org_id"] == "org-7" and r["rate_limits"] == {"requests-limit": "1000"}
    assert cc.probe(key, spend=False)["status"] == "unknown"


# ---------- your real OS keychain (opt-in) ----------

@pytest.mark.skipif(os.environ.get("CLAUDE_CREDITS_REAL_KEYRING") != "1",
                    reason="set CLAUDE_CREDITS_REAL_KEYRING=1 to touch the real OS keychain")
def test_real_keyring_round_trip(monkeypatch):
    keyring.core.init_backend()  # back to the platform's own keychain
    assert not isinstance(keyring.get_keyring(), MemoryKeyring)
    monkeypatch.setattr(cc, "KEYRING_SERVICE", f"claude-credits-test-{secrets.token_hex(4)}")
    store, dummy = cc.KeyringStore(), fake_key()
    try:
        store.put("t", dummy)
        assert store.get("t") == dummy
    finally:
        assert store.delete("t") is True
    assert store.get("t") is None and store.delete("t") is False


# ---------- regressions from the 2026-10-22 review ----------

def reg_key(tmp, name="default"):
    return json.loads(registry_text(tmp))["keys"][name]


def test_a_second_add_under_the_same_name_needs_replace(env, capsys):
    api, kr, _, tmp = env
    _, k1 = add(env, "default", "org-a")
    rc, k2 = add(env, "default", "org-b")
    assert rc == 2 and kr.items[("claude-credits", "default")] == k1
    assert "--replace" in capsys.readouterr().err
    rc, k2 = add(env, "default", "org-b", k2, "--replace")
    assert rc == 0 and kr.items[("claude-credits", "default")] == k2
    assert reg_key(tmp)["org_id"] == "org-b" and "replaced sha256:" in capsys.readouterr().out


def test_exhaustion_seen_on_the_renewal_day_is_rechecked_the_next_day(env, monkeypatch):
    api, kr, _, tmp = env
    _, key = add(env, "default", "org-1", None, "--renews-day", "17")
    api.exhausted.add(key)
    monkeypatch.setattr(cc, "today", lambda: dt.date(2026, 11, 17))
    assert cc.main(["check"]) == 1
    assert reg_key(tmp)["status_until"] == "2026-11-18"
    api.exhausted.discard(key)
    monkeypatch.setattr(cc, "today", lambda: dt.date(2026, 11, 18))
    assert cc.main(["pick", "--verify"]) == 0


def test_exhaustion_without_a_renewal_day_is_rechecked_daily_and_set_recomputes(env):
    api, kr, _, tmp = env
    _, key = add(env)
    api.exhausted.add(key)
    cc.main(["check"])
    assert reg_key(tmp)["status_until"] == "2026-10-23"
    cc.main(["set", "default", "--renews-day", "1"])
    assert reg_key(tmp)["status_until"] == "2026-11-01"


def test_choose_retests_parked_keys_before_giving_up(env):
    api, *_ = env
    _, key = add(env, "default", "org-1", None, "--renews-day", "17")
    api.exhausted.add(key)
    cc.main(["check"])
    api.exhausted.discard(key)
    assert cc.main(["pick", "--verify"]) == 0


def test_mark_exhausted_refuses_while_the_key_works(env, capsys):
    api, kr, _, tmp = env
    _, key = add(env)
    assert cc.main(["mark", "default", "exhausted"]) == 2 and "isn't used up" in capsys.readouterr().err
    api.exhausted.add(key)
    assert cc.main(["mark", "default", "exhausted"]) == 0 and reg_key(tmp)["status"] == "exhausted"
    assert cc.main(["mark", "default", "ok"]) == 0
    n = len(api.calls)
    assert cc.main(["mark", "default", "exhausted", "--force"]) == 0 and len(api.calls) == n


@pytest.mark.parametrize("argv", [["add", "--renews-day", "32"], ["add", "--renews-day", "0"],
                                  ["set", "default", "--renews-day", "x"],
                                  ["mark", "default", "exhausted", "--until", "soon"]])
def test_bad_dates_and_days_are_refused_before_anything_happens(env, argv):
    api, kr, pasted, _ = env
    pasted["v"] = fake_key()
    with pytest.raises(SystemExit):
        cc.main(argv)
    assert kr.items == {} and api.calls == []


def test_a_free_check_keeps_a_recorded_exhaustion(env):
    api, kr, _, tmp = env
    _, key = add(env, "default", "org-1", None, "--renews-day", "17")
    api.exhausted.add(key)
    cc.main(["check"])
    cc.main(["check", "--free"])
    assert (reg_key(tmp)["status"], reg_key(tmp)["status_until"]) == ("exhausted", "2026-11-17")


def test_soonest_expiry_is_tried_first_even_when_unchecked(env, tmp_path):
    add(env, "ok-later", "org-1", None, "--renews-day", "12")
    add(env, "unknown-soon", "org-2", None, "--renews-day", "24", "--no-spend-probe")
    assert cc.order(json.loads(registry_text(env[3]))["keys"], REF)[0] == "unknown-soon"


def test_least_recently_used_breaks_ties():
    ks = {"a": {"status": "ok", "renews_day": 5, "last_used": "2026-10-21T00:00:00Z"},
          "b": {"status": "ok", "renews_day": 5, "last_used": "2026-10-01T00:00:00Z"}}
    assert cc.order(ks, REF) == ["b", "a"]


def test_add_refuses_when_the_api_cannot_be_reached_unless_offline(env, monkeypatch, capsys):
    api, kr, pasted, _ = env
    monkeypatch.setattr(cc, "http", lambda *a, **k: (0, {}, "network error: URLError"))
    pasted["v"] = fake_key()
    assert cc.main(["add"]) == 2 and kr.items == {} and "couldn't check" in capsys.readouterr().err
    assert cc.main(["add", "--offline"]) == 0 and kr.items


def test_stdin_on_a_terminal_uses_the_hidden_prompt(env, monkeypatch):
    api, kr, pasted, _ = env
    key = fake_key()
    api.orgs[key] = "org-t"
    pasted["v"] = key
    import io

    class TTY(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr(cc.sys, "stdin", TTY("this would echo\n"))
    assert cc.main(["add", "--stdin"]) == 0 and kr.items[("claude-credits", "default")] == key


def test_a_stored_value_that_isnt_a_key_is_never_sent(env, capsys):
    api, kr, _, _ = env
    _, key = add(env)
    kr.items[("claude-credits", "default")] = key + "\n"
    n = len(api.calls)
    assert cc.main(["check"]) == 2 and len(api.calls) == n
    assert key not in capsys.readouterr().err


def test_an_unreachable_keychain_gives_a_clean_refusal_and_remove_force_forgets(env, capsys):
    api, kr, _, tmp = env
    add(env)

    class Locked(MemoryKeyring):
        def get_password(self, s, u):
            raise keyring.errors.KeyringLocked("locked")

        def delete_password(self, s, u):
            raise keyring.errors.KeyringLocked("locked")
    keyring.set_keyring(Locked())
    assert cc.main(["check"]) == 2 and "keychain refused" in capsys.readouterr().err
    assert cc.main(["remove", "default", "--yes"]) == 2
    assert cc.main(["remove", "default", "--yes", "--force"]) == 0
    assert json.loads(registry_text(tmp))["keys"] == {}


def test_relative_xdg_config_home_is_ignored(monkeypatch):
    monkeypatch.delenv("CLAUDE_CREDITS_HOME", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    assert os.path.isabs(cc.config_dir())
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/dir")
    assert os.path.isabs(cc.config_dir())


def test_file_store_temp_files_are_private_and_cleaned_up(env):
    api, kr, _, tmp = env
    add(env, "srv", "org-f", None, "--file-store")
    cfg = tmp / "cfg"
    assert not [p for p in os.listdir(cfg) if p.endswith(".tmp")]
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(cfg / "keys.json").st_mode) == 0o600


def test_exec_refuses_a_missing_command_before_touching_the_key(env, capsys):
    api, *_ = env
    add(env)
    n = len(api.calls)
    assert cc.main(["exec", "--", "no-such-program-xyz"]) == 2
    assert "command not found" in capsys.readouterr().err and len(api.calls) == n


def test_examples_are_copied_into_a_folder(env, tmp_path):
    dest = tmp_path / "ex"
    assert cc.main(["examples", str(dest)]) == 0
    assert sorted(os.listdir(dest)) == ["batch.py", "hello.py"]
    (dest / "hello.py").write_text("mine")
    assert cc.main(["examples", str(dest)]) == 0 and (dest / "hello.py").read_text() == "mine"


def test_http_refuses_a_base_url_other_than_anthropic(monkeypatch):
    monkeypatch.setattr(cc, "API_BASE", "http://evil.example.com")
    with pytest.raises(cc.Refused, match="only go to"):
        cc.http("GET", "/v1/models", "k")


def test_http_never_follows_a_redirect_with_the_key(monkeypatch):
    import http.server
    import threading
    seen = []

    class Target(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("x-api-key"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    target = http.server.HTTPServer(("127.0.0.1", 0), Target)

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{target.server_port}/stolen")
            self.end_headers()

        def log_message(self, *a):
            pass

    redirector = http.server.HTTPServer(("127.0.0.1", 0), Redirector)
    for srv in (target, redirector):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(cc, "API_BASE", f"http://127.0.0.1:{redirector.server_port}")
        status, _, _ = cc.http("GET", "/v1/models", "sk-ant-usr-secret")
        assert status == 302 and seen == []
    finally:
        target.shutdown()
        redirector.shutdown()


# ---------- regressions from the Sol review of the shared logic ----------

def test_concurrent_adds_cannot_both_claim_one_organization(env, monkeypatch):
    import threading
    import time as _time
    api, kr, pasted, tmp = env
    k1, k2 = fake_key(), fake_key()
    api.orgs.update({k1: "org-same", k2: "org-same"})
    real_put = cc.STORES["keyring"].put
    monkeypatch.setattr(cc.STORES["keyring"], "put", lambda n, v: (_time.sleep(0.3), real_put(n, v)))
    values = {"a": k1, "b": k2}
    monkeypatch.setattr(cc, "read_key", lambda args: values[args.name])
    rcs = {}
    threads = [threading.Thread(target=lambda n=n: rcs.__setitem__(n, cc.main(["add", n]))) for n in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(rcs.values()) == [0, 2]
    assert len(json.loads(registry_text(tmp))["keys"]) == 1


def test_a_parked_key_is_not_revived_by_an_inconclusive_test(env, monkeypatch):
    api, *_ = env
    _, key = add(env, "default", "org-1", None, "--renews-day", "17")
    api.exhausted.add(key)
    cc.main(["check"])
    monkeypatch.setattr(cc, "http", lambda *a, **k: (0, {}, "network error: URLError"))
    assert cc.main(["pick", "--verify"]) == 2


def test_exec_name_is_verified_unless_no_verify(env, monkeypatch):
    api, *_ = env
    _, key = add(env)
    api.exhausted.add(key)
    ran = {}
    monkeypatch.setattr(cc.os, "execve", lambda f, a, e: ran.update(argv=a))
    assert cc.main(["exec", "--name", "default", "--", "python3", "-c", "0"]) == 2 and not ran
    cc.main(["exec", "--name", "default", "--no-verify", "--", "python3", "-c", "0"])
    assert ran["argv"][1:] == ["-c", "0"]


def test_unrelated_checks_do_not_move_an_exhaustion_into_the_next_cycle(env, monkeypatch):
    api, kr, _, tmp = env
    _, key = add(env, "default", "org-1", None, "--renews-day", "17")
    api.exhausted.add(key)
    cc.main(["check"])
    monkeypatch.setattr(cc, "today", lambda: dt.date(2026, 11, 20))
    cc.main(["check", "--free"])
    cc.main(["set", "default", "--renews-day", "18"])
    assert reg_key(tmp)["status_until"] == "2026-11-18"


# ---------- regressions from Sol round 2 ----------

def test_a_test_result_for_a_replaced_key_is_discarded(env, monkeypatch):
    api, kr, _, tmp = env
    _, k1 = add(env)
    api.exhausted.add(k1)
    real_http = cc.http

    def swap_mid_test(method, path, key, body=None):
        out = real_http(method, path, key, body)
        if method == "POST" and key == k1:
            with cc.registry_txn() as reg:
                reg["keys"]["default"].update(fingerprint="sha256:new", status="ok", org_id="org-2")
        return out
    monkeypatch.setattr(cc, "http", swap_mid_test)
    cc.main(["check"])
    assert (reg_key(tmp)["status"], reg_key(tmp)["org_id"]) == ("ok", "org-2")


def test_exec_name_refuses_a_recorded_exhaustion_on_an_inconclusive_test(env, monkeypatch):
    api, *_ = env
    _, key = add(env, "default", "org-1", None, "--renews-day", "17")
    api.exhausted.add(key)
    cc.main(["check"])
    monkeypatch.setattr(cc, "http", lambda *a, **k: (0, {}, "network error: URLError"))
    assert cc.main(["exec", "--name", "default", "--", "python3", "-c", "0"]) == 2


def test_readding_with_a_new_renewal_day_recomputes_the_exhaustion(env):
    api, kr, pasted, tmp = env
    _, key = add(env, "default", "org-1", None, "--renews-day", "17")
    api.exhausted.add(key)
    cc.main(["check"])
    pasted["v"] = key
    assert cc.main(["add", "--replace", "--renews-day", "18", "--no-spend-probe"]) == 0
    assert (reg_key(tmp)["status"], reg_key(tmp)["status_until"]) == ("exhausted", "2026-11-18")


def test_a_legacy_exhaustion_record_is_not_stretched_by_a_later_check(env, monkeypatch):
    api, kr, _, tmp = env
    add(env, "default", "org-1", None, "--renews-day", "17")
    with cc.registry_txn() as reg:
        reg["keys"]["default"].update(status="exhausted", status_until="2026-11-17",
                                      last_check="2026-10-22T12:00:00Z")
        reg["keys"]["default"].pop("exhausted_seen", None)
    monkeypatch.setattr(cc, "today", lambda: dt.date(2026, 11, 20))
    cc.main(["check", "--free"])
    cc.main(["set", "default", "--renews-day", "18"])
    assert reg_key(tmp)["status_until"] == "2026-11-18"
