#!/usr/bin/env python3
"""claude-credits: keep your Claude Max plan's monthly API credit key safe, and run jobs on it.

Max 20x plans include $200 of Claude API credit each billing cycle, and Max 5x plans $100
(platform.claude.com/docs/en/about-claude/api-credits-for-subscribers). You claim it by
linking a Console organization to your plan; every API key in that organization draws on
the credit. Unused credit expires when the cycle ends, with no rollover. When it runs out,
requests fail with "credit balance is too low" and nothing is charged, unless the
organization also holds purchased credit or has auto-reload on.

What it pays for, tested 2026-10-22: your own Messages and Batch API calls, and Managed
Agents sessions. It refused every request carrying the Claude Code or Agent SDK system
prompt, so `claude -p` and the Agent SDK fail, although Anthropic's help center says both
are covered when run with a key from the linked organization.

Keys live in your operating system's keychain (macOS Keychain, Windows Credential Locker,
Linux Secret Service) through the `keyring` library, or, on a headless machine and only if
you ask, in a file only you can read. A registry next to it holds what isn't secret:
organization id, key fingerprint, status, renewal day. `exec` hands the key to one command
through its environment; it never prints it or puts it on a command line, and it only
ever sends it to https://api.anthropic.com.
"""
from __future__ import annotations

import argparse
import calendar
import contextlib
import datetime as dt
import getpass
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

__version__ = "0.2.0"


def config_dir() -> str:
    """Where the registry (and the file store, if used) live. Empty or relative
    XDG_CONFIG_HOME / APPDATA values are ignored, so keys never land in the current folder."""
    if os.environ.get("CLAUDE_CREDITS_HOME"):
        return os.environ["CLAUDE_CREDITS_HOME"]
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", "")
        return os.path.join(base if os.path.isabs(base) else os.path.expanduser("~"), "claude-credits")
    base = os.environ.get("XDG_CONFIG_HOME", "")
    return os.path.join(base if os.path.isabs(base) else os.path.expanduser("~/.config"), "claude-credits")


API_BASE = os.environ.get("CLAUDE_CREDITS_API_BASE", "https://api.anthropic.com")
ALLOWED_API_BASES = re.compile(r"https://api\.anthropic\.com|https?://(127\.0\.0\.1|localhost)(:\d+)?")
PROBE_MODEL = os.environ.get("CLAUDE_CREDITS_PROBE_MODEL", "claude-haiku-5-5")
KEYRING_SERVICE = "claude-credits"
TIMEOUT = 60
DEFAULT_NAME = "default"

# Console keys: personal (usr), service-account (svc) and legacy (api03).
KEY_RE = re.compile(r"sk-ant-(?P<kind>usr|svc|api)(?P<ver>\d{2})?-[A-Za-z0-9_-]{40,}")
NOT_A_CONSOLE_KEY = {
    "oat": "a subscription OAuth token (from `claude setup-token`), not an API key. Create an API key "
           "at platform.claude.com > API keys.",
    "ort": "an OAuth refresh token, not an API key.",
    "admin": "an Admin API key. It manages the organization but can't send messages.",
}
NAME_RE = re.compile(r"[A-Za-z0-9._@+-]{1,64}")
EXHAUSTED_MARK = "credit balance is too low"
STRIP_ENV = ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
             "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")
EXAMPLES = ("hello.py", "batch.py")


class Refused(Exception):
    """A check failed: say why, change nothing."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today() -> dt.date:
    return dt.datetime.now().date()


def renewal_day(text: str) -> int:
    """argparse type for --renews-day."""
    try:
        day = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a day of the month")
    if not 1 <= day <= 31:
        raise argparse.ArgumentTypeError("the renewal day must be 1-31")
    return day


def iso_date(text: str) -> str:
    """argparse type for --until."""
    try:
        return dt.date.fromisoformat(text).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a date like 2026-11-17")


# ---------- keys ----------

def classify(value: str) -> str:
    """Kind of Console key (usr, svc, api), or raise Refused naming what it is instead."""
    m = KEY_RE.fullmatch(value)
    if m:
        return m["kind"]
    for kind, why in NOT_A_CONSOLE_KEY.items():
        if value.startswith(f"sk-ant-{kind}"):
            raise Refused(f"that is {why}")
    raise Refused("that doesn't look like a Claude Console API key (expected sk-ant-usr-…)")


def fingerprint(value: str) -> str:
    """Short sha256 of a high-entropy key: enough to tell keys apart, useless for guessing one."""
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()[:16]


# ---------- secret stores ----------

class KeyringStore:
    """The OS keychain through `keyring`."""
    kind = "keyring"

    def _kr(self):
        try:
            import keyring
            from keyring.backends import fail
        except ImportError:
            raise Refused("the `keyring` package isn't installed. Install claude-credits with "
                          "`uv tool install …` or `pipx install …`, which bring it along.")
        if isinstance(keyring.get_keyring(), fail.Keyring):
            raise Refused("no OS keychain is available here (common on headless Linux). Use "
                          "`add --file-store` to keep the key in a file only you can read.")
        return keyring

    @contextlib.contextmanager
    def _errors(self):
        try:
            yield
        except Refused:
            raise
        except Exception as e:  # keyring.errors.KeyringError and backend errors (locked, denied)
            raise Refused(f"the OS keychain refused ({type(e).__name__}). Unlock it or allow access, "
                          "then rerun.")

    def get(self, name: str) -> str | None:
        with self._errors():
            return self._kr().get_password(KEYRING_SERVICE, name)

    def put(self, name: str, value: str) -> None:
        with self._errors():
            kr = self._kr()
            kr.set_password(KEYRING_SERVICE, name, value)
            if kr.get_password(KEYRING_SERVICE, name) != value:
                raise Refused("the keychain didn't return the key as written")

    def delete(self, name: str) -> bool:
        with self._errors():
            kr = self._kr()
            from keyring.errors import PasswordDeleteError
            try:
                kr.delete_password(KEYRING_SERVICE, name)
                return True
            except PasswordDeleteError:
                return False


class FileStore:
    """A JSON file only you can read, in the config folder, for machines with no keychain.
    Writes go through a fresh private temp file (mkstemp) under the registry lock. On
    Windows the file's protection comes from your profile folder's permissions."""
    kind = "file"

    def path(self) -> str:
        return os.path.join(config_dir(), "keys.json")

    def _load(self) -> dict:
        p = self.path()
        if not os.path.exists(p):
            return {}
        if sys.platform != "win32" and os.stat(p).st_mode & 0o077:
            raise Refused(f"{p} is readable by other users; run: chmod 600 {p}")
        with open(p) as fh:
            return json.load(fh)

    def _save(self, data: dict) -> None:
        os.makedirs(config_dir(), mode=0o700, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=config_dir(), prefix=".keys-", suffix=".tmp")  # 0600, O_EXCL
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, self.path())
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def get(self, name: str) -> str | None:
        return self._load().get(name)

    def put(self, name: str, value: str) -> None:
        with _locked(os.path.join(config_dir(), "keys.lock")):
            data = self._load()
            data[name] = value
            self._save(data)

    def delete(self, name: str) -> bool:
        with _locked(os.path.join(config_dir(), "keys.lock")):
            data = self._load()
            if name not in data:
                return False
            del data[name]
            self._save(data)
            return True


STORES = {"keyring": KeyringStore(), "file": FileStore()}


def store_for(entry: dict):
    return STORES[entry.get("store", "keyring")]


def stored_key(name: str, entry: dict) -> str | None:
    """The key for `name` from its store, re-checked: a value edited outside the tool
    (a stray newline, say) is refused here rather than sent anywhere."""
    value = store_for(entry).get(name)
    if value is None:
        return None
    try:
        classify(value)
    except Refused:
        raise Refused(f"the value stored for {name!r} isn't an API key; add it again")
    return value


# ---------- registry (no secrets) ----------

@contextlib.contextmanager
def _locked(path: str):
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    with open(path, "a") as fh:
        if sys.platform == "win32":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


@contextlib.contextmanager
def registry_txn(write: bool = True):
    """Load the registry under its lock; save atomically (0600) if `write` and it changed."""
    path = os.path.join(config_dir(), "registry.json")
    with _locked(path + ".lock"):
        if os.path.exists(path):
            with open(path) as fh:
                reg = json.load(fh)
        else:
            reg = {"version": 1, "keys": {}}
        before = json.dumps(reg, sort_keys=True)
        yield reg
        if write and json.dumps(reg, sort_keys=True) != before:
            fd, tmp = tempfile.mkstemp(dir=config_dir(), prefix=".registry-", suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(reg, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, path)


def entry(name: str) -> dict:
    with registry_txn(write=False) as reg:
        if name not in reg["keys"]:
            raise Refused(f"no key named {name!r}. Run: claude-credits list")
        return dict(reg["keys"][name])


# ---------- billing cycle ----------

def renewal_on_or_after(day: int, ref: dt.date) -> dt.date:
    """The first renewal date on or after `ref` for a plan that renews on `day` (1-31);
    short months renew on their last day."""
    if not 1 <= day <= 31:
        raise ValueError("renewal day must be 1-31")
    y, m = ref.year, ref.month
    for _ in range(2):
        d = dt.date(y, m, min(day, calendar.monthrange(y, m)[1]))
        if d >= ref:
            return d
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    raise AssertionError("unreachable: some renewal falls within two months")


def days_left(e: dict, ref: dt.date) -> int | None:
    """Days until this key's credit expires (its plan's next renewal), or None if unknown."""
    day = e.get("renews_day")
    if not day:
        return None
    nxt = renewal_on_or_after(day, ref)
    if nxt == ref:  # credit refreshes on the renewal day itself
        nxt = renewal_on_or_after(day, ref + dt.timedelta(days=1))
    return (nxt - ref).days


def exhausted_until(e: dict, seen: dt.date) -> str:
    """How long an exhaustion seen on `seen` holds: until the plan's next renewal, when new
    credit arrives. On the renewal day itself the new credit may not have landed yet (the
    time of day isn't documented), and with no renewal day known there's no date to wait
    for, so both re-check the next day."""
    day = e.get("renews_day")
    if day:
        nxt = renewal_on_or_after(day, seen)
        if nxt > seen:
            return nxt.isoformat()
    return (seen + dt.timedelta(days=1)).isoformat()


def effective_status(e: dict, ref: dt.date) -> str:
    """ok | exhausted | invalid | unknown. An exhaustion whose `until` has passed reads unknown."""
    st = e.get("status", "unknown")
    if st == "exhausted":
        until = e.get("status_until")
        if not until or dt.date.fromisoformat(until) <= ref:
            return "unknown"
    return st


def order(keys: dict, ref: dt.date) -> list[str]:
    """Spending order over usable keys (ok or unknown) when you have more than one: soonest
    expiry first, unknown renewal last, because credit that expires unspent is lost; then
    keys known to work; then least recently used."""
    rank = {"ok": 0, "unknown": 1}
    usable = [n for n, e in keys.items() if effective_status(e, ref) in rank]

    def key(n: str):
        e = keys[n]
        dl = days_left(e, ref)
        return (dl is None, dl or 0, rank[effective_status(e, ref)], e.get("last_used") or "", n)
    return sorted(usable, key=key)


def set_status(e: dict, status: str, ref: dt.date) -> None:
    e["status"] = status
    if status == "exhausted":
        e["status_until"] = exhausted_until(e, ref)
    else:
        e.pop("status_until", None)


def renewal_changed(e: dict) -> None:
    """Recompute a recorded exhaustion's end after the renewal day changes."""
    if e.get("status") == "exhausted" and e.get("last_check"):
        e["status_until"] = exhausted_until(e, dt.date.fromisoformat(e["last_check"][:10]))


# ---------- API probes ----------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would resend x-api-key to whatever host it names."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def http(method: str, path: str, key: str, body: dict | None = None) -> tuple[int, dict, dict | str]:
    if not ALLOWED_API_BASES.fullmatch(API_BASE):
        raise Refused(f"CLAUDE_CREDITS_API_BASE={API_BASE!r}: keys only go to https://api.anthropic.com "
                      "(or localhost, for tests)")
    try:
        req = urllib.request.Request(
            API_BASE + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json", "user-agent": f"claude-credits/{__version__}"})
        with _OPENER.open(req, timeout=TIMEOUT) as r:
            status, headers, raw = r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        status, headers, raw = e.code, dict(e.headers or {}), e.read()
    except ValueError:  # its message would quote the offending header, i.e. the key
        return 0, {}, "the key contains characters that can't go in a header"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, {}, f"network error: {type(e).__name__}"
    headers = {k.lower(): v for k, v in headers.items()}
    try:
        return status, headers, json.loads(raw or b"{}")
    except ValueError:
        return status, headers, raw.decode("utf-8", "replace")[:300]


def error_text(payload) -> str:
    if isinstance(payload, dict):
        err = payload.get("error") or {}
        return f"{err.get('type', '')}: {err.get('message', '')}".strip(": ")
    return str(payload)


def probe(key: str, spend: bool = True) -> dict:
    """Ask the API about a key. `GET /v1/models` is free and names the organization; with
    `spend`, a 1-token Haiku message (about $0.00001) also tells whether credit remains
    and returns the organization's rate limits. `verified` means the API accepted the key."""
    out: dict = {"checked": now(), "verified": False}
    status, headers, payload = http("GET", "/v1/models?limit=100", key)
    out["org_id"] = headers.get("anthropic-organization-id")
    if status == 401:
        return {**out, "status": "invalid", "detail": error_text(payload)}
    if status != 200:
        return {**out, "status": "unknown", "detail": f"models HTTP {status}: {error_text(payload)}"}
    out["verified"] = True
    models = [m.get("id", "") for m in payload.get("data", [])] if isinstance(payload, dict) else []
    if not spend:
        return {**out, "status": "unknown", "detail": "key accepted (no spend probe)"}
    model = PROBE_MODEL if PROBE_MODEL in models or not models else next(
        (m for m in models if "haiku" in m), models[-1])
    status, headers, payload = http("POST", "/v1/messages", key, {
        "model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "ok"}]})
    out["org_id"] = out["org_id"] or headers.get("anthropic-organization-id")
    limits = {k[len("anthropic-ratelimit-"):]: v for k, v in headers.items()
              if k.startswith("anthropic-ratelimit-") and k.endswith("-limit")}
    if limits:
        out["rate_limits"] = limits
    if status == 200:
        return {**out, "status": "ok", "detail": f"{model} answered"}
    text = error_text(payload)
    if EXHAUSTED_MARK in text.lower():
        return {**out, "status": "exhausted", "detail": text}
    if status == 401:
        return {**out, "status": "invalid", "detail": text}
    if status == 429:
        return {**out, "status": "ok", "detail": f"rate limited right now ({text})"}
    return {**out, "status": "unknown", "detail": f"messages HTTP {status}: {text}"}


def apply_probe(e: dict, result: dict, ref: dt.date) -> None:
    e["last_check"] = result["checked"]
    e["last_check_detail"] = result.get("detail", "")
    if result.get("org_id"):
        e["org_id"] = result["org_id"]
    if result.get("rate_limits"):
        e["rate_limits"] = result["rate_limits"]
    if result["status"] == "unknown" and e.get("status") in ("ok", "exhausted", "invalid"):
        return  # a free or failed check says nothing about the balance: keep what was known
    set_status(e, result["status"], ref)


def probe_stored(name: str, e: dict, spend: bool = True) -> dict:
    key = stored_key(name, e)
    if key is None:
        return {"checked": now(), "status": "invalid", "detail": "missing from the store", "verified": False}
    return probe(key, spend=spend)


# ---------- reading the key ----------

def read_clipboard() -> str:
    for cmd in (["pbpaste"], ["wl-paste", "--no-newline"], ["xclip", "-o", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--output"],
                ["powershell", "-NoProfile", "-Command", "Get-Clipboard"]):
        if shutil.which(cmd[0]):
            r = subprocess.run(cmd, capture_output=True, timeout=10)
            if r.returncode == 0:
                return r.stdout.decode("utf-8", "replace").strip()
    raise Refused("couldn't read the clipboard here; run `claude-credits add` and paste at the prompt")


def clear_clipboard_if(value: str) -> bool:
    try:
        if read_clipboard() != value:
            return False
    except Refused:
        return False
    for cmd in (["pbcopy"], ["wl-copy", "--clear"], ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
                ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $null"]):
        if shutil.which(cmd[0]):
            return subprocess.run(cmd, input=b"", timeout=10).returncode == 0
    return False


def prompt_key() -> str:
    return getpass.getpass("Paste your Claude API key (sk-ant-usr-…; input hidden): ").strip()


def read_key(args) -> str:
    if args.clipboard:
        return read_clipboard()
    if args.stdin:
        if sys.stdin.isatty():  # typing into --stdin would echo the key: hide it instead
            return prompt_key()
        return sys.stdin.read().strip()
    if not sys.stdin.isatty():
        raise Refused("no terminal to prompt on; pass --stdin or --clipboard")
    return prompt_key()


# ---------- commands ----------

def cmd_add(args) -> int:
    name = args.name
    if not NAME_RE.fullmatch(name):
        raise Refused("names may use letters, digits and . _ @ + - (up to 64)")
    with registry_txn(write=False) as reg:
        previous = dict(reg["keys"].get(name, {}))
    if previous and not args.replace:
        raise Refused(f"a key named {name!r} is already stored (org {previous.get('org_id') or '?'}). "
                      f"Pass --replace to swap it, or give this one another name: claude-credits add plan2")
    value = read_key(args)
    kind = classify(value)
    if kind == "api" and not args.force:
        raise Refused("that's a legacy sk-ant-api key. The credit program issues sk-ant-usr keys. Pass --force "
                      "if you're sure it's from the organization linked to your plan.")
    fp, ref = fingerprint(value), today()
    if args.offline:
        result = {"checked": now(), "status": "unknown", "detail": "not checked (--offline)", "verified": False}
    else:
        result = probe(value, spend=not args.no_spend_probe)
        if result["status"] == "invalid":
            raise Refused(f"the API rejected this key ({result.get('detail', '')}). Nothing was stored.")
        if not result["verified"]:
            raise Refused(f"couldn't check the key with the API ({result.get('detail', '')}). Nothing was "
                          "stored. Try again, or pass --offline to store it unchecked.")
    with registry_txn(write=False) as reg:
        for other, e in reg["keys"].items():
            if other == name:
                continue
            if e.get("fingerprint") == fp:
                raise Refused(f"this key is already stored as {other!r}")
            if result.get("org_id") and e.get("org_id") == result["org_id"] and not args.force:
                raise Refused(f"this key's organization is already stored as {other!r}. Keys in one organization "
                              "share one balance, so a second one adds nothing. Pass --force to keep both.")
    store = STORES["file" if args.file_store else "keyring"]
    store.put(name, value)
    with registry_txn() as reg:
        e = reg["keys"].setdefault(name, {})
        if e.get("fingerprint") != fp:  # a new key starts with a clean status
            for k in ("status", "status_until", "last_used", "org_id", "rate_limits"):
                e.pop(k, None)
        old_store = e.get("store")
        e.update({"store": store.kind, "kind": kind, "fingerprint": fp,
                  "added": e.get("added") or now(), "updated": now()})
        if args.renews_day:
            e["renews_day"] = args.renews_day
        apply_probe(e, result, ref)
        renews = e.get("renews_day")
    if old_store and old_store != store.kind:  # moved between stores: don't leave a copy behind
        try:
            STORES[old_store].delete(name)
        except Refused as err:
            print(f"warning  couldn't delete the old copy from the {old_store} store ({err}); "
                  "remove it by hand", file=sys.stderr)
    cleared = args.clipboard and not args.keep_clipboard and clear_clipboard_if(value)
    value = ""
    where = "your OS keychain" if store.kind == "keyring" else FileStore().path()
    if previous.get("fingerprint") and previous["fingerprint"] != fp:
        print(f"replaced {previous['fingerprint']} (org {previous.get('org_id') or '?'})")
    print(f"stored   {name!r} in {where}")
    print(f"key      {kind} key, {fp}")
    print(f"org      {result.get('org_id') or 'unknown'}")
    print(f"status   {result['status']}: {result.get('detail', '')}")
    if result.get("rate_limits"):
        print("limits   " + ", ".join(f"{k} {v}" for k, v in sorted(result["rate_limits"].items())))
    if cleared:
        print("clipboard cleared (clipboard history apps may still hold it)")
    print("note     this key spends whatever organization it came from. Make sure that's the one you\n"
          "         linked to your Max plan, not an employer's or one with a card on file.")
    if not renews:
        print("next     tell it the day your plan renews, so it knows when this cycle's credit expires:\n"
              f"         claude-credits set {name} --renews-day N   (see claude.ai > Settings > Billing)")
    return 0


def describe(name: str, e: dict, ref: dt.date) -> dict:
    return {"name": name, "status": effective_status(e, ref), "store": e.get("store", "keyring"),
            "kind": e.get("kind"), "org_id": e.get("org_id"), "renews_day": e.get("renews_day"),
            "days_left": days_left(e, ref), "last_check": e.get("last_check"), "last_used": e.get("last_used"),
            "detail": e.get("last_check_detail", ""), "fingerprint": e.get("fingerprint"),
            "rate_limits": e.get("rate_limits", {})}


def cmd_list(args) -> int:
    ref = today()
    with registry_txn(write=False) as reg:
        keys = reg["keys"]
    rows = [describe(n, keys[n], ref) for n in sorted(keys)]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no keys yet. Run: claude-credits add")
        return 0
    print(f"{'name':20} {'status':9} {'credit expires':20} {'store':8} {'org':10} {'last check':20}")
    for r in rows:
        exp = f"in {r['days_left']}d" if r["days_left"] is not None else "? (set --renews-day)"
        print(f"{r['name']:20} {r['status']:9} {exp:20} {r['store']:8} {(r['org_id'] or '?')[:8]:10} "
              f"{r['last_check'] or '-':20}")
    return 0


def cmd_check(args) -> int:
    ref = today()
    with registry_txn(write=False) as reg:
        names = args.names or sorted(reg["keys"])
        missing = [n for n in names if n not in reg["keys"]]
        entries = {n: dict(reg["keys"][n]) for n in names if n in reg["keys"]}
    if missing:
        raise Refused("no such key: " + ", ".join(missing))
    bad = 0
    for name, e in entries.items():
        key = stored_key(name, e)
        if key is None:
            result = {"checked": now(), "status": "invalid", "detail": "missing from the store"}
        else:
            if fingerprint(key) != e.get("fingerprint"):
                print(f"warning  {name}: stored key doesn't match its registered fingerprint", file=sys.stderr)
            result = probe(key, spend=not args.free)
        key = None
        with registry_txn() as reg:
            apply_probe(reg["keys"][name], result, ref)
        bad += result["status"] in ("invalid", "exhausted")
        print(f"{name:20} {result['status']:9} {result.get('detail', '')}")
    return 1 if bad else 0


def choose(verify: bool) -> str:
    """The next key to spend, by `order`. With `verify`, test each candidate and skip (and
    record) any that's exhausted or invalid; if none works, re-test the ones recorded as
    exhausted before giving up, in case their credit has arrived."""
    ref = today()
    with registry_txn(write=False) as reg:
        keys = {n: dict(e) for n, e in reg["keys"].items()}
    candidates = order(keys, ref)
    parked = sorted((n for n, e in keys.items() if effective_status(e, ref) == "exhausted"),
                    key=lambda n: (days_left(keys[n], ref) is None, days_left(keys[n], ref) or 0, n))
    if not verify:
        if candidates:
            return candidates[0]
    else:
        for name in candidates + parked:
            result = probe_stored(name, keys[name])
            with registry_txn() as reg:
                apply_probe(reg["keys"][name], result, ref)
            if result["status"] in ("ok", "unknown"):
                return name
    raise Refused("no usable key: every stored key is exhausted, invalid or missing. Run: claude-credits check")


def cmd_pick(args) -> int:
    print(choose(args.verify))
    return 0


def resolve_command(cmd: list[str]) -> list[str]:
    """Find the program the way a shell would, so Windows `.cmd`/`.bat` launchers work too."""
    path = shutil.which(cmd[0])
    if path is None:
        raise Refused(f"command not found: {cmd[0]}")
    return [path, *cmd[1:]]


def cmd_exec(args) -> int:
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        raise Refused("usage: claude-credits exec [--name NAME] -- COMMAND [ARGS...]")
    if os.path.basename(cmd[0]).split(".")[0] == "claude":
        print("claude-credits: warning: on 2026-10-22 the credit refused every `claude -p` request "
              "(\"Credit balance is too low\"). It covers your own API calls and Managed Agents.", file=sys.stderr)
    resolved = resolve_command(cmd)
    name = args.name or choose(verify=not args.no_verify)
    key = stored_key(name, entry(name))
    if key is None:
        raise Refused(f"{name!r} is missing from its store; add it again")
    with registry_txn() as reg:
        reg["keys"][name]["last_used"] = now()
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    env.update(ANTHROPIC_API_KEY=key, CLAUDE_CREDITS_NAME=name)
    if sys.platform == "win32":
        return subprocess.call(resolved, env=env)
    os.execve(resolved[0], resolved, env)
    return 127  # unreachable


def cmd_set(args) -> int:
    with registry_txn() as reg:
        if args.name not in reg["keys"]:
            raise Refused(f"no key named {args.name!r}")
        e = reg["keys"][args.name]
        if args.renews_day is not None:
            e["renews_day"] = args.renews_day
            renewal_changed(e)
        if args.note is not None:
            e["note"] = args.note
    print(f"updated {args.name}")
    return 0


def cmd_mark(args) -> int:
    """Record a status by hand. `exhausted` tests first: the same "credit balance is too
    low" error also comes back for requests the credit refuses by kind (Claude Code, the
    Agent SDK), and parking a key that still has credit would waste it."""
    ref, e = today(), entry(args.name)
    if args.state == "exhausted" and not args.force:
        result = probe_stored(args.name, e)
        if result["status"] == "ok":
            raise Refused(f"a test request on {args.name!r} just worked, so its credit isn't used up. The error "
                          "you saw was probably about the request: the credit refuses anything sent as Claude "
                          "Code or the Agent SDK. Pass --force to mark it anyway.")
        with registry_txn() as reg:
            k = reg["keys"][args.name]
            apply_probe(k, result, ref)
            if result["status"] == "unknown":  # the test itself failed: take your word for a day
                set_status(k, "exhausted", ref)
                k["status_until"] = (ref + dt.timedelta(days=1)).isoformat()
            if args.until and k.get("status") == "exhausted":
                k["status_until"] = args.until
            status = k["status"]
        print(f"{args.name}: {status} ({result.get('detail', '')})")
        return 0
    with registry_txn() as reg:
        k = reg["keys"][args.name]
        set_status(k, args.state, ref)
        k["last_check_detail"] = "marked by hand"
        if args.state == "exhausted" and args.until:
            k["status_until"] = args.until
    print(f"{args.name}: {args.state}")
    return 0


def cmd_remove(args) -> int:
    if not args.yes:
        raise Refused(f"this deletes the key {args.name!r} from this computer. Rerun with --yes.")
    e = entry(args.name)
    try:
        gone = store_for(e).delete(args.name)
    except Refused as err:
        if not args.force:
            raise Refused(f"{err} To forget the key here anyway, rerun with --force "
                          "(and delete the copy from your keychain by hand).")
        gone = False
    with registry_txn() as reg:
        reg["keys"].pop(args.name, None)
    print(f"removed {args.name}" + ("" if gone else " (the store had no copy, or couldn't be reached)"))
    print("note     this only deletes it here. To revoke the key itself, delete it at platform.claude.com > API keys.")
    return 0


def examples_dir() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "claude_credits_examples"), os.path.join(here, "examples")):
        if all(os.path.isfile(os.path.join(candidate, f)) for f in EXAMPLES):
            return candidate
    return None


def cmd_examples(args) -> int:
    src = examples_dir()
    if src is None:
        raise Refused("the example scripts aren't installed with this copy of claude-credits")
    os.makedirs(args.dir, exist_ok=True)
    written = []
    for f in EXAMPLES:
        dest = os.path.join(args.dir, f)
        if os.path.exists(dest) and not args.overwrite:
            print(f"kept     {dest} (exists; --overwrite replaces it)")
            continue
        shutil.copyfile(os.path.join(src, f), dest)
        written.append(dest)
    for d in written:
        print(f"wrote    {d}")
    print(f"\ntry:     claude-credits exec -- uv run {os.path.join(args.dir, 'hello.py')} \"What is a tax credit?\"")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="claude-credits", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=f"claude-credits {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="store your API key (paste at a hidden prompt)")
    a.add_argument("name", nargs="?", default=DEFAULT_NAME, help=f"a name for this key (default: {DEFAULT_NAME})")
    src = a.add_mutually_exclusive_group()
    src.add_argument("--clipboard", action="store_true", help="read the key from the clipboard and clear it")
    src.add_argument("--stdin", action="store_true", help="read the key from a pipe")
    a.add_argument("--keep-clipboard", action="store_true", help="with --clipboard, leave it alone")
    a.add_argument("--renews-day", type=renewal_day, help="day of the month your plan renews (1-31)")
    a.add_argument("--file-store", action="store_true", help="keep the key in a private file, not the OS keychain")
    a.add_argument("--replace", action="store_true", help="swap out the key already stored under this name")
    a.add_argument("--no-spend-probe", action="store_true", help="skip the 1-token test message")
    a.add_argument("--offline", action="store_true", help="store without checking the key with the API")
    a.add_argument("--force", action="store_true", help="accept a legacy key or a second key in one organization")
    a.set_defaults(fn=cmd_add)

    l = sub.add_parser("list", help="show your keys and their state (no secrets)")
    l.add_argument("--json", action="store_true")
    l.set_defaults(fn=cmd_list)

    c = sub.add_parser("check", help="test keys against the API (a 1-token Haiku call each)")
    c.add_argument("names", nargs="*")
    c.add_argument("--free", action="store_true", help="only GET /v1/models (can't see an empty balance)")
    c.set_defaults(fn=cmd_check)

    k = sub.add_parser("pick", help="print the key to spend next")
    k.add_argument("--verify", action="store_true", help="test candidates and skip dead ones")
    k.set_defaults(fn=cmd_pick)

    x = sub.add_parser("exec", help="run a command with ANTHROPIC_API_KEY set to your credit key")
    x.add_argument("--name", help="use this key instead of picking")
    x.add_argument("--no-verify", action="store_true", help="skip the 1-token test call before running")
    x.add_argument("command", nargs=argparse.REMAINDER)
    x.set_defaults(fn=cmd_exec)

    s = sub.add_parser("set", help="record a key's renewal day or a note")
    s.add_argument("name")
    s.add_argument("--renews-day", type=renewal_day)
    s.add_argument("--note")
    s.set_defaults(fn=cmd_set)

    m = sub.add_parser("mark", help="record a status by hand (exhausted tests first)")
    m.add_argument("name")
    m.add_argument("state", choices=["exhausted", "ok", "invalid", "unknown"])
    m.add_argument("--until", type=iso_date, help="exhausted until this date (YYYY-MM-DD)")
    m.add_argument("--force", action="store_true", help="mark exhausted without testing")
    m.set_defaults(fn=cmd_mark)

    r = sub.add_parser("remove", help="delete a key from this computer")
    r.add_argument("name")
    r.add_argument("--yes", action="store_true")
    r.add_argument("--force", action="store_true", help="forget it even if its store can't be reached")
    r.set_defaults(fn=cmd_remove)

    e = sub.add_parser("examples", help="copy the example jobs into a folder")
    e.add_argument("dir", nargs="?", default="claude-credits-examples")
    e.add_argument("--overwrite", action="store_true")
    e.set_defaults(fn=cmd_examples)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except Refused as err:
        print(f"claude-credits: {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("claude-credits: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
