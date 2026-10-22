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
prompt, so `claude -p` and the Agent SDK fail, although the docs list them as covered.

Keys live in your operating system's keychain (macOS Keychain, Windows Credential Locker,
Linux Secret Service) through the `keyring` library, or, on a headless machine and only if
you ask, in a 0600 file. A registry next to it holds what isn't secret: organization id,
key fingerprint, status, renewal day. `exec` hands the key to one command through its
environment; it never prints it or puts it on a command line.
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
import urllib.error
import urllib.request

__version__ = "0.1.0"


def config_dir() -> str:
    if os.environ.get("CLAUDE_CREDITS_HOME"):
        return os.environ["CLAUDE_CREDITS_HOME"]
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "claude-credits")
    return os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "claude-credits")


API_BASE = os.environ.get("CLAUDE_CREDITS_API_BASE", "https://api.anthropic.com")
PROBE_MODEL = os.environ.get("CLAUDE_CREDITS_PROBE_MODEL", "claude-haiku-5-5")
KEYRING_SERVICE = "claude-credits"
TIMEOUT = 60
DEFAULT_NAME = "default"

# Console keys: personal (usr), service-account (svc) and legacy (api03). The credit
# program issues personal keys; a legacy key may sit in an organization that bills a card.
KEY_RE = re.compile(r"^sk-ant-(?P<kind>usr|svc|api)(?P<ver>\d{2})?-[A-Za-z0-9_-]{40,}$")
NOT_A_CONSOLE_KEY = {
    "oat": "a subscription OAuth token (from `claude setup-token`), not an API key. Create an API key "
           "at platform.claude.com > API keys.",
    "ort": "an OAuth refresh token, not an API key.",
    "admin": "an Admin API key. It manages the organization but can't send messages.",
}
NAME_RE = re.compile(r"^[A-Za-z0-9._@+-]{1,64}$")
EXHAUSTED_MARK = "credit balance is too low"
STRIP_ENV = ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
             "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")


class Refused(Exception):
    """A check failed: say why, change nothing."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today() -> dt.date:
    return dt.datetime.now().date()


# ---------- keys ----------

def classify(value: str) -> str:
    """Kind of Console key (usr, svc, api), or raise Refused naming what it is instead."""
    m = KEY_RE.match(value)
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
            raise Refused("no OS keychain is available here (common on headless Linux). Rerun "
                          "`add` with --file-store to keep the key in a 0600 file instead.")
        return keyring

    def get(self, name: str) -> str | None:
        return self._kr().get_password(KEYRING_SERVICE, name)

    def put(self, name: str, value: str) -> None:
        kr = self._kr()
        kr.set_password(KEYRING_SERVICE, name, value)
        if kr.get_password(KEYRING_SERVICE, name) != value:
            raise Refused("the keychain didn't return the key as written")

    def delete(self, name: str) -> bool:
        kr = self._kr()
        try:
            kr.delete_password(KEYRING_SERVICE, name)
            return True
        except Exception as e:  # keyring.errors.PasswordDeleteError when absent
            if type(e).__name__ == "PasswordDeleteError":
                return False
            raise


class FileStore:
    """A 0600 JSON file in the config directory, for machines without a keychain."""
    kind = "file"

    def path(self) -> str:
        return os.path.join(config_dir(), "keys.json")

    def _load(self) -> dict:
        p = self.path()
        if not os.path.exists(p):
            return {}
        if sys.platform != "win32" and os.stat(p).st_mode & 0o077:
            raise Refused(f"{p} is readable by other users; run: chmod 600 {p}")
        return json.load(open(p))

    def _save(self, data: dict) -> None:
        os.makedirs(config_dir(), mode=0o700, exist_ok=True)
        tmp = self.path() + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, self.path())

    def get(self, name: str) -> str | None:
        return self._load().get(name)

    def put(self, name: str, value: str) -> None:
        data = self._load()
        data[name] = value
        self._save(data)

    def delete(self, name: str) -> bool:
        data = self._load()
        if name not in data:
            return False
        del data[name]
        self._save(data)
        return True


STORES = {"keyring": KeyringStore(), "file": FileStore()}


def store_for(entry: dict):
    return STORES[entry.get("store", "keyring")]


# ---------- registry (no secrets) ----------

@contextlib.contextmanager
def _locked(path: str):
    with open(path, "a") as fh:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
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
    os.makedirs(config_dir(), mode=0o700, exist_ok=True)
    path = os.path.join(config_dir(), "registry.json")
    with _locked(path + ".lock"):
        reg = json.load(open(path)) if os.path.exists(path) else {"version": 1, "keys": {}}
        before = json.dumps(reg, sort_keys=True)
        yield reg
        if write and json.dumps(reg, sort_keys=True) != before:
            tmp = path + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
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


def effective_status(e: dict, ref: dt.date) -> str:
    """ok | exhausted | invalid | unknown. An exhaustion whose `until` has passed reads unknown."""
    st = e.get("status", "unknown")
    if st == "exhausted":
        until = e.get("status_until")
        if until and dt.date.fromisoformat(until) <= ref:
            return "unknown"
    return st


def order(keys: dict, ref: dt.date) -> list[str]:
    """Spending order when you have more than one key: usable (ok, then unknown) first;
    within that, soonest expiry first (unknown renewal last); then least recently used."""
    rank = {"ok": 0, "unknown": 1}
    usable = [n for n, e in keys.items() if effective_status(e, ref) in rank]

    def key(n: str):
        e = keys[n]
        dl = days_left(e, ref)
        return (rank[effective_status(e, ref)], dl is None, dl or 0, e.get("last_used") or "", n)
    return sorted(usable, key=key)


# ---------- API probes ----------

def http(method: str, path: str, key: str, body: dict | None = None) -> tuple[int, dict, dict | str]:
    req = urllib.request.Request(
        API_BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json", "user-agent": f"claude-credits/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            status, headers, raw = r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        status, headers, raw = e.code, dict(e.headers or {}), e.read()
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
    and returns the organization's rate limits."""
    out: dict = {"checked": now()}
    status, headers, payload = http("GET", "/v1/models?limit=100", key)
    out["org_id"] = headers.get("anthropic-organization-id")
    if status == 401:
        return {**out, "status": "invalid", "detail": error_text(payload)}
    if status != 200:
        return {**out, "status": "unknown", "detail": f"models HTTP {status}: {error_text(payload)}"}
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
    e["status"] = result["status"]
    e["last_check"] = result["checked"]
    e["last_check_detail"] = result.get("detail", "")
    if result.get("org_id"):
        e["org_id"] = result["org_id"]
    if result.get("rate_limits"):
        e["rate_limits"] = result["rate_limits"]
    if result["status"] == "exhausted" and e.get("renews_day"):
        e["status_until"] = (ref + dt.timedelta(days=days_left(e, ref))).isoformat()
    else:
        e.pop("status_until", None)


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
                ["xsel", "--clipboard", "--input"], ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $null"]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, input=b"", timeout=10)
            return True
    return False


def read_key(args) -> str:
    if args.stdin:
        return sys.stdin.read().strip()
    if args.clipboard:
        return read_clipboard()
    if not sys.stdin.isatty():
        raise Refused("no terminal to prompt on; pass --stdin or --clipboard")
    return getpass.getpass("Paste your Claude API key (sk-ant-usr-…; input hidden): ").strip()


# ---------- commands ----------

def cmd_add(args) -> int:
    name = args.name
    if not NAME_RE.match(name):
        raise Refused("names may use letters, digits and . _ @ + - (up to 64)")
    value = read_key(args)
    kind = classify(value)
    if kind == "api" and not args.force:
        raise Refused("that's a legacy sk-ant-api key. The credit program issues sk-ant-usr keys, and a legacy "
                      "key may sit in an organization that bills a card. Pass --force if you're sure.")
    fp, ref = fingerprint(value), today()
    result = {"checked": now(), "status": "unknown", "detail": "not probed (--offline)"} if args.offline \
        else probe(value, spend=not args.no_spend_probe)
    if result["status"] == "invalid":
        raise Refused(f"the API rejected this key ({result.get('detail', '')}). Nothing was stored.")
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
        old = reg["keys"].get(name, {})
        if old and old.get("store", "keyring") != store.kind:
            store_for(old).delete(name)  # moved between stores: don't leave a copy behind
        e = reg["keys"].setdefault(name, {})
        e.update({"store": store.kind, "kind": kind, "fingerprint": fp,
                  "added": e.get("added") or now(), "updated": now()})
        if args.renews_day:
            renewal_on_or_after(args.renews_day, ref)
            e["renews_day"] = args.renews_day
        apply_probe(e, result, ref)
    cleared = args.clipboard and not args.keep_clipboard and clear_clipboard_if(value)
    value = ""
    where = "your OS keychain" if store.kind == "keyring" else FileStore().path()
    print(f"stored   {name!r} in {where}")
    print(f"key      {kind} key, {fp}")
    print(f"org      {result.get('org_id') or 'unknown'}")
    print(f"status   {result['status']}: {result.get('detail', '')}")
    if result.get("rate_limits"):
        print("limits   " + ", ".join(f"{k} {v}" for k, v in sorted(result["rate_limits"].items())))
    if cleared:
        print("clipboard cleared")
    if not args.renews_day and not entry(name).get("renews_day"):
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
    print(f"{'name':20} {'status':9} {'credit expires':15} {'store':8} {'org':10} {'last check':20}")
    for r in rows:
        exp = f"in {r['days_left']}d" if r["days_left"] is not None else "? (set --renews-day)"
        print(f"{r['name']:20} {r['status']:9} {exp:15} {r['store']:8} {(r['org_id'] or '?')[:8]:10} "
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
        key = store_for(e).get(name)
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
    ref = today()
    with registry_txn(write=False) as reg:
        candidates = order(reg["keys"], ref)
        entries = {n: dict(reg["keys"][n]) for n in candidates}
    for name in candidates:
        if not verify:
            return name
        key = store_for(entries[name]).get(name)
        result = probe(key) if key else {"checked": now(), "status": "invalid", "detail": "missing from the store"}
        key = None
        with registry_txn() as reg:
            apply_probe(reg["keys"][name], result, ref)
        if result["status"] in ("ok", "unknown"):
            return name
    raise Refused("no usable key: every stored key is exhausted, invalid or missing. Run: claude-credits list")


def cmd_pick(args) -> int:
    print(choose(args.verify))
    return 0


def cmd_exec(args) -> int:
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        raise Refused("usage: claude-credits exec [--name NAME] -- COMMAND [ARGS...]")
    name = args.name or choose(verify=not args.no_verify)
    key = store_for(entry(name)).get(name)
    if key is None:
        raise Refused(f"{name!r} is missing from its store; add it again")
    with registry_txn() as reg:
        reg["keys"][name]["last_used"] = now()
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    env.update(ANTHROPIC_API_KEY=key, CLAUDE_CREDITS_NAME=name)
    if os.path.basename(cmd[0]).split(".")[0] == "claude":
        print("claude-credits: warning: on 2026-10-22 the credit refused every `claude -p` request "
              "(\"Credit balance is too low\"). It covers your own API calls and Managed Agents.", file=sys.stderr)
    if sys.platform == "win32":
        return subprocess.call(cmd, env=env)
    os.execvpe(cmd[0], cmd, env)
    return 127  # unreachable


def cmd_set(args) -> int:
    with registry_txn() as reg:
        if args.name not in reg["keys"]:
            raise Refused(f"no key named {args.name!r}")
        e = reg["keys"][args.name]
        if args.renews_day is not None:
            renewal_on_or_after(args.renews_day, today())  # validates 1-31
            e["renews_day"] = args.renews_day
        if args.note is not None:
            e["note"] = args.note
    print(f"updated {args.name}")
    return 0


def cmd_mark(args) -> int:
    ref = today()
    with registry_txn() as reg:
        if args.name not in reg["keys"]:
            raise Refused(f"no key named {args.name!r}")
        e = reg["keys"][args.name]
        apply_probe(e, {"checked": now(), "status": args.state, "detail": "marked by hand"}, ref)
        if args.state == "exhausted" and args.until:
            e["status_until"] = dt.date.fromisoformat(args.until).isoformat()
    print(f"{args.name}: {args.state}")
    return 0


def cmd_remove(args) -> int:
    if not args.yes:
        raise Refused(f"this deletes the key {args.name!r}. Rerun with --yes.")
    e = entry(args.name)
    gone = store_for(e).delete(args.name)
    with registry_txn() as reg:
        reg["keys"].pop(args.name, None)
    print(f"removed {args.name}" + ("" if gone else " (the store had no copy)"))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="claude-credits", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=f"claude-credits {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="store your API key (paste at a hidden prompt)")
    a.add_argument("name", nargs="?", default=DEFAULT_NAME, help=f"a name for this key (default: {DEFAULT_NAME})")
    src = a.add_mutually_exclusive_group()
    src.add_argument("--clipboard", action="store_true", help="read the key from the clipboard and clear it")
    src.add_argument("--stdin", action="store_true", help="read the key from stdin")
    a.add_argument("--keep-clipboard", action="store_true", help="with --clipboard, leave it alone")
    a.add_argument("--renews-day", type=int, help="day of the month your plan renews (1-31)")
    a.add_argument("--file-store", action="store_true", help="keep the key in a 0600 file, not the OS keychain")
    a.add_argument("--no-spend-probe", action="store_true", help="skip the 1-token test message")
    a.add_argument("--offline", action="store_true", help="don't contact the API at all")
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
    x.add_argument("--no-verify", action="store_true", help="skip the pre-flight test call")
    x.add_argument("command", nargs=argparse.REMAINDER)
    x.set_defaults(fn=cmd_exec)

    s = sub.add_parser("set", help="record a key's renewal day or a note")
    s.add_argument("name")
    s.add_argument("--renews-day", type=int)
    s.add_argument("--note")
    s.set_defaults(fn=cmd_set)

    m = sub.add_parser("mark", help="record a key as exhausted, ok, invalid or unknown")
    m.add_argument("name")
    m.add_argument("state", choices=["exhausted", "ok", "invalid", "unknown"])
    m.add_argument("--until", help="exhausted until this date (YYYY-MM-DD)")
    m.set_defaults(fn=cmd_mark)

    r = sub.add_parser("remove", help="delete a key and its registry entry")
    r.add_argument("name")
    r.add_argument("--yes", action="store_true")
    r.set_defaults(fn=cmd_remove)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except Refused as e:
        print(f"claude-credits: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
