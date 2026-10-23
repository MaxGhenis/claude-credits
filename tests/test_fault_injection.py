"""Fault injection: random add / replace / store-move / remove sequences, each crashed at a
random point (a store read, write or delete, or a registry write, before or after it
lands). After every step these must hold:

1. No untracked copies, even right after a crash: every slot in any store is either the
   one a registry entry points to or listed in the journal.
2. After the next clean run (reconcile), the journal is empty, every slot in any store is
   live, and every entry points to a stored key whose fingerprint matches.
3. No lost or invented keys: a name holds the key it held before the step or the one the
   step was adding (or nothing, for a remove or a first add that didn't land), and a step
   that reported success holds exactly the new state.
"""
from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import string
import sys
import tempfile
from pathlib import Path

from hypothesis import settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import claude_credits as cc  # noqa: E402

ALPHABET = string.ascii_letters + string.digits + "-_"
NAMES = ["a", "b"]


class Crash(Exception):
    """An injected failure: anything from a power cut to a keychain error mid-write."""


class Plan:
    def __init__(self):
        self.fail_at, self.count = None, 0

    def arm(self, fail_at):
        self.fail_at, self.count = fail_at, 0

    def tick(self, where):
        self.count += 1
        if self.fail_at is not None and self.count == self.fail_at:
            self.fail_at = None
            raise Crash(where)


class MemStore:
    def __init__(self, kind, plan):
        self.kind, self.plan, self.data = kind, plan, {}

    def get(self, slot):
        self.plan.tick(f"{self.kind}.get")
        return self.data.get(slot)

    def put(self, slot, value):
        self.plan.tick(f"{self.kind}.put:before")
        self.data[slot] = value
        self.plan.tick(f"{self.kind}.put:after")

    def delete(self, slot):
        self.plan.tick(f"{self.kind}.delete:before")
        existed = self.data.pop(slot, None) is not None
        self.plan.tick(f"{self.kind}.delete:after")
        return existed


class FakeAPI:
    def __init__(self, keys):
        self.orgs = {k: f"org-{i}" for i, k in enumerate(keys)}

    def http(self, method, path, key, body=None):
        h = {"anthropic-organization-id": self.orgs[key]}
        if path.startswith("/v1/models"):
            return 200, h, {"data": [{"id": "claude-haiku-5-5"}]}
        return 200, h, {"content": []}


class Store(RuleBasedStateMachine):
    @initialize()
    def setup(self):
        self.home = tempfile.mkdtemp(prefix="cc-faults-")
        self.keys = ["sk-ant-usr-" + "".join(secrets.choice(ALPHABET) for _ in range(90)) for _ in range(4)]
        self.plan = Plan()
        self.stores = {"keyring": MemStore("keyring", self.plan), "file": MemStore("file", self.plan)}
        self.allowed = {n: {None} for n in NAMES}  # fingerprints each name may hold now
        self.saved = {name: getattr(cc, name) for name in ("STORES", "http", "read_key", "registry_txn", "today")}
        self.saved_home = os.environ.get("CLAUDE_CREDITS_HOME")
        os.environ["CLAUDE_CREDITS_HOME"] = self.home
        api, real_txn, plan = FakeAPI(self.keys), self.saved["registry_txn"], self.plan

        @contextlib.contextmanager
        def faulty_txn(write=True):
            plan.tick("registry.open")
            with real_txn(write) as reg:
                yield reg
                if write:
                    plan.tick("registry.commit:before")
            if write:
                plan.tick("registry.commit:after")

        cc.STORES, cc.http, cc.registry_txn = self.stores, api.http, faulty_txn
        cc.today = lambda: cc.dt.date(2026, 10, 22)

    def teardown(self):
        for name, value in getattr(self, "saved", {}).items():
            setattr(cc, name, value)
        if getattr(self, "saved_home", None) is None:
            os.environ.pop("CLAUDE_CREDITS_HOME", None)
        else:
            os.environ["CLAUDE_CREDITS_HOME"] = self.saved_home
        shutil.rmtree(getattr(self, "home", ""), ignore_errors=True)

    def _registry(self):
        with self.saved["registry_txn"](write=False) as reg:
            return reg

    def _run(self, argv, fail_at):
        self.plan.arm(fail_at)
        try:
            with open(os.devnull, "w") as null, contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                return cc.main(argv)
        except Crash:
            return "crash"
        finally:
            self.plan.arm(None)

    @rule(name=st.sampled_from(NAMES), k=st.integers(0, 3), store=st.sampled_from(["keyring", "file"]),
          fail_at=st.one_of(st.none(), st.integers(1, 20)))
    def add(self, name, k, store, fail_at):
        key = self.keys[k]
        cc.read_key = lambda args, key=key: key
        before = self._registry()["keys"].get(name, {}).get("fingerprint")
        rc = self._run(["add", name, "--replace"] + (["--file-store"] if store == "file" else []), fail_at)
        if rc == 0:
            self.allowed[name] = {cc.fingerprint(key)}
        elif rc == "crash" or rc == 130:
            self.allowed[name] = {before, cc.fingerprint(key)}
        else:  # refused (e.g. the key is stored under the other name): nothing may change
            self.allowed[name] = {before}

    @rule(name=st.sampled_from(NAMES), fail_at=st.one_of(st.none(), st.integers(1, 14)))
    def remove(self, name, fail_at):
        before = self._registry()["keys"].get(name, {}).get("fingerprint")
        rc = self._run(["remove", name, "--yes"], fail_at)
        if rc == 0:
            self.allowed[name] = {None}
        elif rc == "crash" or rc == 130:
            self.allowed[name] = {before, None}
        else:
            self.allowed[name] = {before}

    @invariant()
    def holds(self):
        if not hasattr(self, "plan"):
            return
        self.plan.arm(None)
        reg = self._registry()
        live = {(e.get("store", "keyring"), cc.secret_id(n, e)) for n, e in reg["keys"].items()}
        journaled = {(i["store"], i["slot"]) for i in reg.get("journal", [])}
        # 1. right after the step, crash or not: nothing untracked
        for kind, store in self.stores.items():
            for slot in store.data:
                assert (kind, slot) in live | journaled, f"untracked copy {kind}:{slot}"
        # 2. after the next clean run: journal settled, only live slots, entries consistent
        with cc.op_lock():
            assert cc.reconcile() == []
        reg = self._registry()
        assert reg.get("journal", []) == []
        live = {(e.get("store", "keyring"), cc.secret_id(n, e)) for n, e in reg["keys"].items()}
        for kind, store in self.stores.items():
            assert {(kind, s) for s in store.data} <= live, f"orphan in {kind}: {set(store.data)}"
        for n, e in reg["keys"].items():
            value = self.stores[e.get("store", "keyring")].data.get(cc.secret_id(n, e))
            assert value is not None and cc.fingerprint(value) == e["fingerprint"], f"{n} points at a bad slot"
        # 3. no lost or invented keys
        for n in NAMES:
            actual = reg["keys"].get(n, {}).get("fingerprint")
            assert actual in self.allowed[n], f"{n} holds {actual}, expected one of {self.allowed[n]}"
            self.allowed[n] = {actual}


Store.TestCase.settings = settings(max_examples=300, stateful_step_count=25, deadline=None)
TestFaultInjection = Store.TestCase
