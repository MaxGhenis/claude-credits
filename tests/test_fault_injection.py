"""Fault injection: random add / replace / store-move / remove / recovery sequences, each
step failed at a random store or registry operation, in one of two ways:

- error: that one operation raises and the code's own error handling runs (finally
  blocks, cleanup, rollback);
- kill: the process dies there. Every later operation in that step fails too, so no
  cleanup runs, as with a power cut or `kill -9`.

Steps start from whatever the last one left (an unsettled journal, a half-finished
replace), and recovery itself can be interrupted. After every step these must hold:

1. No untracked copies: every slot in any store is either the one a registry entry points
   to or listed in the journal.
2. Every entry points to a stored key whose fingerprint matches.
3. No lost or invented keys: a name holds the key it held before the step or the one the
   step was adding (or nothing, for a remove or a first add that didn't land), and a step
   that reported success holds exactly the new state.

And after an uninterrupted recovery: the journal is empty and only live slots remain.
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
faults = st.one_of(st.none(), st.tuples(st.integers(1, 24), st.booleans()))  # (operation number, kill?)


class Crash(Exception):
    """An injected failure."""


class Plan:
    def __init__(self):
        self.arm(None)

    def arm(self, fault):
        self.fail_at, self.kill = fault if fault else (None, False)
        self.count, self.dead = 0, False

    def tick(self, where):
        if self.dead:  # killed: nothing after the kill point happens
            raise Crash(f"{where} (process gone)")
        self.count += 1
        if self.fail_at is not None and self.count == self.fail_at:
            self.fail_at, self.dead = None, self.kill
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

    def _quietly(self, fn, fault):
        self.plan.arm(fault)
        try:
            with open(os.devnull, "w") as null, contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                return fn()
        except Crash:
            return "crash"
        finally:
            self.plan.arm(None)

    @rule(name=st.sampled_from(NAMES), k=st.integers(0, 3), store=st.sampled_from(["keyring", "file"]), fault=faults)
    def add(self, name, k, store, fault):
        key = self.keys[k]
        cc.read_key = lambda args, key=key: key
        before = self._registry()["keys"].get(name, {}).get("fingerprint")
        argv = ["add", name, "--replace"] + (["--file-store"] if store == "file" else [])
        rc = self._quietly(lambda: cc.main(argv), fault)
        if rc == 0:
            self.allowed[name] = {cc.fingerprint(key)}
        elif rc in ("crash", 130):
            self.allowed[name] = {before, cc.fingerprint(key)}
        else:  # refused (e.g. the key is stored under the other name): nothing may change
            self.allowed[name] = {before}

    @rule(name=st.sampled_from(NAMES), fault=faults)
    def remove(self, name, fault):
        before = self._registry()["keys"].get(name, {}).get("fingerprint")
        rc = self._quietly(lambda: cc.main(["remove", name, "--yes"]), fault)
        self.allowed[name] = {None} if rc == 0 else {before, None}

    @rule(fault=faults)
    def recover(self, fault):
        """A later run's recovery, possibly interrupted itself."""
        def run():
            with cc.op_lock():
                cc.reconcile()
        self._quietly(run, fault)

    @rule()
    def recover_cleanly(self):
        with cc.op_lock():
            assert cc.reconcile() == []
        reg = self._registry()
        assert reg.get("journal", []) == []
        live = {(e.get("store", "keyring"), cc.secret_id(n, e)) for n, e in reg["keys"].items()}
        for kind, store in self.stores.items():
            assert {(kind, s) for s in store.data} <= live, f"orphan in {kind}: {set(store.data)}"

    @invariant()
    def holds(self):
        if not hasattr(self, "plan"):
            return
        reg = self._registry()
        live = {(e.get("store", "keyring"), cc.secret_id(n, e)) for n, e in reg["keys"].items()}
        journaled = {(i["store"], i["slot"]) for i in reg.get("journal", [])}
        for kind, store in self.stores.items():  # 1. nothing untracked
            for slot in store.data:
                assert (kind, slot) in live | journaled, f"untracked copy {kind}:{slot}"
        for n, e in reg["keys"].items():  # 2. every entry points at its key
            value = self.stores[e.get("store", "keyring")].data.get(cc.secret_id(n, e))
            assert value is not None and cc.fingerprint(value) == e["fingerprint"], f"{n} points at a bad slot"
        for n in NAMES:  # 3. no lost or invented keys
            actual = reg["keys"].get(n, {}).get("fingerprint")
            assert actual in self.allowed[n], f"{n} holds {actual}, expected one of {self.allowed[n]}"
            self.allowed[n] = {actual}


Store.TestCase.settings = settings(max_examples=300, stateful_step_count=30, deadline=None)
TestFaultInjection = Store.TestCase
