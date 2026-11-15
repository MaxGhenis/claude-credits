# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic==2.8.0"]
# ///
"""Review files or a diff with Claude in one request on your plan's API credit.

    claude-credits exec -- uv run review.py --brief brief.md --diff change.diff --out review.md
    gh pr diff 123 | claude-credits exec -- uv run review.py --brief brief.md --diff - \\
        --out review.md src/app.py

It sends one Messages API request (Claude Fable 5.2 at high effort unless you pick another
model), writes the review to --out, and writes its token counts and list-price cost beside
it in --out + ".json". Before sending, it counts the prompt's tokens (a free call) and
refuses if the most the request could cost is more than --cap dollars.

The model sees only what you pass. It can't open other files, run tests or follow links,
and it gets no project instructions, so put everything the review needs in the brief and
the files. The prompt is your own, so the credit pays for it; requests carrying the
Claude Code or Agent SDK system prompt are refused by the credit.

Refusal fallbacks are on: if a safety classifier declines the request, the API reruns it
on the model Anthropic recommends for that refusal (for Fable 5.2, Opus 4.8 or Opus 5).
The review's header names the model that wrote it. --no-fallback turns this off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

# USD per million tokens (input, output) at standard rates, from Anthropic's price table
# (cached 2026-11-03). The last three are refusal-fallback targets.
PRICES = {
    "claude-fable-5-2": (8.00, 40.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
}
HAIKU_LONG_PROMPT = 100_000           # Haiku 5.5 costs more above this many input tokens
HAIKU_LONG_PRICES = (0.50, 2.50)
CACHE_WRITE, CACHE_READ = 1.25, 0.10  # multiples of the input price
MODELS = ("claude-fable-5-2", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
# Models a declined request may be rerun on. Haiku 5.5 has no server-side fallback.
FALLBACK_TARGETS = {
    "claude-fable-5-2": ("claude-opus-4-8", "claude-opus-5"),
    "claude-opus-5-5": ("claude-opus-5", "claude-opus-4-8"),
    "claude-sonnet-5-5": ("claude-sonnet-5",),
    "claude-haiku-5-5": (),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CONTEXT_WINDOW = 1_000_000
EFFORTS = ("low", "medium", "high", "xhigh", "max")

SYSTEM = """You review work for its author. The materials come first: whole files (with line \
numbers) and diffs, each as its own document. The brief comes last and says what to review \
and what matters.

Review only what you are given. Where a judgment depends on something you can't see, such \
as code that isn't included or whether the tests pass, say so instead of guessing.

Start with one line: VERDICT: APPROVE or VERDICT: REQUEST_CHANGES. Then list findings, most \
severe first. Give each one a level (P0 breaks correctness, security or data; P1 is a real \
bug or missing safeguard; P2 is a gap worth fixing; P3 is minor), its location (file and \
line, or the diff hunk), what is wrong, a concrete input or situation where it fails, and \
a fix. After the findings, list missing tests and the risks that remain. If you find \
nothing worth fixing, say so plainly. The brief may change any of this."""

# Patterns that look like credentials. A match stops the request before anything is sent.
SECRET_PATTERNS = (
    ("an Anthropic API key or token", re.compile(r"sk-ant-[a-z]+\d*-[A-Za-z0-9_\-]{16,}")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("a GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("an AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
)


class Refused(Exception):
    """Stopped before anything was sent (or, for --out, before anything was written)."""


@dataclass
class Material:
    kind: str        # "file", "diff" or "brief"
    name: str
    text: str
    sha256: str = field(default="")

    @property
    def title(self) -> str:
        return f"{self.kind}: {self.name}"


def read_text(path: str, kind: str, stdin) -> Material:
    if path == "-":
        raw, name = stdin.buffer.read(), "(stdin)"
    else:
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as e:
            raise Refused(f"can't read {kind} {path}: {e.strerror or e}") from None
        name = path
    if b"\x00" in raw:
        raise Refused(f"{kind} {name} looks binary; pass text files only")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise Refused(f"{kind} {name} isn't UTF-8 text") from None
    if not text.strip():
        raise Refused(f"{kind} {name} is empty")
    return Material(kind, name, text, hashlib.sha256(raw).hexdigest())


def numbered(text: str) -> str:
    lines = text.splitlines()
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}  {line}" for i, line in enumerate(lines, 1))


def load_inputs(args, stdin) -> tuple[Material, list[Material]]:
    if args.brief == "-" and args.diff == "-":
        raise Refused("only one of --brief and --diff can read stdin")
    brief = read_text(args.brief, "brief", stdin)
    materials = [read_text(p, "file", stdin) for p in args.files]
    if args.diff:
        materials.append(read_text(args.diff, "diff", stdin))
    if not materials:
        raise Refused("nothing to review: pass files, --diff, or both")
    return brief, materials


def check_secrets(items: list[Material], env_key: str | None, allow_key_like: bool) -> None:
    for m in items:
        if env_key and len(env_key) >= 20 and env_key in m.text:
            raise Refused(f"{m.kind} {m.name} contains the API key this job runs with; remove it")
        if allow_key_like:
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(m.text):
                raise Refused(f"{m.kind} {m.name} contains what looks like {label}; redact it, "
                              "or pass --allow-key-like if it's a dummy value")


def build_request(brief: Material, materials: list[Material], model: str, effort: str,
                  max_tokens: int, fallback: bool, line_numbers: bool) -> dict:
    content = []
    for m in materials:
        body = numbered(m.text) if (m.kind == "file" and line_numbers) else m.text
        content.append({"type": "document", "title": m.title,
                        "source": {"type": "text", "media_type": "text/plain", "data": body}})
    content.append({"type": "text", "text": "Brief:\n\n" + brief.text})
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": content}],
        "output_config": {"effort": effort},
    }
    if fallback and FALLBACK_TARGETS[model]:
        params["betas"] = [FALLBACK_BETA]
        params["fallbacks"] = "default"
    return params


def price_key(model: str) -> str | None:
    """The PRICES entry for a served model name: exact, else the longest prefix."""
    if model in PRICES:
        return model
    hits = [k for k in PRICES if model.startswith(k + "-")]
    return max(hits, key=len) if hits else None


def prices_for(model: str, input_tokens: int) -> tuple[float, float]:
    key = price_key(model)
    if key is None:
        raise KeyError(model)
    if key == "claude-haiku-5-5" and input_tokens > HAIKU_LONG_PROMPT:
        return HAIKU_LONG_PRICES
    return PRICES[key]


def billable_input_bound(counted: int) -> int:
    """Token counting is an estimate, so the cap assumes up to 10% (plus 200) more input."""
    return -(-counted * 11 // 10) + 200


def worst_case_cost(model: str, input_tokens: int, max_tokens: int, fallback: bool) -> float:
    """The most the request can cost: the requested model's attempt, plus one attempt on
    every permitted fallback target, each billed for the input bound and max_tokens of
    output (thinking counts as output)."""
    attempts = [model] + (list(FALLBACK_TARGETS[model]) if fallback else [])
    bound = billable_input_bound(input_tokens)
    total = 0.0
    for m in attempts:
        pin, pout = prices_for(m, bound)
        total += (bound * pin + max_tokens * pout) / 1e6
    return total


def _as_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def attempt_cost(model: str, u: dict) -> dict:
    inp = u.get("input_tokens") or 0
    out = u.get("output_tokens") or 0
    cw = u.get("cache_creation_input_tokens") or 0
    cr = u.get("cache_read_input_tokens") or 0
    row = {"model": model, "type": u.get("type", "message"), "input_tokens": inp,
           "output_tokens": out, "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr}
    try:
        pin, pout = prices_for(model, inp + cw + cr)
    except KeyError:
        row["cost_usd"] = None
        return row
    row["cost_usd"] = (inp * pin + cw * pin * CACHE_WRITE + cr * pin * CACHE_READ + out * pout) / 1e6
    return row


def usage_rows(message, requested_model: str) -> list[dict]:
    """One row per model attempt. usage.iterations lists every attempt when a fallback
    ran; otherwise the top-level usage covers the one attempt."""
    usage = _as_dict(getattr(message, "usage", None))
    iterations = usage.get("iterations") or []
    if iterations:
        rows = []
        for it in iterations:
            it = _as_dict(it)
            model = it.get("model") or (requested_model if it.get("type") == "message" else message.model)
            rows.append(attempt_cost(model, it))
        return rows
    return [attempt_cost(getattr(message, "model", None) or requested_model, usage)]


def review_text(message) -> tuple[str, bool]:
    """The review's text, and whether text from a declined attempt was dropped. A
    `fallback` block marks where a declined model stopped and the next one began; text
    before the last such block is the declined model's partial output, so it's dropped."""
    parts, dropped = [], False
    for b in message.content:
        kind = getattr(b, "type", "")
        if kind == "fallback":
            dropped = dropped or any(p.strip() for p in parts)
            parts = []
        elif kind == "text":
            parts.append(getattr(b, "text", ""))
    return "".join(parts), dropped


def api_error_message(status: int | None, text: str) -> str:
    low = text.lower()
    if "credit balance is too low" in low or "workspace api usage limits" in low:
        name = os.environ.get("CLAUDE_CREDITS_ACCOUNT") or os.environ.get("CLAUDE_CREDITS_NAME") or "default"
        return ("Anthropic says this credit (or your share of it) is used up. If so, run: "
                f"claude-credits mark {name} exhausted (it checks first).")
    return f"the API returned an error{f' ({status})' if status else ''}: {text[:300]}"


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def run(args, client, *, stdin=None, env=None, clock=time.monotonic, now=None, err=None) -> int:
    stdin = stdin if stdin is not None else sys.stdin
    env = env if env is not None else os.environ
    err = err if err is not None else sys.stderr
    if args.model not in MODELS:
        raise Refused(f"--model must be one of {', '.join(MODELS)}")
    if args.cap <= 0:
        raise Refused("--cap must be more than 0")
    brief, materials = load_inputs(args, stdin)
    check_secrets([brief, *materials], env.get("ANTHROPIC_API_KEY"), args.allow_key_like)
    sidecar = args.out + ".json"
    if not args.dry_run and not args.overwrite:
        for p in (args.out, sidecar):
            if os.path.exists(p):
                raise Refused(f"{p} exists; pass --overwrite to replace it")
    fallback = not args.no_fallback and bool(FALLBACK_TARGETS[args.model])
    params = build_request(brief, materials, args.model, args.effort, args.max_tokens,
                           fallback, not args.no_line_numbers)

    counted = client.messages.count_tokens(model=params["model"], system=params["system"],
                                           messages=params["messages"]).input_tokens
    if counted + args.max_tokens > CONTEXT_WINDOW:
        raise Refused(f"{counted:,} input tokens plus --max-tokens {args.max_tokens:,} exceed the "
                      f"{CONTEXT_WINDOW:,}-token context window; send fewer files")
    worst = worst_case_cost(args.model, counted, args.max_tokens, fallback)
    print(f"{counted:,} input tokens, at most {args.max_tokens:,} output tokens: worst case "
          f"${worst:.2f} at list price{' counting fallbacks' if fallback else ''} (cap ${args.cap:.2f})",
          file=err)
    if worst > args.cap:
        raise Refused(f"the worst case (${worst:.2f}) is above --cap ${args.cap:.2f}; lower "
                      "--max-tokens, send less, or raise --cap")
    if args.dry_run:
        print("dry run: nothing sent", file=err)
        return 0

    started = clock()
    with client.beta.messages.stream(**params) as stream:
        message = stream.get_final_message()
        request_id = getattr(stream, "request_id", None)
    seconds = clock() - started

    rows = usage_rows(message, args.model)
    known = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    cost = sum(known) if len(known) == len(rows) else None
    served = getattr(message, "model", None) or args.model
    fell_back = any(r["type"] == "fallback_message" for r in rows)
    stop = getattr(message, "stop_reason", None)
    details = _as_dict(getattr(message, "stop_details", None))
    text, dropped_partial = review_text(message)
    when =(now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")

    if stop == "refusal":
        status = (f"DECLINED (category: {details.get('category') or 'none given'})"
                  + ("; the fallback declined too" if fell_back else "") + ". There is no review below.")
        body = ""
        code = 3
    elif stop == "max_tokens":
        status = f"CUT OFF at --max-tokens {args.max_tokens:,}; the review below is incomplete."
        body = text
        code = 4
    else:
        status = "complete"
        body = text
        code = 0

    inp = sum(r["input_tokens"] for r in rows)
    out = sum(r["output_tokens"] for r in rows)
    attempts = "; ".join(f"{r['model']} ({r['type']}): {r['input_tokens']:,} in, {r['output_tokens']:,} out, "
                         + (f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "price unknown")
                         for r in rows)
    account = env.get("CLAUDE_CREDITS_ACCOUNT") or env.get("CLAUDE_CREDITS_NAME")
    header = [
        f"# Review: {os.path.basename(args.out)}",
        "",
        f"- Status: {status}",
        f"- Asked: {args.model}, effort {args.effort}, max {args.max_tokens:,} output tokens"
        + ("" if fallback else ", no fallback"),
        f"- Written by: {served}" + (" (a fallback, after the requested model declined"
                                     + ("; its partial text is left out" if dropped_partial else "") + ")"
                                     if fell_back and code != 3 else ""),
        f"- Tokens: {inp:,} input, {out:,} output (thinking included)",
        f"- Cost at list price: " + (f"${cost:.4f}" if cost is not None else "unknown (a model outside the price table ran)"),
        f"- Attempts: {attempts}",
        f"- Request id: {request_id or 'not given'}",
    ]
    if account:
        header.append(f"- Paid from the credit of: {account}")
    header.append(f"- Inputs: " + ", ".join(f"{m.title} (sha256 {m.sha256[:12]})" for m in [brief, *materials]))
    header.append(f"- Finished: {when}, after {seconds:.0f} s")
    _atomic_write(args.out, "\n".join(header) + "\n\n---\n\n" + body + ("\n" if body and not body.endswith("\n") else ""))
    record = {
        "status": status, "exit_code": code, "requested_model": args.model, "served_model": served,
        "fallback_enabled": fallback, "fell_back": fell_back, "effort": args.effort,
        "max_tokens": args.max_tokens, "stop_reason": stop, "stop_details": details or None,
        "counted_input_tokens": counted, "worst_case_usd": round(worst, 6), "cap_usd": args.cap,
        "cost_usd": round(cost, 6) if cost is not None else None, "attempts": rows,
        "output_tokens_details": _as_dict(_as_dict(getattr(message, "usage", None)).get("output_tokens_details")) or None,
        "dropped_declined_partial": dropped_partial, "request_id": request_id, "account": account,
        "inputs": [{"kind": m.kind, "name": m.name, "sha256": m.sha256} for m in [brief, *materials]],
        "finished_at": when, "seconds": round(seconds, 1), "review_file": args.out,
    }
    _atomic_write(sidecar, json.dumps(record, indent=2) + "\n")
    print(f"{status} Written by {served}; {inp:,} in, {out:,} out; "
          + (f"${cost:.4f}" if cost is not None else "cost unknown") + f"; {args.out}", file=err)
    return code


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("files", nargs="*", help="files to include whole (numbered by line)")
    p.add_argument("--brief", required=True, help="what to review and how (a file, or - for stdin)")
    p.add_argument("--diff", help="a diff to review (a file, or - for stdin)")
    p.add_argument("--out", required=True, help="write the review here, and its record to OUT.json")
    p.add_argument("--model", default="claude-fable-5-2", choices=MODELS)
    p.add_argument("--effort", default="high", choices=EFFORTS)
    p.add_argument("--max-tokens", type=int, default=32_000, help="output cap, thinking included")
    p.add_argument("--cap", type=float, default=5.0, help="refuse if the worst case costs more (USD)")
    p.add_argument("--dry-run", action="store_true", help="count tokens and price the worst case; send nothing")
    p.add_argument("--no-fallback", action="store_true", help="don't rerun a declined request on another model")
    p.add_argument("--no-line-numbers", action="store_true")
    p.add_argument("--allow-key-like", action="store_true",
                   help="send text that looks like a credential (never the job's own key)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.max_tokens < 1:
        print("review: --max-tokens must be at least 1", file=sys.stderr)
        return 2
    import anthropic  # imported here so the tests run without the SDK

    client = anthropic.Anthropic(max_retries=2, timeout=1800.0)
    try:
        return run(args, client)
    except Refused as e:
        print(f"review: {e}", file=sys.stderr)
        return 2
    except anthropic.APIStatusError as e:
        print(f"review: {api_error_message(e.status_code, str(e))}", file=sys.stderr)
        return 5
    except anthropic.APIConnectionError as e:
        print(f"review: couldn't reach the API ({type(e).__name__}); nothing was written. "
              "A request cut off mid-stream may still be billed for what it produced.", file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
