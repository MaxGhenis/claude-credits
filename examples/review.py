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
it in --out + ".json". Before sending, it counts the prompt's tokens (free calls) and
refuses if the most the request could cost is more than --cap dollars.

The model sees only what you pass. It can't open other files, run tests or follow links,
and it gets no project instructions, so put everything the review needs in the brief and
the files. The prompt is your own, so the credit pays for it; requests carrying the
Claude Code or Agent SDK system prompt are refused by the credit.

Refusal fallbacks are on: if a safety classifier declines the request, the API reruns it
on the next permitted model (for Fable 5.2, Opus 4.8 and then Opus 5), and the cap counts
those attempts. The review's header names the model that wrote it. --no-fallback turns
fallbacks off.

The credential screen is best-effort: it stops the job's own key and common token shapes,
not every secret. Exit codes: 0 complete, 2 refused before sending, 3 declined, 4 stopped
early or empty, 5 API or network error, 6 paid but the files couldn't be written (the
review then goes to stdout).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
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
# Each model's permitted fallbacks, in the order they're tried, as the Models API lists
# them (allowed_fallback_models, read 2026-11-15). The request names these explicitly, so
# the cap prices exactly the models that can run. Haiku 5.5 has none.
FALLBACK_TARGETS = {
    "claude-fable-5-2": ("claude-opus-4-8", "claude-opus-5"),
    "claude-opus-5-5": ("claude-opus-4-8", "claude-opus-5"),
    "claude-sonnet-5-5": ("claude-sonnet-5",),
    "claude-haiku-5-5": (),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CONTEXT_WINDOW = 1_000_000
EFFORTS = ("low", "medium", "high", "xhigh", "max")
COMPLETE_STOPS = ("end_turn", "stop_sequence")

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

# Shapes of common credentials. A match stops the request before anything is sent.
SECRET_PATTERNS = (
    ("an Anthropic API key or token", re.compile(r"sk-ant-[a-z]+\d*-[A-Za-z0-9_\-]{16,}")),
    ("an OpenAI key", re.compile(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_\-]{20,}")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----")),
    ("a GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("an AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("a Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("a Stripe live key", re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{16,}")),
    ("a Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("an npm token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
)


class Refused(Exception):
    """Stopped before anything was sent."""


@dataclass
class Material:
    kind: str        # "file", "diff" or "brief"
    name: str
    text: str
    sha256: str = field(default="")

    @property
    def title(self) -> str:
        return f"{self.kind}: {self.name}"


def decode(raw: bytes, label: str) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):   # UTF-16, as Windows PowerShell 5.1 writes
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            raise Refused(f"{label} has a UTF-16 mark but isn't valid UTF-16") from None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise Refused(f"{label} isn't UTF-8 text") from None


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
    text = decode(raw, f"{kind} {name}")
    if "\x00" in text:
        raise Refused(f"{kind} {name} looks binary; pass text files only")
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
    key = (env_key or "").strip()
    for m in items:
        if len(key) >= 20 and key in m.text:
            raise Refused(f"{m.kind} {m.name} contains the API key this job runs with; remove it")
        if allow_key_like:
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(m.text):
                raise Refused(f"{m.kind} {m.name} contains what looks like {label}; redact it, "
                              "or pass --allow-key-like if it's a dummy value")


def check_writable(out: str, overwrite: bool) -> None:
    """Refuse before anything is paid for if the review couldn't be saved."""
    folder = os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(folder):
        raise Refused(f"the folder for --out ({folder}) doesn't exist")
    for p in (out, out + ".json"):
        if os.path.isdir(p):
            raise Refused(f"{p} is a folder")
        if os.path.exists(p) and not overwrite:
            raise Refused(f"{p} exists; pass --overwrite to replace it")
    try:
        fd, probe = tempfile.mkstemp(prefix=".review-write-test-", dir=folder)
        os.close(fd)
        os.unlink(probe)
    except OSError as e:
        raise Refused(f"can't write in {folder}: {e.strerror or e}") from None


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
        params["fallbacks"] = [{"model": t} for t in FALLBACK_TARGETS[model]]
    return params


def attempt_models(params: dict) -> list[str]:
    return [params["model"]] + [f["model"] for f in params.get("fallbacks") or []]


def price_key(model: str | None) -> str | None:
    """The PRICES entry for a served model name: exact, or that name plus a date suffix."""
    if not model:
        return None
    if model in PRICES:
        return model
    m = re.fullmatch(r"(.+)-\d{8}", model)
    return m.group(1) if m and m.group(1) in PRICES else None


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


def worst_case_cost(counts: dict[str, int], max_tokens: int) -> float:
    """The most the request can cost: one attempt on each model that may run, each billed
    for its input bound and max_tokens of output (thinking counts as output)."""
    total = 0.0
    for model, counted in counts.items():
        bound = billable_input_bound(counted)
        pin, pout = prices_for(model, bound)
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


def attempt_cost(model: str | None, u: dict) -> dict:
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


def usage_rows(message, requested_model: str) -> tuple[list[dict], bool]:
    """One row per model attempt, and whether the rows came from usage.iterations (which
    lists every attempt). Without iterations the top-level usage covers one attempt."""
    usage = _as_dict(getattr(message, "usage", None))
    iterations = usage.get("iterations") or []
    if iterations:
        rows = []
        for it in iterations:
            it = _as_dict(it)
            model = it.get("model") or (requested_model if it.get("type") == "message" else None)
            rows.append(attempt_cost(model, it))
        return rows, True
    return [attempt_cost(getattr(message, "model", None) or requested_model, usage)], False


def review_text(message) -> tuple[str, bool, bool]:
    """The review's text, whether a `fallback` block appeared, and whether text from a
    declined attempt was dropped. Text before the last `fallback` block is the declined
    model's partial output."""
    parts, saw_switch, dropped = [], False, False
    for b in message.content:
        kind = getattr(b, "type", "")
        if kind == "fallback":
            saw_switch = True
            dropped = dropped or any(p.strip() for p in parts)
            parts = []
        elif kind == "text":
            parts.append(getattr(b, "text", ""))
    return "".join(parts), saw_switch, dropped


def api_error_message(status: int | None, text: str) -> str:
    low = text.lower()
    if "credit balance is too low" in low or "workspace api usage limits" in low:
        name = os.environ.get("CLAUDE_CREDITS_ACCOUNT") or os.environ.get("CLAUDE_CREDITS_NAME") or "default"
        return ("Anthropic says this credit (or your share of it) is used up. If so, run: "
                f"claude-credits mark {name} exhausted (it checks first).")
    if status in (429, 529):
        return f"the API is rate-limited or overloaded ({status}); nothing was billed, so try again later"
    return f"the API returned an error{f' ({status})' if status else ''}: {text[:300]}"


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def run(args, client, *, stdin=None, env=None, clock=time.monotonic, now=None, err=None, out=None) -> int:
    stdin = stdin if stdin is not None else sys.stdin
    env = env if env is not None else os.environ
    err = err if err is not None else sys.stderr
    out = out if out is not None else sys.stdout
    if args.model not in MODELS:
        raise Refused(f"--model must be one of {', '.join(MODELS)}")
    if not (math.isfinite(args.cap) and args.cap > 0):
        raise Refused("--cap must be a number above 0")
    if args.max_tokens < 1:
        raise Refused("--max-tokens must be at least 1")
    brief, materials = load_inputs(args, stdin)
    check_secrets([brief, *materials], env.get("ANTHROPIC_API_KEY"), args.allow_key_like)
    if not args.dry_run:
        check_writable(args.out, args.overwrite)
    fallback = not args.no_fallback and bool(FALLBACK_TARGETS[args.model])
    params = build_request(brief, materials, args.model, args.effort, args.max_tokens,
                           fallback, not args.no_line_numbers)
    allowed = attempt_models(params)

    # Count for each model that may run: models can tokenize the same text differently.
    counts = {m: client.messages.count_tokens(model=m, system=params["system"],
                                              messages=params["messages"]).input_tokens
              for m in allowed}
    for m, n in counts.items():
        if billable_input_bound(n) + args.max_tokens > CONTEXT_WINDOW:
            raise Refused(f"{n:,} input tokens on {m} (plus the counting margin) and --max-tokens "
                          f"{args.max_tokens:,} exceed the {CONTEXT_WINDOW:,}-token context window; send less")
    worst = worst_case_cost(counts, args.max_tokens)
    counted = counts[args.model]
    print(f"{counted:,} input tokens, at most {args.max_tokens:,} output tokens: worst case "
          f"${worst:.2f} at list price{f' over {len(allowed)} possible attempts' if fallback else ''} "
          f"(cap ${args.cap:.2f})", file=err)
    if worst > args.cap:
        raise Refused(f"the worst case (${worst:.2f}) is above --cap ${args.cap:.2f}; lower "
                      "--max-tokens, send less, or raise --cap")
    if args.dry_run:
        print("dry run: nothing sent", file=err)
        return 0

    started = clock()
    # No SDK retries on the paid call: a retried request could be billed twice, past the cap.
    with client.with_options(max_retries=0).beta.messages.stream(**params) as stream:
        message = stream.get_final_message()
        request_id = getattr(stream, "request_id", None)
    seconds = clock() - started

    rows, itemized = usage_rows(message, args.model)
    text, saw_switch, dropped_partial = review_text(message)
    iter_fallback = any(r["type"] == "fallback_message" for r in rows)
    reported = getattr(message, "model", None)
    served = rows[-1]["model"] if itemized else reported
    fell_back = saw_switch or iter_fallback or (price_key(served) or served) != args.model
    if fell_back and not itemized:
        rows.append({"model": None, "type": "unreported", "input_tokens": 0, "output_tokens": 0,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "cost_usd": None})
        if (price_key(served) or served) == args.model:
            served = None
    known = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    cost = sum(known) if len(known) == len(rows) else None
    strays = sorted({str(r["model"]) for r in rows if r["model"] and (price_key(r["model"]) or r["model"]) not in allowed})
    over = cost is not None and cost > worst + 1e-9
    stop = getattr(message, "stop_reason", None)
    details = _as_dict(getattr(message, "stop_details", None))
    when = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")

    if stop == "refusal":
        status = (f"DECLINED (category: {details.get('category') or 'none given'})"
                  + ("; the fallback declined too" if fell_back else "") + ". There is no review below.")
        body, code = "", 3
    elif stop in COMPLETE_STOPS and text.strip():
        status, body, code = "complete", text, 0
    elif stop == "max_tokens":
        status, body, code = f"CUT OFF at --max-tokens {args.max_tokens:,}; the review below is incomplete.", text, 4
    elif not text.strip():
        status, body, code = f"NO REVIEW TEXT (stop reason: {stop}).", "", 4
    else:
        status, body, code = f"STOPPED EARLY (stop reason: {stop}); the review below may be incomplete.", text, 4

    writer = served or "unknown (a fallback ran, and the API didn't say which model)"
    inp = sum(r["input_tokens"] for r in rows)
    outp = sum(r["output_tokens"] for r in rows)
    attempts = "; ".join(f"{r['model'] or 'unreported attempt'} ({r['type']}): {r['input_tokens']:,} in, "
                         f"{r['output_tokens']:,} out, "
                         + (f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "cost unknown")
                         for r in rows)
    account = env.get("CLAUDE_CREDITS_ACCOUNT") or env.get("CLAUDE_CREDITS_NAME")
    header = [f"# Review: {os.path.basename(args.out)}", ""]
    if strays or over:
        header.append("- WARNING: " + "; ".join(
            ([f"a model outside the priced set ran ({', '.join(strays)})"] if strays else [])
            + ([f"the bill (${cost:.4f}) is above the worst case the cap was checked against (${worst:.4f})"]
               if over else [])))
    header += [
        f"- Status: {status}",
        f"- Asked: {args.model}, effort {args.effort}, max {args.max_tokens:,} output tokens"
        + ("" if fallback else ", no fallback"),
        f"- Written by: {writer}" + (" (a fallback, after the requested model declined"
                                     + ("; its partial text is left out" if dropped_partial else "") + ")"
                                     if fell_back and code != 3 else ""),
        f"- Tokens: {inp:,} input, {outp:,} output (thinking included)",
        "- Cost at list price: " + (f"${cost:.4f}" if cost is not None else "unknown (an attempt wasn't priced)"),
        f"- Attempts: {attempts}",
        f"- Request id: {request_id or 'not given'}",
    ]
    if account:
        header.append(f"- Paid from the credit of: {account}")
    header.append("- Inputs: " + ", ".join(f"{m.title} (sha256 {m.sha256[:12]})" for m in [brief, *materials]))
    header.append(f"- Finished: {when}, after {seconds:.0f} s")
    document = "\n".join(header) + "\n\n---\n\n" + body + ("\n" if body and not body.endswith("\n") else "")
    record = {
        "status": status, "exit_code": code, "requested_model": args.model, "served_model": served,
        "reported_model": reported, "fallback_enabled": fallback, "fell_back": fell_back,
        "allowed_models": allowed, "effort": args.effort, "max_tokens": args.max_tokens,
        "stop_reason": stop, "stop_details": details or None, "counted_input_tokens": counts,
        "worst_case_usd": round(worst, 6), "cap_usd": args.cap,
        "cost_usd": round(cost, 6) if cost is not None else None, "over_worst_case": over,
        "unpriced_models_ran": strays, "attempts": rows,
        "output_tokens_details": _as_dict(_as_dict(getattr(message, "usage", None)).get("output_tokens_details")) or None,
        "dropped_declined_partial": dropped_partial, "request_id": request_id, "account": account,
        "inputs": [{"kind": m.kind, "name": m.name, "sha256": m.sha256} for m in [brief, *materials]],
        "finished_at": when, "seconds": round(seconds, 1), "review_file": args.out,
    }
    record_text = json.dumps(record, indent=2) + "\n"
    try:
        _atomic_write(args.out, document)
        _atomic_write(args.out + ".json", record_text)
    except OSError as e:
        print(f"review: paid, but couldn't write {args.out} ({e.strerror or e}); "
              "the review and its record follow on stdout", file=err)
        out.write(document + "\n" + record_text)
        return 6
    print(f"{status} Written by {writer}; {inp:,} in, {outp:,} out; "
          + (f"${cost:.4f}" if cost is not None else "cost unknown") + f"; {args.out}", file=err)
    if strays or over:
        print(f"review: {header[2][2:]}", file=err)
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


def main(argv: list[str] | None = None, *, sdk=None, transport_errors=None) -> int:
    args = parser().parse_args(argv)
    if sdk is None:
        import anthropic as sdk  # imported here so the tests run without the SDK
    if transport_errors is None:
        try:
            import httpx2  # the SDK's HTTP library; a connection lost mid-stream raises its errors
            transport_errors = (httpx2.HTTPError,)
        except ImportError:
            transport_errors = ()
    client = sdk.Anthropic(max_retries=2, timeout=1800.0)
    try:
        return run(args, client)
    except Refused as e:
        print(f"review: {e}", file=sys.stderr)
        return 2
    except sdk.APIStatusError as e:
        print(f"review: {api_error_message(e.status_code, str(e))}", file=sys.stderr)
        return 5
    except (sdk.APIError, *transport_errors) as e:
        print(f"review: the request failed ({type(e).__name__}); nothing was written. A request "
              "cut off mid-stream may still be billed for what it produced.", file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
