# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic"]
# ///
"""Run many prompts through the Message Batches API, which costs half the normal price.

    claude-credits exec -- uv run examples/batch.py prompts.txt --out results.jsonl

`prompts.txt` holds one prompt per line. The script submits them as one batch, waits for
it to finish (most finish within an hour, all within 24), and writes one JSON line per
prompt: {"line": n, "prompt": ..., "text": ...} or {"line": n, "prompt": ..., "error": ...}.
If you stop waiting, pick the results up later with --results BATCH_ID.
"""
import argparse
import json
import os
import sys
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("prompts", nargs="?", help="file with one prompt per line")
p.add_argument("--out", help="write results here as JSON lines (default: stdout)")
p.add_argument("--model", default="claude-opus-5-5")
p.add_argument("--system", help="system prompt for every request")
p.add_argument("--max-tokens", type=int, default=16000)
p.add_argument("--results", metavar="BATCH_ID", help="fetch the results of an earlier batch")
p.add_argument("--poll", type=int, default=30, help="seconds between status checks")
args = p.parse_args()
client = anthropic.Anthropic()

if args.results:
    batch_id, prompts = args.results, None
else:
    if not args.prompts:
        p.error("give a prompts file, or --results BATCH_ID")
    with open(args.prompts, encoding="utf-8-sig") as fh:  # not the locale's encoding (cp1252 on Windows)
        prompts = [line.strip() for line in fh if line.strip()]
    params = {"model": args.model, "max_tokens": args.max_tokens}
    if args.system:
        params["system"] = args.system
    try:
        batch = client.messages.batches.create(requests=[
            Request(custom_id=f"line-{i}", params=MessageCreateParamsNonStreaming(
                **params, messages=[{"role": "user", "content": prompt}]))
            for i, prompt in enumerate(prompts, 1)])
    except anthropic.BadRequestError as e:
        if "credit balance is too low" in str(e).lower() or "workspace api usage limits" in str(e).lower():
            name = os.environ.get("CLAUDE_CREDITS_NAME", "default")
            sys.exit(f"Anthropic says this credit (or your share of it) is used up. Run: claude-credits mark {name} exhausted")
        raise
    batch_id = batch.id
    print(f"submitted {len(prompts)} prompts as batch {batch_id}", file=sys.stderr)

while True:
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status == "ended":
        break
    print(f"{batch.processing_status}: {batch.request_counts.processing} still processing "
          f"(Ctrl-C is safe; resume with --results {batch_id})", file=sys.stderr)
    time.sleep(args.poll)

out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
counts = {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
for result in client.messages.batches.results(batch_id):  # arrives in any order: key by custom_id
    n = int(result.custom_id.split("-", 1)[1])
    row = {"line": n}
    if prompts:
        row["prompt"] = prompts[n - 1]
    counts[result.result.type] += 1
    if result.result.type == "succeeded":
        msg = result.result.message
        row["text"] = next((b.text for b in msg.content if b.type == "text"), "")
        row["stop_reason"] = msg.stop_reason
    elif result.result.type == "errored":
        err = getattr(result.result.error, "error", result.result.error)  # ErrorResponse wraps the error
        row["error"] = f"{getattr(err, 'type', '')}: {getattr(err, 'message', '')}"
    else:
        row["error"] = result.result.type
    out.write(json.dumps(row) + "\n")
if args.out:
    out.close()
print(", ".join(f"{v} {k}" for k, v in counts.items() if v), file=sys.stderr)
