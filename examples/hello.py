# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic"]
# ///
"""One request on your plan's API credit.

    claude-credits exec -- uv run examples/hello.py "What is a refundable tax credit?"

The Anthropic SDK reads ANTHROPIC_API_KEY, which `claude-credits exec` sets for this
process only. Refusal fallbacks are on: if a safety classifier declines the request, the
API reruns it on another model in the same call.
"""
import os
import sys

import anthropic

question = " ".join(sys.argv[1:]) or "In two sentences, what is a refundable tax credit?"
client = anthropic.Anthropic()
try:
    response = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": question}],
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
    )
except anthropic.BadRequestError as e:
    if "credit balance is too low" in str(e).lower() or "workspace api usage limits" in str(e).lower():
        name = os.environ.get("CLAUDE_CREDITS_NAME", "default")
        sys.exit(f"Anthropic says this credit (or your share of it) is used up. Run: claude-credits mark {name} exhausted")
    raise

if response.stop_reason == "refusal":
    sys.exit("The request was declined, and so was the fallback.")
print(next((b.text for b in response.content if b.type == "text"), ""))
print(f"\n[{response.model}: {response.usage.input_tokens} input, {response.usage.output_tokens} output tokens]",
      file=sys.stderr)
