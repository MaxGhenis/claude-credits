# claude-credits

Your Claude Max plan comes with monthly Claude API credit: $200 per billing cycle on Max
20x, $100 on Max 5x. Unused credit expires at the end of each cycle. `claude-credits`
keeps your API key in your operating system's keychain and runs your jobs on that
credit, so you can use it before it lapses.

## 1. Claim your credit (once, about 5 minutes)

1. On [claude.ai](https://claude.ai), open **Settings > Billing**. Under **API credits**,
   choose **Link organization**, create a new Console organization, and accept the terms.
   You need to have been on the plan for 7 days.
2. On [platform.claude.com](https://platform.claude.com), signed in with the same
   account, open **API keys** and create a key. Copy it.
3. **Don't add a payment method or auto-reload** to that organization. Without one, an
   empty balance just stops requests until the next cycle and you're never charged. With
   one, usage keeps going on your card.

Your credit appears under **Billing > Promotional credits** in the Console, with its
expiry date
([Anthropic's guide](https://platform.claude.com/docs/en/about-claude/api-credits-for-subscribers)).

## 2. Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv`
on a Mac). Then:

```bash
uv tool install git+https://github.com/MaxGhenis/claude-credits
```

`pipx install git+https://github.com/MaxGhenis/claude-credits` works too. For a downloaded
copy, run `uv tool install ./claude-credits`. It runs on macOS, Windows and Linux with
Python 3.10 or later.

## 3. Add your key

```bash
claude-credits add                      # paste the key at the hidden prompt
claude-credits set default --renews-day 17
```

`add` checks the key with Anthropic before storing it. The check is free, plus one
1-token test message that costs about a thousandth of a cent. The key then goes into your
macOS Keychain, Windows Credential Locker or Linux Secret Service.

Set `--renews-day` to the day of the month your plan renews (shown in claude.ai >
Settings > Billing), so `claude-credits list` can show when this cycle's credit expires.

On a server with no keychain, use `claude-credits add --file-store`. The key then sits in
a file only you can read (`~/.config/claude-credits/keys.json`).

## 4. Run jobs on it

`claude-credits exec` runs any command with `ANTHROPIC_API_KEY` set to your credit key,
for that command only:

```bash
claude-credits exec -- uv run examples/hello.py "What is a refundable tax credit?"
claude-credits exec -- uv run examples/batch.py prompts.txt --out results.jsonl
claude-credits exec -- python my_script.py
```

- **[`examples/hello.py`](examples/hello.py)** sends one request to Claude Opus 5.5.
- **[`examples/batch.py`](examples/batch.py)** sends a file of prompts (one per line)
  through the Message Batches API. It costs half the normal price, and most batches finish
  within an hour.
- **Your own code** with the Anthropic SDK needs no changes: `anthropic.Anthropic()`
  reads `ANTHROPIC_API_KEY` by itself.

Prices per million tokens (input/output), at standard rates: Opus 5.5 $4/$20, Sonnet 5.5
$2/$10, Haiku 5.5 $0.10/$0.50. Batch halves them. As a rough guide, $200 buys 25 million
input plus 5 million output tokens on Opus 5.5.

## What the credit pays for

Tested on 2026-10-22:

| Use | Covered? |
|---|---|
| Your own Messages API calls (Anthropic SDK, raw HTTP) | yes |
| Message Batches API | yes |
| [Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) (Anthropic-hosted agents with a sandbox), including tools | yes |
| `claude -p` or the Claude Agent SDK run with this key | **no**: "Credit balance is too low" |
| Claude Code, claude.ai or the desktop app | no (they use your plan, not this credit) |

Anthropic's docs say `claude -p` and the Agent SDK are covered, but every request that
carries their system prompt was refused in testing. `exec` warns when you run `claude`.

## When it runs out

Requests fail with "Your credit balance is too low", and nothing is charged. Run
`claude-credits mark default exhausted`. `list` then shows the key as exhausted until
your renewal day, and `check` re-tests it. The examples print that command for you.

## Keep it yours

Anthropic's [credit terms](https://www.anthropic.com/legal/credit-terms) let credits be
used only by the holder of the account and forbid transferring them. Use your key for
your own work, and don't give it to anyone else.

## How it keeps the key safe

- The key lives only in your OS keychain, or in the 0600 file if you chose
  `--file-store`.
- `~/.config/claude-credits/registry.json` (0600; `%APPDATA%\claude-credits` on Windows)
  holds what isn't secret: a short fingerprint of the key, your Console organization id,
  status, renewal day and rate limits.
- `exec` passes the key to one child process through its environment. It never prints
  the key and never puts it on a command line.
- `add` refuses things that aren't API keys: a `claude setup-token` token, an Admin key,
  or a legacy `sk-ant-api` key (unless you pass `--force`).

## Commands

| Command | What it does |
|---|---|
| `claude-credits add [NAME] [--clipboard \| --stdin] [--renews-day N] [--file-store]` | store a key (default name `default`) |
| `claude-credits list [--json]` | your keys, status and days until the credit expires |
| `claude-credits check [NAME...]` | re-test keys with Anthropic |
| `claude-credits exec [--name NAME] -- COMMAND...` | run a command on the credit |
| `claude-credits set NAME --renews-day N` | record the renewal day |
| `claude-credits mark NAME exhausted\|ok` | record that the credit ran out, or is back |
| `claude-credits remove NAME --yes` | delete a key |

With more than one key (say, a second plan), `exec` uses the one whose credit expires
soonest, then the least recently used.

## Development

```bash
uv run --no-project --with pytest --with hypothesis --with keyring pytest -q
CLAUDE_CREDITS_REAL_KEYRING=1 uv run --no-project --with pytest --with hypothesis --with keyring pytest -q -k real_keyring
```

The tests run against a fake API and an in-memory keychain. The second command also
round-trips a dummy key through your real keychain and deletes it.
