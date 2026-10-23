# claude-credits

[![tests](https://github.com/MaxGhenis/claude-credits/actions/workflows/test.yml/badge.svg)](https://github.com/MaxGhenis/claude-credits/actions/workflows/test.yml)

Your Claude Max plan comes with monthly Claude API credit: $200 per billing cycle on Max
20x, $100 on Max 5x. Unused credit expires at the end of each cycle. `claude-credits`
keeps your API key in your operating system's keychain and runs your jobs on that
credit, so you can use it before it lapses.

## 1. Claim your credit (once, about 5 minutes)

1. On [claude.ai](https://claude.ai), open **Settings > Billing**. Under **API credits**,
   choose **Link organization**, create a new Console organization, and accept the terms.
   You need to have been on the plan for 7 days. You can't change the linked organization
   yourself afterwards (only support can), so a new one just for this is simplest.
2. On [platform.claude.com](https://platform.claude.com), signed in with the same
   account, open **API keys** and create a key in that organization. Copy it. The Console
   shows a key only once.
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

`pipx install git+https://github.com/MaxGhenis/claude-credits` works too. If someone sent
you the `.whl` file, run `uv tool install claude_credits-0.2.0-py3-none-any.whl`. It runs
on macOS, Windows and Linux with Python 3.10 or later.

## 3. Add your key

```bash
claude-credits add                      # paste the key at the hidden prompt
claude-credits set default --renews-day 17
```

`add` checks the key with Anthropic before storing it. The check is free, plus one
1-token test message that costs about a thousandth of a cent. If Anthropic can't be
reached, nothing is stored. The key then goes into your macOS Keychain, Windows
Credential Locker or Linux Secret Service.

Set `--renews-day` to the day of the month your plan renews (shown in claude.ai >
Settings > Billing), so `claude-credits list` can show when this cycle's credit expires.

A key spends whichever organization it came from. Make sure yours came from the
organization you linked to your Max plan, not an employer's organization or one with a
card on file.

On a server with no keychain, use `claude-credits add --file-store`. The key then sits in
`~/.config/claude-credits/keys.json`, readable only by you. On Windows, that file is
protected by your user profile's folder permissions.

## 4. Run jobs on it

`claude-credits exec` runs any command with `ANTHROPIC_API_KEY` set to your credit key.
The key reaches that command and any programs it starts, and nothing else. It first sends
a 1-token test request so a dead key fails fast; `--no-verify` skips that.

```bash
claude-credits examples                  # copies two example jobs into ./claude-credits-examples
claude-credits exec -- uv run claude-credits-examples/hello.py "What is a refundable tax credit?"
claude-credits exec -- uv run claude-credits-examples/batch.py prompts.txt --out results.jsonl
claude-credits exec -- python my_script.py
```

- **`hello.py`** sends one request to Claude Opus 5.5.
- **`batch.py`** sends a file of prompts (one per line) through the Message Batches API.
  It costs half the normal price, and most batches finish within an hour.
- **Your own code** with the Anthropic SDK needs no changes: `anthropic.Anthropic()`
  reads `ANTHROPIC_API_KEY` by itself.

Prices per million tokens (input/output), at standard rates: Opus 5.5 $4/$20, Sonnet 5.5
$2/$10, Haiku 5.5 $0.10/$0.50 for prompts up to 100K tokens ($0.50/$2.50 beyond). Batch
halves them. As a rough guide, $200 buys 25 million input plus 5 million output tokens on
Opus 5.5.

## What the credit pays for

Tested on 2026-10-22:

| Use | Covered? |
|---|---|
| Your own Messages API calls (Anthropic SDK, raw HTTP) | yes |
| Message Batches API | yes |
| [Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) (Anthropic-hosted agents with a sandbox), including tools | yes |
| `claude -p` or the Claude Agent SDK run with this key | **no**: "Credit balance is too low" |
| Claude Code, claude.ai or the desktop app | no (they use your plan, not this credit) |

Anthropic says otherwise for `claude -p` and the Agent SDK. Its
[help center](https://support.claude.com/en/articles/17154008-monthly-api-credits-for-max-and-team-plans)
says both are covered when run with an API key from the linked organization. In testing,
though, every request carrying their system prompt was refused. `exec` warns when you
run `claude`.

## When it runs out

Requests fail with "Your credit balance is too low", and nothing is charged. Run
`claude-credits mark default exhausted`. It sends one neutral test request first and
refuses if that works, because the same error also comes back for requests the credit
won't pay for (above), not only for an empty balance.

`list` then shows the key as exhausted until your renewal day. If it ran out on the
renewal day itself, it's re-tested the next day. The examples print the command for you.

## Keep it yours

Anthropic's [credit terms](https://www.anthropic.com/legal/credit-terms) let credits be
used only by the holder of the account and forbid transferring them. Use your key for
your own work, and don't give it to anyone else.

## How it keeps the key safe

- The key lives only in your OS keychain, or in the private file if you chose
  `--file-store`.
- `~/.config/claude-credits/registry.json` (`%APPDATA%\claude-credits` on Windows) holds
  what isn't secret: a short fingerprint of the key, your Console organization id,
  status, renewal day and rate limits.
- `exec` passes the key through the environment. It never prints the key and never puts
  it on a command line.
- The key is only ever sent to `https://api.anthropic.com`, and redirects are never
  followed.
- `add` refuses things that aren't API keys: a `claude setup-token` token, an Admin key,
  or a legacy `sk-ant-api` key (unless you pass `--force`). It won't replace a stored key
  without `--replace`.
- The hidden prompt keeps the key off your screen. If you pasted it there, your clipboard
  still holds it. `--clipboard` reads the key from the clipboard and clears it after
  storing; otherwise copy something else afterwards. Clipboard history tools (Win+V,
  Paste, Klipper) may keep a copy either way. If a terminal can't hide input, `add`
  refuses rather than showing the key.
- `remove` deletes the key from this computer only. To revoke it, delete it at
  platform.claude.com > API keys.

## Commands

| Command | What it does |
|---|---|
| `claude-credits add [NAME] [--clipboard \| --stdin] [--renews-day N] [--file-store] [--replace]` | store a key (default name `default`) |
| `claude-credits list [--json]` | your keys, status and days until the credit expires |
| `claude-credits check [NAME...]` | re-test keys with Anthropic |
| `claude-credits exec [--name NAME] [--no-verify] -- COMMAND...` | run a command on the credit |
| `claude-credits examples [DIR]` | copy the example jobs into a folder |
| `claude-credits set NAME --renews-day N` | record the renewal day |
| `claude-credits mark NAME exhausted\|ok` | record that the credit ran out, or is back |
| `claude-credits remove NAME --yes` | delete a key from this computer |

With more than one key, say from a second plan (`claude-credits add plan2`), `exec` uses
the one whose credit expires soonest, then one known to work, then the least recently
used. If none works, it re-tests the ones recorded as exhausted before giving up.

## Development

```bash
uv run --no-project --with pytest --with hypothesis --with keyring pytest -q
CLAUDE_CREDITS_REAL_KEYRING=1 uv run --no-project --with pytest --with hypothesis --with keyring pytest -q -k real_keyring
```

The tests run against a fake API and an in-memory keychain. The second command also
round-trips a dummy key through your real keychain and deletes it.
