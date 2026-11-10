# Review Housemate in five minutes

You need Hermes 0.23.0 or newer, Chrome and a microphone (headphones help). Hermes must have speech-to-text turned on (`stt.enabled: true`, any provider; local faster-whisper works) and a text-to-speech voice. No phone is needed: two browser windows on one PC are two rooms.

Every claim below is also listed in [CLAIMS.md](CLAIMS.md) with the file or script that backs it. Screenshots are in [docs/screens/](docs/screens/).

## 0. Install (a few minutes)

```bash
hermes plugins install housemate
hermes plugins enable housemate
hermes config set platforms.housemate.enabled true
hermes config set webapp.enabled true
hermes housemate rooms group Kitchen desk    # both windows hear one microphone:
hermes housemate rooms group Office desk     # one earshot group lets the hub pick one
hermes gateway restart
hermes webapp                                # prints a one-use sign-in link
```

The dependency install takes a minute or two, and the first `hermes webapp` may build the Webapp first (about five minutes on our test machine).

Hermes ships a plugin's Desktop page switched off. In the Webapp open **Capabilities → Plugins**, click **Housemate** and switch **Desktop** on. **Rooms** then appears in the sidebar.

## 1. Two rooms on one PC

Open `https://localhost:8742` in one Chrome window and `https://127.0.0.1:8742` in another, side by side. Accept the certificate warning in each (the fingerprint is on the Rooms page). Name one "Kitchen tab" in **Kitchen** and the other "Office tab" in **Office**. Each window shows a six-digit code: on the Rooms page press **Approve** next to each request and type that window's code. Then press **Tap to start** in each.

**Look for:** both devices appear on the Rooms page in their rooms, with overlapping earshot circles and the label "Within earshot · Kitchen + Office".

## 2. The arbitration moment

Say **"Hey Jarvis, what's two plus two?"**

**Look for:** both windows heard it, exactly one answers out loud ("Four."), and for about three seconds the Rooms page marks one device *answered* and the other *deferred*. Only the answering window shows the caption. Both windows hear the same microphone at the same level here, so either one may win. The hub settles this within 400 ms ([core/arbiter.py](core/arbiter.py), [tests/test_arbiter.py](tests/test_arbiter.py)).

From here on, **hold the talk button** in a window to talk to that room. No wake word is needed.

## 3. One household conversation

- Kitchen (hold to talk): "Set a tea timer for twenty seconds."
- Office (hold to talk): "Which room is the tea timer in?"

**Look for:** the office answers "the kitchen". Both rooms share one Hermes session, the room travels with each question, and each answer plays only where it was asked.

## 4. A timer rings where it was set

Twenty seconds after step 3, the **Kitchen** window rings: a chime, "Your tea timer is done." and the label in large type. The Rooms page shows the timer ringing, and Desktop shows a notification.

**Look for:** the Office window stays silent. Say **"stop"** without the wake word, or tap the Kitchen window. No model call is involved in the ring ([core/timers.py](core/timers.py)).

## 5. A spoken approval

Run `mkdir ~/housemate-demo` (on Windows, `mkdir %USERPROFILE%\housemate-demo`), then in the Kitchen: "Delete the housemate-demo folder in my home directory."

**Look for:** with Hermes's command approvals on, the Kitchen window shows a card with the whole command, exactly as Hermes will run it (a long one scrolls inside the card), and Approve / Deny, and reads the question out loud. Answer after the question ends (no talk button needed for the answer):

- Say **"deny"**: Hermes reports it was denied and the folder is still there.
- Ask again (hold to talk) and say **"yes, but what does it do?"**: that is not a clean answer, so you are asked again.
- Say **"approve"**: the folder is gone.
- Say nothing: Hermes's approval timeout denies it.

The spoken answer goes through a side chat, Hermes's own STT and the plugin's `pre_llm_call` hook, which resolves that card by its request id with `resolve_gateway_approval_card`, naming the device that answered ([adapter.py](adapter.py), [core/answers.py](core/answers.py), [core/intents.py](core/intents.py); tests in [tests/test_approvals.py](tests/test_approvals.py), [tests/test_answers.py](tests/test_answers.py), [tests/test_intents.py](tests/test_intents.py)).

## 6. Intercom

Office (hold to talk): "Tell the kitchen dinner's ready."

**Look for:** a chime in the Kitchen window, the caption "Dinner is ready." and the spoken message, while the Office hears a short confirmation.

## 7. Measure it yourself

With the gateway running:

```bash
hermes housemate bench latency --runs 5 --out ./bench-out
hermes housemate bench wake --max-hours 0.5 --no-generate --out ./bench-out
```

- `latency` pairs a scripted device called "Bench satellite" into the first room, speaks a wake-word question five times (after one warm-up), and prints p50 / p95 and the time spent in each stage. Our run of 20: p50 <!-- num:latency_p50 -->2.6 s<!-- /num -->, p95 <!-- num:latency_p95 -->3.2 s<!-- /num --> ([results/latency.json](results/latency.json)). Your numbers depend on your STT, voice and model.
- `wake` scores the committed wake clips and counts false wakes on the first half hour of the LibriVox set (it downloads those files). Our full run: <!-- num:wake_recall -->100 %<!-- /num --> recall clean, <!-- num:wake_fa_per_hour -->0.11<!-- /num --> false wakes per hour ([results/wake.json](results/wake.json)).
- `hermes housemate revoke housemate-bench` removes the bench device afterwards.

## 8. Tests and validation

The tests are in [tests/](tests/): <!-- num:tests_passed -->566<!-- /num --> pass against the real Hermes <!-- num:hermes_version -->0.23.0<!-- /num --> ([results/tests.json](results/tests.json)). From a checkout:

```bash
python -m pytest -m "not slow"
```

Good places to start:

| File | What it pins down |
|---|---|
| `tests/test_arbiter.py` | One winner per earshot group, close scores go to the louder device, separate groups each answer. |
| `tests/test_router.py` | Household and per-room routing, the queue, a turn held while its prompt is open. |
| `tests/test_approvals.py`, `tests/test_answers.py`, `tests/test_intents.py` | Spoken and tapped approvals, hedged answers never approve, no answers from our own audio, taps only from the right room, "stop" while a timer rings. |
| `tests/test_timers.py` | Timers ring once in their room, repeat every 20 s, stop after 3 minutes, survive a restart. |
| `tests/test_hub_ws.py`, `tests/fake_satellite.py` | The satellite protocol end to end over a real WebSocket, pairing and refusals. |
| `tests/test_hermes_contract.py` | Runs Hermes's own private-import and private-attribute lints on this repository. |
| `tests/test_desktop_contract.py` | The Rooms page imports only the Desktop SDK and React, and passes Hermes's desktop surface lint. |
| `tests/test_manifest.py`, `tests/test_tools.py` | The declared tools and every env setting read match the manifest; tools never raise. |

Catalog validation:

```bash
hermes plugins validate <path to the checkout> --install-deps
```

Expected output (abridged):

```text
✓ python dependencies — 5 requirements from pyproject
✓ capability probe — register() ran in isolation
✓ declared tools — matches registrations
✓ declared hooks — matches registrations
✓ built-in tool collisions — no collisions
✓ security scan — safe
✓ no core override — no runtime rebinds of Hermes core
◇ Isolation — in-process only: kind 'platform': gateway platform adapters run
in the Hermes process

Validation passed.
```

CI runs the same suite, `hermes plugins doctor` and `hermes plugins validate --install-deps` against Hermes v0.23.0 on Linux ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Where to look in the code

| Part | File |
|---|---|
| Gateway platform (inbound voice, streamed replies, prompts, queue) | [adapter.py](adapter.py) |
| LAN hub, pairing, satellite protocol | [core/hub.py](core/hub.py), [core/protocol.py](core/protocol.py) |
| Wake word, VAD, per-device listener | [core/wake.py](core/wake.py), [core/vad.py](core/vad.py), [core/listener.py](core/listener.py), [core/endpointer.py](core/endpointer.py) |
| Arbitration and echo guard | [core/arbiter.py](core/arbiter.py) |
| Household routing and queue | [core/router.py](core/router.py) |
| Timers and alarms | [core/timers.py](core/timers.py) |
| Spoken answers side chat and hook | [core/answers.py](core/answers.py), [core/intents.py](core/intents.py) |
| Streamed Piper voice | [core/tts_piper.py](core/tts_piper.py) |
| Agent tools, commands, CLI, blueprints | [tools.py](tools.py), [commands.py](commands.py), [cli.py](cli.py) |
| Rooms page | [desktop/plugin.js](desktop/plugin.js), [dashboard/plugin_api.py](dashboard/plugin_api.py), [desktop_api.py](desktop_api.py) |
| Satellite web app | [web/](web/) |
| Bench and number rendering | [bench/run.py](bench/run.py), [scripts/render_numbers.py](scripts/render_numbers.py) |
| "Hey Hermes" training recipe | [wakeword/](wakeword/) |
