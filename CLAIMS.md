# Claims

Every public statement about Housemate, with how much evidence stands behind it and where to check it.

| Tier | Meaning |
|---|---|
| **REPRODUCIBLE** | A script or test in this repository reproduces it. |
| **VERIFIED-LIVE** | Seen working on a real Hermes 0.23.0 gateway on our test machine; the steps to repeat it are given. |
| **MODELED** | Follows from the design or the code, but has not been measured or walked through end to end. |
| **NOT-CLAIMED** | Listed so nobody reads it into the docs. |

Test machine for every VERIFIED-LIVE line: Windows 11, Ryzen 5 5600X, RTX 4060 8 GB, Chrome, Hermes 0.23.0 on Python 3.14. The full setup is recorded in the `environment` block of each file in [results/](results/).

## Numbers

| Claim | Tier | Reproduce with |
|---|---|---|
| The answer starts <!-- num:latency_p50 -->2.6 s<!-- /num --> (median) and <!-- num:latency_p95 -->3.2 s<!-- /num --> (p95) after the end of speech, over <!-- num:latency_runs -->20<!-- /num --> warm runs. | REPRODUCIBLE | `hermes housemate bench latency` → [results/latency.json](results/latency.json) (raw runs, per-stage times, setup). One machine, one-word answers, satellite on the same PC as the hub. |
| Wake word recall <!-- num:wake_recall -->100 %<!-- /num --> on <!-- num:wake_recall_n -->58<!-- /num --> clean clips, <!-- num:wake_recall_noisy -->95 %<!-- /num --> with speech 10 dB lower in the background. | REPRODUCIBLE | `hermes housemate bench wake` → [results/wake.json](results/wake.json); corpus in [bench/corpus.md](bench/corpus.md), [bench/corpus.py](bench/corpus.py). Synthetic voices, not recordings of people. |
| <!-- num:wake_fa_per_hour -->0.11<!-- /num --> false wakes per hour (<!-- num:wake_false_accepts -->1<!-- /num --> in <!-- num:wake_negative_hours -->9.2<!-- /num --> h of LibriVox speech). | REPRODUCIBLE | Same run; the one event is listed with its file and second in `results/wake.json`. |
| <!-- num:tests_passed -->547<!-- /num --> tests pass against Hermes <!-- num:hermes_version -->0.23.0<!-- /num -->. | REPRODUCIBLE | `python -m pytest -m "not slow"` or `hermes housemate bench tests` → [results/tests.json](results/tests.json); CI in [.github/workflows/ci.yml](.github/workflows/ci.yml). |
| `hermes plugins validate --install-deps` passes: declared tools and hooks match, security scan safe, no core override. | REPRODUCIBLE | The "Catalog validation" step in CI; locally `hermes plugins validate <checkout> --install-deps`. |
| The first answer after a gateway restart takes about 7 s. | REPRODUCIBLE | `warm_up_s` in `results/latency.json` (6.4 s in the committed run; 6.5 to 7.5 s in earlier runs). |
| Whole-file voices start about 2.7 s later than `housemate-piper`; a streaming cloud voice about 1.4 s later. | VERIFIED-LIVE | Measured on the test machine with an earlier model. Repeat: change `tts.provider`, restart the gateway, run `hermes housemate bench latency`. Not in `results/`. |
| `end_silence_ms: 500` answers about 0.2 s sooner. | VERIFIED-LIVE | Same bench with the setting changed; it saves exactly the shorter silence wait. |
| The custom "Hey Hermes" head reached 89.4 % recall at 0.34 false wakes per hour at best, below our bar. | VERIFIED-LIVE | [wakeword/MODEL_CARD.md](wakeword/MODEL_CARD.md); recipe and evaluator in [wakeword/](wakeword/). The raw reports and weights are not in the repository. |

## Behaviour

| Claim | Tier | Where to check |
|---|---|---|
| The wake word and VAD run on the hub, per device; the phone only runs a web page. | REPRODUCIBLE | [core/listener.py](core/listener.py), [core/wake.py](core/wake.py), [web/](web/); tests in `tests/test_listener.py`, `tests/test_wake.py`. |
| The microphone stream stays in memory; only the question, from just after the wake word (or a held talk button or an open window) to the end of speech, is handed to Hermes. | REPRODUCIBLE | `tests/test_listener.py` (`test_wake_capture_starts_after_the_wake_word`, `test_speech_without_wake_or_window_is_ignored`); `HousemateAdapter._dispatch` in [adapter.py](adapter.py). |
| The nearest device answers: detections in one earshot group within 400 ms form one round; the highest wake score wins and close scores go to the louder device. | REPRODUCIBLE, VERIFIED-LIVE | `tests/test_arbiter.py`, `tests/test_listener.py::test_two_devices_in_earshot_only_the_winner_answers`; the Rooms page showing *answered* / *deferred* ([screenshot](docs/screens/rooms-arbitration-light.png)). |
| While a device in the group plays, the wake threshold rises and no follow-up window opens. | REPRODUCIBLE | `tests/test_listener.py::test_playback_raises_threshold_and_blocks_followup`, `tests/test_arbiter.py::test_group_playing_covers_the_whole_earshot_group`. |
| While one device's talk button is held, the other devices in its earshot group take no follow-up question. | REPRODUCIBLE, VERIFIED-LIVE | `tests/test_listener.py::test_holding_talk_on_one_device_closes_follow_up_windows_in_its_earshot`. Live, two windows on one PC: a Kitchen question held right after the Office had answered set one timer, in the Kitchen only. |
| One household conversation: a timer set in the kitchen is known in the office; each answer plays only in the room that asked. | REPRODUCIBLE, VERIFIED-LIVE | `tests/test_router.py`, `tests/test_adapter.py`. Live: two scripted satellites on a real gateway; the office asked which room the tea timer was in and heard "the kitchen". |
| A room that speaks mid-turn is queued and never interrupts the running turn or its approval. | REPRODUCIBLE, VERIFIED-LIVE | `tests/test_router.py::test_household_queue_holds_second_room_until_turn_end`, `tests/test_approvals.py::test_open_approval_holds_the_household_turn_past_its_timeout`. Live: the office question was answered after the kitchen's, with no interruption. |
| After a gateway that was killed or crashed mid-answer, the paused conversation comes back with one spoken word: "continue" (or a tap) drops the unfinished request, which is never re-run and stays in the history, and the waiting question is answered. Nothing is dropped until someone says it. | REPRODUCIBLE, VERIFIED-LIVE | `tests/test_restart_recovery.py`. Live on the test machine: gateway killed while an approval was open, then a question → spoken notice and Continue card; "Continue." (no wake word), a tap, and an Office question asked during the pause were all answered; the lost request stayed in the transcript. `hermes gateway restart --yes` interrupted a waiting approval cleanly and nothing paused. |
| Timers and alarms ring in their own room with no model call, repeat every 20 s for up to 3 minutes, survive a restart, and stop by "stop" or a tap. | REPRODUCIBLE, VERIFIED-LIVE | `tests/test_timers.py`, `tests/test_approvals.py::test_spoken_stop_while_ringing`. Live: a voice-set timer rang in the kitchen only and a bare "Stop." silenced it. |
| A spoken "approve" resolves Hermes's approval card by its request id; "deny" denies; no answer is Hermes's timeout, which denies. | REPRODUCIBLE, VERIFIED-LIVE | `tests/test_approvals.py` (`test_spoken_approve_goes_through_side_chat`, `test_spoken_deny`, `test_unanswered_approval_never_resolves_approve`). Live: "Approve." ran `rm -rf` on a scratch folder; "Deny." kept it. |
| Hedged or longer answers ("yes, but what does it delete?") never approve. | REPRODUCIBLE | `tests/test_intents.py`, `tests/test_approvals.py::test_hedged_spoken_answers_never_approve`. |
| Audio that overlaps the hub's own playback, a timer labelled "approve", and text the hub announced never count as an answer. | REPRODUCIBLE | `tests/test_approvals.py` (`test_answer_heard_while_another_room_plays_is_not_an_answer`, `test_timer_labelled_approve_cannot_approve`, `test_words_we_announced_are_not_an_answer`). |
| A command flagged by Hermes's smart approval can only be approved by a tap; taps count only from allowed devices in that room that were shown the card. | REPRODUCIBLE | `tests/test_approvals.py` (`test_smart_denied_approval_is_tap_only`, `test_tap_from_another_room_is_ignored`, `test_tap_from_a_device_outside_the_allowlist_is_ignored`). |
| Several approvals are asked and answered one at a time, in order. | REPRODUCIBLE | `tests/test_approvals.py::test_three_pending_approvals_are_answered_in_order`. |
| Clarify questions are read out and answered by voice, by tap or with "skip". | REPRODUCIBLE, VERIFIED-LIVE | `tests/test_approvals.py` (clarify tests). Live: voice, tap and skip on a real gateway. |
| "Tell the kitchen dinner's ready" is spoken in the kitchen with a chime and a caption. | REPRODUCIBLE, VERIFIED-LIVE | `tests/test_announce.py`, `tests/test_tools.py`. Live with scripted satellites. |
| Tapping another room on a speaker sends a voice message that is not saved and not sent to Hermes. | REPRODUCIBLE | `tests/test_intercom.py`; `HousemateAdapter._start_intercom` in [adapter.py](adapter.py). |
| Recipes and lists go on the room's screen as a card. | VERIFIED-LIVE | `housemate_show` in [tools.py](tools.py); [screenshot](docs/screens/satellite-card.png). |
| Replies stream to the asking device as soon as the voice produces audio; `housemate-piper` streams Piper sentence by sentence. | REPRODUCIBLE, VERIFIED-LIVE | `tests/test_playout.py`, `tests/test_tts_piper.py`; stage times `tts_begin` and `satellite_first_pcm` in `results/latency.json`. |
| Only paired devices are heard; an unpaired device gets a pairing code and nothing else. | REPRODUCIBLE | `tests/test_hub_ws.py` (`test_unknown_device_gets_pairing_code`, `test_unpaired_device_audio_ignored`, `test_bad_device_key_rejected`, `test_cross_origin_ws_refused`). |
| The control API listens on 127.0.0.1 only, needs a fresh token and refuses browsers. | REPRODUCIBLE | `tests/test_control.py` (`test_control_binds_loopback_only`, `test_control_rejects_missing_token`, `test_control_rejects_browser_origin`, `test_token_file_is_fresh_and_private`). |
| Model downloads come from the official releases and are pinned by SHA-256; no model file is committed. | REPRODUCIBLE | [core/models.py](core/models.py); `tests/test_models.py` (`test_manifest_pins_every_model`, `test_ensure_model_checksum_mismatch_raises`, `test_no_model_files_in_the_repo`). |
| No telemetry. The only network requests the plugin makes are the model downloads, a missing Piper voice, and the bench corpus when you run the bench. | REPRODUCIBLE | Read [core/models.py](core/models.py), [core/tts_piper.py](core/tts_piper.py), [bench/corpus.py](bench/corpus.py); a search for `urllib`, `ClientSession` and `requests` finds nothing else that leaves the machine. |
| The Rooms page uses only the Desktop plugin SDK and React. | REPRODUCIBLE | `tests/test_desktop_contract.py`; the "desktop surface" check of `hermes plugins validate`. |
| On Hermes 0.23.0 the Rooms page works through the plugin's REST routes. | VERIFIED-LIVE | Live in the 0.23.0 Webapp: arbitration display, drag-to-pair, announce, timer cancel and stop, mode switch saved to `config.yaml`. Tests: `tests/test_desktop.py::test_rest_routes_mirror_the_rpc_methods`. |
| On builds with plugin RPC the Rooms page uses `plugin.housemate.rooms.*`. | REPRODUCIBLE (unit only) | `tests/test_desktop.py` (`test_register_desktop_registers_every_rooms_method`, `test_rpc_handlers_forward_to_control_ops_with_only_their_params`). Not run on a live build with plugin RPC. |
| Two Chrome windows on one PC (`https://localhost:8742` and `https://127.0.0.1:8742`) act as two rooms. | VERIFIED-LIVE | [JUDGES.md](JUDGES.md) steps 0 to 6, walked on the test machine: two windows of one Chromium as Kitchen and Office in one earshot group, paired by typing their codes on the Rooms page of the 0.23.0 Webapp. One simulated microphone fed every spoken line to both windows at once, as with headphones on. One window answered while the Rooms page showed *answered* and *deferred*; the held-to-talk timer rang in the Kitchen only, with the Desktop notification; a bare "stop", "deny", a hedged answer that was asked again, "approve" and the intercom all worked. Browser storage is per address, so each window is its own device ([core/hub.py](core/hub.py)). |
| The satellite works on any phone, tablet or laptop with a current browser. | VERIFIED-LIVE (one phone) | Verified on a Samsung Galaxy A52 (Android 14, Chrome 156): pairing, wake word, spoken answers, timers and approval cards, over USB port-forwarding (`adb reverse`) because the test network isolates Wi-Fi clients. Also Chrome on Windows 11. Other phones are MODELED: the page needs HTTPS, `getUserMedia`, AudioWorklet and WebSocket. |
| Screenshots show the real UI. | VERIFIED-LIVE | Satellite screens: the web app in Chrome on the test machine. Desktop screens: the Hermes 0.23.0 Webapp, with rooms filled by scripted test satellites. |

## Not claimed

- **No "Hey Hermes" wake word.** The trained model missed our bar, so the default is openWakeWord's stock "Hey Jarvis".
- **No voice recognition.** Housemate does not know who is speaking; anyone in earshot of a paired speaker can talk to the agent.
- **"Nearest" is not a distance measurement.** It is the highest wake score, then the louder device.
- **Latency is not a general figure.** It is one machine, short answers, a warm gateway and the setup recorded in `results/latency.json`. Long answers, cold starts, other hardware and other providers differ.
- **Wake numbers are not from real homes.** They use synthetic voices and audiobook speech, not TV, music or children.
- **Not tested** on macOS, on Linux desktop audio, in Safari or Firefox, in the native Desktop app, or on a live Hermes build with plugin RPC.
- **No parallel answers** in household mode: one question at a time.
- **No access from outside the LAN.** No relay, no NAT traversal; a VPN or reverse proxy of your own is your choice.
- **No commercial-use wake pipeline.** openWakeWord's models are CC BY-NC-SA 4.0.
- **Spoken approve, deny, stop and skip words are English only.**
- **No defence against a hostile paired device** beyond the approval rules above. Paired devices are treated as household members; pair only devices you control.
- **Not a music player and not a smart-home driver.** Other Hermes plugins cover those, and Housemate makes them reachable by voice.

## Public surfaces used

Housemate extends Hermes only through documented surfaces. `hermes plugins validate` reports no private imports, no private attribute access and no core override, and `tests/test_hermes_contract.py` runs the same lints on every test run.

### Registrations (PluginContext)

| Surface | Where | Why it is public |
|---|---|---|
| `ctx.register_platform` | [`__init__.py:79`](__init__.py#L79) | Plugin platform API (adding-platform-adapters). Optional `PlatformEntry` fields are feature-tested with `dataclasses.fields`, as the guide shows. |
| `ctx.register_tts_provider` + `agent.tts_provider:TTSProvider` | [`__init__.py:83`](__init__.py#L83), [`core/tts_piper.py:179`](core/tts_piper.py#L179) | PluginContext registrar for TTS providers; `agent.tts_provider` is on the validator's public list. Registers `housemate-piper`. |
| `ctx.register_hook("pre_llm_call")` | [`__init__.py:100`](__init__.py#L100) | Documented hook, declared in `provides_hooks`. It acts only on the plugin's side chat. |
| `ctx.register_tool` | [`tools.py:301`](tools.py#L301) | Tool registration; the seven tools are declared in `provides_tools`. |
| `ctx.dispatch_tool` | [`tools.py:77`](tools.py#L77) | Lets the hub call Hermes's own `text_to_speech` tool for announcements and prompts. |
| `ctx.register_command` | [`commands.py:128`](commands.py#L128) | `/housemate` and `/announce`. |
| `ctx.register_cli_command` | [`commands.py:134`](commands.py#L134) | `hermes housemate …`. |
| `ctx.register_automation_blueprint` | [`commands.py:145`](commands.py#L145) | Morning, bedtime and reminder blueprints. |
| `ctx.register_skill` | [`commands.py:152`](commands.py#L152) | The bundled `housemate` skill. |
| `ctx.register_rpc` (when present) | [`desktop_api.py:66`](desktop_api.py#L66) | Plugin RPC for the Rooms page on builds that have it; feature-tested, absent on 0.23.0. |
| Dashboard plugin backend (`dashboard/manifest.json`, `plugin_api.py`, FastAPI `APIRouter`) | [`dashboard/plugin_api.py:20`](dashboard/plugin_api.py#L20) | The documented `/api/plugins/<id>` backend namespace shared by the Desktop and dashboard SDKs; the tab is hidden. |
| Desktop plugin SDK (`@hermes/plugin-sdk`, `react`) | [`desktop/plugin.js:35`](desktop/plugin.js#L35) | The only imports in `desktop/plugin.js`. |

### Imports of Hermes modules

| Import | Where | Why it is public |
|---|---|---|
| `gateway.config` (`Platform`, `PlatformConfig`) | [`adapter.py:38`](adapter.py#L38) | Public module on the validator's list (adapter guide). |
| `gateway.platforms.base` (`BasePlatformAdapter`, `AudioFormat`, `ExecApprovalPrompt`, `SendResult`, `StreamingTTSHandle`, `cache_audio_from_bytes_async`) | [`adapter.py:39`](adapter.py#L39) | "Adapter base class, send results, media cache helpers" in the adapter guide. |
| `gateway.platforms.event` (`MessageEvent`, `MessageType`) | [`adapter.py:47`](adapter.py#L47) | Public module. |
| `gateway.platforms._shared:get_scoped_secret` | [`adapter.py:2206`](adapter.py#L2206) | Public module; the adapter guide names it for profile-scoped settings. |
| `gateway.platform_registry:PlatformEntry` | [`__init__.py:71`](__init__.py#L71) | Public module; read only to feature-test optional fields. |
| `gateway.session_context:get_session_env` | [`tools.py:48`](tools.py#L48) | Named in the plugin guide for reading session identity from the task-local context; the tools use it to find the speaking room. |
| `tools.approval:resolve_gateway_approval_card` | [`adapter.py:1858`](adapter.py#L1858) | Adapter guide, "Resolve an approval from a native button". |
| `tools.clarify_gateway:resolve_gateway_clarify`, `SKIPPED` | [`adapter.py:1864`](adapter.py#L1864), [`adapter.py:2224`](adapter.py#L2224) | Adapter guide, "Native clarify buttons". |
| `tools.registry:registry` | [`tools.py:87`](tools.py#L87) | Public module; fallback for `dispatch_tool`. |
| `plugins.plugin_storage` (`plugin_data_dir`, `plugin_db`) | [`adapter.py:303`](adapter.py#L303), [`tools.py:34`](tools.py#L34), [`cli.py:137`](cli.py#L137), [`dashboard/plugin_api.py:41`](dashboard/plugin_api.py#L41) | The plugin's own data folder and SQLite database. |
| `hermes_cli.config` (`load_config`, `write_platform_config_field`) | [`adapter.py:2194`](adapter.py#L2194), [`core/tts_piper.py:43`](core/tts_piper.py#L43), [`core/desk.py:223`](core/desk.py#L223) | Public module. Reads the approval timeout and TTS voice; writes only `platforms.housemate.conversation`. |
| `hermes_cli.plugin_events:broadcast_plugin_event` | [`core/desk.py:242`](core/desk.py#L242) | Public module; pushes `plugin.housemate.rooms.changed` to the Rooms page. |
| `hermes_constants:get_hermes_home` | [`core/tts_piper.py:37`](core/tts_piper.py#L37) | Public module; locates Hermes's shared Piper voice cache. |
| Bench only, not loaded at runtime: `hermes_cli.config`, `plugins.plugin_storage` | [`bench/run.py:216`](bench/run.py#L216), [`bench/run.py:603`](bench/run.py#L603) | Public modules. |

### Adapter methods

Housemate's adapter is a subclass of `BasePlatformAdapter`. It overrides methods in its own class and never replaces anything on Hermes's classes or modules.

| Method | Where | Why |
|---|---|---|
| `connect`, `disconnect`, `send`, `edit_message`, `send_typing`, `stop_typing`, `get_chat_info`, `send_voice`, `on_processing_complete` | [`adapter.py:300`](adapter.py#L300), [`adapter.py:1199`](adapter.py#L1199), [`adapter.py:803`](adapter.py#L803) | The adapter contract. |
| `supports_streaming_tts`, `begin_streaming_tts`, `write_streaming_tts`, `finish_streaming_tts`, `abort_streaming_tts` | [`adapter.py:1297`](adapter.py#L1297) | Hermes's streaming TTS contract for adapters. |
| `_send_exec_approval_prompt` (override) | [`adapter.py:1649`](adapter.py#L1649) | The base class's seam for native approval buttons ("Render prompt with the platform's native buttons; the default has none"). |
| `send_clarify`, `retire_clarify_card` (override) | [`adapter.py:1672`](adapter.py#L1672), [`adapter.py:1738`](adapter.py#L1738) | Clarify contract for adapters with native choices. |
| `_should_auto_tts_for_chat` (override) | [`adapter.py:607`](adapter.py#L607) | Makes replies on this platform spoken by default (`speak_replies`, default on); a speaker has no other way to deliver an answer. Hermes resets the base class's auto-TTS default at start-up, so an override in our own subclass is the per-platform way to set it. Other platforms are untouched. |
| `prepare_tts_text` (override) | [`adapter.py:610`](adapter.py#L610) | Public base-class method that turns a reply into the script Hermes's voice reads. Ours says Hermes's "conversation is paused" notices in plain words instead of "send /discard"; every other reply goes to the base class unchanged. |
| `_mark_connected`, `_mark_disconnected` (called) | [`adapter.py:342`](adapter.py#L342), [`adapter.py:539`](adapter.py#L539) | Shown in the adapter guide's examples. |
| `_set_fatal_error` (called) | [`adapter.py:327`](adapter.py#L327) | Base-class helper that reports a port clash to the gateway as non-retryable. |
| `_event_session_key`, `_active_sessions` (read only) | [`adapter.py:765`](adapter.py#L765), [`adapter.py:768`](adapter.py#L768) | Base-class state read in our own subclass to wait until a household session is idle before handing over a queued question, so it never interrupts a running turn. Never written. A public "is this session busy" helper would replace it. |
| `build_source`, `handle_message` (called) | [`adapter.py:495`](adapter.py#L495) | The documented way to turn an inbound message into a Hermes turn. |
| `self.config.extra["allow_from"]` (set on our own config) | [`adapter.py:561`](adapter.py#L561) | The platform's own settings dict. Paired devices become the allowlist that Hermes's access check reads, so pairing grants access and revoking takes it away at once. In memory only. |

### Other notes for reviewers

- **TLS context.** Hermes swaps `ssl.SSLContext` for a client-only class at start-up. The hub builds its server context from the standard library's own class ([`core/hub.py:66`](core/hub.py#L66)) and does not modify Hermes's.
- **Spoken answers.** Approval and clarify answers go through a side chat ("Housemate answers") as voice events with a one-time marker in `channel_context`. Hermes's STT transcribes them, and the `pre_llm_call` hook resolves the prompt through the public resolvers and vetoes the side turn ([core/answers.py](core/answers.py)). No private transcription API is imported.
- **Paused conversations.** Hermes 0.23 pauses a conversation whose turn was lost in an unclean gateway exit and lifts it with the in-chat `/discard` command (GW-10). When the adapter receives that notice it asks for a spoken or tapped "continue", then sends `/discard` as an ordinary text message from the device that said it ([`adapter.py:982`](adapter.py#L982)), exactly as a person would type it in another chat. The only exception is the plugin's own answer side chat, which never holds a model turn and is unpaused at once. Hermes has no structured signal for the pause, so the adapter recognises it from the notice text ("⏸" and "/discard").
- **Model-initiated speech.** `housemate_announce` is speech on the user's own speakers inside the home, not an outbound message to a third party. Announced text is never accepted as an approval answer.
