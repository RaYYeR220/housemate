# Model card: `hey_hermes` wake-word head

> **Status: did not pass. Not shipped.**
>
> - **Our bar:** recall ≥ 90 % on held-out positives **and** ≤ 0.5 false accepts per hour at one threshold, measured with Housemate's own runtime.
> - **Four training runs; none meets both halves.**
>   - Run 3 on the original held-out set: 89.4 % at 0.34 FA/h, or 92.1 % at 0.60 FA/h.
>   - Run 4 on the stricter, voice-disjoint held-out set (below): 77.6 % at 0.43 FA/h.
> - **Consequence:** `hey_hermes.onnx` is **not in this repository**, and Housemate keeps openWakeWord's stock `hey_jarvis` as the default wake word.
>
> This card documents the attempt so it can be reproduced or continued. The numbers come from runs on 2026-11-08/09 on an RTX 4060.

## Model

- **What it is:** an openWakeWord-format classification head for the phrase "Hey Hermes".
  - Input: `[1, 16, 96]`, 16 consecutive embeddings at 80 ms each (1.28 s of audio).
  - Output: a score in `[0, 1]`.
- **Architecture:** `Flatten → Linear(1536, 32) → LayerNorm → ReLU → [Linear(32, 32) → LayerNorm → ReLU] → Linear(32, 1) → Sigmoid`. openWakeWord's `dnn` model with `layer_size: 32`, exported to ONNX (opset 13).
- **Runtime:** it runs on top of openWakeWord's v0.5.1 `melspectrogram.onnx` and `embedding_model.onnx`, through Housemate's `core/wake.py`. The runtime is onnxruntime + numpy and is score-identical to openWakeWord 0.6.0. The head does not work on its own.
- **Recipe:** [README.md](README.md), with config [train_config.yaml](train_config.yaml).

## Training data (all synthetic or pre-computed; no recordings of real users)

- **Positives (run 3):** 55,000 clips of "hey hermes" plus 6,000 for validation.
  - All are voiced by the Piper LibriTTS-R multi-speaker generator: 25,000 from `train.py`'s 50/50 speaker mixes, and 30,000 from unmixed speakers covering all 904 voices.
  - **Run 4 adds** 2,200 Microsoft Edge neural TTS clips (×5 copies, 10 % to validation). They come from the 22 TRAIN voices of the split below and vary punctuation, rate (-20…+20 %) and pitch (-10…+10 Hz).
  - Speaking rates 0.75–1.25, stochastic durations, punctuation variants.
  - Augmented with EQ, distortion, pitch shift, band-stop filtering, coloured noise, room impulse responses (MIT), and background audio at -10 to 15 dB SNR. The background audio is AudioSet, FMA music, and 10 s chunks of LibriSpeech dev-clean read speech.
- **Adversarial negatives:** 25,000 clips plus 3,000 for validation, from the same generator. Run 4 adds 2,800 unmixed-speaker Piper clips and 439 Edge TRAIN-voice clips (×5) of the near-misses that fired in run 3: "hey, her mess", "hey herman", "hair mess", "hey, hermit", "hey, herb".
  - Phoneme-overlap phrases chosen automatically by openWakeWord.
  - 30 curated near-misses × 40 each, such as "hey herman", "hey hermit", "hair mess", "hey there miss", "hey her mess", "hermes" on its own, and "I bought a hermes scarf". The full list is in the config.
- **General negatives:** openWakeWord's pre-computed ACAV100M features (about 2,000 h of real-world speech, music and noise).
- **Checkpoint selection:** openWakeWord's ~11 h false-positive validation features (DiPCo, Santa Barbara Corpus, MUSDB).

## Evaluation

**Method.** Every number below comes from [`eval_wake.py`](eval_wake.py). It streams audio through Housemate's `core/wake.py` in 80 ms chunks from a fresh detector state, as a satellite does; these are not `train.py`'s metrics.

There are two held-out sets:
- **v1, runs 1–3:** all 47 English Edge voices. The Edge engine was not used in training then.
- **v2, runs 3–4:** once run 4 trained on Edge voices, the 47 voices were split into fixed TRAIN and TEST personas ([`edge_voices.py`](edge_voices.py)). v2 uses only the TEST voices.
  - A voice's "Multilingual" variant stays on the same side as its base voice.
  - The voices of the 14 earlier Housemate fixtures are TEST.
  - **TEST (25):** AU Natasha; CA Clara; GB Ryan, Sonia, Thomas; HK Sam; IE Emily; IN Prabhat; KE Asilia; NG Abeo; NZ Molly; PH James; SG Luna; TZ Elimu; US Andrew (+Multilingual), Aria, Brian (+Multilingual), Christopher, Emma (+Multilingual), Guy, Jenny; ZA Leah.
  - **TRAIN (22):** AU William (Multilingual); CA Liam; GB Libby, Maisie; HK Yan; IE Connor; IN Neerja (+Expressive); KE Chilemba; NG Ezinne; NZ Mitchell; PH Rosa; SG Wayne; TZ Imani; US Ana, Ava (+Multilingual), Eric, Michelle, Roger, Steffan; ZA Luke.
  - v2 positives: 25 TEST voices × 4 lines ("Hey Hermes." / "Hey Hermes, <command>" / "Hey, Hermes." / "Hey Hermes!") at random rate and pitch, plus the 14 fixtures. That is 114 clips and 228 trials.
  - v2 near-misses: 12 phrases × 5 TEST voices, 60 clips in all. The 3 older near-miss fixtures were dropped because their voices are TRAIN.
  - v2 is harder than v1: it is accent-heavy, and it shares the Edge engine (not the speakers) with run 4's training data.

The v1 set in detail:

- **Held-out positives:** 108 clips of "Hey Hermes" from Microsoft Edge neural TTS (no Edge voice is used in training).
  - 47 English voices across 14 locales (US, GB, AU, NZ, IE, CA, ZA, IN, SG, HK, PH, KE, NG, TZ), at rate -15/0/+15 % and pitch ±6 Hz.
  - Each voice says "Hey Hermes." and "Hey Hermes, <command>". There are also 14 earlier Housemate fixtures.
  - Each clip is scored twice: *clean* (over a -65 dBFS noise floor) and *speech-bg* (inside continuous LibriSpeech speech at a random 15–20 dB SNR). That makes 216 trials.
  - A clip is detected if any chunk ≥ threshold falls between its start and 1 s after its end.
  - No recordings of real people were available. All positives are synthetic speech from an engine and voices the model never saw.
- **Negatives:**
  - 11.21 h of LibriSpeech test-clean + test-other (177 chapter streams, public-domain LibriVox readings);
  - 25 min of AudioSet + FMA music/noise held out from training.

  A false accept is a crossing ≥ threshold, with 2 s ignored afterwards.
- **Near-misses:** 75 clips, scored like clean positives and reported as a count: "Hey Herman", "Hey, hermit", "Hair mess", "Hey there, miss", "Hermes.", "Hey, her mess", "Hey Hermione!", "Hey, Herb…", "I love my Hermes scarf", and others, across 6 voices each.
  - Several of these phrases are also training negatives, voiced there by Piper.

**Run 3 on v1 (head sha256 `debe67cc…`)**

| threshold | recall (all 216) | clean | speech-bg 15–20 dB | false accepts / h (11.6 h) | near-miss fires (of 75) |
|---|---|---|---|---|---|
| 0.10 | 92.1 % | 91.7 % | 92.6 % | **0.60** (7) | 9 |
| 0.15 | **89.4 %** | 88.0 % | 90.7 % | 0.34 (4) | 7 |
| 0.30 | 86.6 % | 87.0 % | 86.1 % | 0.34 (4) | 4 |
| 0.50 | 82.4 % | 84.3 % | 80.6 % | 0.26 (3) | 3 |
| 0.70 | 79.2 % | 82.4 % | 75.9 % | 0.17 (2) | 1 |
| 0.90 | 72.2 % | 76.9 % | 67.6 % | 0.17 (2) | 1 |

- **Latency.** Measured on the plain "Hey Hermes." clips, from end of speech to the first chunk ≥ 0.5: p50 100 ms, p95 300 ms, max 500 ms.
- **In-context check.** On the Housemate 10-minute test tracks (LibriVox reading with 14 spliced "hey hermes" plus 10 "hey jarvis" and 15 other phrases):
  - at 0.5: 12/14 detected and 0 false triggers;
  - at 0.3: 13/14.
- **Operating threshold.** If this head were used anyway, 0.15 gives the best recall within 0.5 FA/h. But the FA estimate rests on 4 events in 11.6 h (95 % Poisson interval roughly 0.1–0.9/h), so the margin is thin.

**Runs 3 and 4 on v2 (voice-disjoint)**

| threshold | run 3: recall (clean / speech-bg) | run 3: FA/h | run 4: recall (clean / speech-bg) | run 4: FA/h | near-miss fires r3 / r4 (of 60) |
|---|---|---|---|---|---|
| 0.05 | 84.6 % (87.7 / 81.6) | 1.38 | 80.7 % (86.0 / 75.4) | 0.60 | 10 / 3 |
| 0.10 | 79.4 % (81.6 / 77.2) | 0.60 | **77.6 %** (83.3 / 71.9) | 0.43 | 7 / 3 |
| 0.15 | **78.5 %** (80.7 / 76.3) | 0.34 | 76.8 % (83.3 / 70.2) | 0.34 | 7 / 2 |
| 0.30 | 75.4 % (78.1 / 72.8) | 0.34 | 74.1 % (80.7 / 67.5) | 0.34 | 7 / 2 |
| 0.50 | 73.2 % (76.3 / 70.2) | 0.26 | 71.9 % (78.1 / 65.8) | 0.26 | 6 / 1 |
| 0.70 | 71.1 % (74.6 / 67.5) | 0.17 | 69.7 % (74.6 / 64.9) | 0.17 | 5 / 1 |
| 0.90 | 64.5 % (66.7 / 62.3) | 0.17 | 64.5 % (68.4 / 60.5) | 0.09 | 2 / 0 |

- **Run 4 latency** (from end of speech, threshold 0.5): p50 140 ms, p95 390 ms.
- **Run 4 on the 10-minute tracks:** 12/14, 0 false triggers.
- **What changed.** Run 4 fixed the near-misses it was aimed at:

  | phrase | run 3 | run 4 |
  |---|---|---|
  | "Hey, her mess" | 0.98 | 0.004 |
  | "Hey Herman" | 0.67 | 0.006 |
  | "Hermes." | 0.41 | 0.015 |

  "Hey, hermit" now reaches 0.74, and "Hey, Herb" 0.41.
- **What did not change.** Recall on unseen speakers of the weak accents stayed where it was. Edge TRAIN voices from the same locales did not transfer to other speakers of those locales.

**Training history**

| run | change | recall @0.5 (clean / speech-bg) | best recall with FA/h ≤ 0.5 | FA/h @0.5 |
|---|---|---|---|---|
| 1 | upstream recipe: 25k positives, 30 curated near-misses | 67.6 % / 45.4 % | 72.2 % (thr 0.05) | 0.09 |
| 2 | + 15k unmixed-speaker positives, 100 positives/batch, negative weight 1000 | 85.2 % / 59.3 % | 79.6 % (thr 0.10; clean 90.7 %) | 0.26 |
| 3 | + 15k more unmixed positives, + LibriSpeech dev-clean background speech | 84.3 % / 80.6 % | **89.4 %** (thr 0.15) | 0.26 |
| 4 (v2 set) | + 2.2k Edge TRAIN-voice positives ×5, + 5.3k near-miss negatives | 78.1 % / 65.8 % (r3 on v2: 76.3 / 70.2) | 77.6 % (thr 0.10; r3 on v2: 78.5 % at 0.15) | 0.26 |

Runs 1–3 took 60–80 min each on the RTX 4060: about 20 min of clip generation (run 1 only), 25–40 min of augmentation and feature extraction, and 25–37 min of training. Run 4 took about 2 h 10 min, because another job was saturating the host CPU during its augmentation.

**Where it fails**

- **Accents.** Misses concentrate on non-US English voices. On v1 these were AU (Natasha, William), NZ (Mitchell, Molly), the GB child voice (Maisie), IN (Prabhat, Neerja), ZA (Luke), KE, NG, TZ and SG. On v2 (run 4) they are IN Prabhat, NZ Molly, AU Natasha, GB Thomas, NG Abeo, KE Asilia, TZ Elimu and PH James. US, IE and CA voices are almost always detected.
  - The Piper LibriTTS-R generator is mostly North-American read speech. Run 2 showed that unmixed speakers help, but synthetic coverage of these accents stays thin.
- **Known false triggers (run 3, LibriSpeech).** "…HE HURRIED FORTH…" (0.83), "…OBTAIN REMISSION…" (0.57), "…SOME HOURS AFTER…" (0.55), and "…HE HURLED HIMSELF…" (0.35). All are "h + er/ur + m/s" sequences.
- **Near-miss phrases that fire.**

  | phrase | worst-case score |
  |---|---|
  | "Hey, her mess" | 0.98 |
  | "Hey Herman" (one voice) | 0.67 |
  | "Hey, Herb, is that yours?" | 0.58 |
  | "Hermes." alone | 0.41 |
  | "Hey Hermione!" | 0.16 |

  "Hair mess", "Hey there, miss", "Hey, hermit" and "Her messages…" stay below 0.02.

- **A structural conflict.** In non-rhotic accents (GB, AU, NZ, ZA, IN, much of Africa), "Hermes" is /hɜːmiːz/ and "her mess" is /hɜː mes/. They differ only in the last vowel and the final s/z. Suppressing "hey, her mess" as a hard negative therefore works against exactly the accents that are missed.

**Next steps if this is picked up again:**
- real recordings of the target users (even 20–50 people) as positives;
- Piper voices with non-US speakers (for example `en_GB-vctk`) next to LibriTTS-R;
- a larger near-miss set for "her mess" / "Herman" / "Herb".

The raw reports (`r1.json`–`r3.json`, per-chunk scores) are kept with the training data, outside the repo.

## Licence

The head's weights are our own training output, but several inputs carry share-alike, non-commercial terms. **The head is therefore released under CC BY-NC-SA 4.0 and must be treated as non-commercial.** It is not under the MIT licence of the code in this repository.

| Input | Licence | Effect on the head |
|---|---|---|
| ACAV100M pre-computed features (general negatives) | CC BY-NC-SA 4.0 | Training data under NonCommercial + ShareAlike, so the head inherits **BY-NC-SA** |
| False-positive validation features (checkpoint selection) | CC BY-NC-SA 4.0 | same |
| openWakeWord melspectrogram + embedding models (v0.5.1) | CC BY-NC-SA 4.0. The README covers "all included pre-trained models"; the embedding network derives from Google's Apache-2.0 `speech_embedding` | The head is trained on, and only works on top of, these features. Housemate downloads them at runtime with a SHA-256 pin and never vendors them |
| openWakeWord training code | Apache-2.0 | none beyond attribution |
| piper-sample-generator v2.0.0 | MIT | none |
| LibriTTS-R (data behind the Piper generator) | CC BY 4.0 | attribution |
| AudioSet clips (background augmentation) | dataset CC BY 4.0; the audio itself comes from YouTube uploads | possible third-party rights in the audio; augmentation input only |
| FMA `fma_small` clips (background augmentation) | per-track Creative Commons licences, some NC | augmentation input only; consistent with NC |
| LibriSpeech dev-clean (background speech augmentation) | CC BY 4.0 (public-domain LibriVox recordings) | attribution |
| MIT impulse responses (reverb augmentation) | "unknown" per the dataset card | augmentation input only |
| Microsoft Edge neural TTS clips (run 4 training; v1/v2 evaluation) | generated through `edge-tts`, an unofficial client of the Edge browser's read-aloud service. Microsoft publishes no explicit licence or terms for training on that output | **unclear**: check before any use beyond a non-commercial hobby model; runs 1–3 used Edge audio for evaluation only |

Attribution:
- openWakeWord by David Scripka (github.com/dscripka/openWakeWord);
- Piper and piper-sample-generator by Michael Hansen / Rhasspy;
- LibriTTS-R (Koizumi et al., 2023);
- LibriSpeech (Panayotov et al., 2015);
- ACAV100M (Lee et al., 2021);
- AudioSet (Gemmeke et al., 2017);
- FMA (Defferrard et al., 2017);
- MIT IR Survey (Traer & McDermott, 2016).

The whole wake pipeline is non-commercial either way. Housemate's default stock head, openWakeWord's `hey_jarvis`, and the feature models it runs on are all CC BY-NC-SA 4.0.
