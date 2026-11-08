# Model card: `hey_hermes` wake-word head

> **Status: did not pass. Not shipped.**
>
> - **Our bar:** recall ≥ 90 % on held-out positives **and** ≤ 0.5 false accepts per hour, measured with Housemate's own runtime.
> - **Best candidate (run 3):**
>   - at threshold 0.15: 89.4 % recall at 0.34 FA/h;
>   - at threshold 0.10: 92.1 % recall at 0.60 FA/h;
>   - no threshold meets both halves.
> - **Consequence:** `hey_hermes.onnx` is **not in this repository**, and Housemate keeps openWakeWord's stock `hey_jarvis` as the default wake word.
>
> This card documents the attempt so it can be reproduced or continued. The numbers come from 2026-11-08 runs on an RTX 4060.

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
  - Speaking rates 0.75–1.25, stochastic durations, punctuation variants.
  - Augmented with EQ, distortion, pitch shift, band-stop filtering, coloured noise, room impulse responses (MIT), and background audio at -10 to 15 dB SNR. The background audio is AudioSet, FMA music, and 10 s chunks of LibriSpeech dev-clean read speech.
- **Adversarial negatives:** 25,000 clips plus 3,000 for validation, from the same generator.
  - Phoneme-overlap phrases chosen automatically by openWakeWord.
  - 30 curated near-misses × 40 each, such as "hey herman", "hey hermit", "hair mess", "hey there miss", "hey her mess", "hermes" on its own, and "I bought a hermes scarf". The full list is in the config.
- **General negatives:** openWakeWord's pre-computed ACAV100M features (about 2,000 h of real-world speech, music and noise).
- **Checkpoint selection:** openWakeWord's ~11 h false-positive validation features (DiPCo, Santa Barbara Corpus, MUSDB).

## Evaluation

**Method.** Every number below comes from [`eval_wake.py`](eval_wake.py). It streams audio through Housemate's `core/wake.py` in 80 ms chunks from a fresh detector state, as a satellite does; these are not `train.py`'s metrics. The test material shares no voices or engine with training.

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

**Run 3 (best candidate, head sha256 `debe67cc…`)**

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

**Training history**

| run | change | recall @0.5 (clean / speech-bg) | best recall with FA/h ≤ 0.5 | FA/h @0.5 |
|---|---|---|---|---|
| 1 | upstream recipe: 25k positives, 30 curated near-misses | 67.6 % / 45.4 % | 72.2 % (thr 0.05) | 0.09 |
| 2 | + 15k unmixed-speaker positives, 100 positives/batch, negative weight 1000 | 85.2 % / 59.3 % | 79.6 % (thr 0.10; clean 90.7 %) | 0.26 |
| 3 | + 15k more unmixed positives, + LibriSpeech dev-clean background speech | 84.3 % / 80.6 % | **89.4 %** (thr 0.15) | 0.26 |

Each run took 60–80 min on the RTX 4060: about 20 min of clip generation (run 1 only), 25–40 min of augmentation and feature extraction, and 25–37 min of training.

**Where it fails**

- **Accents.** Misses concentrate on non-US English voices: AU (Natasha, William), NZ (Mitchell, Molly), the GB child voice (Maisie), IN (Prabhat, Neerja), ZA (Luke), KE, NG, TZ and SG. US, IE and CA voices are almost always detected.
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
