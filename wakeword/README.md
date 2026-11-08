# "Hey Hermes" wake model: training recipe

This directory trains `hey_hermes.onnx`, the wake-word head that Housemate's runtime
(`core/wake.py`) runs on top of openWakeWord's melspectrogram and embedding models.

The head has the same interface as openWakeWord's stock heads:
- input: `[1, 16, 96]`, which is 16 embeddings of 80 ms each (1.28 s);
- output: one score in `[0, 1]`.

Any openWakeWord-format head plugs into the same pipeline.

**Result: the trained head did not pass our bar, so it is not shipped.**
- **The bar:** recall ≥ 90 % on held-out positives **and** ≤ 0.5 false accepts per hour, measured with Housemate's runtime.
- **Best candidate after three runs:** 89.4 % recall at 0.34 FA/h (threshold 0.15), or 92.1 % at 0.60 FA/h (threshold 0.10).
- **Consequence:** Housemate keeps openWakeWord's stock `hey_jarvis` as its default wake word. A head you train yourself can be dropped in as `wakeword/hey_hermes.onnx`, or set by path as `wake_model`.

See [MODEL_CARD.md](MODEL_CARD.md) for the full evaluation, the failure modes and the licence. A head trained with this recipe is **CC BY-NC-SA 4.0 (non-commercial)**, inherited from the training features, unlike the MIT code in this repo.

## What the recipe does

It follows openWakeWord's official automatic training pipeline:
- the [`automatic_model_training.ipynb`](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb) notebook;
- `openwakeword/train.py` driven by a YAML config ([`train_config.yaml`](train_config.yaml)).

The pipeline runs inside a CUDA container ([`Dockerfile`](Dockerfile)). In order:

1. **Synthetic positives.** "hey hermes" voiced by the Piper LibriTTS-R multi-speaker generator (`piper-sample-generator` v2.0.0, 904 speakers). Three speaking rates and stochastic durations add variety.
   - 25,000 clips come from `train.py`'s own call. Each of those is a 50/50 SLERP mix with one of the first few dozen speakers.
   - 30,000 more come from `oww_train.py extra_positives`. This uses the same `generate_samples()` but with unmixed speakers (`slerp_weights: [1.0]`, so all 904 voices) and punctuation variants ("hey, hermes.", "hey hermes!").
   - Without these extra clips, run 1 missed most non-US accents; see the model card.
2. **Adversarial negatives.**
   - Phrases chosen automatically by phoneme overlap with the target (CMUdict).
   - Curated near-misses from the config, such as "hey herman", "hair mess", "hey there miss", "hermes" on its own and "hey her mess".
   - [`oww_train.py`](oww_train.py) repeats the curated list `custom_negative_repeat` times, so those phrases make up about 5 % of the negatives instead of 30 clips out of 25,000.
3. **Augmentation.** EQ, distortion, pitch shift, band-stop filtering, coloured noise, background audio at -10 to 15 dB SNR, and room impulse responses (MIT IR survey). Then openWakeWord features are computed on the GPU.
   - The background audio is AudioSet, Free Music Archive, and LibriSpeech dev-clean read speech in 10 s chunks.
   - The read speech is our addition. It lifted recall with background speech from 59 % to 81 % at threshold 0.5.
4. **Training.** A 2-layer fully connected head (32 units) on:
   - the clips above;
   - about 2,000 h of pre-computed ACAV100M negative features.

   It uses openWakeWord's `auto_train` schedule: three cycles, a rising negative weight, and checkpoint averaging selected on the ~11 h false-positive validation set.
5. **Evaluation with our runtime, not the trainer's metrics.** [`eval_wake.py`](eval_wake.py) streams audio through `core/wake.py` in 80 ms chunks, as a satellite does. The material:
   - held-out real-TTS positives from a different engine ([`make_eval_corpus.py`](make_eval_corpus.py): Microsoft Edge neural voices, 47 English voices across 14 locales);
   - near-miss phrases;
   - more than 10 h of LibriSpeech test speech plus unseen AudioSet/FMA music and noise.

### Changes from the upstream notebook (and why)

| Upstream | Here | Why |
|---|---|---|
| `git clone piper-sample-generator` (HEAD) | tag `v2.0.0` | `train.py` imports the top-level `generate_samples.py`; v3 moved it into a package |
| AudioSet `bal_train09.tar` | `data/bal_train/09.parquet` (same HF dataset) | the tar files were replaced by parquet shards |
| `rudraml/fma` loader (streams `fma_small.zip`) | the same zip, read member by member over HTTP ranges | avoids the dataset loading script; same audio |
| TensorFlow + onnx-tf for `.tflite` export | not installed | Housemate only uses the ONNX head |
| `torch` from Colab | `torch==2.4.1` (CUDA 12.1) + `onnxruntime-gpu==1.19.2` | one cuDNN 9 for both, so feature extraction runs on the GPU |
| `custom_negative_phrases` listed once | repeated via `custom_negative_repeat` | `train.py` voices the list in order, one clip per entry |
| positives from `train.py` only | plus `extra_positives` (unmixed speakers, punctuation variants) | accent coverage; see above |
| AudioSet + FMA backgrounds | plus LibriSpeech dev-clean speech chunks (`prepare_data.py speechbg`) | wake word spoken over a TV/podcast; dev-clean speakers are disjoint from the test sets used for evaluation |
| `--convert_to_tflite` default `"False"` (a truthy string) | patched to `is True` in the image | otherwise `train.py` always attempts the TFLite conversion and exits non-zero after saving the ONNX head |

The DeepPhonemizer model that `train.py` downloads for out-of-vocabulary target words is gone (HTTP 403). The target phrase therefore has to use CMUdict words, and "hey" and "hermes" both are.

## Requirements

- An NVIDIA GPU with Docker GPU support. Tested on an RTX 4060 8 GB with Docker Desktop (WSL2) on Windows 11.
- About 25 GB of free disk for the data directory (17.5 GB of it is the openWakeWord features), plus 12 GB for the image.
- For evaluation on the host: Python 3.11 with `onnxruntime` and `numpy` (the plugin's runtime deps), `edge-tts`, and `ffmpeg` on PATH.

## Commands

The commands below use bash syntax. In PowerShell, replace `$PWD` with `${PWD}` and use backtick line continuations.
`DATA` holds downloads (reusable across runs). `WORK` holds generated clips, features and the trained head.

```bash
docker build -t housemate-wakeword wakeword/

DATA=/path/to/wakeword-train/data
WORK=/path/to/wakeword-train/work
RUN="docker run --rm --gpus all --shm-size 4g -v $DATA:/data -v $WORK:/work -v $PWD/wakeword:/recipe:ro housemate-wakeword"

# 1. data: RIRs, AudioSet shard 09, 160 FMA clips, LibriSpeech dev-clean speech chunks, openWakeWord features (17.5 GB),
#    LibriSpeech test sets (eval only)
$RUN python /recipe/prepare_data.py rirs audioset fma speechbg features librispeech

# 2. Piper clips -> augmentation + features -> training (writes $WORK/hey_hermes/hey_hermes.onnx)
#    One call runs the stages in order; they can also be run one at a time.
$RUN python /recipe/oww_train.py --config /recipe/train_config.yaml \
    extra_positives generate_clips augment_clips train_model
cp $WORK/hey_hermes/hey_hermes.onnx wakeword/hey_hermes.onnx   # only if it passes step 3

# 3. held-out evaluation with the Housemate runtime (on the host)
python wakeword/make_eval_corpus.py --out $WORK/../eval/edge
python wakeword/eval_wake.py --head wakeword/hey_hermes.onnx --models-dir <dir with melspectrogram.onnx + embedding_model.onnx> \
    --positives $WORK/../eval/edge/positives <your own recordings> --nearmiss $WORK/../eval/edge/nearmiss \
    --speech $DATA/eval/librispeech --music-noise $DATA/eval/audioset_16k $DATA/eval/fma_16k \
    --out $WORK/../eval/report.json
```

Notes on the commands:
- Every stage is resumable. `extra_positives` and `generate_clips` top up missing clips, and `augment_clips` skips features that already exist (pass `--overwrite` to recompute them).
- Expect about 25 min of clip generation, 40 min of augmentation and feature extraction, and 35 min of training on an RTX 4060.
- `--models-dir` needs the two feature models. Housemate downloads them with SHA-256 pins into `<plugin data dir>/models/` on first start (`core/models.py`), or fetch them from the [openWakeWord v0.5.1 release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1).

## Data sources and licences

| Data | Used for | Source | Licence |
|---|---|---|---|
| openWakeWord code (`train.py`, data utils) | training pipeline | github.com/dscripka/openWakeWord @ `368c037` | Apache-2.0 |
| melspectrogram + embedding models (v0.5.1) | features in training and at runtime | openWakeWord release v0.5.1 | CC BY-NC-SA 4.0. All of openWakeWord's pre-trained models are under it; the embedding network itself derives from Google's Apache-2.0 `speech_embedding` |
| ACAV100M pre-computed features (~2,000 h) | general negatives | HF `davidscripka/openwakeword_features` | **CC BY-NC-SA 4.0** |
| False-positive validation features (~11 h: DiPCo, Santa Barbara Corpus, MUSDB) | checkpoint selection | same HF dataset | **CC BY-NC-SA 4.0** |
| piper-sample-generator v2.0.0 | synthetic speech | github.com/rhasspy/piper-sample-generator | MIT |
| LibriTTS-R Piper generator (`en_US-libritts_r-medium.pt`) | synthetic speech | piper-sample-generator release v2.0.0; trained on LibriTTS-R | LibriTTS-R is CC BY 4.0 |
| MIT environmental impulse responses | reverb augmentation | HF `davidscripka/MIT_environmental_impulse_responses` (McDermott lab IR survey) | unknown (stated as such by the dataset card) |
| AudioSet balanced train, shard 09 | background augmentation | HF `agkphysics/AudioSet` | dataset CC BY 4.0; the audio clips come from YouTube uploads |
| Free Music Archive `fma_small` (160 clips) | background augmentation | os.unil.cloud.switch.ch/fma | per-track Creative Commons licences, some NC |
| LibriSpeech dev-clean (400 × 10 s) | background speech augmentation | openslr.org/12 | CC BY 4.0; the recordings are public-domain LibriVox readings |
| LibriSpeech test-clean + test-other | **evaluation only** | openslr.org/12 | CC BY 4.0; the recordings are public-domain LibriVox readings |
| Microsoft Edge neural TTS clips | **evaluation only** | `edge-tts` | generated test audio, not redistributed |

A deterministic hash split keeps 15 % of the AudioSet and FMA clips out of training. They are used only as unseen music/noise in the false-accept evaluation.
