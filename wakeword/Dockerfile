# Training image for the "Hey Hermes" wake-word head.
#
# Runs openWakeWord's automatic training pipeline (openwakeword/train.py + a YAML config) with
# piper-sample-generator v2.0.0 (LibriTTS-R multi-speaker Piper generator) on an NVIDIA GPU.
#
#   docker build -t housemate-wakeword wakeword/
#   docker run --rm --gpus all -v <data>:/data -v <work>:/work -v $PWD/wakeword:/recipe housemate-wakeword ...
#
# See README.md in this directory for the full recipe.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git curl ca-certificates ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# PyTorch 2.4.1 (the PyPI Linux wheel is the CUDA 12.1 build; it pulls cuDNN 9 as nvidia-* wheels,
# which onnxruntime-gpu 1.19 also needs).
RUN pip install torch==2.4.1 torchaudio==2.4.1

# openWakeWord training dependencies (versions from the upstream automatic_model_training notebook
# where they still install; TensorFlow/onnx-tf are left out because we only need the ONNX export).
RUN pip install \
        numpy==1.26.4 scipy==1.11.4 scikit-learn==1.3.2 tqdm==4.66.5 pyyaml==6.0.2 requests==2.32.3 \
        onnxruntime-gpu==1.19.2 onnx==1.16.2 \
        mutagen==1.47.0 torchinfo==1.8.0 torchmetrics==1.2.0 speechbrain==0.5.14 \
        audiomentations==0.33.0 torch-audiomentations==0.11.0 acoustics==0.2.6 pronouncing==0.2.0 \
        librosa==0.10.1 soundfile==0.12.1 \
        piper-phonemize==1.1.0 webrtcvad==2.0.10 \
        huggingface_hub==0.25.2 pyarrow==17.0.0 fsspec==2024.6.1 aiohttp==3.10.10

# onnxruntime-gpu 1.19 does not preload the CUDA libraries that come with the torch wheels.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib

# openWakeWord at a pinned commit (main, 0.6.0), installed without its runtime extras.
ARG OWW_COMMIT=368c03716d1e92591906a84949bc477f3a834455
RUN git clone https://github.com/dscripka/openWakeWord /opt/openwakeword \
    && git -C /opt/openwakeword checkout ${OWW_COMMIT} \
    && pip install --no-deps -e /opt/openwakeword

# Feature extractor models (official v0.5.1 release; the same files the Housemate runtime pins).
RUN set -e; d=/opt/openwakeword/openwakeword/resources/models; mkdir -p $d; \
    curl -fsSL -o $d/melspectrogram.onnx https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx; \
    curl -fsSL -o $d/embedding_model.onnx https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx; \
    echo "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f  $d/melspectrogram.onnx" | sha256sum -c -; \
    echo "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f  $d/embedding_model.onnx" | sha256sum -c -

# piper-sample-generator v2.0.0 (train.py imports its top-level generate_samples.py) + LibriTTS-R generator.
RUN git clone --depth 1 --branch v2.0.0 https://github.com/rhasspy/piper-sample-generator /opt/piper-sample-generator \
    && curl -fsSL -o /opt/piper-sample-generator/models/en_US-libritts_r-medium.pt \
        https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt \
    && echo "$(sha256sum /opt/piper-sample-generator/models/en_US-libritts_r-medium.pt)"
# sha256 observed when this recipe was run (2026-11-08):
# e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6  en_US-libritts_r-medium.pt

# Upstream bug: train.py declares `--convert_to_tflite` with default="False" (a truthy string), so it always
# tries the TensorFlow conversion after saving the ONNX head and exits non-zero without onnx-tf.
RUN sed -i 's/if args.convert_to_tflite:/if args.convert_to_tflite is True:/' /opt/openwakeword/openwakeword/train.py \
    && grep -q "if args.convert_to_tflite is True:" /opt/openwakeword/openwakeword/train.py

# Fail the build early if the training stack does not import or cannot see CUDA providers.
RUN python -c "import torch, torchaudio, onnxruntime, speechbrain, audiomentations, torch_audiomentations, acoustics, pronouncing, piper_phonemize, webrtcvad; import openwakeword.train; print('torch', torch.__version__, 'ort', onnxruntime.__version__, onnxruntime.get_available_providers())"

# The recipe scripts and config are mounted at /recipe (not baked in), data at /data, outputs at /work.
WORKDIR /work
