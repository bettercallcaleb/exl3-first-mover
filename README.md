# EXL3 First Mover

A conservative, resumable workflow for checking Hugging Face models and making EXL3 quants with ExLlamaV3. It detects unsupported architectures **before** downloading model weights. Source revision, model license, source size, disk budget and existing EXL3 releases are recorded. Quantization calls ExLlamaV3's official converter; this repository does not implement a new quantization algorithm.

## Quick start: preflight only

Requires Python 3.10+.

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python fm.py preflight Aleph-Alpha/Kolibri-1.1

To check architecture support, set `EXLLAMAV3_REPO` to an existing checkout. It defaults to `~/exllamav3`. A missing checkout yields `null` (unknown), not a claim of support. Preflight only fetches metadata and `config.json`, not weights.

## Full conversion (advanced)

1. Install a compatible NVIDIA driver and CUDA-enabled PyTorch in `.venv`. Install [ExLlamaV3](https://github.com/turboderp-org/exllamav3) into the same environment, following its current official instructions. Set `EXLLAMAV3_REPO` to that checkout.
2. Optionally install [TabbyAPI](https://github.com/theroyallab/tabbyAPI) and configure its path in local `config.json`. TabbyAPI validation is required for automatic publication by default.
3. Copy `job.example.json` to `job.json`, set the source model and target bitrate. The example has `publish: false`.
4. Run `./run.sh job job.json` in the foreground, or `./run.sh launch job.json` to detach and `./run.sh status OWNER/MODEL` to inspect the stored state. Conversion can resume from the working directory.

The source-weight and parameter size limits are safety gates, not estimates of GPU memory requirements. Large models are blocked by default, including 75B Kolibri 1.1; bypassing the size gate does **not** add architecture support. Never publish an artifact that fails smoke tests.

## Commands

    ./run.sh doctor
    ./run.sh preflight OWNER/MODEL
    ./run.sh job job.json
    ./run.sh launch job.json
    ./run.sh status OWNER/MODEL
    ./run.sh publish OWNER/MODEL --dest-repo NAMESPACE/REPOSITORY --bits 4.0

`publish` requires a prior conversion, generation probe, integrity manifest and (by default) TabbyAPI smoke test. It is never automatic unless a job explicitly sets `publish: true`. Your Hugging Face authentication remains managed by the local Hugging Face client; no token is stored in this repository.

## Kolibri 1.1 reproducible preflight (November 24, 2026)

On an ExLlamaV3 v1.5.3 checkout, `Aleph-Alpha/Kolibri-1.1` reports architecture `Kolibri1_1ForCausalLM`, 75.113B parameters, ~72.211 GiB source weights, Apache-2.0 license, and `exllamav3_arch_supported: false`. It is blocked by both the 40B size gate and the absent model implementation. **No Kolibri EXL3 conversion or decode benchmark is claimed.** The result is specific to this pinned checkout; update ExLlamaV3 and rerun the check as support evolves.

## What is and is not validated

Preflight validates metadata and inspects the architecture registry. It cannot guarantee that a supported model will quantize correctly. A completed run separately requires a successful converter, finite-logit generation probe, artifact checksum manifest, and (if enabled) TabbyAPI compatibility tests. A smoke test is not a full quality benchmark.

The source code is MIT licensed. Quantized model artifacts remain subject to their original model licenses.

The optional TabbyAPI smoke test refuses to overwrite an existing `config.yml`. Use a separate TabbyAPI checkout for validation to protect your configuration.
