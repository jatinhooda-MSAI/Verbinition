# Day 1 — NLA inference smoke test

**Goal**: validate that the released Qwen-7B NLA AV produces sensible
explanations of activations from a fresh Qwen-7B load on your hardware.
If this doesn't work cleanly, nothing downstream works.

## Hardware

- 1× H100 80GB or A100 80GB is sufficient for this step
- ~40 GB free disk (Qwen2.5-7B-Instruct + the AV are each ~14 GB)
- On Northwestern QUEST, use the `gengpu` partition and request one GPU. For
  80 GB A100/H100 cards, QUEST docs say to add `#SBATCH --constraint=sxm` in
  addition to the GPU request.

## Setup

```bash
# 1. Clone this repo and an upstream NLA inference repo side by side
git clone <this-repo> vac
git clone https://github.com/kitft/natural_language_autoencoders nla-upstream
# Alternative: git clone https://github.com/kitft/nla-inference nla-inference

# 2. Create env, install core deps
cd vac
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[sglang]"

# 3. Download the AV checkpoint locally.
# NLAClient reads nla_meta.yaml and the embedding weights from this directory.
huggingface-cli download kitft/nla-qwen2.5-7b-L20-av \
    --local-dir data/models/nla-qwen2.5-7b-L20-av
```

If `uv` isn't installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
Equivalent `pip` commands work fine.

## Run the smoke test

### Step 1 — extract activations

```bash
python smoke_test_nla.py generate
```

Loads Qwen2.5-7B-Instruct, runs 8 probe prompts through it, extracts the
layer-20 last-token activation for each, writes `data/smoke_test.parquet`.
Takes ~2 minutes after the model is cached.

### Step 2 — launch the AV server

In a separate terminal:

```bash
python -m sglang.launch_server \
    --model-path data/models/nla-qwen2.5-7b-L20-av \
    --port 30000 \
    --disable-radix-cache \
    --trust-remote-code
```

Startup loads the AV weights (~14 GB). Wait for `server is fired up` or an
equivalent ready line before the next step.

### Step 3 — run NLA inference

```bash
python smoke_test_nla.py decode \
    --checkpoint data/models/nla-qwen2.5-7b-L20-av \
    --sglang-url http://localhost:30000 \
    --results data/smoke_test_results.jsonl
```

This injects each activation as a token embedding into the AV's prompt and
autoregresses an explanation. Output is jsonl with one explanation per probe.

If the upstream repo is somewhere other than `../nla-upstream` or
`../nla-inference`, pass `--nla-inference-py /path/to/nla_inference.py`.

### Step 4 — inspect

```bash
python smoke_test_nla.py inspect --results data/smoke_test_results.jsonl
```

## What success looks like

Explanations should visibly differ across categories. Specifically:

| probe                  | should mention                              |
|------------------------|---------------------------------------------|
| `python_code`          | code / programming / function / recursion   |
| `strong_emotion`       | anger / frustration / negative emotion      |
| `math_word_problem`    | math / trains / speed / arithmetic          |
| `deception_intent`     | deception / lying / covering up             |
| `refusal_trigger`      | harmful request / unauthorized access       |
| `multilingual_spanish` | Spanish / non-English / library / schedule  |
| `evaluation_aware`     | testing / evaluation / cautious response    |
| `neutral_factual`      | landmark / Paris / factual statement        |

Uniform or generic explanations across all probes mean the infrastructure
isn't working — see troubleshooting.

## Known failure modes (from upstream `docs/inference.md`)

- **"Output is in Chinese"** — activation scale factor is wrong. The AV ships
  an `nla_meta.yaml` sidecar with the correct `injection_scale`; confirm the
  upstream `nla_inference.py` is loading it. Don't hardcode scale values.
- **Explanations are gibberish** — usually a layer mismatch (we want L20) or
  d_model mismatch (we want 3584). Both are pinned in the smoke-test script;
  check that SGLang loaded the right AV checkpoint.
- **SGLang OOM on startup** — the AV is a full 7B model in bf16. Needs
  ≥30 GB free on the GPU. Other processes can be holding memory.
- **"input_embeds not supported"** — SGLang version too old. The repo
  requires `>=0.5.6`.

Read `docs/inference.md` in the upstream NLA repo before debugging beyond
these.

## One QUEST batch job shape

This is the rough shape for an `sbatch` script. Fill in your account/email and
module/conda setup for your QUEST environment.

```bash
#!/bin/bash
#SBATCH -A <account>
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=sxm
#SBATCH -t 02:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=nla-smoke
#SBATCH --output=logs/nla-smoke-%j.out
#SBATCH --error=logs/nla-smoke-%j.err

set -euo pipefail

source .venv/bin/activate
mkdir -p logs data/models

huggingface-cli download kitft/nla-qwen2.5-7b-L20-av \
    --local-dir data/models/nla-qwen2.5-7b-L20-av

python smoke_test_nla.py generate

python -m sglang.launch_server \
    --model-path data/models/nla-qwen2.5-7b-L20-av \
    --port 30000 \
    --disable-radix-cache \
    --trust-remote-code > logs/sglang-smoke.log 2>&1 &

SERVER_PID=$!
trap 'kill "$SERVER_PID"' EXIT

until python - <<'PY'
import httpx
try:
    httpx.get("http://localhost:30000/health", timeout=2).raise_for_status()
except Exception:
    raise SystemExit(1)
PY
do
    sleep 10
done

python smoke_test_nla.py decode \
    --checkpoint data/models/nla-qwen2.5-7b-L20-av \
    --results data/smoke_test_results.jsonl

python smoke_test_nla.py inspect --results data/smoke_test_results.jsonl
```

## Decision point

After eyeballing the explanations:

- **Sensible differentiation across categories** → green light. Move to D2-3
  (agent harness + local MCP server).
- **Some categories work, some don't** → yellow. Note which fail. Could
  indicate distribution issues that will resurface on agent activations;
  proceed cautiously.
- **Nothing makes sense** → red. Debug before writing agent code. The
  Day 2-3 work is wasted if NLA doesn't work on text-style activations.
