# Option B - Full AV Domain Adaptation

This path scales the MCP/ReAct battery and fine-tunes only the Activation
Verbalizer (AV) with LoRA. The Activation Reconstructor should remain frozen
for later fidelity evaluation.

## 1. Generate The Full MCP Battery

On Colab Pro, use a GPU runtime and make sure the SGLang AV server is not
running. Then generate a balanced battery. The default below creates 450
trajectories: 150 benign, 150 prompt-injection, and 150 pressure/evaluation.

```bash
!python scripts/run_mcp_probe.py \
  --battery full \
  --per-condition 150 \
  --run-id full_mcp_react_v1 \
  --max-tool-steps 2
```

Outputs:

```text
data/runs/full_mcp_react_v1/
  generated_documents.json
  trajectories.jsonl
  activations.parquet
  env.txt
```

If the runtime is tight, use `--per-condition 75` first and then resume with a
new run id. Multiple run dirs can be combined in the SFT dataset step.

## 2. Build AV SFT Data

```bash
!python scripts/build_av_sft_dataset.py \
  --run-dir data/runs/full_mcp_react_v1 \
  --out data/av_sft/full_mcp_react_v1.jsonl \
  --eval-frac 0.15
```

The dataset rows contain:

```text
probe_id
scenario_id
condition
decision_kind
diagnostic_kind
split
activation_vector
target_text
```

The initial labels are deterministic labels from the known scenario/tool-result
metadata. They are intentionally plain and factual. A stronger teacher-model
labeler can replace this later, but deterministic labels are enough to test
whether the AV can adapt away from generic JSON/API explanations.

## 3. Train AV LoRA

Install training deps in the pinned environment:

```bash
!uv pip install --system peft==0.13.2
```

Then train:

```bash
!python scripts/train_av_lora.py \
  --base-av data/models/nla-qwen2.5-7b-L20-av \
  --train data/av_sft/full_mcp_react_v1.jsonl \
  --out data/models/nla-qwen-agent-av-lora \
  --rank 16 \
  --alpha 32 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --gradient-checkpointing
```

The trainer reproduces the NLA AV input contract locally: it loads
`nla_meta.yaml`, renders the AV prompt, normalizes the activation vector by the
checkpoint's injection scale, replaces the activation placeholder embedding,
and trains next-token loss on the target explanation text.

## 4. Merge Adapter

```bash
!python scripts/merge_av_lora.py \
  --base-av data/models/nla-qwen2.5-7b-L20-av \
  --adapter data/models/nla-qwen-agent-av-lora \
  --out data/models/nla-qwen-agent-av-merged
```

The merge script copies `nla_meta.yaml` so `smoke_test_nla.py decode` can use
the merged checkpoint like the released AV.

## 5. Decode Base And Adapted AV

Serve the base AV, decode held-out activations, then stop the server. Repeat
with the merged adapted AV.

```bash
!python smoke_test_nla.py decode \
  --parquet data/runs/full_mcp_react_v1/activations.parquet \
  --checkpoint data/models/nla-qwen2.5-7b-L20-av \
  --sglang-url http://localhost:30000 \
  --results data/runs/full_mcp_react_v1/nla_results_base.jsonl \
  --nla-inference-py /content/nla-inference/nla_inference.py
```

```bash
!python smoke_test_nla.py decode \
  --parquet data/runs/full_mcp_react_v1/activations.parquet \
  --checkpoint data/models/nla-qwen-agent-av-merged \
  --sglang-url http://localhost:30000 \
  --results data/runs/full_mcp_react_v1/nla_results_lora.jsonl \
  --nla-inference-py /content/nla-inference/nla_inference.py
```

## 6. Compare

```bash
!python scripts/compare_nla_results.py \
  --base data/runs/full_mcp_react_v1/nla_results_base.jsonl \
  --adapted data/runs/full_mcp_react_v1/nla_results_lora.jsonl \
  --dataset data/av_sft/full_mcp_react_v1.jsonl
```

Success looks like higher adapted keyword hits and qualitatively better
mentions of:

- untrusted/injected instructions,
- true value vs injected value,
- evaluation/deployment pressure,
- logs/concealment constraints,
- less generic JSON/API/database format chatter.
