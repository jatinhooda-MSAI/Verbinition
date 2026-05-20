# Day 2 — One Benign Agent Tool-Use Trajectory

**Goal**: move from static prompt activations to the first real agent-style
decision point. This is still deliberately small: one benign task, one local
calculator tool, one extracted activation before the tool call, and one
extracted activation before the final answer.

## What This Tests

The smoke test showed that the NLA can verbalize Qwen layer-20 activations.
This probe asks whether we can now capture the same kind of activation at an
agent decision boundary:

```text
Qwen reads task + tool instructions
        ↓
last prompt token before assistant tool-call turn
        ↓
layer-20 activation
        ↓
model emits JSON tool call
        ↓
local calculator executes
        ↓
last prompt token before final answer
        ↓
second layer-20 activation
```

This is not the full MCP/ReAct harness yet. It is the smallest end-to-end
trajectory that exercises the core measurement surface.

## Important Runtime Shape

Do not keep the SGLang AV server running while extracting agent activations on
a 40 GB GPU. The workflow is sequential:

1. Stop/kill the SGLang AV server.
2. Load Qwen with `agent_probe.py` and write activations.
3. Exit `agent_probe.py`, which frees Qwen GPU memory.
4. Start the SGLang AV server again.
5. Decode the agent activations with `smoke_test_nla.py decode`.

## Generate The Agent Trajectory

```bash
python agent_probe.py \
    --trajectory data/agent_probe_trajectory.jsonl \
    --parquet data/agent_probe_activations.parquet
```

Outputs:

- `data/agent_probe_trajectory.jsonl`: full trajectory, including tool call,
  tool result, final answer, and activation vectors.
- `data/agent_probe_activations.parquet`: compact NLA input with two rows:
  `benign_calculator_total:tool_call` and
  `benign_calculator_total:final_response`.

The first row is the main milestone. It captures the model state immediately
before it starts emitting the tool call.

## Start The NLA AV Server

```bash
python -m sglang.launch_server \
    --model-path data/models/nla-qwen2.5-7b-L20-av \
    --port 30000 \
    --disable-radix-cache \
    --trust-remote-code \
    --context-length 2048 \
    --mem-fraction-static 0.75
```

If FlashInfer gives trouble, use the slower fallback backends:

```bash
python -m sglang.launch_server \
    --model-path data/models/nla-qwen2.5-7b-L20-av \
    --port 30000 \
    --disable-radix-cache \
    --trust-remote-code \
    --context-length 2048 \
    --mem-fraction-static 0.75 \
    --attention-backend triton \
    --sampling-backend pytorch
```

## Decode Agent Decision Activations

```bash
python smoke_test_nla.py decode \
    --parquet data/agent_probe_activations.parquet \
    --checkpoint data/models/nla-qwen2.5-7b-L20-av \
    --sglang-url http://localhost:30000 \
    --results data/agent_probe_nla_results.jsonl
```

If running in Colab and the upstream repo is at `/content/nla-inference`, add:

```bash
    --nla-inference-py /content/nla-inference/nla_inference.py
```

Then inspect:

```bash
python smoke_test_nla.py inspect \
    --parquet data/agent_probe_activations.parquet \
    --results data/agent_probe_nla_results.jsonl
```

## What Success Looks Like

For `benign_calculator_total:tool_call`, the NLA explanation should mention
tool use, arithmetic, prices, cost/total, calculator-like computation, or a
structured JSON/action format.

For `benign_calculator_total:final_response`, it should mention using a result,
giving the computed total, or summarizing a calculation.

If both activations verbalize as unrelated generic text, the NLA infrastructure
still works from Day 1, but our agent extraction point or prompt format needs
adjustment.
