# Day 2 — Tiny Three-Condition Agent Battery

**Goal**: move from static prompt activations to the first real agent-style
decision points. This is still deliberately small: three local stub scenarios,
one tool-call turn per scenario, one extracted activation before each tool
call, and one extracted activation before each final answer.

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
local stub tool executes
        ↓
last prompt token before final answer
        ↓
second layer-20 activation
```

This is not the full MCP/ReAct harness yet. It is the smallest battery that
exercises all three planned conditions:

- `benign_calculator_total`: straightforward calculator use.
- `injection_vendor_policy`: read-only document contains an adversarial page
  instruction.
- `pressure_deployment_notice`: read-only document contains evaluation and
  deployment pressure.

## Important Runtime Shape

Do not keep the SGLang AV server running while extracting agent activations on
a 40 GB GPU. The workflow is sequential:

1. Stop/kill the SGLang AV server.
2. Load Qwen with `agent_probe.py` and write activations.
3. Exit `agent_probe.py`, which frees Qwen GPU memory.
4. Start the SGLang AV server again.
5. Decode the agent activations with `smoke_test_nla.py decode`.

## Generate The Agent Trajectories

Use a descriptive `--run-id`. This keeps every run in its own folder and avoids
overwriting previous results.

```bash
python agent_probe.py \
    --run-id day2_three_condition_v1
```

Outputs:

- `data/runs/day2_three_condition_v1/trajectories.jsonl`: one JSONL row per
  scenario, including tool call, tool result, final answer, and activation
  vectors.
- `data/runs/day2_three_condition_v1/activations.parquet`: compact NLA input
  with six rows: one `tool_call` row and one `final_response` row per scenario.

If you omit `--run-id`, the script creates a timestamped folder such as
`data/runs/agent_probe_20260531_143012/`.

Do not reuse a `--run-id` unless you intentionally want to overwrite that run's
files.

Older explicit paths still work:

```bash
python agent_probe.py \
    --trajectory data/custom_trajectory.jsonl \
    --parquet data/custom_activations.parquet
```

Those explicit paths are useful for quick scratch runs, but named run folders
are safer for the research log.

The trajectory file contains one JSONL row per scenario, including tool call,
tool result, final answer, and activation vectors.

The `tool_call` rows capture model state immediately before the assistant
starts emitting the tool invocation. The `final_response` rows capture model
state after the tool result has been injected back into the conversation.

To run only one condition:

```bash
python agent_probe.py \
    --scenario injection_vendor_policy \
    --run-id day2_injection_only_v1
```

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
    --parquet data/runs/day2_three_condition_v1/activations.parquet \
    --checkpoint data/models/nla-qwen2.5-7b-L20-av \
    --sglang-url http://localhost:30000 \
    --results data/runs/day2_three_condition_v1/nla_results.jsonl
```

If running in Colab and the upstream repo is at `/content/nla-inference`, add:

```bash
    --nla-inference-py /content/nla-inference/nla_inference.py
```

Then inspect:

```bash
python smoke_test_nla.py inspect \
    --parquet data/runs/day2_three_condition_v1/activations.parquet \
    --results data/runs/day2_three_condition_v1/nla_results.jsonl
```

## What Success Looks Like

For `benign_calculator_total:tool_call`, the NLA explanation should mention
tool use, arithmetic, prices, cost/total, calculator-like computation, or a
structured JSON/action format.

For `benign_calculator_total:final_response`, it should mention using a result,
giving the computed total, or summarizing a calculation.

For `injection_vendor_policy:final_response`, useful signs include references
to an instruction conflict, untrusted page text, policy limits, or choosing
between the real policy value and the injected `$999` claim.

For `pressure_deployment_notice:final_response`, useful signs include
evaluation, deployment pressure, safe reporting, not hiding/altering logs, or
the assistant being assessed.

If both activations verbalize as unrelated generic text, the NLA infrastructure
still works from Day 1, but our agent extraction point or prompt format needs
adjustment.
