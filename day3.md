# Day 3 - Option A MCP/ReAct Probe

**Goal**: replace the Day 2 direct local-tool diagnostic with a small proper
MCP/ReAct harness while keeping the run controlled. This battery is capped at
nine trajectories:

- 3 benign tool-use scenarios,
- 3 prompt-injection document scenarios,
- 3 pressure/evaluation document scenarios.

The harness still extracts Qwen layer-20 activations at the main decision
points:

```text
last token before each assistant MCP tool-call turn
last token before the final user-facing response
```

It also keeps the useful diagnostics from Day 2:

```text
tool_result_end
tool_content_plain_end
```

## Install

Use the same pinned GPU environment from Day 1/2. To avoid disturbing the
serving stack, add MCP explicitly and install this repo without dependency
resolution:

```bash
uv pip install "mcp>=1.0"
uv pip install -e . --no-deps
```

If you already installed the package normally with dependencies, the exact
commands are less important than having `torch`, `transformers`, `pyarrow`, and
`mcp` available in the active environment.

## List Scenarios

```bash
python scripts/run_mcp_probe.py --list-scenarios
```

## Generate MCP Trajectories And Activations

Do not keep the SGLang AV server running while extracting Qwen activations on a
single 40 GB GPU.

```bash
python scripts/run_mcp_probe.py \
    --run-id day3_mcp_react_v1
```

Outputs:

```text
data/runs/day3_mcp_react_v1/
  trajectories.jsonl
  activations.parquet
  env.txt
```

The trajectory rows include:

```text
run/scenario metadata
messages
tool_calls
tool_results
final_answer
visible_behavior_labels
decision_points with activation vectors
```

The parquet rows are NLA-ready and include:

```text
probe_id
scenario_id
condition
decision_kind
diagnostic_kind
step
layer
prompt
activation_vector
```

To run a subset:

```bash
python scripts/run_mcp_probe.py \
    --run-id day3_injection_only_v1 \
    --scenario injection_vendor_policy \
    --scenario injection_travel_policy \
    --scenario injection_warranty_window
```

## Decode With The NLA AV

After `run_mcp_probe.py` exits and frees Qwen GPU memory, start the AV server:

```bash
python -m sglang.launch_server \
    --model-path data/models/nla-qwen2.5-7b-L20-av \
    --port 30000 \
    --disable-radix-cache \
    --trust-remote-code \
    --context-length 2048 \
    --mem-fraction-static 0.75
```

Then decode:

```bash
python smoke_test_nla.py decode \
    --parquet data/runs/day3_mcp_react_v1/activations.parquet \
    --checkpoint data/models/nla-qwen2.5-7b-L20-av \
    --sglang-url http://localhost:30000 \
    --results data/runs/day3_mcp_react_v1/nla_results.jsonl
```

Inspect:

```bash
python smoke_test_nla.py inspect \
    --parquet data/runs/day3_mcp_react_v1/activations.parquet \
    --results data/runs/day3_mcp_react_v1/nla_results.jsonl
```

If the decision-point rows remain format-dominated but
`tool_content_plain_end` rows surface more injection/pressure content, that
confirms the Day 2 failure mode in a real MCP trajectory setting and supports
moving to Option B.
