# Project Handoff: Verbalizing Hidden Cognition in Tool-Using Agents

This document is a complete context handoff for a fresh chat. It covers the
research framing, all key decisions and their rationale, deferred items, the
current state of the codebase, and what comes next.

---

## 1. Project in one sentence

Apply the recently released Natural Language Autoencoders (NLAs, Fraser-
Taliente et al., 2026) to the activation streams of a Qwen2.5-7B agent at
tool-call decision points, and measure the gap between the agent's stated
chain-of-thought and its NLA-revealed internal cognition.

---

## 2. Upstream artifacts

- **Paper**: https://transformer-circuits.pub/2026/nla/
- **Code**: https://github.com/kitft/natural_language_autoencoders
- **Lightweight inference package**: https://github.com/kitft/nla-inference
- **AV/AR checkpoints**: https://huggingface.co/collections/kitft/nla-models

The released checkpoints cover: Qwen2.5-7B, Gemma-3-12B, Gemma-3-27B,
Llama-3.3-70B. We are using **Qwen2.5-7B** (model `kitft/nla-qwen2.5-7b-L20-av`
for the AV, and the corresponding AR checkpoint).

---

## 3. What NLAs are (brief)

Two jointly-trained components on top of a frozen target model:

- **Activation Verbalizer (AV)**: takes a residual-stream vector injected as
  a single token embedding, autoregresses a natural-language description of
  what the vector encodes.
- **Activation Reconstructor (AR)**: takes the AV's text description, maps
  it back to a vector. Round-trip MSE on L2-normalized vectors is the fidelity
  metric (called FVE — fraction of variance explained).

The AV and AR were jointly trained by Anthropic using GRPO RL with
reconstruction fidelity as the reward. The released AV for Qwen-7B was trained
on FineWeb-style text and is **out-of-distribution on agentic activations** —
this is addressed in the project's training component (see §7).

---

## 4. Research question

What is a tool-using agent "thinking" at the decision point immediately before
it takes a consequential action? Specifically:

- Does NLA-verbalized cognition during tool calls differ from the agent's
  visible chain-of-thought (faithfulness question)?
- In prompt-injection scenarios (adversarial tool outputs), does the NLA
  reveal whether the agent internally "notices" the injection even when it
  doesn't say so?
- In agentic-pressure scenarios (variants of the published agentic-
  misalignment evals), are there internal indicators of suspicion, goal-drift,
  or evaluation-awareness that NLAs surface but CoT doesn't?

---

## 5. Why this project

- NLAs were released ~1 week before project start. Agentic application is
  genuinely uncharted — nobody has run them on tool-call decision points.
- Agentic safety is the live frontier at Anthropic, OpenAI, and FAANG.
- The project naturally covers safety/interp (core) + agents + MCP + text
  generation without forcing components together.
- Compute-feasible: inference-only on 2-4 GPUs; no RL training required
  (LoRA fine-tune of AV is the only training step, and it's lightweight).

---

## 6. Target and constraints

- **Target audience**: portfolio piece for Anthropic / OpenAI / FAANG
  applications (all labs equally weighted).
- **Compute**: 2–4 H100s or A100s.
- **Timeline**: 3 weeks hard deadline. A summer extension for a paper is
  possible but not in scope for current planning.
- **Codebase**: GitHub repo (`vac/`), Python 3.11, uv, pyproject.toml with
  optional extras for sglang / agent / train.

---

## 7. Technical design

### 7a. Frozen agent backbone

**Qwen2.5-7B-Instruct** served locally. Activations extracted at **layer 20**
(pinned to match the released AV). Extraction point: the **last token
immediately preceding each tool invocation** and the last token preceding the
final user-facing response. These are the two "decision points."

### 7b. NLA inference

AV served via SGLang (`sglang.launch_server`). Inference via
`nla_inference.py` from the upstream repo. The AV's `nla_meta.yaml` sidecar
carries the correct `injection_scale` — must be loaded, never hardcoded (known
failure mode: "output is in Chinese" when scale is wrong).

### 7c. Agent harness

Hand-rolled ReAct loop (~200-300 lines), not a framework, for full control
over activation extraction points. Tools exposed to the agent via a **local
MCP server** (Python `mcp` package, `>=1.0`). Tool set: file operations,
HTTP fetching, shell-style command. MCP is a genuine component, not decorative
— it satisfies the "Extra Criteria: MCP" requirement.

### 7d. Scenario taxonomy (three conditions, ~500 trajectories total)

1. **Benign tool use**: agent completes straightforward tasks without pressure.
   Establishes baseline NLA explanation patterns.
2. **Prompt injection in tool outputs**: adversarial instructions embedded in
   tool return values. Measures whether NLA reveals injection-awareness that
   CoT doesn't express.
3. **Agentic-pressure scenarios**: variants of the Anthropic agentic-
   misalignment eval pattern (survival-relevant pressure + destructive action
   option). Measures whether NLA surfaces goal-drift or deception signals.

### 7e. Training component

**LoRA fine-tune of the AV** on agent decision-point activations. Motivation:
the released AV was trained on FineWeb text; agent activations are out-of-
distribution. Fine-tuning with reconstruction MSE as the loss signal (AR is
frozen) should improve explanation quality on the agentic domain.

Data construction: generate a held-out set of agent activation / explanation
pairs; explanations bootstrapped from a stronger LLM and filtered by AR
reconstruction score to keep only high-fidelity examples. Library: `peft`
(LoRA), `datasets`, `trl`.

### 7f. Analysis pipeline

- **Judge model** (Claude API or GPT-4) classifies NLA explanations along
  axes: evaluation-awareness, deception-signal, goal-drift, faithfulness to
  visible CoT. Produces a binary/scalar score per trajectory per axis.
- **Linear probe baselines**: probes trained directly on hidden states for the
  same axes. Key question: does NLA add information beyond what linear probes
  recover? If not, the contribution shrinks.
- **Faithfulness metric**: per-trajectory comparison of NLA explanation
  content vs. visible CoT content. Quantifies the gap we're trying to measure.

---

## 8. Extra Criteria (course rubric)

The project satisfies:

1. **Explainable AI** — the core methodology. NLAs are an unsupervised
   interpretability method; the analysis is about gaps between stated and
   internal cognition. This is the center of gravity, not an add-on.
2. **MCP** — tools exposed to the agent via a local MCP server.

Deliberately excluded from scope: Multimodal (no natural fit), RAG (not
needed), MLOps (out of scope for 3 weeks), Chatbot GUI (cut to stay on track).

---

## 9. Key decisions and rationale

| Decision | Choice | Why |
|---|---|---|
| Base model | Qwen2.5-7B-Instruct | Released NLA checkpoint exactly matches this model; lower setup friction than Gemma-3-12B; Gemma has documented gotchas (√d embed-scale) |
| Full replication | No | Qwen-7B NLA required 2×8×H100 for the RL stage; not feasible on 2-4 GPUs |
| SAE intervention | Deferred | No confirmed Qwen SAE coverage; would extend timeline to 4-5 weeks; revisit at week-3 milestone if NLA surfaces a specific finding worth causal validation |
| Paper trajectory | Deferred | Possible summer extension but not in scope for 3-week deadline planning |
| Framework for agent | Hand-rolled | Need exact control over activation extraction points; frameworks add abstraction that complicates this |
| LoRA target | AV only (AR frozen) | AR is the fidelity evaluator; fine-tuning it would corrupt the metric |
| Layer | 20 (fixed) | Pinned by the released checkpoint; not a tunable hyperparameter |
| Activation position | Last token before tool call | Semantically richest position for the agent's "decision state" |

---

## 10. What was considered and rejected

- **1:1 NLA replication from scratch**: requires 2×8×H100 + weeks of RL
  iteration, not feasible on 2-4 GPUs.
- **GPT-2 / Pythia toy replication with SFT only**: SFT-only NLA produces
  uninformative explanations without RL; weak portfolio signal; rejected.
- **Gemma-3-12B**: has a marginal edge for SAE coverage later (Gemma Scope),
  but more setup friction and larger memory footprint. Revisit if SAE
  intervention becomes the path in summer.
- **SAE-based steering as parallel track**: would extend scope to 4-5 weeks.
  Added as a conditional week-3 step if NLA surfaces something specific worth
  causal validation.
- **Kitchen-sink approach (spanning multimodal + RAG + MCP + explainability
  deeply)**: rejected — dilutes signal, shallow on all. Two focused extra
  criteria beats four shallow ones.
- **Portfolio project optimized for a single lab**: user targets all labs
  equally; the cross-cutting optimum is safety/interp work.

---

## 11. Three-week plan

### Week 1
- **D1–3**: NLA inference working end-to-end on Qwen-7B. Smoke test with
  curated probe prompts. Decision point: green/yellow/red before writing
  agent code.
- **D4–6**: Agent harness (ReAct loop) + local MCP server. One benign scenario
  running end-to-end with activation extraction and NLA verbalization.
- **D7**: One trajectory from each condition (benign / injection / pressure)
  producing NLA explanations. Qualitative inspection.

### Week 2
- **D8–9**: Scale scenario battery to ~150 trajectories per condition.
- **D10–11**: Linear probe baselines. Judge-model classification pipeline.
- **D12–13**: Initial quantitative results. Does NLA explanation diverge from
  CoT more in pressure/injection conditions than benign? Is the divergence
  statistically consistent?
- **D14**: Decision point — are findings interesting? If yes, proceed to LoRA
  fine-tune. If results are flat, pivot to tighter scope (e.g., NLA failure-
  mode characterization).

### Week 3
- **D15–17**: LoRA fine-tune AV on agent activations. Validate with held-out
  reconstruction MSE. Compare explanation quality pre/post fine-tune.
- **D18–19**: Final analysis. Human spot-check of judge-model classifications
  on sample of 50 trajectories. Ablations (layer choice, prompt template).
- **D20–21**: Writeup. Code cleanup. README.

---

## 12. Current status

**Day 1. Scaffold complete. Smoke test not yet run.**

Repo scaffolded at `vac/` with the following files:

```
vac/
├── README.md
├── .gitignore
├── pyproject.toml
├── docs/
│   └── day1.md          # setup, run, expected outcomes, troubleshooting
├── scripts/
│   └── smoke_test_nla.py
└── src/
    └── vac/
        └── __init__.py
```

The Day 1 smoke test (`scripts/smoke_test_nla.py`) generates layer-20
activations from 8 semantically diverse probe prompts (neutral factual, strong
emotion, Python code, math word problem, deception intent, refusal trigger,
multilingual, evaluation-aware) and writes a parquet for the upstream
`nla_inference.py` to run against. The `docs/day1.md` has the full 4-step
workflow.

**Immediate next step**: run the smoke test on GPU hardware. Report back with
the NLA explanations for the 8 probes. Decision branches:

- **Green** (visibly different explanations across probe categories): proceed
  to D2-3 agent harness + MCP server.
- **Yellow** (partial): note which probes fail; decide whether to proceed or
  investigate `injection_scale` / layer issues.
- **Red** (gibberish): debug against upstream `docs/inference.md` before
  writing agent code.

---

## 13. Formal proposal (course submission)

> **Title**: Verbalizing Hidden Cognition in Tool-Using Agents
>
> **Text source**: A purpose-built corpus of ~500 agentic trajectories
> generated by Qwen2.5-7B-Instruct under three conditions — benign tool use,
> prompt-injected tool outputs, and agentic-pressure scenarios adapted from
> recent agentic-misalignment evaluations. Each trajectory logs reasoning,
> tool calls, tool returns, and residual-stream activations at decision points.
>
> **Model architecture**: Qwen2.5-7B-Instruct as the frozen agent backbone,
> instrumented with the released Natural Language Autoencoders (Anthropic /
> Transformer Circuits, 2026) for Qwen-7B at layer 20. The NLA pair consists
> of an Activation Verbalizer (AV) that takes a residual-stream vector
> injected as a token embedding and autoregresses a description of what the
> vector encodes, and an Activation Reconstructor (AR) that maps the
> description back to a vector; round-trip MSE on L2-normalized vectors is the
> fidelity metric. The released AV was trained on FineWeb text and is out-of-
> distribution on agentic activations, so I will LoRA fine-tune it on agent
> decision-point activations to close the domain gap, with linear probes on
> hidden states as baselines.
>
> **Extra Criteria**: Explainable AI (the project's center of gravity —
> measuring the gap between agents' stated chain-of-thought and NLA-revealed
> internal cognition during consequential decisions) and MCP (tools exposed to
> the agent via a local MCP server).
