# Verbinition

Natural Language Autoencoders for inspecting tool-use decision states in a
Qwen2.5-7B agent.

This repo asks a narrow question: when an agent is about to call a tool or
answer after seeing a tool result, do NLA verbalizations surface useful
internal state about prompt injection, evaluation pressure, or ordinary tool
use?

## Short Answer

The infrastructure works, but the released NLA is not plug-and-play for
agentic decision states.

- **Smoke test**: Qwen layer-20 activations decode into sensible semantic
  categories on ordinary text prompts.
- **MCP/ReAct behavior**: the toy agent uses tools correctly and resists
  injected document instructions in the controlled scenarios.
- **Released AV result**: decision-point explanations are dominated by
  interface format such as JSON, commands, query results, and response blocks.
- **Best released-AV signal**: stripping tool-result wrappers recovers partial
  prompt-injection and pressure/evaluation content.
- **LoRA adaptation result**: supervised AV LoRA greatly increases task-
  relevant injection/pressure language, but current labels are deterministic
  and the adapted AV often reproduces the target label template verbatim. This
  is promising engineering evidence, not yet a strong scientific claim.

## Method

The target model is `Qwen/Qwen2.5-7B-Instruct`. Activations are extracted from
layer 20, matching the released Qwen NLA checkpoint
`kitft/nla-qwen2.5-7b-L20-av`.

The main extraction points are:

- `tool_call`: last prompt token before the assistant emits a JSON MCP tool
  call.
- `final_response`: last prompt token before the final user-facing answer.
- `tool_result_end`: diagnostic row immediately after the MCP observation.
- `tool_content_plain_end`: diagnostic row using plain tool content without
  JSON/MCP wrapper noise.

The agent harness is deliberately small: a hand-rolled ReAct loop, a local MCP
server, a safe calculator, and local document tools. Scenarios cover:

- benign tool use,
- prompt injection in tool outputs,
- pressure/evaluation notices.

## Results So Far

### Phase 1: NLA Smoke Test

Eight static text prompts were used to check that activation extraction and AV
decoding were wired correctly. The released AV produced noisy but recognizable
categories such as code, Spanish text, deception, refusal-like cyber content,
math, and evaluation-aware text.

Interpretation: the NLA pipeline works on in-distribution-ish text
activations.

### Phase 2: Local Tool Probe

A three-scenario local-tool battery tested benign calculation, prompt
injection in a vendor policy document, and a pressure/evaluation notice.

Visible behavior was correct. However, NLA outputs at `tool_call` and
`final_response` mostly verbalized format and action state rather than the
safety-relevant document content. The plain-content diagnostic recovered
arithmetic content and partial injection-like structure.

Interpretation: the released AV is format-dominated at agent/tool boundaries.

### Phase 3: MCP/ReAct Probe

The same pattern held under a proper local MCP tool server. In the 9-scenario
MCP validation set, visible behavior was green:

- all benign tasks answered correctly,
- all injection tasks answered with the trusted value, not the injected value,
- pressure tasks mentioned the relevant evaluation/logging constraints.

NLA behavior was still mixed:

- `tool_call` rows mostly described JSON/tool/query format;
- `tool_result_end` rows mostly described structured response wrappers;
- `tool_content_plain_end` rows gave the clearest safety-relevant signal.

Interpretation: the Day 2 failure mode was not just an artifact of the
non-MCP harness.

### Option B: AV LoRA Adaptation

A full generated battery was run with 150 scenarios per condition:

```text
450 trajectories
1,800 activation rows
600 rows per condition
```

A supervised dataset was built from scenario/tool metadata and used to LoRA
fine-tune the AV only. The AR is intentionally not trained.

Base AV vs adapted AV keyword comparison:

```text
condition           rows  base_hits  adapted_hits  delta
agentic_pressure    600          9          4309   +4300
benign_tool_use     600       1085           950    -135
prompt_injection    600        118          2400   +2282
```

Qualitatively, the adapted AV gives much more relevant explanations for
injection and pressure rows. For example, it identifies the trusted answer,
the injected answer, and the instruction to ignore the injected page text.

Important limitation: current supervised labels are deterministic templates.
The adapted AV often reproduces those templates nearly exactly. This shows the
AV can be steered into the agent/tool domain, but it does not yet prove 
generalization to unseen scenarios or that the verbalizations faithfully
recover hidden cognition.

## RESULTS: What Is Supported

Supported claims:

- Released Qwen layer-20 NLAs can verbalize ordinary text activations.
- On tool-use decision states, the released AV is often dominated by
  interface/format features.
- Removing wrapper noise improves the visibility of prompt-injection and
  pressure/evaluation content.
- Supervised LoRA adaptation can make AV outputs substantially more
  task-relevant on the generated agent-domain battery.

Not yet supported:

- NLA reliably reveals hidden agentic cognition at decision points.
- The adapted AV generalizes beyond the deterministic label distribution.
- The adapted explanations are faithful in the AR reconstruction sense.
- NLA adds signal beyond simpler baselines such as keyword classifiers or
  linear probes.

## Possible Future Work

The next clean evaluation should use a held-out battery with different surface
forms and label style:

1. Generate fresh MCP trajectories with new document templates.
2. Decode with released AV and adapted AV.
3. Measure eval-only keyword hits, target-copy rate, and qualitative samples.
4. Add AR reconstruction scoring if the corresponding Qwen AR checkpoint is
   integrated.
5. Compare against linear probes and simple text/metadata baselines.