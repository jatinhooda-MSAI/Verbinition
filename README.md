# Verbalizing Hidden Cognition in Tool-Using Agents

Apply Natural Language Autoencoders (Fraser-Taliente et al., 2026) to the
activation streams of a Qwen2.5-7B agent during tool-use decisions. Measure
the gap between the agent's stated chain-of-thought and the internal cognition
surfaced by the NLA.

## Status

Day 2 of 21. Day 1 smoke test is green; current focus is one benign
tool-use trajectory with NLA-verbalized decision activations.

## Quick start

See [day1.md](day1.md) for the NLA smoke-test workflow and [day2.md](day2.md)
for the first agent tool-use trajectory.

## Three-week plan

- **Week 1** — NLA inference working end-to-end on Qwen-7B (D1–3). Agent
  harness with local MCP server (D4–6). One scenario end-to-end (D7).
- **Week 2** — Full scenario battery (benign / prompt-injection / agentic-
  pressure). ~500 trajectories. Initial NLA analysis. Linear-probe baselines.
- **Week 3** — LoRA fine-tune AV on agent activations to close the
  text→agent domain gap. Judge-model classification. Writeup.

## Layout

```
vac/
├── day1.md             setup notes and smoke-test workflow
├── day2.md             first benign agent trajectory workflow
├── smoke_test_nla.py   Day 1 activation extraction / NLA decode check
├── agent_probe.py      Day 2 one-tool agent trajectory extraction
├── src/vac/            library package namespace
├── pyproject.toml      Python dependencies and optional extras
└── data/               generated activations, checkpoints, results (gitignored)
```

## Upstream artifacts

- Paper: https://transformer-circuits.pub/2026/nla/
- Code: https://github.com/kitft/natural_language_autoencoders
- AV/AR checkpoints: https://huggingface.co/collections/kitft/nla-models
