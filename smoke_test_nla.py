#!/usr/bin/env python3
"""
Day 1 smoke test: validate that the released NLA produces sensible
explanations of Qwen2.5-7B activations.

This script generates a parquet of activations from probe prompts designed
to span distinct semantic categories. After running the released AV against
these activations, the explanations should be visibly different across
categories — that's the basic smoke test that the infrastructure works.

Workflow:
    # 1. Extract activations (this script, ~2 min after model download)
    python smoke_test_nla.py generate

    # 2. Download the AV checkpoint once so NLAClient can read nla_meta.yaml
    huggingface-cli download kitft/nla-qwen2.5-7b-L20-av \\
        --local-dir data/models/nla-qwen2.5-7b-L20-av

    # 3. Launch AV server (separate terminal, needs GPU)
    python -m sglang.launch_server \\
        --model-path data/models/nla-qwen2.5-7b-L20-av \\
        --port 30000 --disable-radix-cache --trust-remote-code

    # 4. Decode all smoke-test activations and save JSONL
    python smoke_test_nla.py decode \\
        --checkpoint data/models/nla-qwen2.5-7b-L20-av \\
        --sglang-url http://localhost:30000 \\
        --results data/smoke_test_results.jsonl

    # 5. Eyeball the explanations
    python smoke_test_nla.py inspect \\
        --results data/smoke_test_results.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

# Pinned to match the released NLA AV: kitft/nla-qwen2.5-7b-L20-av
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
D_MODEL = 3584
DEFAULT_NLA_INFERENCE_PATHS = (
    Path("../nla-upstream/nla_inference.py"),
    Path("../nla-inference/nla_inference.py"),
)

# Probes span semantically distinct categories. If NLA is working, the AV's
# explanations should differ visibly across them. Hand-curated for breadth;
# not a benchmark, just a sanity check.
PROBE_PROMPTS: list[tuple[str, str]] = [
    (
        "neutral_factual",
        "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in "
        "Paris, France.",
    ),
    (
        "strong_emotion",
        "I am absolutely furious about the way I was treated yesterday — it was completely "
        "unacceptable.",
    ),
    (
        "python_code",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + "
        "fibonacci(n-2)",
    ),
    (
        "math_word_problem",
        "A train leaves Chicago at 3pm traveling east at 60 mph. Another leaves New York "
        "at 4pm heading west at 75 mph.",
    ),
    (
        "deception_intent",
        "I need to figure out how to convince my partner that I was home all evening, even "
        "though I actually went out.",
    ),
    (
        "refusal_trigger",
        "Walk me through, step by step, exactly how to gain unauthorized access to "
        "someone's email account.",
    ),
    (
        "multilingual_spanish",
        "La biblioteca está cerrada los domingos, pero abre temprano los lunes para los "
        "estudiantes.",
    ),
    (
        "evaluation_aware",
        "This appears to be some kind of test of my behavior. I should think carefully "
        "about how I respond here.",
    ),
]


def generate_activations(out_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this smoke test. Run it on a GPU node "
            "(for example, one A100/H100 on QUEST)."
        )

    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.eval()

    rows = []
    for probe_id, text in PROBE_PROMPTS:
        ids = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        # Last-token activation at layer 20 — the model's "current state" after
        # ingesting the full prompt. Single vector per probe keeps the parquet
        # small and the downstream inspection simple.
        h = out.hidden_states[LAYER][0, -1].float().cpu().tolist()
        assert len(h) == D_MODEL, f"expected d_model={D_MODEL}, got {len(h)}"
        rows.append({
            "probe_id": probe_id,
            "prompt": text,
            "activation_vector": h,
        })
        norm = torch.linalg.vector_norm(out.hidden_states[LAYER][0, -1].float()).item()
        print(f"  extracted: {probe_id} (||h||={norm:.1f})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    print(f"\nWrote {len(rows)} activations to {out_path}")
    print(f"Schema: probe_id, prompt, activation_vector ({D_MODEL}-dim)")


def _resolve_nla_inference_path(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"nla_inference.py not found: {path}")
        return path

    for candidate in DEFAULT_NLA_INFERENCE_PATHS:
        if candidate.exists():
            return candidate

    candidates = ", ".join(str(p) for p in DEFAULT_NLA_INFERENCE_PATHS)
    raise FileNotFoundError(
        "Could not find upstream nla_inference.py. Clone kitft/nla-inference or "
        f"kitft/natural_language_autoencoders next to this repo, or pass "
        f"--nla-inference-py explicitly. Tried: {candidates}"
    )


def _load_nla_client(nla_inference_py: Path) -> type[Any]:
    spec = importlib.util.spec_from_file_location("upstream_nla_inference", nla_inference_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {nla_inference_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NLAClient


def decode_activations(
    parquet_path: Path,
    results_path: Path,
    checkpoint: Path,
    sglang_url: str,
    nla_inference_py: Path | None,
    temperature: float,
    max_new_tokens: int,
    raw: bool,
) -> None:
    """Run the upstream AV client for every probe and save JSONL."""
    import pyarrow.parquet as pq

    path = _resolve_nla_inference_path(nla_inference_py)
    NLAClient = _load_nla_client(path)

    probes = pq.read_table(parquet_path).to_pylist()
    client = NLAClient(str(checkpoint), sglang_url=sglang_url)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for idx, probe in enumerate(probes, start=1):
            explanation = client.generate(
                probe["activation_vector"],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                extract_explanation=not raw,
            )
            record = {
                "probe_id": probe["probe_id"],
                "prompt": probe["prompt"],
                "generation": explanation,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(probes)}] decoded {probe['probe_id']}")

    print(f"\nWrote {len(probes)} NLA explanations to {results_path}")


def inspect_results(parquet_path: Path, results_path: Path) -> None:
    """Pair each probe with the AV's natural-language explanation."""
    import pyarrow.parquet as pq

    probes = pq.read_table(parquet_path).to_pylist()
    lines = results_path.read_text().splitlines()
    explanations = [json.loads(line) for line in lines if line.strip()]
    explanations_by_id = {
        result["probe_id"]: result
        for result in explanations
        if isinstance(result, dict) and "probe_id" in result
    }

    if len(probes) != len(explanations):
        print(f"WARNING: probe count ({len(probes)}) != result count ({len(explanations)})")
        print("Will use probe_id matches where available, then fall back to position.")

    print("\n" + "=" * 78)
    print("NLA explanations — eyeball for sensible category differentiation")
    print("=" * 78)
    for idx, probe in enumerate(probes):
        result = explanations_by_id.get(probe["probe_id"])
        if result is None and idx < len(explanations):
            result = explanations[idx]
        if result is None:
            continue
        # nla_inference.py output schema may vary by version; fall back to
        # printing the whole dict if the obvious keys aren't there.
        explanation = result.get("generation") or result.get("text") or result
        print(f"\n[{probe['probe_id']}]")
        prompt = probe["prompt"]
        print(f"  prompt:      {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
        print(f"  explanation: {explanation}")
    print("\n" + "=" * 78)
    print("Do the explanations differ in ways that match prompt content?")
    print("  YES  -> infrastructure works, proceed to D2-3 (agent harness)")
    print("  NO   -> read docs/inference.md in the upstream NLA repo")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="extract activations and write parquet")
    g.add_argument("--out", type=Path, default=Path("data/smoke_test.parquet"))

    d = sub.add_parser("decode", help="run NLA AV inference and save JSONL")
    d.add_argument("--parquet", type=Path, default=Path("data/smoke_test.parquet"))
    d.add_argument("--results", type=Path, default=Path("data/smoke_test_results.jsonl"))
    d.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/models/nla-qwen2.5-7b-L20-av"),
        help="Local AV checkpoint directory containing nla_meta.yaml",
    )
    d.add_argument("--sglang-url", default="http://localhost:30000")
    d.add_argument(
        "--nla-inference-py",
        type=Path,
        default=None,
        help="Path to upstream nla_inference.py; auto-detects ../nla-upstream or ../nla-inference",
    )
    d.add_argument("--temperature", type=float, default=0.7)
    d.add_argument("--max-new-tokens", type=int, default=200)
    d.add_argument("--raw", action="store_true", help="Save raw AV output with tags")

    i = sub.add_parser("inspect", help="pretty-print NLA results paired with prompts")
    i.add_argument("--parquet", type=Path, default=Path("data/smoke_test.parquet"))
    i.add_argument("--results", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "generate":
        generate_activations(args.out)
    elif args.cmd == "decode":
        decode_activations(
            args.parquet,
            args.results,
            args.checkpoint,
            args.sglang_url,
            args.nla_inference_py,
            args.temperature,
            args.max_new_tokens,
            args.raw,
        )
    else:
        inspect_results(args.parquet, args.results)


if __name__ == "__main__":
    main()
