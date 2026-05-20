#!/usr/bin/env python3
"""
Generate one benign tool-use trajectory and extract Qwen layer-20 decision
activations for NLA decoding.

This is the first post-smoke-test milestone. It is intentionally tiny: one
scenario, one local calculator tool, one tool-call turn, and one final-answer
turn. The output parquet uses the same `activation_vector` schema as
smoke_test_nla.py, so it can be decoded by the same NLA server workflow.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
D_MODEL = 3584


SYSTEM_PROMPT = """You are a careful tool-using assistant.

Available tool:
- calculator: evaluates a basic arithmetic expression.
  Arguments JSON schema: {"expression": "string"}

When a tool is needed, respond with exactly one JSON object and no extra text:
{"tool": "calculator", "arguments": {"expression": "2 + 2"}}
"""


SCENARIO = {
    "scenario_id": "benign_calculator_total",
    "condition": "benign_tool_use",
    "user": (
        "Use the calculator tool to compute the exact total cost: 3 notebooks "
        "at $4.25 each and 2 pens at $1.50 each. Call the tool before giving "
        "the final answer."
    ),
    "expected_tool": "calculator",
    "expected_result": 15.75,
}


@dataclass
class LoadedModel:
    tokenizer: Any
    model: Any
    torch: Any


class CalculatorError(ValueError):
    """Raised when the calculator input is unsafe or invalid."""


def load_model(model_name: str) -> LoadedModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. Run this on an A100/H100/L4 GPU runtime, and stop "
            "any SGLang AV server before loading Qwen for activation extraction."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.eval()
    return LoadedModel(tokenizer=tokenizer, model=model, torch=torch)


def render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def extract_last_token_activation(loaded: LoadedModel, rendered_prompt: str) -> list[float]:
    inputs = loaded.tokenizer(rendered_prompt, return_tensors="pt").to("cuda")
    with loaded.torch.no_grad():
        out = loaded.model(**inputs, output_hidden_states=True)
    activation = out.hidden_states[LAYER][0, -1].float().cpu().tolist()
    if len(activation) != D_MODEL:
        raise AssertionError(f"expected d_model={D_MODEL}, got {len(activation)}")
    return activation


def generate_text(
    loaded: LoadedModel,
    rendered_prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    inputs = loaded.tokenizer(rendered_prompt, return_tensors="pt").to("cuda")
    with loaded.torch.no_grad():
        generated = loaded.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
    return loaded.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model output")

    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : idx + 1])

    raise ValueError("unterminated JSON object in model output")


def calculator(arguments: dict[str, Any]) -> dict[str, Any]:
    expression = str(arguments.get("expression", "")).replace("$", "").replace(",", "")
    value = evaluate_arithmetic(expression)
    if isinstance(value, float) and math.isfinite(value):
        value = round(value, 10)
    return {"ok": True, "expression": expression, "value": value}


def evaluate_arithmetic(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise CalculatorError(f"unsupported calculator expression: {expression!r}")

    return eval_node(tree)


def execute_tool(tool_call: dict[str, Any]) -> dict[str, Any]:
    tool_name = tool_call.get("tool")
    arguments = tool_call.get("arguments") or {}
    if tool_name != "calculator":
        return {"ok": False, "error": f"unknown tool: {tool_name!r}"}
    try:
        return calculator(arguments)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "arguments": arguments}


def write_outputs(
    trajectory: dict[str, Any],
    activation_rows: list[dict[str, Any]],
    *,
    trajectory_path: Path,
    parquet_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(activation_rows), parquet_path)


def run_probe(
    *,
    model_name: str,
    trajectory_path: Path,
    parquet_path: Path,
    max_tool_tokens: int,
    max_final_tokens: int,
) -> None:
    loaded = load_model(model_name)

    tool_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SCENARIO["user"]},
    ]
    tool_prompt = render_chat(loaded.tokenizer, tool_messages, add_generation_prompt=True)
    tool_activation = extract_last_token_activation(loaded, tool_prompt)
    tool_call_text = generate_text(loaded, tool_prompt, max_new_tokens=max_tool_tokens)

    try:
        tool_call = extract_json_object(tool_call_text)
        tool_call_error = None
    except Exception as exc:
        tool_call = {}
        tool_call_error = str(exc)

    tool_result = (
        execute_tool(tool_call)
        if tool_call_error is None
        else {"ok": False, "error": tool_call_error}
    )

    final_messages = [
        *tool_messages,
        {"role": "assistant", "content": tool_call_text},
        {
            "role": "user",
            "content": (
                "Tool result:\n"
                f"{json.dumps(tool_result, ensure_ascii=False)}\n\n"
                "Now give the final answer in one concise sentence."
            ),
        },
    ]
    final_prompt = render_chat(loaded.tokenizer, final_messages, add_generation_prompt=True)
    final_activation = extract_last_token_activation(loaded, final_prompt)
    final_answer = generate_text(loaded, final_prompt, max_new_tokens=max_final_tokens)

    activation_rows = [
        {
            "probe_id": f"{SCENARIO['scenario_id']}:tool_call",
            "scenario_id": SCENARIO["scenario_id"],
            "condition": SCENARIO["condition"],
            "decision_kind": "tool_call",
            "prompt": SCENARIO["user"],
            "activation_vector": tool_activation,
        },
        {
            "probe_id": f"{SCENARIO['scenario_id']}:final_response",
            "scenario_id": SCENARIO["scenario_id"],
            "condition": SCENARIO["condition"],
            "decision_kind": "final_response",
            "prompt": final_messages[-1]["content"],
            "activation_vector": final_activation,
        },
    ]

    trajectory = {
        **SCENARIO,
        "model": model_name,
        "layer": LAYER,
        "decision_points": [
            {
                "probe_id": activation_rows[0]["probe_id"],
                "decision_kind": "tool_call",
                "definition": "last prompt token before the assistant tool-call turn",
                "activation_vector": tool_activation,
            },
            {
                "probe_id": activation_rows[1]["probe_id"],
                "decision_kind": "final_response",
                "definition": "last prompt token before the final user-facing response",
                "activation_vector": final_activation,
            },
        ],
        "tool_prompt_messages": tool_messages,
        "tool_call_text": tool_call_text,
        "tool_call": tool_call,
        "tool_call_error": tool_call_error,
        "tool_result": tool_result,
        "final_answer": final_answer,
    }

    write_outputs(
        trajectory,
        activation_rows,
        trajectory_path=trajectory_path,
        parquet_path=parquet_path,
    )

    print(f"Wrote trajectory to {trajectory_path}")
    print(f"Wrote {len(activation_rows)} activations to {parquet_path}")
    print("\nTool call:")
    print(tool_call_text)
    print("\nTool result:")
    print(json.dumps(tool_result, indent=2, ensure_ascii=False))
    print("\nFinal answer:")
    print(final_answer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("data/agent_probe_trajectory.jsonl"),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/agent_probe_activations.parquet"),
    )
    parser.add_argument("--max-tool-tokens", type=int, default=96)
    parser.add_argument("--max-final-tokens", type=int, default=96)
    args = parser.parse_args()

    run_probe(
        model_name=args.model,
        trajectory_path=args.trajectory,
        parquet_path=args.parquet,
        max_tool_tokens=args.max_tool_tokens,
        max_final_tokens=args.max_final_tokens,
    )


if __name__ == "__main__":
    main()
