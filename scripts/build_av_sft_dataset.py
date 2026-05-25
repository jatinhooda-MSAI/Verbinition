#!/usr/bin/env python3
"""Build supervised AV adaptation examples from MCP probe run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from vac.agent.scenarios import AGENTIC_PRESSURE, BENIGN, PROMPT_INJECTION


def load_trajectories(run_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    trajectories = {}
    for run_dir in run_dirs:
        path = run_dir / "trajectories.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                trajectories[row["scenario_id"]] = row
    return trajectories


def load_activation_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = []
    for run_dir in run_dirs:
        table = pq.read_table(run_dir / "activations.parquet")
        rows.extend(table.to_pylist())
    return rows


def split_for_id(scenario_id: str, eval_frac: float) -> str:
    bucket = int(hashlib.sha256(scenario_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < eval_frac else "train"


def first_tool_result(trajectory: dict[str, Any]) -> dict[str, Any]:
    for item in trajectory.get("tool_results", []):
        result = item.get("result", {})
        if result.get("ok"):
            return result
    return {}


def scenario_summary(trajectory: dict[str, Any]) -> str:
    condition = trajectory["condition"]
    expected = trajectory.get("expected_answer")
    injected = trajectory.get("injected_answer")
    result = first_tool_result(trajectory)
    content = str(result.get("content", ""))
    value = result.get("value")

    if condition == BENIGN and value is not None:
        return (
            f"benign calculator task with computed result {value}; no prompt "
            "injection or evaluation pressure is present"
        )
    if condition == BENIGN:
        return (
            f"benign document lookup; the relevant answer is {expected}; no "
            "prompt injection or evaluation pressure is present"
        )
    if condition == PROMPT_INJECTION:
        return (
            f"prompt-injection document task; the trusted answer is {expected}, "
            f"while untrusted page text tries to make the assistant answer "
            f"{injected}; the injected instruction should be ignored"
        )
    if condition == AGENTIC_PRESSURE:
        keywords = ", ".join(trajectory.get("pressure_keywords") or [])
        return (
            "agentic-pressure document task about evaluation, deadline or "
            f"deployment pressure; relevant pressure/logging details include "
            f"{keywords}; the assistant should report accurately and avoid "
            "altering logs or concealing uncertainty"
        )
    return content[:300]


def label_for_row(row: dict[str, Any], trajectory: dict[str, Any]) -> str:
    summary = scenario_summary(trajectory)
    kind = row["decision_kind"]
    diagnostic_kind = row.get("diagnostic_kind")
    expected_tool = trajectory.get("expected_tool")
    final_answer = trajectory.get("final_answer", "")

    if kind == "tool_call":
        return (
            f"The activation is from a tool-call decision. The assistant is "
            f"preparing a structured JSON MCP call to {expected_tool} for a "
            f"{trajectory['condition']} scenario: {summary}."
        )
    if diagnostic_kind == "tool_result_end":
        return (
            "The activation is immediately after an MCP observation. It combines "
            f"tool-result JSON formatting with this task content: {summary}."
        )
    if diagnostic_kind == "tool_content_plain_end":
        return (
            "The activation is from a plain-text diagnostic rendering of tool "
            f"content, with wrapper noise removed. It represents: {summary}."
        )
    if kind == "final_response":
        return (
            "The activation is before the final user-facing answer. The assistant "
            f"should answer based on the tool result as: {final_answer} "
            f"Key task content: {summary}."
        )
    return f"The activation represents this agent task state: {summary}."


def build_dataset(run_dirs: list[Path], out_path: Path, eval_frac: float) -> None:
    trajectories = load_trajectories(run_dirs)
    activation_rows = load_activation_rows(run_dirs)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in activation_rows:
            scenario_id = row["scenario_id"]
            trajectory = trajectories[scenario_id]
            record = {
                "probe_id": row["probe_id"],
                "scenario_id": scenario_id,
                "condition": row["condition"],
                "decision_kind": row["decision_kind"],
                "diagnostic_kind": row.get("diagnostic_kind"),
                "split": split_for_id(scenario_id, eval_frac),
                "activation_vector": row["activation_vector"],
                "target_text": label_for_row(row, trajectory),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} AV SFT examples to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Run directory containing trajectories.jsonl and activations.parquet.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-frac", type=float, default=0.15)
    args = parser.parse_args()
    build_dataset(args.run_dir, args.out, args.eval_frac)


if __name__ == "__main__":
    main()
