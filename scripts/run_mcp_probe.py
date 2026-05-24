#!/usr/bin/env python3
"""
Run the Option A MCP/ReAct probe battery and extract Qwen layer-20 activations.

This is the first proper MCP harness after the Day 2 local-tool diagnostic.
It launches a local stdio MCP server, asks Qwen2.5-7B-Instruct to make
ReAct-style JSON tool calls, executes those calls through MCP, and saves both
full trajectories and NLA-ready activation rows under a named run folder.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from vac.agent.extraction import MODEL_NAME
from vac.agent.harness import default_run_id, resolve_output_paths, run_probe
from vac.agent.scenarios import SCENARIOS, scenario_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Name for this artifact set. Defaults to "
            "mcp_probe_YYYYMMDD_HHMMSS and writes under data/runs/<run-id>/."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario id to run; repeat for multiple. Use 'all' for the full small battery.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenario ids and exit.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Optional explicit trajectory JSONL path.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Optional explicit activation parquet path.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=None,
        help="Optional explicit env metadata path.",
    )
    parser.add_argument("--max-tool-steps", type=int, default=2)
    parser.add_argument("--max-tool-tokens", type=int, default=128)
    parser.add_argument("--max-final-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.list_scenarios:
        for scenario in SCENARIOS:
            print(f"{scenario.scenario_id}\t{scenario.condition}")
        return

    selected = args.scenario or ["all"]
    run_id = args.run_id or default_run_id()
    trajectory_path, parquet_path, env_path = resolve_output_paths(
        run_id=run_id,
        trajectory_path=args.trajectory,
        parquet_path=args.parquet,
        env_path=args.env,
    )

    print(f"Run id: {run_id}")
    print(f"Model: {args.model}")
    print(f"Scenarios: {', '.join(scenario_ids() if 'all' in selected else selected)}")
    print(f"Trajectory path: {trajectory_path}")
    print(f"Activation parquet path: {parquet_path}")
    print(f"Env path: {env_path}")

    asyncio.run(
        run_probe(
            run_id=run_id,
            model_name=args.model,
            scenario_ids=selected,
            trajectory_path=trajectory_path,
            parquet_path=parquet_path,
            env_path=env_path,
            max_tool_steps=args.max_tool_steps,
            max_tool_tokens=args.max_tool_tokens,
            max_final_tokens=args.max_final_tokens,
        )
    )


if __name__ == "__main__":
    main()
