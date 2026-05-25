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
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from vac.agent.extraction import MODEL_NAME
from vac.agent.generated_battery import generate_battery
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
        "--battery",
        choices=("static", "full"),
        default="static",
        help="Use the original 9 static scenarios or generate a balanced full battery.",
    )
    parser.add_argument(
        "--per-condition",
        type=int,
        default=150,
        help="Number of scenarios per condition for --battery full.",
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
        if args.battery == "full":
            scenarios = generate_battery(args.per_condition).scenarios
        else:
            scenarios = list(SCENARIOS)
        for scenario in scenarios:
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
    scenarios_override = None
    documents_path = None
    if args.battery == "full":
        battery = generate_battery(args.per_condition)
        scenario_pool = {scenario.scenario_id: scenario for scenario in battery.scenarios}
        if "all" in selected:
            scenarios_override = battery.scenarios
        else:
            unknown = [scenario_id for scenario_id in selected if scenario_id not in scenario_pool]
            if unknown:
                valid_preview = ", ".join(["all", *list(scenario_pool)[:20]])
                raise ValueError(
                    f"unknown generated scenario(s): {unknown}. "
                    f"Valid examples: {valid_preview}, ..."
                )
            scenarios_override = [scenario_pool[scenario_id] for scenario_id in selected]
        documents_path = trajectory_path.parent / "generated_documents.json"
        documents_path.parent.mkdir(parents=True, exist_ok=True)
        documents_path.write_text(
            json.dumps(battery.documents, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"Run id: {run_id}")
    print(f"Model: {args.model}")
    if scenarios_override is None:
        printable_scenarios = scenario_ids() if "all" in selected else selected
        scenario_label = ", ".join(printable_scenarios)
    else:
        printable_scenarios = [scenario.scenario_id for scenario in scenarios_override]
        preview = ", ".join(printable_scenarios[:12])
        suffix = " ..." if len(printable_scenarios) > 12 else ""
        scenario_label = f"{len(printable_scenarios)} total: {preview}{suffix}"
    print(f"Battery: {args.battery}")
    print(f"Scenarios: {scenario_label}")
    print(f"Trajectory path: {trajectory_path}")
    print(f"Activation parquet path: {parquet_path}")
    print(f"Env path: {env_path}")
    if documents_path is not None:
        print(f"Generated documents path: {documents_path}")

    asyncio.run(
        run_probe(
            run_id=run_id,
            model_name=args.model,
            scenario_ids=selected,
            trajectory_path=trajectory_path,
            parquet_path=parquet_path,
            env_path=env_path,
            documents_path=documents_path,
            scenarios_override=scenarios_override,
            max_tool_steps=args.max_tool_steps,
            max_tool_tokens=args.max_tool_tokens,
            max_final_tokens=args.max_final_tokens,
        )
    )


if __name__ == "__main__":
    main()
