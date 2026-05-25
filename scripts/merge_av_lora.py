#!/usr/bin/env python3
"""Merge a trained AV LoRA adapter into a standalone HF checkpoint."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def merge(base_av: Path, adapter: Path, out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_av),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(base_av), trust_remote_code=True)
    tokenizer.save_pretrained(str(out))

    meta_path = base_av / "nla_meta.yaml"
    if meta_path.exists():
        shutil.copy2(meta_path, out / "nla_meta.yaml")
    print(f"Saved merged AV checkpoint to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-av", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    merge(args.base_av, args.adapter, args.out)


if __name__ == "__main__":
    main()
