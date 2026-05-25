#!/usr/bin/env python3
"""LoRA fine-tune the NLA Activation Verbalizer on agent-domain activations."""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class Example:
    probe_id: str
    split: str
    activation_vector: list[float]
    target_text: str


class AVSFTDataset:
    def __init__(self, path: Path, split: str) -> None:
        self.examples = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split", "train") == split:
                    self.examples.append(
                        Example(
                            probe_id=row["probe_id"],
                            split=row.get("split", "train"),
                            activation_vector=row["activation_vector"],
                            target_text=row["target_text"],
                        )
                    )
        if not self.examples:
            raise ValueError(f"no examples found for split={split!r} in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


class AVDataCollator:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        base_av: Path,
        max_target_tokens: int,
        train_tags: bool,
    ) -> None:
        from vac.nla.injection import (
            actor_prompt_ids,
            find_injection_position,
            load_actor_config,
            resolve_embed_scale,
        )

        self.tokenizer = tokenizer
        self.model = model
        self.cfg = load_actor_config(base_av, tokenizer)
        self.embed_scale = resolve_embed_scale(base_av)
        self.prompt_ids = actor_prompt_ids(tokenizer, self.cfg)
        self.injection_pos = find_injection_position(self.prompt_ids, self.cfg)
        self.max_target_tokens = max_target_tokens
        self.train_tags = train_tags

    def __call__(self, examples: list[Example]) -> dict[str, torch.Tensor]:
        import torch

        from vac.nla.injection import normalize_activation

        item_ids = []
        item_labels = []
        item_vectors = []
        for example in examples:
            target_text = example.target_text.strip()
            if self.train_tags:
                target_text = f"<explanation>{target_text}</explanation>"
            if self.tokenizer.eos_token:
                target_text += self.tokenizer.eos_token
            target_ids = self.tokenizer(
                target_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_target_tokens,
            )["input_ids"]
            ids = self.prompt_ids + target_ids
            labels = [-100] * len(self.prompt_ids) + target_ids
            item_ids.append(ids)
            item_labels.append(labels)
            item_vectors.append(example.activation_vector)

        max_len = max(len(ids) for ids in item_ids)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        padded_ids = []
        padded_labels = []
        attention_mask = []
        for ids, labels in zip(item_ids, item_labels, strict=True):
            pad = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad)
            padded_labels.append(labels + [-100] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)

        device = self.model.device
        ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
        labels_t = torch.tensor(padded_labels, dtype=torch.long, device=device)
        mask_t = torch.tensor(attention_mask, dtype=torch.long, device=device)
        vectors_t = torch.tensor(item_vectors, dtype=torch.float32, device=device)
        vectors_t = normalize_activation(vectors_t, self.cfg.injection_scale)

        embeds = self.model.get_input_embeddings()(ids_t) * self.embed_scale
        embeds = embeds.clone()
        embeds[:, self.injection_pos, :] = vectors_t.to(embeds.dtype)
        return {"inputs_embeds": embeds, "attention_mask": mask_t, "labels": labels_t}


def save_training_config(args: argparse.Namespace, out_dir: Path) -> None:
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (out_dir / "training_config.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evaluate(model: Any, loader: Any) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            losses.append(float(out.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AV LoRA training.")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_av), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_av),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(args.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = AVDataCollator(
        tokenizer=tokenizer,
        model=model,
        base_av=args.base_av,
        max_target_tokens=args.max_target_tokens,
        train_tags=not args.no_explanation_tags,
    )
    train_ds = AVSFTDataset(args.train, "train")
    eval_ds = AVSFTDataset(args.train, "eval")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    best_eval = float("inf")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += float(out.loss.detach().cpu())
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = running / args.log_every
                    print(
                        f"epoch={epoch} step={global_step}/{total_steps} "
                        f"train_loss={avg:.4f}"
                    )
                    running = 0.0

                if global_step % args.eval_every == 0:
                    eval_loss = evaluate(model, eval_loader)
                    print(f"epoch={epoch} step={global_step} eval_loss={eval_loss:.4f}")
                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        model.save_pretrained(out_dir / "best")
                        tokenizer.save_pretrained(out_dir / "best")

    final_eval = evaluate(model, eval_loader)
    print(f"final_eval_loss={final_eval:.4f}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    save_training_config(args, out_dir)
    meta_path = args.base_av / "nla_meta.yaml"
    if meta_path.exists():
        shutil.copy2(meta_path, out_dir / "nla_meta.yaml")
    print(f"Saved LoRA adapter to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-av", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-target-tokens", type=int, default=220)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-explanation-tags", action="store_true")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=list(DEFAULT_TARGET_MODULES),
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
