"""Embedding-injection utilities for local AV LoRA training."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoConfig


@dataclass(frozen=True)
class NLAActorConfig:
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    actor_prompt_template: str
    injection_scale: float


SCALED_EMBED_MODEL_TYPES = frozenset({"gemma", "gemma2", "gemma3", "gemma3_text", "t5"})


def load_actor_config(
    checkpoint_dir: str | Path,
    tokenizer: Any,
    injection_scale_override: float | None = None,
) -> NLAActorConfig:
    """Load and validate the released NLA sidecar for AV training."""
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "nla_meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"no nla_meta.yaml at {checkpoint_dir}")

    meta = yaml.safe_load(meta_path.read_text())
    kind = meta["kind"]
    d_model = meta["d_model"] if kind == "nla_model" else meta["extraction"]["d_model"]
    injection_scale = meta.get("extraction", {}).get("injection_scale")
    if injection_scale is None:
        injection_scale = injection_scale_override
    if injection_scale is None:
        raise ValueError("actor sidecar has no extraction.injection_scale")

    tokens = meta["tokens"]
    cfg = NLAActorConfig(
        d_model=int(d_model),
        injection_char=tokens["injection_char"],
        injection_token_id=int(tokens["injection_token_id"]),
        injection_left_neighbor_id=int(tokens["injection_left_neighbor_id"]),
        injection_right_neighbor_id=int(tokens["injection_right_neighbor_id"]),
        actor_prompt_template=meta["prompt_templates"].get("av")
        or meta["prompt_templates"]["actor"],
        injection_scale=float(injection_scale),
    )

    live_injection = tokenizer.encode(cfg.injection_char, add_special_tokens=False)
    if live_injection != [cfg.injection_token_id]:
        raise ValueError(
            f"tokenizer drift: {cfg.injection_char!r} -> {live_injection}, "
            f"sidecar says {[cfg.injection_token_id]}"
        )

    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    positions = [i for i, token_id in enumerate(ids) if token_id == cfg.injection_token_id]
    if len(positions) != 1:
        raise ValueError(f"expected one injection token in actor prompt, found {len(positions)}")
    pos = positions[0]
    if ids[pos - 1] != cfg.injection_left_neighbor_id:
        raise ValueError("left injection neighbor does not match sidecar")
    if ids[pos + 1] != cfg.injection_right_neighbor_id:
        raise ValueError("right injection neighbor does not match sidecar")
    return cfg


def resolve_embed_scale(checkpoint_dir: str | Path) -> float:
    """Return embedding forward scale for architectures such as Gemma."""
    config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    model_type = getattr(text_config, "model_type", "") or ""
    if model_type in SCALED_EMBED_MODEL_TYPES:
        return math.sqrt(text_config.hidden_size)
    return 1.0


def normalize_activation(vectors: torch.Tensor, target_scale: float) -> torch.Tensor:
    norms = vectors.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vectors / (norms / target_scale).to(vectors.dtype)


def actor_prompt_ids(tokenizer: Any, cfg: NLAActorConfig) -> list[int]:
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )


def find_injection_position(input_ids: list[int], cfg: NLAActorConfig) -> int:
    positions = [
        i
        for i, token_id in enumerate(input_ids)
        if token_id == cfg.injection_token_id
        and i > 0
        and i < len(input_ids) - 1
        and input_ids[i - 1] == cfg.injection_left_neighbor_id
        and input_ids[i + 1] == cfg.injection_right_neighbor_id
    ]
    if len(positions) != 1:
        raise ValueError(f"expected one valid injection position, found {len(positions)}")
    return positions[0]
