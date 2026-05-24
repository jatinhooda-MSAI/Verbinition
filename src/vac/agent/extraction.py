"""Qwen loading, chat rendering, generation, and activation extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
D_MODEL = 3584


@dataclass
class LoadedModel:
    tokenizer: Any
    model: Any
    torch: Any


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
