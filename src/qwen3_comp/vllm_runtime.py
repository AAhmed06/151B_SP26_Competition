"""vLLM-based generation for Qwen3-4B-Thinking.

This module is the only place that loads the model. It refuses to load
on a non-CUDA machine so that no inference accidentally runs locally:
the plan requires real generation to happen inside a DSMLP GPU
container.

The vLLM runtime is preferred for speed. ``transformers`` is kept only
as a documented emergency fallback; pass ``fallback="transformers"`` if
vLLM cannot start in the assigned container.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .prompts import build_messages


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


@dataclass
class SamplingConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0
    seed: Optional[int] = None


@dataclass
class GenerationRequest:
    """A single request to the engine.

    ``max_new_tokens`` is per-request so the budget router can give
    short questions a small budget and long questions a large one
    without restarting the engine.
    """

    item: dict
    prompt_id: str
    max_new_tokens: int
    n: int = 1
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


def _require_cuda() -> None:
    """Refuse to start the engine outside a CUDA environment.

    Set ``QWEN3_COMP_ALLOW_NO_GPU=1`` to bypass for tests that mock the
    engine. Never set this when actually generating real submissions.

    Uses device_count() instead of is_available() to avoid fully
    initializing CUDA (which would break vLLM's fork-based engine).
    """
    if os.environ.get("QWEN3_COMP_ALLOW_NO_GPU") == "1":
        return
    try:
        import torch  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - torch always present on GPU host
        raise RuntimeError(
            "torch is not installed. Run this inside a DSMLP GPU container."
        ) from exc
    if torch.cuda.device_count() == 0:
        raise RuntimeError(
            "CUDA is not available. Inference must run inside a DSMLP GPU "
            "container (launch-scipy-ml.sh -g 1). Set "
            "QWEN3_COMP_ALLOW_NO_GPU=1 only for tests that mock the engine."
        )


class VLLMEngine:
    """Thin wrapper around ``vllm.LLM`` that hides chat-template plumbing.

    The engine is constructed lazily so importing this module does not
    require vLLM (handy for unit tests on a laptop). Pass
    ``fallback="transformers"`` to use the slower bitsandbytes path if
    vLLM cannot be installed in the container.
    """

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 16384,
        quantization: Optional[str] = "bitsandbytes",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        seed: int = 0,
        enforce_eager: bool = False,
        backend: str = "vllm",
    ) -> None:
        _require_cuda()
        self.model_id = model_id
        self.backend = backend
        self._tokenizer = None
        self._engine = None
        self._tf_model = None
        self._init_kwargs = {
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "quantization": quantization,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "seed": seed,
            "enforce_eager": enforce_eager,
        }

    def _ensure_loaded(self) -> None:
        if self._engine is not None or self._tf_model is not None:
            return

        from transformers import AutoTokenizer  # noqa: WPS433

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if self.backend == "vllm":
            from vllm import LLM  # noqa: WPS433

            self._engine = LLM(
                model=self.model_id,
                quantization=self._init_kwargs["quantization"],
                load_format=self._init_kwargs["quantization"]
                if self._init_kwargs["quantization"] == "bitsandbytes"
                else "auto",
                gpu_memory_utilization=self._init_kwargs["gpu_memory_utilization"],
                max_model_len=self._init_kwargs["max_model_len"],
                trust_remote_code=self._init_kwargs["trust_remote_code"],
                dtype=self._init_kwargs["dtype"],
                seed=self._init_kwargs["seed"],
                enforce_eager=self._init_kwargs["enforce_eager"],
            )
        elif self.backend == "transformers":
            import torch  # noqa: WPS433
            from transformers import (  # noqa: WPS433
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self._tf_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                quantization_config=bnb,
                device_map="auto",
            )
        else:
            raise ValueError(f"unknown backend: {self.backend!r}")

    def _format_prompt(self, item: dict, prompt_id: str) -> str:
        messages = build_messages(item, prompt_id=prompt_id)
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, requests: list[GenerationRequest]) -> list[list[str]]:
        """Generate ``n`` completions per request.

        Returns a list (one entry per request) of length-``n`` lists of
        decoded response strings, with the chat-template prompt
        stripped. Order matches *requests*.
        """
        if not requests:
            return []

        self._ensure_loaded()

        if self.backend == "vllm":
            return self._generate_vllm(requests)
        return self._generate_transformers(requests)

    def _generate_vllm(
        self, requests: list[GenerationRequest]
    ) -> list[list[str]]:
        from vllm import SamplingParams  # noqa: WPS433

        prompts: list[str] = []
        sampling_list: list[SamplingParams] = []
        for req in requests:
            prompts.append(self._format_prompt(req.item, req.prompt_id))
            sampling_list.append(
                SamplingParams(
                    n=req.n,
                    max_tokens=req.max_new_tokens,
                    temperature=req.sampling.temperature,
                    top_p=req.sampling.top_p,
                    top_k=req.sampling.top_k,
                    repetition_penalty=req.sampling.repetition_penalty,
                    seed=req.sampling.seed,
                )
            )

        outputs = self._engine.generate(prompts, sampling_params=sampling_list)
        # vLLM returns one RequestOutput per prompt in order
        result: list[list[str]] = []
        for out in outputs:
            result.append([cand.text.strip() for cand in out.outputs])
        return result

    def _generate_transformers(
        self, requests: list[GenerationRequest]
    ) -> list[list[str]]:
        import gc

        import torch  # noqa: WPS433

        results: list[list[str]] = []
        for req in requests:
            prompt_text = self._format_prompt(req.item, req.prompt_id)
            inputs = self._tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self._init_kwargs["max_model_len"] // 2,
            ).to(self._tf_model.device)

            samples: list[str] = []
            for _ in range(req.n):
                with torch.no_grad():
                    output_ids = self._tf_model.generate(
                        **inputs,
                        max_new_tokens=req.max_new_tokens,
                        do_sample=True,
                        temperature=req.sampling.temperature,
                        top_p=req.sampling.top_p,
                        top_k=req.sampling.top_k,
                        repetition_penalty=req.sampling.repetition_penalty,
                        pad_token_id=self._tokenizer.eos_token_id,
                        use_cache=True,
                    )
                new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
                samples.append(
                    self._tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                )
                del output_ids, new_tokens
                gc.collect()
                torch.cuda.empty_cache()
            del inputs
            results.append(samples)
        return results
