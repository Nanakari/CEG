"""Hugging Face LLaVA image-text generation adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from ceg_reproduce.models.types import GenerationResult


class LlavaHfGenerator:
    """Generate captions with `transformers` LLaVA checkpoints."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional model deps
            raise RuntimeError(
                "LLaVA generation requires the model stack. Install "
                "`requirements-models-cu12.txt` and a compatible LLaVA checkpoint."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        generation = config.get("generation", {})
        runtime = config.get("runtime", {})
        self.model_name_or_path = str(
            generation.get("model_name_or_path", "llava-hf/llava-1.5-7b-hf")
        )
        self.device = str(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = _resolve_dtype(torch, str(runtime.get("dtype", "float16")), self.device)
        self.temperature = float(generation.get("temperature", 0.0))
        self.top_p = float(generation.get("top_p", 1.0))
        self.do_sample = bool(generation.get("do_sample", False))
        self.max_new_tokens = int(generation.get("caption_max_new_tokens", 64))

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        sample_id: str | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> GenerationResult:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = self._image_cls.open(path).convert("RGB")
        formatted_prompt = _format_llava_prompt(self.processor, prompt)
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])
        effective_do_sample = self.do_sample if do_sample is None else bool(do_sample)
        effective_temperature = self.temperature if temperature is None else float(temperature)
        effective_top_p = self.top_p if top_p is None else float(top_p)
        effective_max_new_tokens = int(max_new_tokens or self.max_new_tokens)
        kwargs: dict[str, Any] = {
            "max_new_tokens": effective_max_new_tokens,
            "do_sample": effective_do_sample,
            "top_p": effective_top_p,
        }
        if effective_do_sample:
            kwargs["temperature"] = effective_temperature

        start = time.perf_counter()
        with self._torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        latency = time.perf_counter() - start
        generated_ids = outputs[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            latency_sec=latency,
            metadata={
                "backend": "llava_hf",
                "model_name_or_path": self.model_name_or_path,
                "sample_id": sample_id,
                "max_new_tokens": effective_max_new_tokens,
                "do_sample": effective_do_sample,
                "temperature": effective_temperature,
                "top_p": effective_top_p,
            },
        )


def _format_llava_prompt(processor: Any, prompt: str) -> str:
    if _looks_preformatted_prompt(prompt):
        return prompt
    text = prompt.replace("<image>", "").replace("<IMAGE>", "").strip()
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
        try:
            return processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    return f"USER: <image>\n{text}\nASSISTANT:"


def _looks_preformatted_prompt(prompt: str) -> bool:
    normalized = prompt.upper()
    return "<IMAGE>" in normalized and ("USER:" in normalized or "ASSISTANT:" in normalized)


def _resolve_dtype(torch: Any, dtype_name: str, device: str) -> Any:
    if device == "cpu":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported runtime.dtype: {dtype_name}")
    return mapping[dtype_name]
