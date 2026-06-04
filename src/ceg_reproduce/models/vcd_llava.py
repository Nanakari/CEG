"""Local VCD-compatible LLaVA adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from ceg_reproduce.models.llava_hf import _format_llava_prompt, _resolve_dtype
from ceg_reproduce.models.types import GenerationResult


class VcdLlavaGenerator:
    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional model deps
            raise RuntimeError(
                "VCD generation requires torch, transformers, and a LLaVA checkpoint."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        generation = config.get("generation", {})
        runtime = config.get("runtime", {})
        vcd = config.get("vcd", {})
        self.model_name_or_path = str(
            generation.get("model_name_or_path", "llava-hf/llava-1.5-7b-hf")
        )
        self.device = str(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = _resolve_dtype(torch, str(runtime.get("dtype", "float16")), self.device)
        self.max_new_tokens = int(generation.get("caption_max_new_tokens", 64))
        self.do_sample = bool(generation.get("do_sample", False))
        self.cd_alpha = float(vcd.get("cd_alpha", 1.0))
        self.cd_beta = float(vcd.get("cd_beta", 0.1))
        self.noise_step = int(vcd.get("noise_step", 500))

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
    ) -> GenerationResult:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = self._image_cls.open(path).convert("RGB")
        formatted_prompt = _format_llava_prompt(self.processor, prompt)
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        pixel_values = inputs["pixel_values"].to(dtype=self.dtype)
        pixel_values_cd = _add_gaussian_noise(self._torch, pixel_values, self.noise_step)
        input_len = int(inputs["input_ids"].shape[-1])
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        max_tokens = int(max_new_tokens or self.max_new_tokens)

        start = time.perf_counter()
        with self._torch.inference_mode():
            for _ in range(max_tokens):
                original = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    return_dict=True,
                )
                distorted = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values_cd,
                    return_dict=True,
                )
                logits = original.logits[:, -1, :]
                logits_cd = distorted.logits[:, -1, :]
                cutoff = self._torch.log(self._torch.tensor(self.cd_beta, device=logits.device))
                cutoff = cutoff + logits.max(dim=-1, keepdim=True).values
                cd_logits = (1 + self.cd_alpha) * logits - self.cd_alpha * logits_cd
                cd_logits = cd_logits.masked_fill(logits < cutoff, -float("inf"))
                if self.do_sample:
                    probs = self._torch.softmax(cd_logits, dim=-1)
                    next_token = self._torch.multinomial(probs, num_samples=1)
                else:
                    next_token = self._torch.argmax(cd_logits, dim=-1, keepdim=True)
                input_ids = self._torch.cat([input_ids, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = self._torch.cat(
                        [attention_mask, self._torch.ones_like(next_token)], dim=-1
                    )
                eos = getattr(self.model.generation_config, "eos_token_id", None)
                if eos is not None and int(next_token[0, 0].item()) == int(eos):
                    break
        latency = time.perf_counter() - start
        generated_ids = input_ids[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            latency_sec=latency,
            metadata={
                "backend": "vcd_llava",
                "model_name_or_path": self.model_name_or_path,
                "cd_alpha": self.cd_alpha,
                "cd_beta": self.cd_beta,
                "noise_step": self.noise_step,
                "sample_id": sample_id,
            },
        )


def _add_gaussian_noise(torch: Any, tensor: Any, noise_step: int) -> Any:
    scale = max(0.0, min(float(noise_step) / 1000.0, 1.0))
    noisy = tensor + torch.randn_like(tensor) * scale
    return noisy.clamp(-3.0, 3.0)
