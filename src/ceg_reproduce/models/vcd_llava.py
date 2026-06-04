"""Local VCD-compatible LLaVA adapter with cache-aware decoding."""

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
        self.do_sample = bool(generation.get("do_sample", True))
        self.temperature = float(generation.get("temperature", 1.0))
        self.top_p = float(generation.get("top_p", 1.0))
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
        pixel_values = inputs["pixel_values"].to(dtype=self.dtype)
        pixel_values_cd = _add_diffusion_noise(self._torch, pixel_values, self.noise_step).to(
            dtype=self.dtype
        )
        effective_max_new_tokens = int(max_new_tokens or self.max_new_tokens)
        effective_do_sample = self.do_sample if do_sample is None else bool(do_sample)
        effective_temperature = self.temperature if temperature is None else float(temperature)
        effective_top_p = self.top_p if top_p is None else float(top_p)
        input_len = int(inputs["input_ids"].shape[-1])
        extra_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"input_ids", "attention_mask", "pixel_values"}
        }

        start = time.perf_counter()
        with self._torch.inference_mode():
            output_ids = self._decode_with_cache(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=pixel_values,
                pixel_values_cd=pixel_values_cd,
                max_new_tokens=effective_max_new_tokens,
                do_sample=effective_do_sample,
                temperature=effective_temperature,
                top_p=effective_top_p,
                extra_inputs=extra_inputs,
            )
        latency = time.perf_counter() - start
        generated_ids = output_ids[0][input_len:]
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
                "max_new_tokens": effective_max_new_tokens,
                "do_sample": effective_do_sample,
                "temperature": effective_temperature,
                "top_p": effective_top_p,
                "vcd_decode_impl": "cache_aware_official_formula",
                "uses_kv_cache": True,
                "noise_impl": "official_diffusion_schedule",
            },
        )

    def _decode_with_cache(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        pixel_values: Any,
        pixel_values_cd: Any,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        extra_inputs: dict[str, Any],
    ) -> Any:
        torch = self._torch
        output_ids = input_ids
        step_input_ids = input_ids
        past = None
        past_cd = None
        eos_ids = _eos_token_ids(self.model.generation_config.eos_token_id)

        for step in range(max_new_tokens):
            common_kwargs = {
                "input_ids": step_input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
                "use_cache": True,
                **extra_inputs,
            }
            if past is not None:
                common_kwargs["past_key_values"] = past
            else:
                common_kwargs["pixel_values"] = pixel_values
            original = self.model(**common_kwargs)

            cd_kwargs = dict(common_kwargs)
            if past_cd is not None:
                cd_kwargs["past_key_values"] = past_cd
                cd_kwargs.pop("pixel_values", None)
            else:
                cd_kwargs["pixel_values"] = pixel_values_cd
            distorted = self.model(**cd_kwargs)

            logits = original.logits[:, -1, :]
            logits_cd = distorted.logits[:, -1, :]
            cd_logits = _contrastive_logits(torch, logits, logits_cd, self.cd_alpha, self.cd_beta)
            next_token = _select_next_token(
                torch,
                cd_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            output_ids = torch.cat([output_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            past = getattr(original, "past_key_values", None)
            past_cd = getattr(distorted, "past_key_values", None)
            step_input_ids = next_token
            if eos_ids and int(next_token[0, 0].item()) in eos_ids:
                break
            if step == 0 and (past is None or past_cd is None):
                # Fail fast instead of silently falling back to the slow full-prefix loop.
                raise RuntimeError("VCD cache-aware decoding did not receive past_key_values.")
        return output_ids


def _contrastive_logits(torch: Any, logits: Any, logits_cd: Any, cd_alpha: float, cd_beta: float) -> Any:
    cutoff = torch.log(torch.tensor(cd_beta, device=logits.device, dtype=logits.dtype))
    cutoff = cutoff + logits.max(dim=-1, keepdim=True).values
    cd_logits = (1 + cd_alpha) * logits - cd_alpha * logits_cd
    return cd_logits.masked_fill(logits < cutoff, -float("inf"))


def _select_next_token(
    torch: Any,
    logits: Any,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Any:
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if temperature and temperature > 0:
        logits = logits / temperature
    if top_p < 1.0:
        logits = _top_p_filter(torch, logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _top_p_filter(torch: Any, logits: Any, top_p: float) -> Any:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    filtered = logits.clone()
    remove_indices = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return filtered.masked_fill(remove_indices, -float("inf"))


def _add_diffusion_noise(torch: Any, tensor: Any, noise_step: int) -> Any:
    num_steps = 1000
    device = tensor.device
    dtype = tensor.dtype
    betas = torch.linspace(-6, 6, num_steps, device=device, dtype=torch.float32)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    t = max(0, min(int(noise_step), num_steps - 1))
    noise = torch.randn_like(tensor)
    noisy = alphas_bar_sqrt[t].to(dtype=dtype) * tensor + one_minus_alphas_bar_sqrt[t].to(dtype=dtype) * noise
    return noisy


def _eos_token_ids(eos_token_id: Any) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {int(eos_token_id)}
    return {int(item) for item in eos_token_id}
