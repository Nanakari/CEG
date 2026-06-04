from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "tensor") or not hasattr(torch, "manual_seed"):
    pytest.skip("VCD adapter tests require a functional torch install.", allow_module_level=True)

from ceg_reproduce.models.vcd_llava import VcdLlavaGenerator, _add_diffusion_noise


def test_diffusion_noise_preserves_shape_dtype_and_changes_values() -> None:
    torch.manual_seed(0)
    tensor = torch.zeros((1, 3, 4, 4), dtype=torch.float32)

    noisy = _add_diffusion_noise(torch, tensor, 500)

    assert noisy.shape == tensor.shape
    assert noisy.dtype == tensor.dtype
    assert not torch.equal(noisy, tensor)


def test_vcd_decode_uses_single_token_after_cache_is_created() -> None:
    generator = VcdLlavaGenerator.__new__(VcdLlavaGenerator)
    generator._torch = torch
    generator.model = CacheRecordingModel()
    generator.cd_alpha = 1.0
    generator.cd_beta = 0.1

    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.ones_like(input_ids)
    output = generator._decode_with_cache(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=torch.zeros((1, 3, 4, 4)),
        pixel_values_cd=torch.ones((1, 3, 4, 4)),
        max_new_tokens=2,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        extra_inputs={},
    )

    assert output.shape[-1] == 4
    assert generator.model.input_lengths == [2, 2, 1, 1]
    assert generator.model.cache_flags == [False, False, True, True]


class CacheRecordingModel:
    def __init__(self) -> None:
        self.generation_config = SimpleNamespace(eos_token_id=99)
        self.input_lengths: list[int] = []
        self.cache_flags: list[bool] = []

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        self.input_lengths.append(int(input_ids.shape[-1]))
        self.cache_flags.append(kwargs.get("past_key_values") is not None)
        logits = torch.zeros((1, int(input_ids.shape[-1]), 8), dtype=torch.float32)
        logits[:, -1, 3] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=("cache", len(self.input_lengths)))
