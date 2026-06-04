# CEG

CEG is a standalone reproduction scaffold for the main experiment of
Claim-Level Counterfactual Verification for LVLM hallucination mitigation.

The repository implements only the COCO Caption / CHAIR workflow:

- `Base`: LLaVA-1.5 caption generation with shared prompt and decoding settings.
- `VCD`: a local Visual Contrastive Decoding-compatible LLaVA adapter.
- `CEG`: post-generation claim-level counterfactual verification and conservative caption revision.

No ablations, POPE, AMBER, or K-sensitivity experiments are included in this project.

## Quick Start

Smoke runs do not load LLaVA, CLIP, or MNLI:

```bash
python -m pip install -e .[dev]
python scripts/smoke_test.py
python -m pytest
```

Real runs require local COCO and model assets configured in `configs/default.yaml`:

```bash
python scripts/check_assets.py --config configs/default.yaml --strict
python scripts/run_caption.py --config configs/default.yaml --method base
python scripts/run_caption.py --config configs/default.yaml --method vcd
python scripts/run_ceg.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
python scripts/export_results.py --config configs/default.yaml
```

The full main sequence is:

```bash
python scripts/run_all.py --config configs/default.yaml --limit 5000
```

## Main Method Contract

CEG reads Base captions, extracts object and attribute noun-phrase claims, ranks
claims with a fixed Top-K policy, locates candidate visual evidence with CLIP
patch-text similarity, erases those regions with a blur mask, regenerates the
caption on the local counterfactual image, and marks the claim as high risk when
MNLI says the counterfactual answer still entails the original claim.

Default main settings:

- `K = 3`
- `top_m = 4` CLIP patches
- blur-mask counterfactuals
- `roberta-large-mnli`
- entailment threshold `0.5`
- greedy decoding for Base and CEG regeneration

For automatic CHAIR evaluation, high-risk object claims are generalized and
high-risk attribute claims lose the attribute. The automatic output does not add
`[unverified]` tags because those tags do not remove object words from CHAIR.

## Repository Layout

```text
configs/                 Main, smoke, method, dataset, and vocabulary configs
examples/smoke/          No-model smoke fixtures
scripts/                 CLI entry points
src/ceg_reproduce/       Python package
tests/                   Unit and smoke tests
```

Ignored local directories include `data/`, `models/`, and `outputs/`. Do not
commit model weights, datasets, or generated experiment outputs.

## Verification

```bash
python -m compileall -q src scripts tests
python -m pytest
python scripts/run_all.py --config configs/smoke.yaml
```
