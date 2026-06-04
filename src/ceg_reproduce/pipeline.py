"""Reusable pipeline steps for scripts and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ceg_reproduce.counterfactual.masking import apply_blur_mask
from ceg_reproduce.extraction.claims import (
    Claim,
    ClaimExtractor,
    claim_hypothesis,
    select_top_claims,
    slugify_claim,
)
from ceg_reproduce.models.types import ImageTextGenerator
from ceg_reproduce.revision.caption import revise_caption
from ceg_reproduce.utils.config import output_root, resolve_path


def generate_for_sample(
    generator: ImageTextGenerator,
    image_path: str | Path,
    prompt: str,
    *,
    sample_id: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    result = generator.generate(
        image_path,
        prompt,
        sample_id=sample_id,
        max_new_tokens=max_new_tokens,
    )
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def run_ceg_record(
    record: Mapping[str, Any],
    generator: ImageTextGenerator,
    grounder: Any,
    nli_scorer: Any,
    extractor: ClaimExtractor,
    config: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    sample_id = str(record.get("sample_id") or record.get("image_id"))
    caption = str(record.get("caption") or record.get("text") or "")
    prompt = str(record.get("prompt") or config.get("prompts", {}).get("caption", ""))
    image_path = str(record.get("image_path") or "")
    counterfactual = config.get("counterfactual", {})
    cf_prompt = str(counterfactual.get("prompt") or prompt)
    max_new_tokens = int(
        counterfactual.get(
            "max_new_tokens",
            config.get("generation", {}).get("caption_max_new_tokens", 64),
        )
    )
    threshold = float(config.get("nli", {}).get("entailment_threshold", 0.5))
    support_threshold = float(config.get("clip_grounding", {}).get("support_norm_threshold", 1.0))
    top_k = int(config.get("ceg", {}).get("top_k", 3))
    image_dir = resolve_path(
        counterfactual.get("image_dir") or output_root(config, project_root) / "counterfactual_images",
        project_root,
    )

    claims = extractor.extract(caption)
    selected = select_top_claims(claims, top_k=top_k)
    evidence_records = []
    evidences = grounder.locate_many(image_path, [claim.text for claim in selected])
    for claim, evidence in zip(selected, evidences):
        claim.support_score = float(evidence.support_score)
        evidence_dict = evidence.to_dict() if hasattr(evidence, "to_dict") else dict(evidence)
        evidence_records.append({"claim_id": claim.claim_id, **evidence_dict})

    verified_claims = []
    counterfactual_answers = []
    high_risk_claims: list[Claim] = []
    extra_latency = sum(float(item.get("latency_sec", 0.0)) for item in evidence_records)
    evidence_by_claim = {item["claim_id"]: item for item in evidence_records}

    for claim in selected:
        evidence = evidence_by_claim.get(claim.claim_id, {})
        claim_slug = slugify_claim(claim.text)
        cf_path = image_dir / f"{sample_id}_{claim.claim_id}_{claim_slug}.jpg"
        masked_path = apply_blur_mask(
            image_path,
            evidence.get("boxes", []),
            cf_path,
            blur_radius=int(counterfactual.get("blur_radius", 12)),
            expand_pixels=int(counterfactual.get("expand_pixels", 8)),
        )
        generation = generator.generate(
            masked_path,
            cf_prompt,
            sample_id=f"{sample_id}::cf::{claim_slug}",
            max_new_tokens=max_new_tokens,
        )
        hypothesis = claim_hypothesis(claim)
        nli = nli_scorer.score(generation.text, hypothesis)
        cps = 1 if float(nli.entailment_prob) > threshold else 0
        metadata = evidence.get("metadata", {}) if isinstance(evidence, dict) else {}
        support_score_norm = float(metadata.get("support_score_norm", evidence.get("support_score", 0.0)))
        support_low = support_score_norm < support_threshold
        risk = bool(cps and support_low)
        if risk:
            high_risk_claims.append(claim)
        generation_dict = generation.to_dict() if hasattr(generation, "to_dict") else dict(generation)
        nli_dict = nli.to_dict() if hasattr(nli, "to_dict") else dict(nli)
        extra_latency += float(generation_dict.get("latency_sec", 0.0))
        extra_latency += float(nli_dict.get("latency_sec", 0.0))
        verified_record = {
            "claim_id": claim.claim_id,
            "claim": claim.text,
            "normalized": claim.normalized,
            "claim_type": claim.claim_type,
            "support_score": claim.support_score,
            "support_score_norm": support_score_norm,
            "support_low": support_low,
            "hypothesis": hypothesis,
            "entailment_prob": float(nli.entailment_prob),
            "entailment_threshold": threshold,
            "support_threshold": support_threshold,
            "cps": cps,
            "risk": risk,
            "counterfactual_image_path": str(masked_path),
        }
        verified_claims.append(verified_record)
        counterfactual_answers.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "counterfactual_answer": generation.text,
                "generation": generation_dict,
                "nli": nli_dict,
            }
        )

    revision = revise_caption(caption, high_risk_claims)
    base_latency = float(record.get("latency_sec", 0.0))
    return {
        "sample_id": sample_id,
        "image_id": record.get("image_id"),
        "image_path": image_path,
        "dataset": record.get("dataset", "coco_chair"),
        "method": "ceg",
        "prompt": prompt,
        "caption": caption,
        "original_caption": caption,
        "revised_caption": revision.revised_caption,
        "gt_objects": list(record.get("gt_objects", [])),
        "claims": [claim.to_dict() for claim in claims],
        "verified_claims": verified_claims,
        "counterfactual_answers": counterfactual_answers,
        "cps": {item["claim_id"]: item["cps"] for item in verified_claims},
        "risk": {item["claim_id"]: item["risk"] for item in verified_claims},
        "revision_actions": revision.actions,
        "latency_sec": base_latency + extra_latency,
        "base_latency_sec": base_latency,
        "external_lvlm_calls": len(verified_claims),
    }
