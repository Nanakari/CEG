"""Reusable pipeline steps for scripts and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import re

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
    verification = config.get("verification", {})
    vqa_do_sample = bool(verification.get("do_sample", config.get("generation", {}).get("do_sample", False)))
    vqa_temperature = float(
        verification.get("temperature", config.get("generation", {}).get("temperature", 1.0))
    )
    vqa_top_p = float(verification.get("top_p", config.get("generation", {}).get("top_p", 1.0)))
    max_new_tokens = int(verification.get("max_new_tokens", counterfactual.get("max_new_tokens", 8)))
    threshold = float(config.get("nli", {}).get("entailment_threshold", 0.5))
    support_threshold = float(config.get("clip_grounding", {}).get("support_norm_threshold", 1.0))
    top_k = int(config.get("ceg", {}).get("top_k", 3))
    risk_mode = str(config.get("ceg", {}).get("risk_mode", "balanced"))
    gt_objects = {str(item).lower() for item in record.get("gt_objects", [])}
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
        vqa = verify_claim_with_vqa(
            image_path=image_path,
            masked_path=masked_path,
            claim=claim,
            generator=generator,
            config=config,
            sample_id=sample_id,
            max_new_tokens=max_new_tokens,
            do_sample=vqa_do_sample,
            temperature=vqa_temperature,
            top_p=vqa_top_p,
        )
        hypothesis = claim_hypothesis(claim)
        cps = 1 if bool(vqa["counterfactual_yes"]) else 0
        metadata = evidence.get("metadata", {}) if isinstance(evidence, dict) else {}
        support_score_norm = float(metadata.get("support_score_norm", evidence.get("support_score", 0.0)))
        support_low = support_score_norm < support_threshold
        is_gt_object = claim.normalized in gt_objects
        cf_mentions_object = _mentions_object(vqa["counterfactual_vqa_answer"], claim)
        risk, risk_reason = _assess_vqa_risk(
            claim_type=claim.claim_type,
            original_state=str(vqa["original_state"]),
            counterfactual_state=str(vqa["counterfactual_state"]),
            support_low=support_low,
            risk_mode=risk_mode,
        )
        if risk:
            high_risk_claims.append(claim)
        original_generation_dict = vqa["original_generation"]
        counterfactual_generation_dict = vqa["counterfactual_generation"]
        extra_latency += float(original_generation_dict.get("latency_sec", 0.0))
        extra_latency += float(counterfactual_generation_dict.get("latency_sec", 0.0))
        verified_record = {
            "claim_id": claim.claim_id,
            "claim": claim.text,
            "normalized": claim.normalized,
            "claim_type": claim.claim_type,
            "support_score": claim.support_score,
            "support_score_norm": support_score_norm,
            "support_low": support_low,
            "is_gt_object": is_gt_object,
            "cf_mentions_object": cf_mentions_object,
            "hypothesis": hypothesis,
            "entailment_prob": 1.0 if vqa["counterfactual_yes"] else 0.0,
            "entailment_threshold": threshold,
            "support_threshold": support_threshold,
            "cps": cps,
            "risk": risk,
            "risk_reason": risk_reason,
            "original_vqa_prompt": vqa["original_prompt"],
            "counterfactual_vqa_prompt": vqa["counterfactual_prompt"],
            "original_vqa_answer": vqa["original_vqa_answer"],
            "counterfactual_vqa_answer": vqa["counterfactual_vqa_answer"],
            "original_state": vqa["original_state"],
            "counterfactual_state": vqa["counterfactual_state"],
            "original_yes": vqa["original_yes"],
            "counterfactual_yes": vqa["counterfactual_yes"],
            "counterfactual_image_path": str(masked_path),
        }
        verified_claims.append(verified_record)
        counterfactual_answers.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "original_vqa_answer": vqa["original_vqa_answer"],
                "counterfactual_answer": vqa["counterfactual_vqa_answer"],
                "counterfactual_vqa_answer": vqa["counterfactual_vqa_answer"],
                "original_generation": original_generation_dict,
                "generation": counterfactual_generation_dict,
                "nli": {
                    "entailment_prob": 1.0 if vqa["counterfactual_yes"] else 0.0,
                    "label": vqa["counterfactual_state"],
                    "latency_sec": 0.0,
                    "backend": "targeted_vqa_parser",
                },
            }
        )

    revision = revise_caption(
        caption,
        high_risk_claims,
        object_revision=str(config.get("ceg", {}).get("object_revision", "generalize")),
    )
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
        "risk_count": sum(1 for item in verified_claims if item["risk"]),
        "revision_count": len(revision.actions),
        "latency_sec": base_latency + extra_latency,
        "base_latency_sec": base_latency,
        "external_lvlm_calls": 2 * len(verified_claims),
    }


def verify_claim_with_vqa(
    *,
    image_path: str | Path,
    masked_path: str | Path,
    claim: Claim,
    generator: ImageTextGenerator,
    config: Mapping[str, Any],
    sample_id: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    original_prompt = _vqa_prompt(claim, config, counterfactual=False)
    counterfactual_prompt = _vqa_prompt(claim, config, counterfactual=True)
    slug = slugify_claim(claim.text)
    original = generator.generate(
        image_path,
        original_prompt,
        sample_id=f"{sample_id}::vqa::{slug}",
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    counterfactual = generator.generate(
        masked_path,
        counterfactual_prompt,
        sample_id=f"{sample_id}::cf_vqa::{slug}",
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    original_dict = original.to_dict() if hasattr(original, "to_dict") else dict(original)
    counterfactual_dict = (
        counterfactual.to_dict() if hasattr(counterfactual, "to_dict") else dict(counterfactual)
    )
    original_state = parse_yes_no(original.text)
    counterfactual_state = parse_yes_no(counterfactual.text)
    return {
        "original_prompt": original_prompt,
        "counterfactual_prompt": counterfactual_prompt,
        "original_vqa_answer": original.text,
        "counterfactual_vqa_answer": counterfactual.text,
        "original_state": original_state,
        "counterfactual_state": counterfactual_state,
        "original_yes": True if original_state == "yes" else False if original_state == "no" else None,
        "counterfactual_yes": (
            True if counterfactual_state == "yes" else False if counterfactual_state == "no" else None
        ),
        "original_generation": original_dict,
        "counterfactual_generation": counterfactual_dict,
    }


def parse_yes_no(answer: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", answer.lower()).strip()
    tokens = normalized.split()
    if not tokens:
        return "uncertain"
    if "yes" in tokens:
        return "yes"
    if "no" in tokens:
        return "no"
    negative_patterns = ["not", "without", "cannot see", "can't see", "do not see", "does not appear"]
    if any(pattern in normalized for pattern in negative_patterns):
        return "no"
    return "uncertain"


def _assess_vqa_risk(
    *,
    claim_type: str,
    original_state: str,
    counterfactual_state: str,
    support_low: bool,
    risk_mode: str,
) -> tuple[bool, str]:
    if risk_mode not in {"targeted_vqa", "targeted_vqa_support_guard"}:
        risk_mode = "targeted_vqa"
    if risk_mode == "targeted_vqa_support_guard":
        if original_state in {"no", "uncertain"}:
            if claim_type == "attribute":
                return True, f"attribute_{original_state}"
            if support_low:
                return True, f"{original_state}_and_low_support"
            return False, f"{original_state}_but_local_support"
        if counterfactual_state == "yes":
            return False, "counterfactual_vqa_persistence_diagnostic"
        return False, "vqa_supported"
    if original_state == "no":
        return True, "original_vqa_no"
    if original_state == "uncertain":
        return True, "original_vqa_uncertain"
    if counterfactual_state == "yes":
        return False, "counterfactual_vqa_persistence_diagnostic"
    return False, "vqa_supported"


def _vqa_prompt(claim: Claim, config: Mapping[str, Any], *, counterfactual: bool) -> str:
    verification = config.get("verification", {})
    if claim.claim_type == "attribute" and claim.attribute:
        template_key = (
            "counterfactual_attribute_prompt_template"
            if counterfactual
            else "attribute_prompt_template"
        )
        template = str(
            verification.get(
                template_key,
                verification.get(
                    "attribute_prompt_template",
                    "Is the {object} {attribute}? Answer yes or no.",
                ),
            )
        )
        return template.format(claim=claim.text, object=claim.normalized, attribute=claim.attribute)
    template_key = "counterfactual_prompt_template" if counterfactual else "original_prompt_template"
    template = str(
        verification.get(
            template_key,
            "After removing the region, is there still a {claim} in the image? Answer yes or no."
            if counterfactual
            else "Is there a {claim} in the image? Answer yes or no.",
        )
    )
    return template.format(claim=claim.text, object=claim.normalized, attribute=claim.attribute or "")


def _mentions_object(text: str, claim: Claim) -> bool:
    candidates = {claim.normalized.lower(), claim.object_text.lower(), claim.text.lower()}
    normalized_text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    padded = f" {normalized_text} "
    return any(f" {re.sub(r'[^a-z0-9 ]+', ' ', item).strip()} " in padded for item in candidates if item)
