"""Rule-based caption revision for automatic CHAIR evaluation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from ceg_reproduce.extraction.claims import Claim


@dataclass
class RevisionResult:
    original_caption: str
    revised_caption: str
    actions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def revise_caption(caption: str, high_risk_claims: Iterable[Claim]) -> RevisionResult:
    actions: list[dict[str, Any]] = []
    edits: list[tuple[int, int, str, Claim, str]] = []
    for claim in high_risk_claims:
        start, end, replacement = _replacement_for_claim(caption, claim)
        edits.append((start, end, replacement, claim, "generalize"))
    revised = caption
    for start, end, replacement, claim, action in sorted(edits, key=lambda item: item[0], reverse=True):
        revised = revised[:start] + replacement + revised[end:]
        actions.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "normalized": claim.normalized,
                "action": action,
                "replacement": replacement,
                "span": [start, end],
                "reason": "counterfactual_persistence",
            }
        )
    revised = _cleanup(revised)
    actions.reverse()
    return RevisionResult(original_caption=caption, revised_caption=revised, actions=actions)


def _replacement_for_claim(caption: str, claim: Claim) -> tuple[int, int, str]:
    start, end = claim.span
    if claim.claim_type == "attribute":
        return start, end, claim.object_text
    article_span = _preceding_article_span(caption, start)
    if article_span is not None:
        article_start, _ = article_span
        return article_start, end, "an object"
    return start, end, "an object"


def _preceding_article_span(caption: str, start: int) -> tuple[int, int] | None:
    match = re.search(r"\b(a|an|the)\s+$", caption[:start], flags=re.IGNORECASE)
    return match.span() if match else None


def _cleanup(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\ba\s+an\s+", "an ", text, flags=re.IGNORECASE)
    text = re.sub(r"\ban\s+a\s+", "a ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text
