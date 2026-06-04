"""Rule-based caption revision for automatic CHAIR evaluation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from ceg_reproduce.extraction.claims import Claim


_COMPOUND_PREFIX_OBJECTS = {"microwave"}


@dataclass
class RevisionResult:
    original_caption: str
    revised_caption: str
    actions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def revise_caption(
    caption: str,
    high_risk_claims: Iterable[Claim],
    *,
    object_revision: str = "generalize",
) -> RevisionResult:
    actions: list[dict[str, Any]] = []
    edits: list[tuple[int, int, str, Claim, str]] = []
    for claim in high_risk_claims:
        start, end, replacement, action = _replacement_for_claim(
            caption, claim, object_revision=object_revision
        )
        edits.append((start, end, replacement, claim, action))
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
                "reason": "claim_verification_risk",
            }
        )
    revised = _cleanup(revised)
    actions.reverse()
    return RevisionResult(original_caption=caption, revised_caption=revised, actions=actions)


def _replacement_for_claim(
    caption: str, claim: Claim, *, object_revision: str
) -> tuple[int, int, str, str]:
    start, end = claim.span
    if claim.claim_type == "attribute":
        return start, end, claim.object_text, "drop_attribute"
    if object_revision == "drop":
        drop_start, drop_end = _object_drop_span(caption, start, end)
        replacement = _drop_replacement(caption, drop_start, drop_end)
        return drop_start, drop_end, replacement, "drop_object"
    following_word = _following_word(caption, end)
    if following_word and " " not in claim.text.strip() and claim.normalized in _COMPOUND_PREFIX_OBJECTS:
        article_span = _preceding_article_span(caption, start)
        if article_span is not None:
            article_start, _ = article_span
            return article_start, end, _article_for(following_word), "generalize"
        return start, end, "", "generalize"
    article_span = _preceding_article_span(caption, start)
    if article_span is not None:
        article_start, _ = article_span
        return article_start, end, "an object", "generalize"
    return start, end, "an object", "generalize"


def _preceding_article_span(caption: str, start: int) -> tuple[int, int] | None:
    match = re.search(r"\b(a|an|the)\s+$", caption[:start], flags=re.IGNORECASE)
    return match.span() if match else None


def _following_word(caption: str, end: int) -> str | None:
    match = re.match(r"\s+([A-Za-z][A-Za-z-]*)", caption[end:])
    return match.group(1).lower() if match else None


def _object_drop_span(caption: str, start: int, end: int) -> tuple[int, int]:
    article_span = _preceding_article_span(caption, start)
    if article_span is not None:
        start = article_span[0]
    return start, end


def _drop_replacement(caption: str, start: int, end: int) -> str:
    before = caption[:start]
    after = caption[end:]
    if before and after and not before[-1].isspace() and not after[:1].isspace() and after[:1] not in ",.!?;:":
        return " "
    return ""


def _article_for(word: str) -> str:
    return "an" if word[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _cleanup(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\ba\s+an\s+", "an ", text, flags=re.IGNORECASE)
    text = re.sub(r"\ban\s+a\s+", "a ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(holding|carrying|with)\s+(beside|near|next to|on|in|at)\b", r"\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(with|holding|carrying)\s*([,.!?;:])", r"\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text
