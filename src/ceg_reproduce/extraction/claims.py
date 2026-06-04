"""Object and attribute noun-phrase claim extraction."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ceg_reproduce.utils.config import resolve_path
from ceg_reproduce.utils.io import load_yaml


@dataclass
class Claim:
    claim_id: str
    text: str
    normalized: str
    object_text: str
    claim_type: str
    span: tuple[int, int]
    attribute: str | None = None
    support_score: float | None = None
    rank_features: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["span"] = list(self.span)
        return value


class ClaimExtractor:
    def __init__(
        self,
        objects: Iterable[str],
        synonyms: Mapping[str, Iterable[str]] | None = None,
        attributes: Mapping[str, Iterable[str]] | None = None,
        *,
        max_left_modifiers: int = 1,
    ) -> None:
        self.objects = sorted({str(item).lower() for item in objects}, key=len, reverse=True)
        self.synonyms = {
            str(key).lower(): [str(item).lower() for item in values]
            for key, values in (synonyms or {}).items()
        }
        self.attribute_words = {
            str(item).lower()
            for values in (attributes or {}).values()
            for item in values
        }
        self.max_left_modifiers = max_left_modifiers
        self.aliases: list[tuple[str, str]] = []
        for obj in self.objects:
            self.aliases.append((obj, obj))
            for alias in self.synonyms.get(obj, []):
                self.aliases.append((alias, obj))
        self.aliases.sort(key=lambda item: len(item[0]), reverse=True)

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], project_root: str | Path
    ) -> "ClaimExtractor":
        extraction = config.get("object_extraction", {})
        vocab_path = resolve_path(extraction.get("vocabulary_path"), project_root)
        vocab = load_yaml(vocab_path)
        return cls(
            vocab.get("objects", []),
            vocab.get("synonyms", {}),
            vocab.get("attributes", {}),
            max_left_modifiers=int(extraction.get("max_left_modifiers", 1)),
        )

    def extract(self, caption: str) -> list[Claim]:
        matches: list[tuple[int, int, str, str]] = []
        occupied: list[tuple[int, int]] = []
        for alias, normalized in self.aliases:
            pattern = re.compile(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", re.IGNORECASE)
            for match in pattern.finditer(caption):
                start, end = match.span()
                if normalized == "orange" and _looks_like_color_modifier(caption, end, self.objects):
                    continue
                if _overlaps(start, end, occupied):
                    continue
                occupied.append((start, end))
                matches.append((start, end, match.group(0), normalized))
        matches.sort(key=lambda item: item[0])

        claims: list[Claim] = []
        for index, (start, end, surface, normalized) in enumerate(matches, start=1):
            claim_start = start
            text = surface
            attr_tuple = self._left_attribute(caption, start)
            claim_type = "object"
            if attr_tuple is not None:
                attr_text, attr_start, attr_end = attr_tuple
                if caption[attr_end:start].strip() == "":
                    claim_start = attr_start
                    text = caption[claim_start:end]
                    claim_type = "attribute"
                else:
                    attr_tuple = None
            attr_value = attr_tuple[0] if claim_type == "attribute" and attr_tuple else None
            claims.append(
                Claim(
                    claim_id=f"c{index}",
                    text=_clean_text(text),
                    normalized=normalized,
                    object_text=surface.lower(),
                    claim_type=claim_type,
                    span=(claim_start, end),
                    attribute=attr_value,
                    rank_features={
                        "object_match": normalized in self.objects,
                        "has_attribute": claim_type == "attribute",
                        "concreteness": _concreteness(normalized),
                    },
                )
            )
        return claims

    def _left_attribute(self, caption: str, object_start: int) -> tuple[str, int, int] | None:
        left = caption[:object_start]
        tokens = list(re.finditer(r"[A-Za-z][A-Za-z-]*", left))
        if not tokens:
            return None
        for token in reversed(tokens[-self.max_left_modifiers :]):
            word = token.group(0).lower()
            if word in self.attribute_words:
                return (word, token.start(), token.end())
        return None


def select_top_claims(claims: Iterable[Claim], *, top_k: int) -> list[Claim]:
    selected = []
    seen_object_keys: set[str] = set()
    for claim in sorted(list(claims), key=_rank_key):
        if claim.claim_type == "object":
            if claim.normalized in seen_object_keys:
                continue
            seen_object_keys.add(claim.normalized)
        selected.append(claim)
        if len(selected) >= max(0, top_k):
            break
    return selected


def slugify_claim(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def claim_hypothesis(claim: Claim) -> str:
    if claim.claim_type == "attribute" and claim.attribute:
        return f"The {claim.normalized} is {claim.attribute}."
    article = "an" if claim.text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"The image contains {article} {claim.text.lower()}."


def _rank_key(claim: Claim) -> tuple[int, int, float, int]:
    features = claim.rank_features or {}
    object_match = 1 if features.get("object_match", True) else 0
    is_object_claim = 1 if claim.claim_type == "object" else 0
    concreteness = float(features.get("concreteness", 1.0))
    return (-object_match, -is_object_claim, -concreteness, claim.span[0])


def _overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end in spans)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _concreteness(normalized: str) -> float:
    abstract = {"scene", "view", "area", "background", "image"}
    return 0.0 if normalized.lower() in abstract else 1.0


def _looks_like_color_modifier(caption: str, object_end: int, objects: Iterable[str]) -> bool:
    match = re.match(r"\s+([A-Za-z][A-Za-z-]*)", caption[object_end:])
    if not match:
        return False
    next_word = match.group(1).lower()
    return next_word not in set(objects)
