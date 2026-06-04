from __future__ import annotations

from ceg_reproduce.extraction.claims import Claim, select_top_claims


def test_topk_prefers_attributes_then_early_claims_without_clip_support() -> None:
    man = Claim(
        claim_id="c1",
        text="man",
        normalized="person",
        object_text="man",
        claim_type="object",
        span=(2, 5),
        support_score=0.95,
        rank_features={"object_match": True, "has_attribute": False, "concreteness": 1.0},
    )
    racket = Claim(
        claim_id="c2",
        text="tennis racket",
        normalized="tennis racket",
        object_text="tennis racket",
        claim_type="object",
        span=(19, 32),
        support_score=0.1,
        rank_features={"object_match": True, "has_attribute": False, "concreteness": 1.0},
    )
    car = Claim(
        claim_id="c3",
        text="red car",
        normalized="car",
        object_text="car",
        claim_type="attribute",
        span=(42, 49),
        attribute="red",
        support_score=0.4,
        rank_features={"object_match": True, "has_attribute": True, "concreteness": 1.0},
    )

    selected = select_top_claims([man, racket, car], top_k=2)

    assert [claim.text for claim in selected] == ["red car", "man"]
