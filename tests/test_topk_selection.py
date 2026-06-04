from __future__ import annotations

from ceg_reproduce.extraction.claims import Claim, select_top_claims


def test_topk_prefers_object_claims_for_chair_budget() -> None:
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

    assert [claim.text for claim in selected] == ["man", "tennis racket"]


def test_topk_deduplicates_repeated_object_claims() -> None:
    first = Claim(
        claim_id="c1",
        text="motorcycle",
        normalized="motorcycle",
        object_text="motorcycle",
        claim_type="object",
        span=(0, 10),
        rank_features={"object_match": True, "concreteness": 1.0},
    )
    duplicate = Claim(
        claim_id="c2",
        text="motorcycle",
        normalized="motorcycle",
        object_text="motorcycle",
        claim_type="object",
        span=(20, 30),
        rank_features={"object_match": True, "concreteness": 1.0},
    )
    car = Claim(
        claim_id="c3",
        text="car",
        normalized="car",
        object_text="car",
        claim_type="object",
        span=(40, 43),
        rank_features={"object_match": True, "concreteness": 1.0},
    )

    selected = select_top_claims([first, duplicate, car], top_k=3)

    assert [claim.claim_id for claim in selected] == ["c1", "c3"]
