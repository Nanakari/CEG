"""Run CEG claim-level counterfactual verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ceg_reproduce.extraction.claims import ClaimExtractor
from ceg_reproduce.grounding import build_grounder
from ceg_reproduce.models import build_generator
from ceg_reproduce.nli import build_nli_scorer
from ceg_reproduce.pipeline import run_ceg_record
from ceg_reproduce.utils.config import config_for_method, load_config, output_root
from ceg_reproduce.utils.io import read_jsonl, write_jsonl
from ceg_reproduce.utils.reproducibility import seed_everything


def main() -> None:
    args = _parse_args()
    base_config = load_config(args.config)
    config = config_for_method(base_config, "ceg")
    seed_everything(config)
    out_root = output_root(config, ROOT)
    base_path = out_root / "coco_base_captions.jsonl"
    if not base_path.exists():
        raise FileNotFoundError(f"Run Base captions first: {base_path}")

    generator = build_generator(config, method="base")
    grounder = build_grounder(config)
    nli_scorer = None if _uses_targeted_vqa(config) else build_nli_scorer(config)
    extractor = ClaimExtractor.from_config(config, ROOT)
    revision_records = []
    claim_records = []
    counterfactual_records = []
    for index, record in enumerate(read_jsonl(base_path)):
        if args.limit is not None and index >= args.limit:
            break
        revised = run_ceg_record(record, generator, grounder, nli_scorer, extractor, config, ROOT)
        revision_records.append(revised)
        claim_records.append(
            {
                "sample_id": revised["sample_id"],
                "image_id": revised["image_id"],
                "caption": revised["caption"],
                "claims": revised["claims"],
                "verified_claims": revised["verified_claims"],
            }
        )
        for item in revised["counterfactual_answers"]:
            counterfactual_records.append(
                {
                    "sample_id": revised["sample_id"],
                    "image_id": revised["image_id"],
                    **item,
                }
            )
    write_jsonl(out_root / "coco_ceg_claims.jsonl", claim_records)
    write_jsonl(out_root / "coco_ceg_counterfactuals.jsonl", counterfactual_records)
    write_jsonl(out_root / "coco_ceg_revisions.jsonl", revision_records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _uses_targeted_vqa(config: dict) -> bool:
    mode = str(config.get("verification", {}).get("mode", ""))
    risk_mode = str(config.get("ceg", {}).get("risk_mode", ""))
    return mode == "targeted_vqa" or risk_mode.startswith("targeted_vqa")


if __name__ == "__main__":
    main()
