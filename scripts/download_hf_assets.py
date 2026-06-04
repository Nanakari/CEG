"""Download Hugging Face model snapshots into the configured cache."""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def main() -> None:
    args = _parse_args()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    for repo_id in args.repo_ids:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=args.cache_dir,
            resume_download=True,
        )
        print(f"{repo_id}: {path}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_ids", nargs="+")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--endpoint", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
