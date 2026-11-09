"""End-to-end Llama-MoTo pipeline.

Runs the five stages in sequence, passing outputs from each stage into the
next.  Stages can be selectively skipped via ``run_stages``.

Usage (CLI):
    python -m moto.llama_moto.pipeline \\
        base_tokenizer=meta-llama/Meta-Llama-3.1-8B \\
        languages='["arb_Arab","cmn_Hani","hin_Deva"]' \\
        data_dir=data/flexitok \\
        corpus_dataset=flexitok/flores-21langs \\
        extra_merges=8000 \\
        active_vocab_size=32000 \\
        output_dir=llama_moto_output

    # With HF publishing:
    python -m moto.llama_moto.pipeline \\
        config=configs/llama_moto/llama3_8b.yaml \\
        hf.publish_to_hf=true \\
        hf.hf_repo_id=gsaltintas/llama-moto-arb-cmn \\
        hf.collections='["gsaltintas/llama-moto"]'

    # Skip byte-fallback analysis and re-use an existing pruned vocab:
    python -m moto.llama_moto.pipeline \\
        config=configs/llama_moto/llama3_8b.yaml \\
        run_stages='["extend_merges","build_moto"]' \\
        pruned_vocab_json=llama_moto_output/picky_bpe/pruned_vocab.json
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import huggingface_hub as hf_hub

from moto.llama_moto.args import LlamaMoToPipelineArgs
from moto.utils.args import HFArgs, dump_config, load_args
from moto.utils.hf_utils import create_or_get_collection_slug

logger = logging.getLogger(__name__)


# ── HF publishing helper ───────────────────────────────────────────────────────

def _publish_to_hf(local_dir: str, hf_args: HFArgs, tag: str = ""):
    """Upload all files under *local_dir* to an HF repo."""
    if not hf_args.publish_to_hf:
        return
    if not hf_args.hf_repo_id:
        raise ValueError("Set hf.hf_repo_id= to publish to HF")

    repo_id = hf_args.hf_repo_id
    if tag:
        repo_id = f"{repo_id}-{tag}"

    logger.info("Publishing %s → HF repo '%s'", local_dir, repo_id)
    hf_hub.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    hf_hub.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=hf_args.commit_message or f"Upload llama-moto {tag}",
    )

    if hf_args.collections:
        api = hf_hub.HfApi()
        for collection in hf_args.collections:
            slug = create_or_get_collection_slug(collection)
            api.add_collection_item(
                collection_slug=slug,
                item_id=repo_id,
                item_type="model",
                exists_ok=True,
            )
            logger.info("Added %s to collection %s", repo_id, collection)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(args: LlamaMoToPipelineArgs):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(args, out / "pipeline_config.yaml")

    stages = set(args.run_stages)
    hf_args: HFArgs = getattr(args, "hf", HFArgs())
    results: Dict = {}

    # ── Stage 1: Byte-fallback analysis ───────────────────────────────────────
    if "byte_fallback_analysis" in stages:
        logger.info("=== Stage 1: Byte-fallback analysis ===")
        from moto.llama_moto.byte_fallback_analysis import main as run_bfa
        from moto.llama_moto.args import ByteFallbackAnalysisArgs
        bfa_args = ByteFallbackAnalysisArgs(
            tokenizer=args.base_tokenizer,
            languages=args.languages,
            data_dir=getattr(args, "data_dir", None),
            sample_size=500,
            output_csv=str(out / "byte_fallback_analysis.csv"),
            seed=args.seed,
        )
        results["byte_fallback_stats"] = run_bfa(bfa_args)
    else:
        logger.info("Skipping stage: byte_fallback_analysis")

    # ── Stage 2: PickyBPE vocabulary pruning ──────────────────────────────────
    picky_dir = out / "picky_bpe"
    pruned_vocab_json = str(picky_dir / "pruned_vocab.json")

    if "picky_bpe" in stages:
        logger.info("=== Stage 2: PickyBPE vocabulary pruning ===")
        from moto.llama_moto.picky_bpe import main as run_picky
        from moto.llama_moto.args import PickyBPEArgs
        picky_args = PickyBPEArgs(
            base_tokenizer=args.base_tokenizer,
            corpus_dataset=args.corpus_dataset,
            corpus_files=args.corpus_files,
            languages=args.languages,
            corpus_split=args.corpus_split,
            corpus_text_column=args.corpus_text_column,
            min_token_freq=args.min_token_freq,
            max_sentences=args.max_sentences,
            output_dir=str(picky_dir),
            seed=args.seed,
        )
        results["pruned_vocab"] = run_picky(picky_args)
    else:
        logger.info("Skipping stage: picky_bpe")
        pruned_vocab_json = getattr(args, "pruned_vocab_json", pruned_vocab_json)

    # ── Stage 4: Extend BPE merges (per language) ─────────────────────────────
    ext_dir = out / "extended_merges"
    language_extensions: Dict[str, str] = {}

    if "extend_merges" in stages:
        logger.info("=== Stage 4: Extend BPE merge rules ===")
        from moto.llama_moto.extend_merges import main as run_ext
        from moto.llama_moto.args import ExtendMergesArgs
        ext_args = ExtendMergesArgs(
            pruned_vocab_json=pruned_vocab_json,
            corpus_dataset=args.corpus_dataset,
            corpus_files=args.corpus_files,
            languages=args.languages,
            corpus_split=args.corpus_split,
            corpus_text_column=args.corpus_text_column,
            extra_merges=args.extra_merges,
            max_sentences=args.max_sentences,
            output_dir=str(ext_dir),
            seed=args.seed,
        )
        summary = run_ext(ext_args)
        results["extension_summary"] = summary
        language_extensions = {
            lang: str(ext_dir / lang)
            for lang in summary.get("languages_processed", [])
        }
    else:
        logger.info("Skipping stage: extend_merges")
        if ext_dir.exists():
            language_extensions = {
                d.name: str(d)
                for d in ext_dir.iterdir()
                if d.is_dir() and (d / "tokenizer.json").exists()
            }

    # ── Stage 3: Multiplicity tokens ──────────────────────────────────────────
    multiplicity_json = str(out / "multiplicity_tokens.json")

    if "multiplicity_tokens" in stages and language_extensions:
        logger.info("=== Stage 3: Multiplicity token identification ===")
        from moto.llama_moto.multiplicity_tokens import main as run_mult
        from moto.llama_moto.args import MultiplicityArgs
        mult_args = MultiplicityArgs(
            base_tokenizer=args.base_tokenizer,
            extension_tokenizers=list(language_extensions.values()),
            output_json=multiplicity_json,
        )
        results["multiplicity"] = run_mult(mult_args)
    else:
        logger.info("Skipping stage: multiplicity_tokens")

    # ── Stage 5: Build MoTo tokenizer ─────────────────────────────────────────
    moto_dir = out / "moto_tokenizer"

    if "build_moto" in stages and language_extensions:
        logger.info("=== Stage 5: Build MoTo tokenizer ===")
        from moto.llama_moto.moto_tokenizer import build_moto_vocab
        moto_config = build_moto_vocab(
            base_tokenizer_path=args.base_tokenizer,
            language_extensions=language_extensions,
            output_dir=str(moto_dir),
            active_vocab_size=args.active_vocab_size,
        )
        results["moto_config"] = moto_config
        logger.info("MoTo tokenizer saved to %s", moto_dir)

        _publish_to_hf(str(moto_dir), hf_args, tag="moto-tokenizer")
    else:
        logger.info("Skipping stage: build_moto")

    logger.info("Pipeline complete. Output dir: %s", out)
    return results


def main(args: LlamaMoToPipelineArgs):
    if not args.languages:
        raise ValueError("Provide languages=[...] (FLORES codes)")
    return run_pipeline(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = load_args(LlamaMoToPipelineArgs)
    main(args)
