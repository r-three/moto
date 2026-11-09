"""Dataclass configs for every stage of the Llama-MoTo pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from moto.utils.args import Args, HFArgs


# ── Stage 1: Byte-fallback analysis ───────────────────────────────────────────

@dataclass
class ByteFallbackAnalysisArgs(Args):
    # Tokenizer to audit (HF repo id or local path)
    tokenizer: str = "meta-llama/Meta-Llama-3.1-8B"
    # FLORES language codes to probe, e.g. ["arb_Arab", "cmn_Hani"]
    languages: List[str] = field(default_factory=list)
    # How many sentences to sample per language
    sample_size: int = 500
    # Local training data root: expects {data_dir}/{lang_code}/*.jsonl
    # Each JSONL line: {"text": "..."}  (override key with text_key)
    # If a language has no local data, falls back to FLORES.
    data_dir: Optional[str] = None
    text_key: str = "text"
    flores_dataset: str = "flexitok/flores-21langs"
    flores_split: str = "devtest"
    seed: int = 42
    output_csv: Optional[str] = "byte_fallback_analysis.csv"
    # Byte-fallback token pattern  (LLaMA-style: <0x41>)
    byte_token_pattern: str = r"^<0x[0-9A-Fa-f]{2}>$"


# ── Stage 2: PickyBPE vocabulary pruning ──────────────────────────────────────

@dataclass
class PickyBPEArgs(Args):
    # Base tokenizer whose vocabulary we will prune
    base_tokenizer: str = "meta-llama/Meta-Llama-3.1-8B"
    # Target corpus: HF dataset id or list of local text files
    corpus_dataset: str = ""
    corpus_files: List[str] = field(default_factory=list)
    # Languages to include when loading the corpus (FLORES codes)
    languages: List[str] = field(default_factory=list)
    corpus_split: str = "train"
    corpus_text_column: str = "text"
    # Token-level minimum frequency to keep in the pruned vocab
    min_token_freq: int = 5
    # Always keep all 256 byte-fallback tokens regardless of frequency
    keep_byte_tokens: bool = True
    # Always keep special tokens (bos, eos, pad, unk …)
    keep_special_tokens: bool = True
    # Max corpus sentences to scan (0 = unlimited)
    max_sentences: int = 0
    seed: int = 42
    output_dir: str = "picky_bpe_output"


# ── Stage 3: Multiplicity tokens ──────────────────────────────────────────────

@dataclass
class MultiplicityArgs(Args):
    # Base tokenizer
    base_tokenizer: str = "meta-llama/Meta-Llama-3.1-8B"
    # One or more extension tokenizer paths (output of Stage 4)
    extension_tokenizers: List[str] = field(default_factory=list)
    output_json: str = "multiplicity_tokens.json"
    # How to handle multiplicity tokens in the final vocabulary:
    #   "base"    — keep base embedding, discard extension duplicate
    #   "ext"     — keep extension embedding, discard base entry
    #   "average" — average the two embeddings (requires model weights)
    #   "both"    — keep both under distinct ids (may inflate vocab)
    resolution_strategy: str = "base"


# ── Stage 4: Merge extension (additional BPE runs) ────────────────────────────

@dataclass
class ExtendMergesArgs(Args):
    # Path to pruned vocabulary produced by Stage 2
    pruned_vocab_json: str = ""
    # Text corpus for BPE training (same format as PickyBPEArgs)
    corpus_dataset: str = ""
    corpus_files: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    corpus_split: str = "train"
    corpus_text_column: str = "text"
    # Number of additional merge operations to learn
    extra_merges: int = 8000
    # Pre-tokenization strategy (mirrors tokenizer_training options)
    pretokenizer: str = "byte_level"
    # Minimum pair frequency to merge
    min_frequency: int = 2
    max_sentences: int = 0
    seed: int = 42
    output_dir: str = "extended_merges_output"


# ── Stage 5: MoTo tokenizer construction ──────────────────────────────────────

@dataclass
class MoToConstructionArgs(Args, HFArgs):
    # Base Llama tokenizer
    base_tokenizer: str = "meta-llama/Meta-Llama-3.1-8B"
    # Mapping {language_code: extended_tokenizer_path}
    # e.g. {"arb_Arab": "output/extended_arb", "cmn_Hani": "output/extended_cmn"}
    language_extensions: Dict[str, str] = field(default_factory=dict)
    # Multiplicity resolution (from Stage 3)
    multiplicity_json: Optional[str] = None
    # How many tokens are active per forward pass (budget)
    active_vocab_size: int = 32000
    output_dir: str = "moto_tokenizer"
    # Language detection: "fasttext", "langdetect", or "manual"
    lang_detect_backend: str = "fasttext"


# ── Full pipeline ──────────────────────────────────────────────────────────────

@dataclass
class LlamaMoToPipelineArgs(Args):
    config: Any = None
    base_tokenizer: str = "meta-llama/Meta-Llama-3.1-8B"
    languages: List[str] = field(default_factory=list)
    # Local JSONL data root (see byte_fallback_analysis / picky_bpe)
    data_dir: Optional[str] = None
    corpus_dataset: str = ""
    corpus_files: List[str] = field(default_factory=list)
    corpus_split: str = "train"
    corpus_text_column: str = "text"
    max_sentences: int = 0
    extra_merges: int = 8000
    active_vocab_size: int = 32000
    min_token_freq: int = 5
    output_dir: str = "llama_moto_output"
    seed: int = 42
    # Optionally skip picky_bpe and supply an existing pruned vocab
    pruned_vocab_json: Optional[str] = None
    # HuggingFace publishing (moto_tokenizer output is published when hf.publish_to_hf=true)
    hf: HFArgs = field(default_factory=HFArgs)
    # Which stages to run (all by default)
    run_stages: List[str] = field(
        default_factory=lambda: [
            "byte_fallback_analysis",
            "picky_bpe",
            "multiplicity_tokens",
            "extend_merges",
            "build_moto",
        ]
    )
