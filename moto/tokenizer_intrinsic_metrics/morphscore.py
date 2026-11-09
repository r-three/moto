"""MorphScore (Arnett et al. 2025, arXiv:2507.06378) for the TokEval settings, including MoTo's routed tokenizers.

    python -m moto.tokenizer_intrinsic_metrics.morphscore
    python -m moto.tokenizer_intrinsic_metrics.morphscore settings=[moto_1b,llama32] context=sentence

Needs the Hub (dataset catherinearnett/morphscore, the tokenizers) or data_path=<csv|parquet|jsonl> with the dataset's
columns. Settings come from the TokEval config, so names match the intrinsic tables: `mode: moto_repo` settings are loaded
through the release MoToTokenizer (set_language + offset mapping, each language routed to its own component, as in
oracle routing); `mode: single` settings through AutoTokenizer; byt5 is scored analytically (a cut at every character).

Definition used here (check against the reference implementation with check_official=true before quoting numbers):
  * gold cuts   = character offsets of the stem boundaries in `wordform`, from `preceding_part` / `following_part`
                  (words with no morpheme boundary are dropped);
  * predicted   = interior token boundaries, from the tokenizer's offset mapping, for the isolated word
                  (context=word) or for the word inside its `sentence` (context=sentence);
  * recall      = |gold & predicted| / |gold|;  precision = |gold & predicted| / |predicted|;
  * variants    = `excl_single`: words the tokenizer keeps as one token are dropped (the paper's default);
                  `incl_single`: they count with recall 0 (precision stays defined over words with >= 1 predicted cut);
  * `common_split`: every setting scored on the words ALL settings split at least once (a like-for-like word set);
  * aggregation = mean over words (weighted by word_freq when freq_scale), then macro over languages.
Persian (fas_Arab/pes_Arab) is excluded by default, like the other intrinsic tables.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from moto.utils.args import Args, load_args

logger = logging.getLogger(__name__)

TARGET_LANGUAGES = ["arb_Arab", "ces_Latn", "cmn_Hans", "dan_Latn", "deu_Latn", "ell_Grek", "eng_Latn", "fas_Arab",
                    "fra_Latn", "hun_Latn", "ind_Latn", "ita_Latn", "jpn_Jpan", "nld_Latn", "pol_Latn", "por_Latn",
                    "rus_Cyrl", "spa_Latn", "swe_Latn", "tur_Latn", "vie_Latn"]
# dataset code (iso_Script) -> our code
ALIASES = {"pes_Arab": "fas_Arab", "zho_Hans": "cmn_Hans", "zho_Hani": "cmn_Hans", "cmn_Hani": "cmn_Hans",
           "pes_Arab_": "fas_Arab"}
FLORES_FALLBACK = {"cmn_Hani": ["cmn_Hans"], "fas_Arab": ["pes_Arab"]}
VARIANTS = ("excl_single", "incl_single")


@dataclass
class MorphScoreArgs(Args):
    tokeval_config: str = "configs/tokeval/moto_eval.yaml"
    settings: List[str] = field(default_factory=list)       # empty = every setting in the config except `skip`
    skip: List[str] = field(default_factory=lambda: ["moto_v1", "byt5_unused"])
    dataset: str = "catherinearnett/morphscore"
    data_path: Optional[str] = None
    output_dir: str = "results/morphscore"
    exclude_languages: List[str] = field(default_factory=lambda: ["fas_Arab"])
    context: str = "word"                                    # word | sentence
    freq_scale: bool = True
    max_items_per_language: int = 0                          # 0 = all
    dump_examples: int = 0                                   # write this many words per (setting, language) with gold/predicted cuts
    check_official: bool = False                             # also run the reference `morphscore` package on HF tokenizers


# ── pure scoring functions (no model needed; unit-tested) ────────────────────────────────────────────────────────────

def to_code(code: str) -> str:
    """'tur_latn' -> 'tur_Latn' (the dataset writes the script in lower case), then ALIASES."""
    iso, _, script = str(code).partition("_")
    out = f"{iso}_{script.capitalize()}" if script else iso
    return ALIASES.get(out, out)


def gold_cuts(wordform: str, preceding: object, following: object) -> Optional[Set[int]]:
    """Interior character offsets of the morpheme boundaries; None if the annotation does not fit the word."""
    pre = "" if pd.isna(preceding) else str(preceding)
    fol = "" if pd.isna(following) else str(following)
    word = str(wordform)
    cuts = set()
    if pre:
        if not word.lower().startswith(pre.lower()):
            return None
        cuts.add(len(pre))
    if fol:
        if not word.lower().endswith(fol.lower()):
            return None
        cuts.add(len(word) - len(fol))
    cuts = {c for c in cuts if 0 < c < len(word)}
    return cuts or None


def pred_cuts(offsets: Sequence[Tuple[int, int]], lo: int, hi: int) -> Set[int]:
    """Token boundaries strictly inside the word span [lo, hi), relative to the word start."""
    return {e - lo for _, e in offsets if lo < e < hi}


def score_item(gold: Set[int], pred: Set[int]) -> Dict[str, float]:
    hit = len(gold & pred)
    return {"recall": hit / len(gold), "precision": hit / len(pred) if pred else float("nan"), "n_pred": len(pred)}


def find_word(sentence: str, word: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", sentence)
    return (m.start(), m.end()) if m else None


def aggregate(items: pd.DataFrame, freq_scale: bool) -> Dict[str, Dict[str, float]]:
    """items columns: recall, precision, n_pred, weight, n_tokens, n_chars -> {variant: {precision, recall, f1, n}}."""
    out = {}
    for variant in VARIANTS:
        d = items if variant == "incl_single" else items[items.n_pred > 0]
        w = d.weight if freq_scale else pd.Series(1.0, index=d.index)
        dp = d[d.n_pred > 0]
        wp = w.loc[dp.index]
        rec = float((d.recall * w).sum() / w.sum()) if len(d) and w.sum() else float("nan")
        prec = float((dp.precision * wp).sum() / wp.sum()) if len(dp) and wp.sum() else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else float("nan")
        out[variant] = {"precision": prec, "recall": rec, "f1": f1, "n": int(len(d))}
    return out


# ── data and tokenizers ──────────────────────────────────────────────────────────────────────────────────────────────

def load_items(args: MorphScoreArgs) -> pd.DataFrame:
    if args.data_path:
        p = Path(args.data_path)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_json(p, lines=True) if p.suffix in (".jsonl", ".json") else pd.read_csv(p)
    else:
        from datasets import concatenate_datasets, load_dataset
        ds = load_dataset(args.dataset)
        df = concatenate_datasets(list(ds.values())).to_pandas() if hasattr(ds, "values") else ds.to_pandas()
    missing = {"language", "wordform", "preceding_part", "following_part"} - set(df.columns)
    if missing:
        raise SystemExit(f"dataset lacks columns {sorted(missing)}; has {df.columns.tolist()}")
    df = df.copy()
    df["lang"] = df.language.map(to_code)
    return df


def prepare(df: pd.DataFrame, args: MorphScoreArgs) -> Dict[str, pd.DataFrame]:
    """{language -> items with gold cuts}, restricted to TARGET_LANGUAGES minus exclude_languages."""
    avail = set(df.lang)
    wanted = [l for l in TARGET_LANGUAGES if l not in args.exclude_languages]
    logger.info("in MorphScore and evaluated: %s", [l for l in wanted if l in avail])
    logger.info("not in MorphScore (reported n/a): %s", [l for l in wanted if l not in avail])
    logger.info("in MorphScore but not ours (ignored): %d languages", len(avail - set(TARGET_LANGUAGES)))
    out = {}
    for lang in wanted:
        d = df[df.lang == lang].copy()
        if not len(d):
            continue
        d["gold"] = [gold_cuts(w, p, f) for w, p, f in zip(d.wordform, d.preceding_part, d.following_part)]
        bad = int(d.gold.isna().sum())
        d = d[d.gold.notna()]
        if args.context == "sentence":
            if "sentence" not in d:
                raise SystemExit("context=sentence needs a `sentence` column")
            d["span"] = [find_word(str(s), str(w)) for s, w in zip(d.sentence, d.wordform)]
            d = d[d.span.notna()]
        if args.max_items_per_language:
            d = d.head(args.max_items_per_language)
        d["weight"] = pd.to_numeric(d["word_freq"], errors="coerce").fillna(1.0) if "word_freq" in d else 1.0
        logger.info("%s: %d items (%d dropped: annotation does not fit the word)", lang, len(d), bad)
        out[lang] = d
    return out


class Segmenter:
    """text -> (token offsets). Subclasses per setting kind."""

    def offsets(self, language: str, texts: List[str]) -> List[List[Tuple[int, int]]]:
        raise NotImplementedError


class HFSegmenter(Segmenter):
    def __init__(self, repo: str, revision: Optional[str] = None, routed: bool = False):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)
        self.tok.model_max_length = int(1e9)
        self.routed = routed
        if not routed and not getattr(self.tok, "is_fast", False):
            raise SystemExit(f"{repo}: needs a fast tokenizer for offset mapping")

    def _route(self, language: str):
        try:
            self.tok.resolve_language(language)
        except ValueError:
            for src, cands in FLORES_FALLBACK.items():
                if language in cands or language == src:
                    self.tok.set_mapping({language: src})
        self.tok.set_language(language)

    def offsets(self, language, texts):
        if self.routed:
            self._route(language)
        enc = self.tok(texts, add_special_tokens=False, return_offsets_mapping=True)
        return [[tuple(o) for o in off if o[1] > o[0]] for off in enc["offset_mapping"]]


class CharSegmenter(Segmenter):
    """Byte-level baseline (ByT5): every character boundary is a token boundary."""

    def offsets(self, language, texts):
        return [[(i, i + 1) for i in range(len(t))] for t in texts]


def build_segmenters(cfg_path: str, names: List[str], skip: List[str]) -> Dict[str, Segmenter]:
    from omegaconf import OmegaConf
    settings = OmegaConf.to_container(OmegaConf.load(cfg_path))["settings"]
    out = {}
    for name, c in settings.items():
        if (names and name not in names) or name in skip:
            continue
        if name.startswith("byt5"):
            out[name] = CharSegmenter()
        elif c["mode"] == "moto_repo":
            out[name] = HFSegmenter(c["repo"], c.get("revision"), routed=True)
        elif c["mode"] == "single":
            out[name] = HFSegmenter(c["path"], c.get("revision"))
        else:
            logger.warning("skipping %s: mode %s not supported", name, c["mode"])
    return out


def score_language(seg: Segmenter, lang: str, d: pd.DataFrame, context: str) -> pd.DataFrame:
    if context == "word":
        texts = d.wordform.astype(str).tolist()
        offs = seg.offsets(lang, texts)
        preds = [pred_cuts(o, 0, len(t)) for o, t in zip(offs, texts)]
        n_tok = [len(o) for o in offs]
    else:
        texts = d.sentence.astype(str).tolist()
        offs = seg.offsets(lang, texts)
        preds, n_tok = [], []
        for o, (lo, hi) in zip(offs, d.span):
            preds.append(pred_cuts(o, lo, hi))
            n_tok.append(sum(1 for s, e in o if s < hi and e > lo))
    rows = [{**score_item(g, p), "weight": w, "n_tokens": n, "n_chars": len(wf), "wordform": wf,
             "gold": sorted(g), "pred": sorted(p)}
            for g, p, w, n, wf in zip(d.gold, preds, d.weight, n_tok, d.wordform)]
    return pd.DataFrame(rows)


def common_split_row(items_by_setting: Dict[str, pd.DataFrame], freq_scale: bool) -> Dict[str, Dict[str, float]]:
    """Score every setting on the words that ALL settings split at least once: `excl_single` compares different word sets
    (a tokenizer with a bigger effective vocabulary keeps whole the common words and is scored on the rare ones)."""
    mask = None
    for it in items_by_setting.values():
        m = (it.n_pred > 0).to_numpy()
        mask = m if mask is None else (mask & m)
    return {name: aggregate(it[mask], freq_scale)["excl_single"] for name, it in items_by_setting.items()}, int(mask.sum())


def main(args: MorphScoreArgs):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = prepare(load_items(args), args)
    segmenters = build_segmenters(args.tokeval_config, args.settings, args.skip)
    rows, kept = [], {}
    for name, seg in segmenters.items():
        for lang, d in data.items():
            items = score_language(seg, lang, d, args.context)
            kept.setdefault(lang, {})[name] = items
            for variant, m in aggregate(items, args.freq_scale).items():
                rows.append({"setting": name, "language": lang, "variant": variant, **m,
                             "token_char_ratio": float(items.n_tokens.sum() / items.n_chars.sum())})
        logger.info("done %s", name)
    for lang, by_setting in kept.items():
        if len(by_setting) > 1:
            scores, n_common = common_split_row(by_setting, args.freq_scale)
            for name, m in scores.items():
                rows.append({"setting": name, "language": lang, "variant": "common_split", **{**m, "n": n_common}})
    if args.dump_examples:
        import json
        with open(out / "examples.jsonl", "w") as fh:
            for lang, by_setting in kept.items():
                for name, it in by_setting.items():
                    for r in it.head(args.dump_examples).itertuples():
                        fh.write(json.dumps({"setting": name, "language": lang, "word": r.wordform, "gold": r.gold,
                                             "pred": r.pred}, ensure_ascii=False) + "\n")
    per_lang = pd.DataFrame(rows)
    per_lang.to_csv(out / "morphscore_per_language.csv", index=False)
    summary = (per_lang.groupby(["setting", "variant"])
               .agg(precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
                    n_languages=("language", "nunique"), n_words=("n", "sum")).reset_index())
    summary.to_csv(out / "morphscore_summary.csv", index=False)
    print(summary[summary.variant.isin(["excl_single", "common_split"])].round(3).to_string(index=False))
    if args.check_official:
        _check_official(args, data, segmenters, per_lang)


def _check_official(args, data, segmenters, per_lang):
    """Run the reference package on plain HF tokenizers and print its numbers next to ours (sanity check)."""
    from morphscore.morphscore import MorphScore
    for name, seg in segmenters.items():
        if not isinstance(seg, HFSegmenter) or seg.routed:
            continue
        ours = per_lang[(per_lang.setting == name) & (per_lang.variant == "excl_single")]
        for lang in ours.language:
            iso = lang[:3] + "_" + lang[4:].lower()
            res = MorphScore(language_subset=[iso], freq_scale=args.freq_scale, exclude_single_tok=True).eval(seg.tok)
            logger.info("%s %s official=%s ours=%s", name, lang, res, ours[ours.language == lang][["precision", "recall"]].to_dict("records"))


if __name__ == "__main__":
    main(load_args(MorphScoreArgs))
