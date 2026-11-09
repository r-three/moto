"""
Regularized BPE trainer with modified merge objective.

At each step the pair (u, w) with the highest score is selected:

    score(u, w) = N_{uw} * λ_seg - λ_tok * (|u| + |w|)

Training stops when:
  - the target vocab_size is reached, OR
  - no pair meets min_frequency, OR
  - the best score falls below λ_voc  (hard early-stopping threshold)

Setting λ_seg=1, λ_tok=0, λ_voc=0 recovers standard BPE.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
WordTuple = Tuple[str, ...]
WordCounts = Dict[WordTuple, int]
Merges = List[Tuple[str, str]]


class RegularizedBPETrainer:
    """
    BPE trainer with a regularized merge-selection objective.

    Parameters
    ----------
    vocab_size:   target vocabulary size (upper bound; may stop earlier if λ_voc > 0)
    lambda_voc:   per-merge cost threshold — stop when best score < λ_voc
    lambda_tok:   token-length penalty  — discourages merging long tokens
    lambda_seg:   frequency reward scale — higher values weight raw frequency more
    min_frequency: pairs seen fewer times than this are never merged
    special_tokens: prepended to the vocab with IDs 0, 1, 2, …
    initial_alphabet: tokens that must be in the vocab before any merges
    show_progress: display a tqdm progress bar if available
    """

    def __init__(
        self,
        vocab_size: int,
        lambda_voc: float = 0.0,
        lambda_tok: float = 0.0,
        lambda_seg: float = 1.0,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        initial_alphabet: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> None:
        self.vocab_size = vocab_size
        self.lambda_voc = lambda_voc
        self.lambda_tok = lambda_tok
        self.lambda_seg = lambda_seg
        self.min_frequency = min_frequency
        self.special_tokens: List[str] = list(special_tokens or [])
        self.initial_alphabet: List[str] = list(initial_alphabet or [])
        self.show_progress = show_progress

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, u: str, w: str, freq: int) -> float:
        # return freq * self.lambda_seg - self.lambda_voc -  (len(u) + len(w)) ** self.lambda_tok 
        return freq * self.lambda_seg - self.lambda_voc - self.lambda_tok * (len(u) + len(w))

    # ------------------------------------------------------------------
    # Pair-count bookkeeping
    # ------------------------------------------------------------------

    def _build_pair_counts(self, words: WordCounts) -> Counter:
        counts: Counter = Counter()
        for word, n in words.items():
            for i in range(len(word) - 1):
                counts[(word[i], word[i + 1])] += n
        return counts

    def _apply_merge(
        self,
        words: WordCounts,
        pair_counts: Counter,
        u: str,
        w: str,
    ) -> WordCounts:
        """
        Merge (u, w) -> uw throughout all words, updating pair_counts in-place.

        For each occurrence of the pair in a word the neighbour pairs change:
            left  neighbour x: (x, u) removed, (x, uw) added
            right neighbour y: (w, y) removed, (uw, y) added
        """
        uw = u + w
        new_words: WordCounts = {}

        for word, n in words.items():
            if u not in word:
                new_words[word] = new_words.get(word, 0) + n
                continue

            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == u and word[i + 1] == w:
                    # update left-neighbour pair
                    if new_word:
                        pair_counts[(new_word[-1], u)] -= n
                        pair_counts[(new_word[-1], uw)] += n
                    new_word.append(uw)
                    i += 2
                    # update right-neighbour pair
                    if i < len(word):
                        pair_counts[(w, word[i])] -= n
                        pair_counts[(uw, word[i])] += n
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_t = tuple(new_word)
            new_words[new_word_t] = new_words.get(new_word_t, 0) + n

        pair_counts.pop((u, w), None)
        return new_words

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, word_counts: WordCounts) -> Tuple[Dict[str, int], Merges]:
        """
        Run the regularized BPE merge loop on pre-counted word frequencies.

        Parameters
        ----------
        word_counts:
            Mapping from (char, char, …) tuples to their corpus frequency.
            Build this with :func:`build_word_counts`.

        Returns
        -------
        vocab_dict : dict[token -> id]   (special tokens assigned IDs first)
        merges     : list[(u, w)]        in the order they were applied
        """
        vocab: set[str] = set(self.initial_alphabet)
        for word in word_counts:
            vocab.update(word)
        for st in self.special_tokens:
            vocab.add(st)

        words: WordCounts = {w: n for w, n in word_counts.items() if n >= self.min_frequency}
        if not words:
            logger.warning(
                "No word types meet min_frequency=%d; returning character-level vocab.",
                self.min_frequency,
            )
            return self._build_vocab(vocab), []

        pair_counts = self._build_pair_counts(words)
        merges: Merges = []

        target = self.vocab_size - len(self.special_tokens)

        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    total=max(0, target - len(vocab)),
                    desc="Regularized BPE merges",
                    unit="merge",
                )
            except ImportError:
                pass

        while len(vocab) < target:
            if not pair_counts:
                logger.info("No more pairs available. Stopping.")
                break

            best_pair: Optional[Tuple[str, str]] = None
            best_freq = 0
            best_score = float("-inf")
            for (u, w), freq in pair_counts.items():
                if freq < self.min_frequency:
                    continue
                frequency = freq / sum(pair_counts.values())
                s = self._score(u, w, frequency)
                if s > best_score:
                    best_score = s
                    best_pair = (u, w)
                    best_freq = frequency
            logger.info(f"Best pair: {best_pair} with score {best_score:.6f} with frequency {best_freq} at vocab size {len(vocab)} / {target}")
            
            if best_pair is None:
                logger.info(
                    "No pairs meet min_frequency=%d. Stopping.", self.min_frequency
                )
                break

            if best_score < self.lambda_voc:
                logger.info(
                    "Early stop: best score %.6f < λ_voc %.6f at vocab size %d / %d.",
                    best_score,
                    self.lambda_voc,
                    len(vocab),
                    target,
                )
                break

            u, w = best_pair
            vocab.add(u + w)
            merges.append((u, w))
            words = self._apply_merge(words, pair_counts, u, w)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        logger.info(
            "Regularized BPE done: %d merges, final vocab size %d (target %d).",
            len(merges),
            len(vocab),
            target,
        )
        return self._build_vocab(vocab), merges

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_vocab(self, vocab: set[str]) -> Dict[str, int]:
        vocab_dict: Dict[str, int] = {}
        for st in self.special_tokens:
            if st not in vocab_dict:
                vocab_dict[st] = len(vocab_dict)
        for token in sorted(vocab - set(self.special_tokens)):
            vocab_dict[token] = len(vocab_dict)
        return vocab_dict


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


def build_word_counts(
    texts: Iterable[str],
    pre_tokenizer,
    max_texts: int = -1,
    log_every: int = 100_000,
) -> WordCounts:
    """
    Stream texts through a HuggingFace pre-tokenizer and count word-type frequencies.

    Each word returned by the pre-tokenizer is split into its constituent characters
    (which are already byte-mapped for ByteLevel pre-tokenizers) to produce the
    character-tuple representation expected by :class:`RegularizedBPETrainer`.

    Parameters
    ----------
    texts:          iterable of raw text strings
    pre_tokenizer:  any HuggingFace pre-tokenizer with a ``pre_tokenize_str`` method
    max_texts:      cap on number of texts processed (-1 = unlimited)
    log_every:      log a progress message every N texts
    """
    counts: Counter = Counter()
    n = 0
    for text in texts:
        if not text:
            continue
        for word_str, _ in pre_tokenizer.pre_tokenize_str(text):
            counts[tuple(word_str)] += 1
        n += 1
        if n % log_every == 0:
            logger.info("build_word_counts: processed %d texts, %d word types", n, len(counts))
        if max_texts > 0 and n >= max_texts:
            break
    logger.info(
        "build_word_counts done: %d texts, %d unique word types, %d total word tokens",
        n,
        len(counts),
        sum(counts.values()),
    )
    return dict(counts)
