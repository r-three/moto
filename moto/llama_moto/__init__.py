"""Llama-MoTo: Mixture-of-Tokenizers extension for Llama.

Pipeline overview
-----------------
1. **byte_fallback_analysis**  — Audit how often Llama falls back to <0x??> byte
   tokens on each target language.  Languages with high byte-fallback rates are
   candidates for vocabulary extension.

2. **picky_bpe**  — PickyBPE: prune the base Llama vocabulary to only the tokens
   that appear on the target corpus, then learn additional merge rules for the
   underserved languages.

3. **multiplicity_tokens**  — Identify tokens whose surface form is shared between
   the base Llama vocab and the newly learned BPE vocab.  These duplicates need
   special handling (shared embedding vs. dedicated row).

4. **extend_merges**  — Run additional BPE iterations on the target corpus starting
   from the pruned vocabulary, producing a language-extended merge list.

5. **moto_tokenizer**  — The MoTo tokenizer wrapper.  It holds the full superset
   vocabulary (Llama vocab ∪ extended tokens) but at inference / training only X
   tokens (y embedding parameters) are active, controlled by a per-input language
   mask.
"""

from moto.llama_moto.moto_tokenizer import MoToTokenizer

__all__ = ["MoToTokenizer"]
