import regex

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

# Latin word rules (Boundless BPE B-3/B-4/B-5).
# Order matters: CamelCase prefix must come before Title/lower.
# \p{M}*): this strips the zerospace chars, relevant for arabic and farsi
_LAT = "|".join(
    [
        r" ?(?:\p{Lu}\p{M}*)+(?=(?:\p{Lu}\p{M}*)(?:\p{Ll}\p{M}*))",  # CamelCase prefix
        r" ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+",  # Title / lower
        r" ?(?:\p{Lu}\p{M}*)+",  # ALL CAPS
    ]
)

_PUNCT = r" ?(?:[\p{P}\p{S}]\p{M}*)+"  # punctuation + symbols
_WS = r"[^\S\r\n]*[\n\r]+|[^\S\r\n]+"  # whitespace / newlines

# Contraction rule: Unicode-general (covers it's, l'homme, etc.)
# Uses \p{L} not [a-z] so works across Latin-script languages.
# Placed before word rules so "it's" is one pre-token not two.
_CONTR = r" ?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+"

# ---------------------------------------------------------------------------
# Per-language pre-tokenizer patterns
# ---------------------------------------------------------------------------


# ------------------------------------------------------------------
# Latin-script languages (fusional / analytic)
# Contraction rule covers both English (it's) and French (l'homme).
# ------------------------------------------------------------------
eng_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
fw_edu = "|".join([_CONTR, _LAT, _PUNCT, _WS])
fra_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
spa_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
por_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
ita_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
deu_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
nld_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
dan_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
swe_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
pol_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
ces_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
hun_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
ind_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
vie_Latn = "|".join([_CONTR, _LAT, _PUNCT, _WS])
# Vietnamese: syllables are whitespace-delimited so _LAT naturally
# produces syllable-level pre-tokens. Multi-syllable words
# (khởi nghiệp) are learned as BPE merges across whitespace.

# ------------------------------------------------------------------
# Turkish (agglutinative, Latin script)
# Turkish uses apostrophe to separate proper noun suffixes
# (Türkiye'de) but not for general contractions. Narrow rule
# instead of general _CONTR to avoid over-splitting.
# ------------------------------------------------------------------
tur_Latn = "|".join(
    [
        r" ?(?:\p{Lu}\p{M}*)?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+",  # Türkiye'de
        _LAT,
        _PUNCT,
        _WS,
    ]
)

# ------------------------------------------------------------------
# Russian (fusional, Cyrillic)
# ------------------------------------------------------------------
rus_Cyrl = "|".join(
    [
        r" ?(?:\p{Cyrillic}\p{M}*)+",
        _PUNCT,
        _WS,
    ]
)

# ------------------------------------------------------------------
# Arabic (non-concatenative, Arabic script)
# Arabic-Indic numerals (٠١٢٣) are \p{Arabic} but number handling
# is done upstream in the script — they should be normalised to
# Western Arabic numerals (0-9) before this pattern fires.
# Arabic clitics (bi-, wa-, li-, al-) are written without separator
# and are part of the word run; BPE segments them if frequent enough.
# Internal diacritics (harakat) captured via \p{M}.
# ------------------------------------------------------------------
arb_Arab = "|".join(
    [
        r" ?(?:\p{Arabic}\p{M}*)+",
        _PUNCT,
        _WS,
    ]
)

# ------------------------------------------------------------------
# Persian (Arabic script)
# Persian-specific codepoints (ک U+06A9, ی U+06CC, etc.) are in
# \p{Arabic} Unicode block so no extra rule needed. Upstream NFC +
# codepoint normalisation (Arabic ك -> Persian ک) handles ambiguity.
# ------------------------------------------------------------------
fas_Arab = "|".join(
    [
        r" ?(?:\p{Arabic}\p{M}*)+",
        _PUNCT,
        _WS,
    ]
)

_ZWNJ = "\u200c"
fas_Arab_w_zwnj = "|".join(
    [
        rf" ?(?:\p{{Arabic}}\p{{M}}*|{_ZWNJ})+",  # allow ZWNJ inside Arabic runs
        _PUNCT,
        _WS,
    ]
)

## handle clitics as separate morphemes by splitting on ZWNJ, but this is not ideal since it breaks up words that are correctly written with ZWNJ (می‌خواهم) and produces many single-character tokens (می + ‌ + خواهم). Better to allow ZWNJ in the word run and let BPE learn the clitics as subword units.
_ZWNJ = "\u200c"
fas_Arab = "|".join(
    [
        rf" ?(?:\p{{Arabic}}\p{{M}}*)+(?={_ZWNJ})",  # Arabic run stopping before ZWNJ
        _ZWNJ,                                         # ZWNJ as its own token
        r" ?(?:\p{Arabic}\p{M}*)+",                   # normal Arabic run
        _PUNCT,
        _WS,
    ]
)
# ------------------------------------------------------------------
# Greek (fusional, Greek script)
# ------------------------------------------------------------------
ell_Grek = "|".join(
    [
        r" ?(?:\p{Greek}\p{M}*)+",
        _PUNCT,
        _WS,
    ]
)

# ------------------------------------------------------------------
# Chinese / Mandarin (isolating, Han script)
# Han runs are NOT hard-split at character boundaries, allowing BPE
# to learn multi-character words and compounds (自然语言处理 can
# become a single token). Splitting each character individually would
# fill the 8k vocab with single characters and bigrams only.
# Latin rule handles loanwords and mixed-script text.
# ------------------------------------------------------------------
cmn_Hani = "|".join(
    [
        r" ?(?:\p{Han})+",
        r" ?(?:\p{Latin}\p{M}*)+",  # loanwords / mixed text
        _PUNCT,
        _WS,
    ]
)

# ------------------------------------------------------------------
# Japanese (agglutinative, mixed script)
# Japanese mixes Han, Hiragana, and Katakana within a single word
# (食べる = Han + Hiragana; テレビ = Katakana). Splitting on script
# boundaries mid-word prevents BPE from learning verb stem +
# inflection as a unit (食べ + る). A single rule therefore covers
# all three scripts plus:
#   \u30FC  katakana-hiragana prolonged sound mark (ー)
#   \u309B  voiced iteration mark (゛)
#   \u309C  semi-voiced iteration mark (゜)
# Latin rule handles romaji and loanword annotations.
# ------------------------------------------------------------------
jpn_Jpan = "|".join(
    [
        r" ?(?:[\p{Han}\p{Hiragana}\p{Katakana}\u30FC\u309B\u309C])+",
        r" ?(?:\p{Latin}\p{M}*)+",  # romaji
        _PUNCT,
        _WS,
    ]
)

boundless_bpe = "|".join(
    [ r" ?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+", # B-1, contraction 
    r"_(?:\p{Ll}\p{M}*)+", # B-2, snake_case 
    r" ?(?:\p{Lu}\p{M}*)+(?=(?:\p{Lu}\p{M}*)(?:\p{Ll}\p{M}*))", # B-3, words 
    r" ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+", # B-4, words 
    r" ?(?:\p{Lu}\p{M}*)+", # B-5, words 
    r" ?(?:[\p{Lt}\p{Lm}\p{Lo}]\p{M}*)+", # B-6, words 
    r"(?:\p{N}\p{M}*){1,3}(?=(?:(?:\p{N}\p{M}*){3})*(?:(?:\P{N}\p{M}*)|$))",# B-7 
    r" ?(?:[\p{P}\p{S}]\p{M}*)+", # B-8, punct and symbols 
    r"[^\S\r\n]*[\n\r]+|[^\S\r\n]+", # B-9, whitespace 
    r"(?:[\p{Z}\p{C}]\p{M}*)+", # B-10, sep or control 
    r"\p{M}+", # B-11, leftover marks 
    ])

addition = "|".join([r'[+=]', _WS])
addition_no_ws = "|".join([r'[+=]'])
addition_split_on_hyphen = "|".join([r'[+=\-]', _WS])

SCRIPT_AWARE_CHAR_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"          # contractions
    r"|[^\r\n\p{L}\p{N}]?\p{Latin}+"          # Latin: word-level (good fertility already)
    r"|[^\r\n\p{L}\p{N}]?\p{Han}"             # Han: CHARACTER-level (no + → single chars)
    r"|[^\r\n\p{L}\p{N}]?\p{Hiragana}"        # Hiragana: character-level
    r"|[^\r\n\p{L}\p{N}]?\p{Katakana}"        # Katakana: character-level
    r"|[^\r\n\p{L}\p{N}]?\p{Arabic}\p{M}*"   # Arabic: character-level
    r"|[^\r\n\p{L}\p{N}]?\p{Cyrillic}\p{M}*" # Cyrillic: character-level
    r"|[^\r\n\p{L}\p{N}]?\p{Greek}\p{M}*"    # Greek: character-level
    r"|\p{N}{1,3}"                             # numbers
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"             # punctuation
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"              # whitespace
)


boundless_bpe_renewed= "|".join([
    # B-0: English contractions — must come first
    # r" ?(?:\p{L}\p{M}*)+(?:'s|'t|'re|'ve|'m|'ll|'d|n't|\u2019s|\u2019t|\u2019re|\u2019ve|\u2019m|\u2019ll|\u2019d)",
    # B-1: single-char clitic (l', d', j', c', n')
    r" ?(?:[ldjLDJcCnN])['\u2019](?=\p{L})",
    # B-1b: multi-char stem before apostrophe
    r" ?(?:\p{L}\p{M}*){2,}(?=['\u2019])",
    # B-1c: apostrophe + suffix ('de, 'nin, 's-Gravenhage)
    r"['\u2019](?:\p{L}\p{M}*)+",
    # B-1f: ZWNJ + following letters
    r"\u200c(?:\p{L}\p{M}*)+",
    r"_(?:\p{Ll}\p{M}*)+",                        # B-2
    r" ?(?:\p{Lu}\p{M}*)+(?=(?:\p{Lu}\p{M}*)(?:\p{Ll}\p{M}*))", # B-3
    r" ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+",      # B-4
    r" ?(?:\p{Lu}\p{M}*)+",                        # B-5
    r" ?(?:[\p{Lt}\p{Lm}\p{Lo}]\p{M}*)+",         # B-6
    # r"(?:\p{N}\p{M}*){1,3}(?=(?:(?:\p{N}\p{M}*){3})*(?:(?:\P{N}\p{M}*)|$))", # B-7
    r" ?(?:[\p{P}\p{S}]\p{M}*)+",                 # B-8
    r"[^\S\r\n]*[\n\r]+|[^\S\r\n]+",              # B-9
    r"(?:[\p{Z}\p{C}]\p{M}*)+",                   # B-10
    r"\p{M}+",                                     # B-11
])