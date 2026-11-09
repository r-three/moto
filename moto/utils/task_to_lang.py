LANG_TO_PRETTY = {
    "eng_Latn": "English",
    "arb_Arab": "Arabic",
    "ces_Latn": "Czech",
    "dan_Latn": "Danish",
    "deu_Latn": "German",
    "ell_Grek": "Greek",
    "fra_Latn": "French",
    "hun_Latn": "Hungarian",
    "ind_Latn": "Indonesian",
    "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese",
    "nld_Latn": "Dutch",    
    "pol_Latn": "Polish",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "spa_Latn": "Spanish",
    "swe_Latn": "Swedish",
    "tur_Latn": "Turkish",
    "vie_Latn": "Vietnamese",
    "zho_Hans": "Chinese",
    "cmn_Hani": "Chinese",
    "fas_Arab": "Farsi",
    "pes_Arab": "Farsi",
}
TASK_TO_LANG = {
    # English baselines
    "arc_challenge/acc_norm":   "eng_Latn",
    "arc_easy/acc_norm":        "eng_Latn",
    "piqa/acc_norm":            "eng_Latn",
    "hellaswag/acc_norm":       "eng_Latn",
    # Belebele
    "belebele_arb_Arab/acc_norm": "arb_Arab",
    "belebele_ces_Latn/acc_norm": "ces_Latn",
    "belebele_dan_Latn/acc_norm": "dan_Latn",
    "belebele_deu_Latn/acc_norm": "deu_Latn",
    "belebele_ell_Grek/acc_norm": "ell_Grek",
    "belebele_eng_Latn/acc_norm": "eng_Latn",
    "belebele_fra_Latn/acc_norm": "fra_Latn",
    "belebele_hun_Latn/acc_norm": "hun_Latn",
    "belebele_ind_Latn/acc_norm": "ind_Latn",
    "belebele_ita_Latn/acc_norm": "ita_Latn",
    "belebele_jpn_Jpan/acc_norm": "jpn_Jpan",
    "belebele_nld_Latn/acc_norm": "nld_Latn",
    "belebele_pes_Arab/acc_norm": "fas_Arab",   # pes -> fas
    "belebele_pol_Latn/acc_norm": "pol_Latn",
    "belebele_por_Latn/acc_norm": "por_Latn",
    "belebele_rus_Cyrl/acc_norm": "rus_Cyrl",
    "belebele_spa_Latn/acc_norm": "spa_Latn",
    "belebele_swe_Latn/acc_norm": "swe_Latn",
    "belebele_tur_Latn/acc_norm": "tur_Latn",
    "belebele_vie_Latn/acc_norm": "vie_Latn",
    "belebele_zho_Hans/acc_norm": "cmn_Hani",   # zho_Hans -> cmn_Hani
    # INCLUDE
    "include_base_44_arabic/acc":     "arb_Arab",
    "include_base_44_chinese/acc":    "cmn_Hani",
    "include_base_44_dutch/acc":      "nld_Latn",
    "include_base_44_french/acc":     "fra_Latn",
    "include_base_44_german/acc":     "deu_Latn",
    "include_base_44_greek/acc":      "ell_Grek",
    "include_base_44_hungarian/acc":  "hun_Latn",
    "include_base_44_indonesian/acc": "ind_Latn",
    "include_base_44_italian/acc":    "ita_Latn",
    "include_base_44_japanese/acc":   "jpn_Jpan",
    "include_base_44_persian/acc":    "fas_Arab",
    "include_base_44_polish/acc":     "pol_Latn",
    "include_base_44_portuguese/acc": "por_Latn",
    "include_base_44_russian/acc":    "rus_Cyrl",
    "include_base_44_spanish/acc":    "spa_Latn",
    "include_base_44_turkish/acc":    "tur_Latn",
    "include_base_44_vietnamese/acc": "vie_Latn",
    # XNLI
    "xnli_ar/acc": "arb_Arab",
    "xnli_zh/acc": "cmn_Hani",
    "xnli_de/acc": "deu_Latn",
    "xnli_el/acc": "ell_Grek",
    "xnli_en/acc": "eng_Latn",
    "xnli_es/acc": "spa_Latn",
    "xnli_fr/acc": "fra_Latn",
    "xnli_ru/acc": "rus_Cyrl",
    "xnli_tr/acc": "tur_Latn",
    "xnli_vi/acc": "vie_Latn",
    # XCOPA
    "xcopa_id/acc": "ind_Latn",
    "xcopa_it/acc": "ita_Latn",
    "xcopa_tr/acc": "tur_Latn",
    "xcopa_vi/acc": "vie_Latn",
    "xcopa_zh/acc": "cmn_Hani",
    # XStoryCloze
    "xstorycloze_ar/acc": "arb_Arab",
    "xstorycloze_en/acc": "eng_Latn",
    "xstorycloze_es/acc": "spa_Latn",
    "xstorycloze_id/acc": "ind_Latn",
    "xstorycloze_ru/acc": "rus_Cyrl",
    "xstorycloze_zh/acc": "cmn_Hani",
    # PAWS
    "paws_de/acc": "deu_Latn",
    "paws_en/acc": "eng_Latn",
    "paws_es/acc": "spa_Latn",
    "paws_fr/acc": "fra_Latn",
    "paws_ja/acc": "jpn_Jpan",
    "paws_zh/acc": "cmn_Hani",
    # Toksuite
    "toksuite_english_canonical/acc_norm": "eng_Latn",
    "toksuite_turkish_canonical/acc_norm": "tur_Latn",
    "toksuite_chinese_canonical/acc_norm": "cmn_Hani",
    "toksuite_italian_canonical/acc_norm": "ita_Latn",
    "toksuite_farsi_canonical/acc_norm":   "fas_Arab",
    "toksuite_math_canonical/acc_norm":    None,  # language-agnostic
    "toksuite_stem_canonical/acc_norm":    None,
    "toksuite_english/acc_norm": "eng_Latn",
    "toksuite_turkish/acc_norm": "tur_Latn",
    "toksuite_chinese/acc_norm": "cmn_Hani",
    "toksuite_italian/acc_norm": "ita_Latn",
    "toksuite_farsi/acc_norm":   "fas_Arab",
    "toksuite_math/acc_norm":    None,  # language-agnostic
    "toksuite_stem/acc_norm":    None,
    # Okapi HellaSwag
    "hellaswag_ar/acc_norm": "arb_Arab",
    "hellaswag_da/acc_norm": "dan_Latn",
    "hellaswag_de/acc_norm": "deu_Latn",
    "hellaswag_fr/acc_norm": "fra_Latn",
    "hellaswag_hu/acc_norm": "hun_Latn",
    "hellaswag_id/acc_norm": "ind_Latn",
    "hellaswag_it/acc_norm": "ita_Latn",
    "hellaswag_nl/acc_norm": "nld_Latn",
    "hellaswag_pt/acc_norm": "por_Latn",
    "hellaswag_ru/acc_norm": "rus_Cyrl",
    "hellaswag_vi/acc_norm": "vie_Latn",
    "hellaswag_es/acc_norm": "spa_Latn",

   "multiblimp_arb/acc_norm": "arb_Arab",
    "multiblimp_ces/acc_norm": "ces_Latn",
    "multiblimp_dan/acc_norm": "dan_Latn",
    "multiblimp_deu/acc_norm": "deu_Latn",
    "multiblimp_ell/acc_norm": "ell_Grek",
    "multiblimp_eng/acc_norm": "eng_Latn",
    "multiblimp_fas/acc_norm": "fas_Arab",
    "multiblimp_fra/acc_norm": "fra_Latn",
    "multiblimp_hun/acc_norm": "hun_Latn",
    "multiblimp_ita/acc_norm": "ita_Latn",
    "multiblimp_nld/acc_norm": "nld_Latn",
    "multiblimp_pol/acc_norm": "pol_Latn",
    "multiblimp_por/acc_norm": "por_Latn",
    "multiblimp_rus/acc_norm": "rus_Cyrl",
    "multiblimp_spa/acc_norm": "spa_Latn",
    "multiblimp_swe/acc_norm": "swe_Latn",
    "multiblimp_tur/acc_norm": "tur_Latn",

   "boolq/acc": "eng_Latn",
   "commonsense_qa/acc": "eng_Latn",
   "xwinograd_en/acc": "eng_Latn",
   "xwinograd_fr/acc": "fra_Latn",
   "xwinograd_jp/acc": "jpn_Jpan",
   "xwinograd_pt/acc": "por_Latn",
   "xwinograd_ru/acc": "rus_Cyrl",
   "xwinograd_zh/acc": "cmn_Hani"
}
flores = {
    "flores200_arb_Arab": "arb_Arab",
    "flores200_ces_Latn": "ces_Latn",
    "flores200_cmn_Hani": "cmn_Hani",
    "flores200_dan_Latn": "dan_Latn",
    "flores200_deu_Latn": "deu_Latn",
    "flores200_ell_Grek": "ell_Grek",
    "flores200_fas_Arab": "fas_Arab",
    "flores200_fra_Latn": "fra_Latn",
    # "flores200_fw_edu": "eng_Latn",
    "flores200_eng_Latn": "eng_Latn",
    "flores200_hun_Latn": "hun_Latn",
    "flores200_ind_Latn": "ind_Latn",
    "flores200_ita_Latn": "ita_Latn",
    "flores200_jpn_Jpan": "jpn_Jpan",
    "flores200_nld_Latn": "nld_Latn",
    "flores200_pol_Latn": "pol_Latn",
    "flores200_por_Latn": "por_Latn",
    "flores200_rus_Cyrl": "rus_Cyrl",
    "flores200_spa_Latn": "spa_Latn",
    "flores200_swe_Latn": "swe_Latn",
    "flores200_tur_Latn": "tur_Latn",
    "flores200_vie_Latn": "vie_Latn",
}
TASK_TO_LANG.update({f"{key}/word_perplexity": val for key, val in flores.items()})
TASK_TO_LANG.update({f"{key}/bits_per_byte": val for key, val in flores.items()})
TASK_TO_LANG.update({f"{key}/byte_perplexity": val for key, val in flores.items()})

LANGS_TO_TASKS = {}
for task, lang in TASK_TO_LANG.items():
    if lang is not None:
        LANGS_TO_TASKS.setdefault(lang, []).append(task)


MODEL_COLORS = {
    "FW2HQ-8K-v3-Oracle":      "#E63946",  # red
    "FW2HQ-v1-Oracle":         "#F4A261",  # orange
    "FW2HQ-v1-Oracle-09":         "#F4A261",  # orange
    "FW2HQ-v2-Oracle":         "#F4A261",  # orange
    "FW2HQ-v1-Random":         "#F4A261",  # orange
    "Llama":                   "#457B9D",  # steel blue (baseline)
    "Multilingual Boundless BPE": "#2A9D8F",  # teal
    "Multilingual GPT4-PreTok": "#6A4C93",  # purple    
    "Llama-43K":                   "#457B9D",  # steel blue (baseline)
    "Llama-Albert":                   "#457B9D",  # steel blue (baseline)
    "SS-Albert":                   "#457B9D",  # steel blue (baseline)
    "Llama-BPEDropout":        "#2A9D8F",  # teal
    "ByT5":                    "#E9C46A",  # yellow
    "FW2HQ-Llama-16K-Oracle":  "#6A4C93",  # purple
    "FW2HQ-Script-Oracle":     "#264653",  # dark teal
    "FW2HQ-16K-Oracle":     "#264653",  # dark teal
}

OLD_LANG_COLORS = {
    # Germanic (blues)
    "English": "#4E79A7",
    "German": "#7EB6D4",
    "Dutch": "#A8D1E7",
    "Swedish": "#C6E4F0",
    "Danish": "#B8DCF0",
    "Norwegian": "#D0EBF7",
    "Afrikaans": "#E3F4FC",

    # Romance (oranges)
    "French": "#F28E2B",
    "Italian": "#F5A85A",
    "Spanish": "#F8C285",
    "Portuguese": "#FAD9AE",
    "Romanian": "#FCE8CE",
    "Catalan": "#FDF0E0",

    # Slavic / Cyrillic (purples)
    "Russian": "#B07AA1",
    "Polish": "#C49DBA",
    "Ukrainian": "#D4B8CE",
    "Bulgarian": "#E2D0E2",
    "Serbian": "#C47EB5",
    "Czech": "#8B5A8B",
    "Slovak": "#A06CA0",

    # Greek (purple-adjacent, distinct)
    "Greek": "#7B2D8B",

    # Arabic script (teals)
    "Arabic": "#2A9D8F",
    "Farsi": "#4DB8AC",
    "Urdu": "#76CEC5",

    # CJK (reds)
    "Chinese": "#E63946",
    "Chinese": "#E63946",  # alias
    "Japanese": "#ED6B74",
    "Korean": "#F49DA3",

    # Turkic (greens)
    "Turkish": "#59A14F",
    "Kazakh": "#7DC474",
    "Uzbek": "#A3D99C",
    "Azerbaijani": "#C5ECC0",

    # Indic / Devanagari (pinks)
    "Hindi": "#E76F51",
    "Bengali": "#EF9778",
    "Marathi": "#F5B89F",
    "Nepali": "#FAD4C3",
    "Punjabi": "#FDEEE8",

    # Other South/SE Asian
    "Tamil": "#D4A017",
    "Telugu": "#E0BA50",
    "Thai": "#F5E6B0",
    "Vietnamese": "#C8E6C9",
    "Indonesian": "#A5D6A7",

    # Hebrew
    "Hebrew": "#6D9F71",

    # Finnish / Uralic (yellow-greens)
    "Finnish": "#EDC948",
    "Hungarian": "#F2D96E",
    "Estonian": "#F7E89A",

    "HellaSwag":      "#E63946",  # red
    "PIQA":           "#F4A261",  # orange
    "ARC-Easy":       "#2A9D8F",  # teal
    "ARC-Challenge":  "#1D6A5A",  # dark teal — paired with ARC-Easy
    "BoolQ":          "#6A4C93",  # purple
    "CommonsenseQA":  "#457B9D",  # blue
    "XWinograd":      "#E9C46A",  # yellow

}


LANG_COLORS = {
    'English': '#9250BE',
    'German': '#E0957A',
 'Dutch': '#C9A227',
 'Swedish': '#3A6FA8',
 'Danish': '#F28E2B',
 'Norwegian': '#A15B92',
 'French': '#C8E6C9',
 'Italian': '#3A7A55',
 'Spanish': '#8C5E3C',
 'Portuguese': '#3A6BD4',
 'Romanian': '#E63946',
 'Russian': '#A03040',
 'Polish': '#4A82B5',
 'Czech': '#4E79A7',
 'Greek': '#C8705C',
 'Arabic': '#C96A2A',
 'Farsi': '#B8961E',
 'Chinese': '#D4773A',
 'Japanese': '#F5A85A',
 'Korean': '#2A7A6A',
 'Turkish': '#7A4F35',
 'Telugu': '#E0BA50',
 'Vietnamese': '#ED6348',
 'Indonesian': '#97CA1E',
 'Finnish': '#5E9E53',
 'Hungarian': '#B07AA1',
 
    "HellaSwag":      "#E63946",  # red
    "PIQA":           "#F4A261",  # orange
    "ARC-Easy":       "#2A9D8F",  # teal
    "ARC-Challenge":  "#1D6A5A",  # dark teal — paired with ARC-Easy
    "BoolQ":          "#6A4C93",  # purple
    "CommonsenseQA":  "#457B9D",  # blue
    "XWinograd":      "#E9C46A",  # yellow

}

RUN_NAME_TO_DISPLAY = {
    "FW2HQ-8K-v3-Oracle":        "MoTo",
    "FW2HQ-v1-Oracle":           "MoTo-GPT4",
    "FW2HQ-v1-Oracle-09":        "MoTo-GPT4-Reg.",
    "FW2HQ-v1-Random":           "MoTo-Random",
    "FW2HQ-Llama-16K-Oracle":    "MoTo-Llama-16K",
    "FW2HQ-16K-Oracle":          "MoTo-16K",
    "FW2HQ-Script-Oracle":       "MoTo-Script",
    "FW2HQ-v2-Oracle":           "FW2HQ-v2",
    "Llama":                     "Llama",
    "Llama-BPEDropout":          "Llama-BPE-Drop.",
    "Llama-Albert":              "Llama-Albert",
    "Llama-43K":                 "Llama-43K",
    "SS-Albert":                 "SS-Albert",
    "Multilingual GPT4-PreTok":  "Multilingual GPT4-PreTok",
    "Multilingual Boundless BPE":"Multilingual Boundless BPE",
    "ByT5":                      "ByT5",
    "SA (wo XGLM)":              "SA (wo XGLM)",
}

# single source of truth — concept → hex
_CONCEPT_COLORS = {
    # MoTo family — reds/oranges
    "MoTo":           "#E63946",
    "MoTo-GPT4":      "#F4A261",
    "MoTo-Random":    "#F4845F",
    "MoTo-Script":    "#E76F51",
    "MoTo-16K":       "#FFBA08",
    "MoTo-Llama-16K": "#FFD166",
    "MoTo-GPT4-Reg.": "#F4A261",  # same hue as GPT4, alpha at plot time
    "FW2HQ-v2":       "#E9C46A",

    # Llama family — blues
    "Llama":          "#457B9D",
    "Llama-BPE-Drop.":"#7EB6D4",
    "Llama-Albert":   "#A8D1E7",
    "Llama-43K":      "#1D3557",

    # Multilingual baselines — greens
    "Multilingual GPT4-PreTok":   "#2A9D8F",
    "Multilingual Boundless BPE": "#57CC99",

    # Other — purples/grays
    "ByT5":           "#6A4C93",
    "SS-Albert":      "#B5838D",
    "SA (wo XGLM)":   "#888780",

    # multilingual addition (paper/plot_maddition.py)
    "MoTo-Reg.":      "#F28482",  # MoTo red, lighter
    "mBPE-2K":        "#80ED99",  # mBPE greens, light to dark with vocab size
    "mBPE-42K":       "#38A3A5",

    "Parity-Aware BPE":  "#4C9354",
    "Demoto":  "#6DB72F",
}

RUN_NAME_TO_COLOR = {
    run: _CONCEPT_COLORS[display]
    for run, display in RUN_NAME_TO_DISPLAY.items()
    if display in _CONCEPT_COLORS
}
# deltas from (release-)evals use display names ("MoTo", "mBPE-GPT4", ...) as run_name, so
# colours are looked up by those too; mBPE-* are the paper's short names of the multilingual BPEs
_CONCEPT_COLORS.setdefault("mBPE-GPT4 (323K)", "#21867A")  # green, darker than mBPE-GPT4
RUN_NAME_TO_COLOR.update(_CONCEPT_COLORS)
RUN_NAME_TO_COLOR.update({
    "mBPE-GPT4": _CONCEPT_COLORS["Multilingual GPT4-PreTok"],
    "mBPE-Boundless": _CONCEPT_COLORS["Multilingual Boundless BPE"],
    "Parity-Aware-BPE": _CONCEPT_COLORS["Parity-Aware BPE"],
    "Demoto-ed": _CONCEPT_COLORS["Demoto"],
    "Moto-Reg.": _CONCEPT_COLORS["MoTo-Reg."],  # spelling in paper/maddition/all_delta_multilingual_addition.py
})
RUN_NAME_TO_DISPLAY.setdefault("Moto-Reg.", "MoTo-Reg.")