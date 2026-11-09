"""FLORES+ rolling-loglikelihood metrics (same shape as lm-eval's wikitext task)."""

import re


def process_results(doc, results):
    (loglikelihood,) = results
    text = doc["text"]
    n_words = len(re.split(r"\s+", text.strip()))
    n_bytes = len(text.encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, n_words),
        "byte_perplexity": (loglikelihood, n_bytes),
        "bits_per_byte": (loglikelihood, n_bytes),
    }
