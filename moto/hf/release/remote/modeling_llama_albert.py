"""Llama with ALBERT-style factorized embeddings, as trained by lingua (model.use_factorized_embeddings).

lingua (apps/main/transformer.py) replaces the vocab x d_model embedding and output matrices with
two factors each, through a small `embedding_dim` (config.json):

    input:  Embedding(vocab, embedding_dim) -> Linear(embedding_dim, d_model)
    output: Linear(d_model, embedding_dim)  -> Linear(embedding_dim, vocab)

Everything else is the standard Llama decoder, so this subclasses transformers' Llama and only
swaps those two modules. Parameter names match the converted checkpoint:
model.embed_tokens.{0,1}.weight and lm_head.{0,1}.weight.

The config is a plain LlamaConfig (model_type "llama") with one extra key, embedding_dim, and
config.json's auto_map names only this model class. So AutoConfig and AutoTokenizer load
without remote code, and only AutoModelForCausalLM needs trust_remote_code=True.

The input and output factors are separate parameters (lingua weight_tying: false), so weight
tying is disabled whatever the config says: tying would overwrite the trained output matrix
with the input embedding. Works with transformers 4.4x-4.57 and 5.x.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM


class LlamaAlbertForCausalLM(LlamaForCausalLM):
    config_class = LlamaConfig
    # transformers 4.x reads a list, 5.x a {target: source} dict; empty in both = never tie
    _tied_weights_keys = {} if isinstance(getattr(LlamaForCausalLM, "_tied_weights_keys", None), dict) else []

    def __init__(self, config: LlamaConfig):
        config.tie_word_embeddings = False
        super().__init__(config)
        e = config.embedding_dim
        self.model.embed_tokens = nn.Sequential(
            nn.Embedding(config.vocab_size, e),
            nn.Linear(e, config.hidden_size, bias=False),
        )
        self.lm_head = nn.Sequential(
            nn.Linear(config.hidden_size, e, bias=False),
            nn.Linear(e, config.vocab_size, bias=False),
        )
        self.post_init()

    def tie_weights(self, *args, **kwargs):  # the factors are trained separately
        return None

    def get_input_embeddings(self):
        return self.model.embed_tokens[0]

    def set_input_embeddings(self, value):
        self.model.embed_tokens[0] = value

    def get_output_embeddings(self):
        return self.lm_head[1]

    def set_output_embeddings(self, new_embeddings):
        self.lm_head[1] = new_embeddings
