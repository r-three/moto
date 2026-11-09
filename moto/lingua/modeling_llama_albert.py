from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class LlamaAlbertConfig(LlamaConfig):
    model_type = "llama_albert"
    architectures = ["LlamaAlbertForCausalLM"]

    def __init__(self, embedding_dim=128, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.embedding_dim = embedding_dim
        self.auto_map={
                "AutoConfig": "modeling_llama_albert.LlamaAlbertConfig",
                "AutoModelForCausalLM": "modeling_llama_albert.LlamaAlbertForCausalLM",
            }
        self._auto_class="modeling_llama_albert.LlamaAlbertForCausalLM"


class LlamaAlbertForCausalLM(LlamaForCausalLM):
    config_class = LlamaAlbertConfig

    def __init__(self, config):
        super().__init__(config)

        # 1. Factorized Embeddings (ALBERT style)
        # Replacing self.model.embed_tokens with a Sequential layer
        self.model.embed_tokens = nn.Sequential(
            nn.Embedding(config.vocab_size, config.embedding_dim),
            nn.Linear(config.embedding_dim, config.hidden_size, bias=False),
        )

        # 2. Factorized LM Head
        # Sequential: Hidden -> Embedding Dim -> Vocab
        self.lm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.embedding_dim, bias=False),
            nn.Linear(config.embedding_dim, config.vocab_size, bias=False),
        )

        # Re-initialize weights for the new layers
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens[0]

    def set_input_embeddings(self, value):
        self.model.embed_tokens[0] = value

    def get_output_embeddings(self):
        return self.lm_head[1]

    def set_output_embeddings(self, new_embeddings):
        self.lm_head[1] = new_embeddings

    def forward(self, input_ids=None, **kwargs):
        # The base LlamaForCausalLM forward calls self.model(...)
        # Since we replaced self.model.embed_tokens with a Sequential,
        # LlamaModel's internal call to embed_tokens(input_ids) will
        # automatically run through both the Embedding and the Linear layer.
        return super().forward(input_ids=input_ids, **kwargs)
