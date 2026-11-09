# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from huggingface_hub import snapshot_download
from tokenizers import AddedToken, processors
from transformers import (
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
)
from transformers.convert_slow_tokenizer import TikTokenConverter

from moto.lingua.modeling_llama_albert import (
    LlamaAlbertConfig,
    LlamaAlbertForCausalLM,
)
from moto.utils.args import Args, HFArgs, load_args

# try:
#     from transformers import LlamaTokenizerFast
# except ImportError as e:
#     warnings.warn(e)
#     warnings.warn(
#         "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
#     )
#     LlamaTokenizerFast = None

"""
Sample usage:

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    input_dir /path/to/downloaded/llama/weights model_size 1B llama_version 3.2 output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).

If you want your tokenizer to add a bos automatically you should update the tokenizer._tokenizers.post_processor:

```py
from tokenizers import processors
bos = "<|begin_of_text|>"
tokenizer._tokenizers.post_processor = processors.Sequence(
    [
        processors.ByteLevel(trim_offsets=False),
        processors.TemplateProcessing(
            single=f"{bos}:0 $A:0",
            pair=f"{bos}:0 $A:0 {bos}:1 $B:1",
            special_tokens=[
                (bos, tokenizer.encode(bos)),
            ],
        ),
    ]
)
```
"""

NUM_SHARDS = {
    "1B": 1,
    "3B": 1,
    "7B": 1,
    "8B": 1,
    "8Bf": 1,
    "7Bf": 1,
    "13B": 2,
    "13Bf": 2,
    "34B": 4,
    "30B": 4,
    "65B": 8,
    "70B": 8,
    "70Bf": 8,
    "405B": 8,
    "405B-MP16": 16,
}

CONTEXT_LENGTH_FOR_VERSION = {
    "Guard-3": 131072,
    "3.2": 131072,
    "3.1": 131072,
    "3": 8192,
    "2": 4096,
    "1": 2048,
}

BOS_ADDED_TOKEN = AddedToken(
    "<|begin_of_text|>",
    single_word=False,
    lstrip=False,
    rstrip=False,
    normalized=False,
    special=True,
)
EOS_ADDED_TOKEN = AddedToken(
    "<|end_of_text|>",
    single_word=False,
    lstrip=False,
    rstrip=False,
    normalized=False,
    special=True,
)
EOT_ADDED_TOKEN = AddedToken(
    "<|eot_id|>",
    single_word=False,
    lstrip=False,
    rstrip=False,
    normalized=False,
    special=True,
)

DEFAULT_LLAMA_SPECIAL_TOKENS = {
    "3": [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ]
    + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)],
    "3.1": [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|reserved_special_token_2|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",  # end of message
        "<|eot_id|>",  # end of turn
        "<|python_tag|>",
    ]
    + [f"<|reserved_special_token_{i}|>" for i in range(3, 256 - 8)],
    "3.2": [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|reserved_special_token_2|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",  # end of message
        "<|eot_id|>",  # end of turn
        "<|python_tag|>",
    ]
    + [f"<|reserved_special_token_{i}|>" for i in range(3, 256 - 8)],
    "Guard-3": [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|reserved_special_token_2|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",  # end of message
        "<|eot_id|>",  # end of turn
        "<|python_tag|>",
    ]
    + [f"<|reserved_special_token_{i}|>" for i in range(3, 256 - 8)],
}


def is_llama_3(version):
    return version in ["3", "3.1", "3.2", "Guard-3"]


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * (
        (int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of
    )


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


@torch.no_grad()
def write_model(
    model_path,
    input_base_path,
    model_size=None,
    safe_serialization=True,
    llama_version="1",
    vocab_size=None,
    num_shards=None,
    instruct=False,
    push_to_hub=False,
    private=False,
    commit_message: str = None,
):
    print("Converting the model.")
    import os

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size] if num_shards is None else num_shards
    params = params.get("model", params)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head)
    )
    if base > 10000.0 and not is_llama_3(llama_version):
        max_position_embeddings = 16384
    else:
        max_position_embeddings = CONTEXT_LENGTH_FOR_VERSION[llama_version]

    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_key_value_heads_per_shard = num_key_value_heads // num_shards
        key_value_dim = dims_per_head * num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_key_value_heads_per_shard = n_heads_per_shard
        key_value_dim = dim

    # permute for sliced rotary
    def permute(w, n_heads, dim1=dim, dim2=dim):
        return (
            w.detach()
            .cpu()
            .view(n_heads, dim1 // n_heads // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    with tempfile.TemporaryDirectory() as tmp_model_path:
        print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
        # Load weights
        if num_shards == 1:
            # Not sharded
            # (The sharded implementation would also work, but this is simpler.)
            p = os.path.join(input_base_path, "consolidated.00.pth")
            if not os.path.exists(p):
                p = os.path.join(input_base_path, "consolidated.pth")

            if not os.path.exists(p):
                p = os.path.join(input_base_path, "model.pth")

            import os

            import psutil

            # Check available memory
            memory = psutil.virtual_memory()
            print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
            print(f"File size: {os.path.getsize(p) / (1024**3):.1f} GB")

            # Load only if you have enough memory
            if memory.available < os.path.getsize(p) * 2:  # 2x safety margin
                print("Insufficient memory!")
            # import code; code.interact(local=dict(globals(), **locals()))
            loaded = torch.load(p, weights_only=True, map_location="cpu", mmap=True)
            if "model" in loaded:
                loaded = loaded["model"]
            # code.interact(local=dict(globals(), **locals()))
        else:
            # Sharded
            checkpoint_list = sorted(
                [file for file in os.listdir(input_base_path) if file.endswith(".pth")]
            )
            print("Loading in order:", checkpoint_list)
            loaded = [
                torch.load(
                    os.path.join(input_base_path, file),
                    map_location="cpu",
                    weights_only=True,
                )
                for file in checkpoint_list
            ]
        param_count = 0
        index_dict = {"weight_map": {}}
        for layer_i in range(n_layers):
            print(f"Processing layer {layer_i + 1}/{n_layers}")
            filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
            if num_shards == 1:
                # Unsharded
                # model.layers.5.self_attn.q_proj.weight
                state_dict = dict()
                state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                    loaded[f"layers.{layer_i}.attention.wq.weight"], n_heads=n_heads
                )
                state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                    loaded.pop(f"layers.{layer_i}.attention.wk.weight"),
                    n_heads=num_key_value_heads,
                    dim1=key_value_dim,
                )
                state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = (
                    loaded.pop(f"layers.{layer_i}.attention.wv.weight")
                )
                state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = (
                    loaded.pop(f"layers.{layer_i}.attention.wo.weight")
                )
                state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = loaded.pop(
                    f"layers.{layer_i}.feed_forward.w1.weight"
                )
                state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = loaded.pop(
                    f"layers.{layer_i}.feed_forward.w2.weight"
                )
                state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = loaded.pop(
                    f"layers.{layer_i}.feed_forward.w3.weight"
                )
                state_dict[f"model.layers.{layer_i}.input_layernorm.weight"] = (
                    loaded.pop(f"layers.{layer_i}.attention_norm.weight")
                )
                state_dict[
                    f"model.layers.{layer_i}.post_attention_layernorm.weight"
                ] = loaded.pop(f"layers.{layer_i}.ffn_norm.weight")
                # Call garbage collector explicitly to ensure memory is released
                gc.collect()
            else:
                # Sharded
                # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
                # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
                # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

                state_dict = {
                    f"model.layers.{layer_i}.input_layernorm.weight": loaded[0][
                        f"layers.{layer_i}.attention_norm.weight"
                    ].clone(),
                    f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                        0
                    ][f"layers.{layer_i}.ffn_norm.weight"].clone(),
                }
                state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                    torch.cat(
                        [
                            loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(
                                n_heads_per_shard, dims_per_head, dim
                            )
                            for i in range(len(loaded))
                        ],
                        dim=0,
                    ).reshape(dim, dim),
                    n_heads=n_heads,
                )
                state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                    torch.cat(
                        [
                            loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(
                                num_key_value_heads_per_shard, dims_per_head, dim
                            )
                            for i in range(len(loaded))
                        ],
                        dim=0,
                    ).reshape(key_value_dim, dim),
                    num_key_value_heads,
                    key_value_dim,
                    dim,
                )
                state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = (
                    torch.cat(
                        [
                            loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(
                                num_key_value_heads_per_shard, dims_per_head, dim
                            )
                            for i in range(len(loaded))
                        ],
                        dim=0,
                    ).reshape(key_value_dim, dim)
                )

                state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = (
                    torch.cat(
                        [
                            loaded[i][f"layers.{layer_i}.attention.wo.weight"]
                            for i in range(len(loaded))
                        ],
                        dim=1,
                    )
                )
                state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"]
                        for i in range(len(loaded))
                    ],
                    dim=0,
                )
                state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"]
                        for i in range(len(loaded))
                    ],
                    dim=1,
                )
                state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"]
                        for i in range(len(loaded))
                    ],
                    dim=0,
                )

            state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = (
                inv_freq
            )
            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            torch.save(state_dict, os.path.join(tmp_model_path, filename))

        filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
        # import code; code.interact(local=dict(globals(), **locals()))
        if params.get("use_factorized_embeddings", False):
            if num_shards == 1:
                # Unsharded
                state_dict = {
                    "model.embed_tokens.0.weight": loaded["tok_embeddings.0.weight"],
                    "model.embed_tokens.1.weight": loaded["tok_embeddings.1.weight"],
                    "model.norm.weight": loaded["norm.weight"],
                    "lm_head.0.weight": loaded["output.0.weight"],
                    "lm_head.1.weight": loaded["output.1.weight"],
                }
            else:
                concat_dim = 0 if is_llama_3(llama_version) else 1
                state_dict = {
                    "model.norm.weight": loaded[0]["norm.weight"],
                    "model.embed_tokens.0.weight": torch.cat(
                        [loaded[i]["tok_embeddings.0.weight"] for i in range(len(loaded))],
                        dim=concat_dim,
                    ),
                    "model.embed_tokens.1.weight": torch.cat(
                        [loaded[i]["tok_embeddings.1.weight"] for i in range(len(loaded))],
                        dim=concat_dim,
                    ),
                    "lm_head.0.weight": torch.cat(
                        [loaded[i]["output.0.weight"] for i in range(len(loaded))], dim=0
                    ),
                    "lm_head.1.weight": torch.cat(
                        [loaded[i]["output.1.weight"] for i in range(len(loaded))], dim=0
                    ),
                }
        else:
            if num_shards == 1:
                # Unsharded
                state_dict = {
                    "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
                    "model.norm.weight": loaded["norm.weight"],
                    "lm_head.weight": loaded["output.weight"],
                }
            else:
                concat_dim = 0 if is_llama_3(llama_version) else 1
                state_dict = {
                    "model.norm.weight": loaded[0]["norm.weight"],
                    "model.embed_tokens.weight": torch.cat(
                        [loaded[i]["tok_embeddings.weight"] for i in range(len(loaded))],
                        dim=concat_dim,
                    ),
                    "lm_head.weight": torch.cat(
                        [loaded[i]["output.weight"] for i in range(len(loaded))], dim=0
                    ),
                }
        if vocab_size is None:
            if params.get("use_factorized_embeddings", False):
                vocab_size = state_dict["model.embed_tokens.0.weight"].shape[0]
            else:
                vocab_size = state_dict["model.embed_tokens.weight"].shape[0]
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

        # Write configs
        index_dict["metadata"] = {"total_size": param_count * 2}
        write_json(
            index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json")
        )
        ffn_dim_multiplier = (
            params["ffn_dim_multiplier"]
            if params.get("ffn_dim_multiplier", None)
            else 1
        )
        multiple_of = params["multiple_of"] if "multiple_of" in params else 256

        # TODO: turn this off and replace
        if is_llama_3(llama_version):
            bos_token_id = 128000

            if instruct:
                eos_token_id = [128001, 128008, 128009]
            else:
                eos_token_id = 128001
        else:
            bos_token_id = 1
            eos_token_id = 2

        if llama_version in ["3.1", "3.2", "Guard-3"]:
            rope_scaling = {
                "factor": 32.0 if llama_version == "3.2" else 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            }
        else:
            rope_scaling = None

        common_args = dict(
            hidden_size=dim,
            intermediate_size=compute_intermediate_size(
                dim, ffn_dim_multiplier, multiple_of
            ),
            num_attention_heads=params["n_heads"],
            num_hidden_layers=params["n_layers"],
            rms_norm_eps=params["norm_eps"],
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            rope_theta=base,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=True if llama_version in ["3.2"] else False,
            # model_type="llama",
        )
        if params.get("use_factorized_embeddings", False):
            config = LlamaAlbertConfig(**common_args, embedding_dim=params["factorized_embedding_dim"])
        else:
            config = LlamaConfig(**common_args)
        config.save_pretrained(tmp_model_path)

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        generation_config.save_pretrained(tmp_model_path)

        # Make space so we can load the model properly now.
        del state_dict
        del loaded
        gc.collect()
        from safetensors.torch import save_file
        print("Loading the checkpoint in a Llama model.")
        if params.get("use_factorized_embeddings", False):
            model = LlamaAlbertForCausalLM.from_pretrained(
                tmp_model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda"
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                tmp_model_path, dtype=torch.bfloat16,  low_cpu_mem_usage=True, device_map="cuda",
            )

        # Avoid saving this as part of the config.
        del model.config._name_or_path
        model.config.dtype = torch.float16

        print("Saving in the Transformers format.")
        if push_to_hub:
            temp_save_dir = Path(tmp_model_path, model_path) / "temp_save_files"
            temp_save_dir.mkdir(
                parents=True, exist_ok=True
            )  # Create the temporary directory
            print(temp_save_dir)
            model.save_pretrained(
                temp_save_dir,
                safe_serialization=safe_serialization,  # True
                max_shard_size="4GB",
            )
            config_kwargs = {}
            if params.get("use_factorized_embeddings", False):
                import shutil
                shutil.copyfile(Path(__file__).parent / "modeling_llama_albert.py", temp_save_dir / "modeling_llama_albert.py")
                config_kwargs["auto_map"] = {
                    "AutoConfig": "modeling_llama_albert.LlamaAlbertConfig",
                    "AutoModelForCausalLM": "modeling_llama_albert.LlamaAlbertForCausalLM"
                }
            # import code; code.interact(local=dict(globals(), **locals()))
            # config.save_pretrained(temp_save_dir, **config_kwargs)
            generation_config.save_pretrained(temp_save_dir)
            torch.cuda.empty_cache()
            gc.collect()
            repo_id = model_path
            import huggingface_hub as hf_hub

            hf_hub.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,  # True if 'private' is True
                exist_ok=True,
            )
            hf_hub.upload_folder(
                folder_path=str(temp_save_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                # ignore_patterns=[".git", ".gitignore"] # If you used git init locally
            )
            print("Pushing to the hub.")
            # import code; code.interact(local=dict(globals(), **locals()))
            # model.push_to_hub(
            #     model_path,
            #     safe_serialization=safe_serialization,
            #     private=private,
            #     use_temp_dir=True,
            #     max_shard_size="4GB",
            #     commit_message=commit_message,
            # )
            config.save_pretrained(
                tmp_model_path, push_to_hub=push_to_hub, repo_id=model_path
            )
            print(f"The model is available at https://huggingface.co/{model_path}")
            del model
        else:
            print("Saving to disk.")
            model.save_pretrained(model_path, safe_serialization=safe_serialization)


class Llama3Converter(TikTokenConverter):
    def __init__(
        self,
        vocab_file,
        special_tokens=None,
        instruct=False,
        llama_version="3.2",
        **kwargs,
    ):
        super().__init__(vocab_file, additional_special_tokens=special_tokens, **kwargs)
        tokenizer = self.converted()

        # References for chat templates in instruct models
        templates_for_version = {
            "2": (
                "meta-llama/Llama-2-7b-chat-hf",
                "f5db02db724555f92da89c216ac04704f23d4590",
            ),
            "3": (
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "5f0b02c75b57c5855da9ae460ce51323ea669d8a",
            ),
            "3.1": (
                "meta-llama/Llama-3.1-8B-Instruct",
                "0e9e39f249a16976918f6564b8830bc894c89659",
            ),
            "3.2": (
                "meta-llama/Llama-3.2-1B-Instruct",
                "e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14",
            ),
            "Guard-3": (
                "meta-llama/Llama-Guard-3-1B",
                "acf7aafa60f0410f8f42b1fa35e077d705892029",
            ),
        }

        # Add chat_template only if instruct is True.
        # Prevents a null chat_template, which triggers
        # a parsing warning in the Hub.
        additional_kwargs = {}
        if instruct or llama_version in ["Guard-3"]:
            model_id, revision = templates_for_version.get(llama_version, (None, None))
            if model_id is not None:
                from transformers import AutoTokenizer

                t = AutoTokenizer.from_pretrained(model_id, revision=revision)
                additional_kwargs["chat_template"] = t.chat_template

        self.converted_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>" if not instruct else "<|eot_id|>",
            model_input_names=["input_ids", "attention_mask"],
            model_max_length=CONTEXT_LENGTH_FOR_VERSION[llama_version],
            clean_up_tokenization_spaces=True,
            **additional_kwargs,
        )
        self.update_post_processor(self.converted_tokenizer)
        # finer special_tokens_map.json
        self.converted_tokenizer._bos_token = BOS_ADDED_TOKEN
        self.converted_tokenizer._eos_token = (
            EOT_ADDED_TOKEN if instruct else EOS_ADDED_TOKEN
        )

    # We can't do this while building the tokenizer because we have no easy access to the bos token id
    def update_post_processor(self, tokenizer):
        tokenizer._tokenizer.post_processor = processors.Sequence(
            [
                processors.ByteLevel(trim_offsets=False),
                processors.TemplateProcessing(
                    single="<|begin_of_text|> $A",
                    pair="<|begin_of_text|>:0 $A:0 <|begin_of_text|>:1 $B:1",
                    special_tokens=[
                        (
                            "<|begin_of_text|>",
                            tokenizer.convert_tokens_to_ids("<|begin_of_text|>"),
                        ),
                    ],
                ),
            ]
        )


def write_tokenizer(
    tokenizer_path,
    input_tokenizer_path,
    llama_version="2",
    special_tokens=None,
    instruct=False,
    push_to_hub=False,
    private=False,
):
    print("Converting the tokenizer.")
    tokenizer_class = (
        LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    )
    if is_llama_3(llama_version):
        tokenizer = Llama3Converter(
            input_tokenizer_path,
            special_tokens,
            instruct,
            llama_version,
        ).converted_tokenizer
    else:
        try:
            tokenizer = tokenizer_class(input_tokenizer_path)
        except Exception:
            raise ValueError(
                "Failed to instantiate tokenizer. Please, make sure you have sentencepiece and protobuf installed."
            )

    if push_to_hub:
        print(
            f"Pushing a {tokenizer_class.__name__} to the Hub repo - {tokenizer_path}."
        )
        tokenizer.push_to_hub(tokenizer_path, private=private, use_temp_dir=True)
    else:
        print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
        tokenizer.save_pretrained(tokenizer_path)
    return tokenizer


def created_consolidated(dcp_directory: Path):
    consolidated_path = dcp_directory.joinpath("consolidated.pth")
    if consolidated_path.exists():
        return
    elif dcp_directory.joinpath("model.pth").exists():
        print("Found model.pth, skipping creating consolidated.pth")
        return
    elif dcp_directory.joinpath("consolidated", "consolidated.pth").exists():
        print("Found consolidated/consolidated.pth, copying to consolidated.pth")
        import shutil
        shutil.copy(
            dcp_directory.joinpath("consolidated", "consolidated.pth"), consolidated_path
        )
        return
    elif dcp_directory.joinpath("consolidated", "model.pth").exists():
        print("Found consolidated/consolidated.pth, copying to consolidated.pth")
        import shutil
        shutil.copy(
            dcp_directory.joinpath("consolidated", "consolidated.pth"), consolidated_path
        )
        return

    print("Creating consolidated.pth")
    # This helps to consolidate also from separate checkpoint directories like 0010000 
    dcp_directory = list(dcp_directory.rglob("__0_0.distcp"))[0].parent
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    dcp_to_torch_save(dcp_directory, consolidated_path)
    # also save model.pth without the optimizer states
    state_dct = torch.load(consolidated_path, map_location="cpu", weights_only=True)
    state_dct.pop("optim")
    torch.save(state_dct, consolidated_path.parent.joinpath("model.pth"))
    if not consolidated_path.parent.joinpath("params.json").exists():
        import shutil
        shutil.copy(dcp_directory.joinpath("params.json"), consolidated_path.parent.joinpath("params.json"))


@dataclass
class ConvertLinguaToHuggingFaceArgs(Args):
    # Location of Llama weights, which contains tokenizer.model and model folders
    input_dir: str = ""
    # 'f' Deprecated in favor of `num_shards`: models correspond to the finetuned versions, and are specific to
    # the Llama2 official release. For more details on Llama2, check out the original repo:
    # https://huggingface.co/meta-llama
    model_size: Optional[str] = None
    # Location to write HF model and tokenizer
    output_dir: str = ""
    # # Whether or not to push the model to the hub at `output_dir` instead of saving it locally.
    # push_to_hub: bool = False
    # Whether or not to save using `safetensors`.
    safe_serialization: bool = True
    # Version of the Llama model to convert. Currently supports Llama1 and Llama2. Controls the context size.
    # Allowed: 1, 2, 3, 3.1, 3.2, Guard-3
    llama_version: str = "1"
    # Version of the tokenizer to convert. Currently supports Llama1 and Llama2.
    # Allowed: 1, 2, 3, 3.1, 3.2, Guard-3
    tokenizer_version: Optional[str] = None
    # The number of individual shards used for the model. Does not have to be the same as the number of
    # consolidated_xx.pth
    num_shards: Optional[int] = None
    # The list of special tokens that should be added to the model.
    special_tokens: Optional[List[str]] = None
    # Whether the model is an instruct model or not. Will affect special tokens and chat template.
    instruct: bool = False
    # Path to the tokenizer.model file.
    tokenizer_path: Optional[str] = None
    # Only convert the model, not the tokenizer.
    only_model: bool = False
    # # Whether to push the model to the hub as public.
    # public: bool = False
    # The name of the collection to which the model belongs.
    collection: Optional[str] = None
    # Whether to ignore the vocabulary size, be careful with this option.
    ignore_vocab_size: bool = False
    # The list of datasets to add to the model card.
    datasets: Optional[List[str]] = None
    # Whether to push the tokenizer to the hub as well. If False, the tokenizer will be saved locally to the
    # output directory, but not pushed to the hub.
    push_tokenizer_to_hub: bool = False
    hf: HFArgs = field(default_factory=HFArgs)


def main():
    args: ConvertLinguaToHuggingFaceArgs = load_args(ConvertLinguaToHuggingFaceArgs)  # type: ignore[assignment]
    allowed_versions = {"1", "2", "3", "3.1", "3.2", "Guard-3"}
    if args.llama_version not in allowed_versions:
        raise ValueError(
            f"Invalid llama_version '{args.llama_version}'. Expected one of {sorted(allowed_versions)}"
        )
    if args.tokenizer_version is not None and args.tokenizer_version not in allowed_versions:
        raise ValueError(
            f"Invalid tokenizer_version '{args.tokenizer_version}'. Expected one of {sorted(allowed_versions)}"
        )
    if args.tokenizer_version is None:
        args.tokenizer_version = args.llama_version
    if args.model_size is None and args.num_shards is None:
        raise ValueError(
            "You have to set at least `num_shards` if you are not giving the `model_size`"
        )
    if args.special_tokens is None:
        # no special tokens by default
        args.special_tokens = DEFAULT_LLAMA_SPECIAL_TOKENS.get(
            str(args.llama_version), []
        )
    created_consolidated(Path(args.input_dir))

    if args.hf.publish_to_hf:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            args.output_dir,
            repo_type="model",
            private=not args.hf.public,
            exist_ok=True,
        )
    vocab_size = None

    if not args.only_model:
        if args.tokenizer_path is None:
            spm_path = os.path.join(args.input_dir, "tokenizer.model")
        else:
            spm_path = args.tokenizer_path
            spm_path = list(Path(spm_path).rglob("*vocab.json"))[0].as_posix()

        vocab_size = len(
            write_tokenizer(
                args.output_dir,
                spm_path,
                llama_version=args.tokenizer_version,
                special_tokens=args.special_tokens,
                instruct=args.instruct,
                push_to_hub=args.push_tokenizer_to_hub,
                private=not args.hf.public,
            )
        )
    elif args.tokenizer_path is not None:
        try:
            spm_path = list(Path(args.tokenizer_path).rglob("*vocab.json"))[0].as_posix()
        except:
            spm_path = list(Path(args.tokenizer_path).rglob("*tokenizer.json"))[0].as_posix()

        with open(spm_path, "r") as f:
            vocab = json.load(f)
            if "model" in vocab:
                try:
                    vocab = vocab["model"]["vocab"]
                except:
                    pass
            vocab_size = len(vocab)
        
        if args.push_tokenizer_to_hub:
            print("Copying tokenizerfiles to the output directory.")
            files_to_push = list(Path(args.tokenizer_path).rglob("*.json")) + list(
                Path(args.tokenizer_path).rglob("*.yaml")
            )
            for file in files_to_push:
                name = file.stem
                if "vocab" in file.name:
                    name = "tokenizer"
                if "info" in file.name:
                    name = "tokenizer_config"
                try:
                    api.upload_file(
                        path_or_fileobj=str(file),
                        path_in_repo=f"{name}{file.suffix}",  # or specify custom path
                        repo_id=args.output_dir,
                        repo_type="model",
                        commit_message=f"Upload tokenizer file {file.name} - {args.hf.commit_message}",
                    )
                except Exception as e:
                    print(f"Error uploading {file.name}: {e}")

    if args.model_size != "tokenizer_only":
        input_base_path = args.input_dir
        print(f"Writing model files from {input_base_path} to {args.output_dir}")
        write_model(
            model_path=args.output_dir,
            input_base_path=input_base_path,
            model_size=args.model_size,
            safe_serialization=args.safe_serialization,
            llama_version=args.llama_version,
            vocab_size=vocab_size if not args.ignore_vocab_size else None,
            num_shards=args.num_shards,
            instruct=args.instruct,
            push_to_hub=args.hf.publish_to_hf,
            private=not args.hf.public,
            commit_message=args.hf.commit_message,
        )
    if args.hf.publish_to_hf:
        if args.datasets is not None:
            from huggingface_hub import metadata_update

            repo_id = args.output_dir
            metadata_update(repo_id, {"datasets": args.datasets}, repo_type="model", overwrite=True)
            print(f"Updated model card with datasets: {args.datasets}")

    if args.collection:
        from huggingface_hub import add_collection_item, create_collection

        collection = create_collection(
            title=args.collection,
            namespace=args.output_dir.split("/")[0],
            exists_ok=True,
        )
        add_collection_item(
            collection.slug, item_id=args.output_dir, item_type="model", exists_ok=True
        )


if __name__ == "__main__":
    main()
