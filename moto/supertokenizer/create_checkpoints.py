"""Create a checkpoint for a specific tokenizer based on the super set vocabulary, needs a llama base model with supervocab size to be created already with lingua."""

import copy
import gc
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import huggingface_hub as hf_hub
import torch
import torch.distributed.checkpoint as dcp

from moto.lingua.convert_lingua_to_huggingface import write_model
from moto.lingua.lm_transformer import LMTransformer, LMTransformerArgs
from moto.utils.args import (
    Args,
    HFArgs,
    dataclass_from_dict,
    dataclass_to_dict,
    load_args,
)


@dataclass
class SuperVocabCkptArgs(Args):
    config: Any = None
    tokenizers: List[str] = field(default_factory=list)
    tokenizer_groups: Optional[str] = None  # Path to a yaml/json file mapping a supertok name to a list of component tokenizer names, e.g. {"supertok1": ["a", "b", "c", "d"], "supertok2": ["e", "f"]}. Each group's checkpoint is built from the union of its components' super_mapping.json vocabularies.
    vocab_dir: str = "/tmp/vocabs"
    model_dir: str = "/tmp/models"
    superset: str = "/tmp/models/superset.pt"
    embedding_path: Optional[str] = None
    create_superset: bool = False
    embedding_layer: str = "tok_embeddings.weight"
    output_layer: str = "output.weight"
    lingua_base_config: str = "configs/lingua/llama_1B.yaml"
    supervocab_size: int = 0
    create_distributed: bool = False
    hf_upload_repo_prefix: Optional[str] = None
    seed: int = 42
    hf: HFArgs = field(default_factory=HFArgs)
    superset_code_name: str = "super_vocab" # The code name to use for the super vocab in the mapping files. This is just a string and does not have to match any actual tokenizer name, but it should be unique from the other superset tokenizers in the mapping files. It is used to identify the super vocab in the mapping files and should be consistent across all mapping files. Default is "super_vocab".
    save_hf: bool = False  # Also save each checkpoint in HuggingFace format (loadable with AutoModelForCausalLM)
    # prob not needed, TODO: revisit and remove after deadline
    # Init superset from a base model (HF repo id or local directory).
    # Backbone layers are copied and base embeddings are scattered into the superset
    # table at positions given by the base tokenizer's super_mapping; non-overlapping
    # positions keep the random init.
    base_model: Optional[str] = None       # HF repo id or local dir, e.g. "meta-llama/Llama-3.2-1B"
    base_tokenizer: Optional[str] = None   # tokenizer used by base_model; defaults to base_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_hf_as_lingua(model_path: str, embedding_layer: str, output_layer: str) -> dict:
    """Load a base model and return its state dict in Lingua key format.

    Priority order for loading:
      1. ``consolidated.pth`` in the directory (Lingua native format, no remapping).
      2. Safetensors shards (HF format) — keys are remapped to Lingua equivalents.

    Key mapping for HF → Lingua (single-shard, non-tensor-parallel):
      model.embed_tokens.weight                  → {embedding_layer}
      lm_head.weight                             → {output_layer}
      model.norm.weight                          → norm.weight
      model.layers.{i}.input_layernorm.weight    → layers.{i}.attention_norm.weight
      model.layers.{i}.post_attention_layernorm  → layers.{i}.ffn_norm.weight
      model.layers.{i}.self_attn.q_proj.weight   → layers.{i}.attention.wq.weight  (+ inv-permute)
      model.layers.{i}.self_attn.k_proj.weight   → layers.{i}.attention.wk.weight  (+ inv-permute)
      model.layers.{i}.self_attn.v_proj.weight   → layers.{i}.attention.wv.weight
      model.layers.{i}.self_attn.o_proj.weight   → layers.{i}.attention.wo.weight
      model.layers.{i}.mlp.gate_proj.weight      → layers.{i}.feed_forward.w1.weight
      model.layers.{i}.mlp.down_proj.weight      → layers.{i}.feed_forward.w2.weight
      model.layers.{i}.mlp.up_proj.weight        → layers.{i}.feed_forward.w3.weight
    """
    mp = Path(model_path)

    # 1. Check for Lingua-native consolidated checkpoint
    consolidated = mp / "consolidated.pth"
    if not consolidated.exists():
        # check huggingface
        consolidated = Path(hf_hub.hf_hub_download(repo_id=model_path, filename="consolidated.pth", repo_type="model"))

    if consolidated.exists():
        logging.info("Found consolidated.pth at '%s' — loading Lingua format directly.", consolidated)
        state = torch.load(consolidated, map_location="cpu")
        logging.info("Loaded %d keys from consolidated.pth.", len(state))
        return state

    # 2. Fall back to HF safetensors
    logging.info("No consolidated.pth found — loading HF safetensors from '%s'.", mp)
    from safetensors import safe_open
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(mp)
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = config.hidden_size // n_heads

    def inv_permute(w, n_h, hd):
        # The HF permute in convert_lingua_to_huggingface.py is its own inverse.
        return w.view(n_h, 2, hd // 2, -1).transpose(1, 2).reshape(n_h * hd, -1)

    raw: dict = {}
    for f in sorted(mp.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                raw[k] = sf.get_tensor(k)

    lingua: dict = {}
    for hf_key, tensor in raw.items():
        if hf_key == "model.embed_tokens.weight":
            lingua[embedding_layer] = tensor
        elif hf_key == "lm_head.weight":
            lingua[output_layer] = tensor
        elif hf_key == "model.norm.weight":
            lingua["norm.weight"] = tensor
        elif hf_key.startswith("model.layers."):
            rest = hf_key[len("model.layers."):]
            layer_i, suffix = rest.split(".", 1)
            if suffix == "self_attn.q_proj.weight":
                lingua[f"layers.{layer_i}.attention.wq.weight"] = inv_permute(tensor, n_heads, head_dim)
            elif suffix == "self_attn.k_proj.weight":
                lingua[f"layers.{layer_i}.attention.wk.weight"] = inv_permute(tensor, n_kv_heads, head_dim)
            elif suffix == "self_attn.v_proj.weight":
                lingua[f"layers.{layer_i}.attention.wv.weight"] = tensor
            elif suffix == "self_attn.o_proj.weight":
                lingua[f"layers.{layer_i}.attention.wo.weight"] = tensor
            elif suffix == "mlp.gate_proj.weight":
                lingua[f"layers.{layer_i}.feed_forward.w1.weight"] = tensor
            elif suffix == "mlp.down_proj.weight":
                lingua[f"layers.{layer_i}.feed_forward.w2.weight"] = tensor
            elif suffix == "mlp.up_proj.weight":
                lingua[f"layers.{layer_i}.feed_forward.w3.weight"] = tensor
            elif suffix == "input_layernorm.weight":
                lingua[f"layers.{layer_i}.attention_norm.weight"] = tensor
            elif suffix == "post_attention_layernorm.weight":
                lingua[f"layers.{layer_i}.ffn_norm.weight"] = tensor
            # skip rotary_emb.inv_freq and other HF-only keys

    logging.info("Loaded HF model from '%s': %d Lingua keys mapped.", model_path, len(lingua))
    return lingua


@torch.no_grad()
def scatter_base_into_superset(
    super_set: dict,
    base_state: dict,
    base_mapping: dict,
    embedding_layer: str,
    output_layer: str,
) -> None:
    """Copy base-model embeddings/output rows into the superset tables in-place.

    For each base token index ``i``, the row ``base_state[embedding_layer][i]``
    is placed at ``super_set[embedding_layer][super_id]`` where ``super_id =
    base_mapping[str(i)]``.  All other backbone weights are also overwritten from
    ``base_state`` when their shapes agree.

    Args:
        super_set:       Mutable state-dict of the freshly-created superset model.
        base_state:      State-dict of the base model (Lingua .pth format).
        base_mapping:    Per-tokenizer super_mapping  {str(local_id): superset_id}.
        embedding_layer: Key for the embedding table (e.g. "tok_embeddings.weight").
        output_layer:    Key for the output projection (e.g. "output.weight").
    """
    # 1. Copy backbone (non-embedding) layers from base to superset.
    copied, skipped = 0, 0
    for key, tensor in base_state.items():
        if key in (embedding_layer, output_layer):
            continue
        if key in super_set and super_set[key].shape == tensor.shape:
            super_set[key] = tensor.clone()
            copied += 1
        else:
            skipped += 1
    logging.info("Backbone: copied %d layers, skipped %d (shape mismatch / missing).", copied, skipped)
    # 2. Scatter base embeddings into the (larger) superset embedding tables.
    base_embed = base_state[embedding_layer]
    base_output = base_state[output_layer]
    super_embed = super_set[embedding_layer]
    super_output = super_set[output_layer]

    local_ids  = torch.tensor([int(k) for k in base_mapping], dtype=torch.long)
    super_ids  = torch.tensor([base_mapping[k] for k in base_mapping], dtype=torch.long)

    super_embed[super_ids]  = base_embed[local_ids].to(super_embed.dtype)
    super_output[super_ids] = base_output[local_ids].to(super_output.dtype)

    super_set[embedding_layer] = super_embed
    super_set[output_layer]    = super_output
    logging.info(
        "Scattered %d base-tokenizer rows into the superset embedding table (size %d).",
        len(local_ids), super_embed.shape[0],
    )


def load_tokenizer_groups(path: str) -> dict:
    """Load a {supertok_name: [component_tokenizer_name, ...]} mapping from yaml or json."""
    p = Path(path)
    with open(p) as f:
        if p.suffix in (".yaml", ".yml"):
            import yaml
            groups = yaml.safe_load(f)
        else:
            groups = json.load(f)
    return groups


def load_super_mapping(tokenizer: str, vocab_dir: str) -> dict:
    path = os.path.join(vocab_dir, f"{tokenizer.replace('/', '--')}_super_mapping.json")
    logging.info("Loading superset vocab mapping for %s from '%s'", tokenizer, path)
    with open(path) as f:
        return json.load(f)


def merge_super_mappings(tokenizers: List[str], vocab_dir: str) -> dict:
    """Union several tokenizers' super_mapping.json files into one contiguous mapping.

    Each component mapping is local_id (str, contiguous from 0) -> superset_id. The
    merged result re-indexes local ids from 0, keeping the first occurrence of each
    superset id in the order the component tokenizers are listed, so a token shared
    by several tokenizers only occupies one row in the resulting checkpoint.
    """
    seen = set()
    merged_superset_ids = []
    for tokenizer in tokenizers:
        mapping = load_super_mapping(tokenizer, vocab_dir)
        for local_id in range(len(mapping)):
            super_id = mapping[str(local_id)]
            if super_id not in seen:
                seen.add(super_id)
                merged_superset_ids.append(super_id)
    logging.info(
        "Merged %d tokenizers into a supertok vocab of %d entries.",
        len(tokenizers), len(merged_superset_ids),
    )
    return {str(i): super_id for i, super_id in enumerate(merged_superset_ids)}


def select_specific(t: torch.Tensor, mapping: dict[int, int]) -> torch.Tensor:
    # Assumes that we contiguous, otherwise we need to do a gather and scatter.
    idx = [mapping[str(i)] for i in range(len(mapping))]
    # Use fancy indexing.
    return t[idx]


def create_superset(superset_vocab_size: int, lingua_base_config: str) -> Tuple[LMTransformerArgs, dict]:
    logging.info("Creating superset model checkpoint.")
    lingua_base_config = Path(lingua_base_config).resolve().absolute()
    if not lingua_base_config.exists():
        logging.error("Lingua base config file '%s' does not exist.", lingua_base_config)
        exit(1)

    import yaml

    with open(lingua_base_config) as f:
        config_dct = yaml.safe_load(f)
    lm_args = LMTransformerArgs(**config_dct["model"])
    lm_args.vocab_size = superset_vocab_size
    lm_args.seed = args.seed

    base_model = LMTransformer(lm_args)
    return lm_args, base_model.state_dict()

def convert_to_distributed(state_dct, base_path):
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    dist_ckpt = base_path/'model_dcp'
    import torch
    from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
    ckpt={'model': state_dct}
    tmp_path = Path("tmp.pth")
    torch.save(ckpt, tmp_path)
    del ckpt
    import gc; gc.collect()
    torch_save_to_dcp(tmp_path, dist_ckpt)
    tmp_path.unlink()
    import pathlib
    import pickle
    import sys

    with open(f"{dist_ckpt}/.metadata", "rb") as f:
        meta = pickle.load(f)

    with open(f"{dist_ckpt}/.metadata", "wb") as f:
        pickle.dump(meta, f, protocol=4)  # protocol 4 = Python 3.8+ compatible

    print("Metadata re-serialized with protocol 4 (compatible with python 3.10)")
    return dist_ckpt

def main(args):
    args.model_dir = Path(args.model_dir
                          )
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if args.create_superset:
        torch.manual_seed(args.seed)
        lm_args,super_set = create_superset(superset_vocab_size=args.supervocab_size, lingua_base_config=args.lingua_base_config)
        if args.embedding_path is not None:
            logging.info("Loading embedding parameters from '%s'", args.embedding_path)
            embedding_params = torch.load(args.embedding_path)
            logging.info("Replacing embedding and output layers with provided embedding path.")
            super_set[args.embedding_layer] = embedding_params[args.embedding_layer]
            super_set[args.output_layer] = embedding_params[args.output_layer]
        if args.base_model is not None:
            effective_base_tokenizer = args.base_tokenizer or args.base_model
            logging.info(
                "Initialising superset from base model '%s' (tokenizer: '%s')",
                args.base_model, effective_base_tokenizer,
            )
            base_state = load_hf_as_lingua(
                args.base_model, args.embedding_layer, args.output_layer
            )
            if "model" in base_state:
                base_state = base_state["model"] 
            base_mapping_path = os.path.join(
                args.vocab_dir,
                f"{effective_base_tokenizer.replace('/', '--')}_super_mapping.json",
            )
            logging.info("Loading base tokenizer super_mapping from '%s'", base_mapping_path)
            with open(base_mapping_path) as f:
                base_mapping = json.load(f)
            scatter_base_into_superset(
                super_set=super_set,
                base_state=base_state,
                base_mapping=base_mapping,
                embedding_layer=args.embedding_layer,
                output_layer=args.output_layer,
            )
            del base_state
        superset_path = Path(args.model_dir, "model.pth")
        with open(superset_path.parent/"params.json", "w") as f:
            json.dump(dataclass_to_dict(lm_args), f, indent=2)
        logging.info("Saving superset model at '%s'", superset_path)
        torch.save(super_set, superset_path)
        if args.create_distributed:
            dcp.save(super_set, checkpoint_id=args.model_dir/"supertokenizer"/'model_dcp')
        if args.hf_upload_repo_prefix is not None:
            if args.hf.hf_repo_id is not None:
                repo_name = args.hf.hf_repo_id
            else:
                repo_name = f"{args.hf_upload_repo_prefix}super_vocab"
            hf_hub.create_repo(
                repo_id=repo_name,
                exist_ok=True,
            )
            logging.info("Uploading superset checkpoint to HF repo '%s'", repo_name)
            model_size = math.floor(sum([t.numel() for k, t in super_set.items() if "tok_embed" not in k and "output" not in k])/1e9)
            write_model(repo_name, input_base_path=superset_path.parent,model_size=f"{model_size}B", push_to_hub=args.hf.publish_to_hf)
            if args.create_distributed:
                hf_hub.upload_folder(
                folder_path=args.model_dir/"supertokenizer"/'model_dcp',
                repo_id=repo_name,
                commit_message=f"Upload superset distributed checkpoint",
            )
    else:
        logging.info("Loading superset model from '%s'", args.superset)
        super_set = torch.load(args.superset)
    # exit(0)
    logging.info("Extracting embedding table at '%s'", args.embedding_layer)
    super_set_embedding = super_set[args.embedding_layer]
    logging.info("Extracting output layer at '%s'", args.output_layer)
    super_set_output = super_set[args.output_layer]

    jobs: List[Tuple[str, List[str]]] = [(tokenizer, [tokenizer]) for tokenizer in args.tokenizers]
    if args.tokenizer_groups is not None:
        logging.info("Loading tokenizer groups from '%s'", args.tokenizer_groups)
        groups = load_tokenizer_groups(args.tokenizer_groups)
        jobs.extend((name, components) for name, components in groups.items())

    for tokenizer, components in jobs:
        if len(components) == 1:
            mapping = load_super_mapping(components[0], args.vocab_dir)
        else:
            logging.info("Building supertok '%s' from tokenizers: %s", tokenizer, components)
            mapping = merge_super_mappings(components, args.vocab_dir)
            merged_mapping_path = os.path.join(
                args.vocab_dir, f"{tokenizer.replace('/', '--')}_super_mapping.json"
            )
            with open(merged_mapping_path, "w") as f:
                json.dump(mapping, f)
            logging.info("Saved merged super_mapping for '%s' at '%s'", tokenizer, merged_mapping_path)

        logging.info("Extracting values from embedding table.")
        new_embedding = select_specific(super_set_embedding, mapping)
        logging.info("Extracting values from output layer.")
        new_output = select_specific(super_set_output, mapping)

        logging.info("Adding embedding/output to the checkpoint.")
        # Don't use deep copy so we have only have one copy of tensors.
        new_model = copy.copy(super_set)
        new_model[args.embedding_layer] = new_embedding
        new_model[args.output_layer] = new_output

        os.makedirs(args.model_dir, exist_ok=True)
        checkpoint = os.path.join(args.model_dir, f"{tokenizer.replace('/', '--')}.pt")
        logging.info("Saving model for %s at '%s'", tokenizer, checkpoint)
        torch.save(new_model, checkpoint)
        repo_name = f"{args.hf_upload_repo_prefix}{tokenizer.replace('/', '-')}"
        if args.hf_upload_repo_prefix is not None:
            logging.info("Uploading checkpoint to HF repo '%s'", repo_name)
            hf_hub.create_repo(
                repo_id=repo_name,
                exist_ok=True,
            )
            hf_hub.upload_file(
                path_or_fileobj=checkpoint,
                path_in_repo=f"{args.superset_code_name}/model.pth",
                repo_id=repo_name,
                commit_message=f"Upload checkpoint for {tokenizer}",
            )
        if args.create_distributed:
            # Also create a distributed version
            # dcp.save(new_model, checkpoint_id=args.model_dir/f"{tokenizer.replace('/', '--')}_dcp")
            dist_ckpt = convert_to_distributed(new_model, os.path.join(
                args.model_dir, f"{tokenizer.replace('/', '--')}"))
            logging.info(
                "Saved distributed model for %s at '%s'", tokenizer, dist_ckpt
            )
            if args.hf_upload_repo_prefix is not None:
                logging.info("Uploading distributed checkpoint to HF repo '%s'", repo_name)
                hf_hub.upload_folder(
                    repo_id=repo_name,
                    folder_path=dist_ckpt,
                    path_in_repo=f"{args.superset_code_name}/model_dcp",
                    commit_message=f"Upload distributed checkpoint for {tokenizer}",
                )

                # hf_hub.upload_file(
                #     path_or_fileobj=dist_ckpt,
                #     repo_id=repo_name,
                #     path_in_repo=f"model_dcp/{dist_ckpt.name}",
                #     commit_message=f"Upload distributed checkpoint for {tokenizer}",
                # )


if __name__ == "__main__":
    args = load_args(SuperVocabCkptArgs)
    main(args)
