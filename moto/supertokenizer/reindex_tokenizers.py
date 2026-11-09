import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import huggingface_hub as hf_hub
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from moto.utils.args import Args, dataclass_from_dict, load_args

logging.basicConfig(level=logging.INFO)


from omegaconf import DictConfig, OmegaConf


@dataclass
class TokenizerArgs:
    name: str = ""
    path: str = ""

@dataclass
class ReindexingArgs(Args):
    config: Any = None
    tokenizers: List[TokenizerArgs] = field(default_factory=list)
    input_dir: str = "vocabs"
    hf_upload_repo_prefix: str = "reindexed-"
    publish_to_hf: bool = True
    temp_dir: str = f"{os.environ.get('TMPDIR', './temp_tokenizers')}"

def load_super_mapping(file_path):
    """Loads the Internal ID -> Super ID mapping."""
    with open(file_path, 'r') as f:
        str_map = json.load(f)
        # Convert internal ID keys back to integers for easy lookup
        return {int(k): v for k, v in str_map.items()}

def load_super_vocab(file_path):
    """Loads the Token String -> Super ID mapping."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_max_super_id(super_vocab):
    """Calculates the new vocabulary size."""
    max_id = max(super_vocab.values())
    return max_id + 1 # Vocab size is max ID + 1

def verify_parity(tok_orig, tok_new, internal_to_super_map):
    """
    Checks if the re-indexed tokenizer preserves original behavior.
    1. Length Check
    2. Subword (String) Check
    3. Mapping (ID) Check
    """
    test_sentences = [
        "Hello world!",
        "Tokenization is hard but super-mapping is harder.",
        "1234567890 !@#$%^&*()",
        "The quick brown fox jumps over the lazy dog.",
        "   Leading and trailing spaces   ",
        "JSON_mapping_test: {'key': 'value'}",
        "\n\tMixed whitespace and tabs\t\n",
        "Gül Sena Altıntaş İstanbullu ineǧin yanında."
    ]

    for text in test_sentences:
        # Get raw outputs
        orig_enc = tok_orig(text, add_special_tokens=False)
        new_enc = tok_new(text, add_special_tokens=False)

        orig_ids = orig_enc["input_ids"]
        new_ids = new_enc["input_ids"]
        
        orig_tokens = tok_orig.convert_ids_to_tokens(orig_ids)
        new_tokens = tok_new.convert_ids_to_tokens(new_ids)

        # Length Parity 
        if len(orig_ids) != len(new_ids):
            logging.error(f"Lengths don't match")
            logging.error(f"Text: {text}")
            logging.error(f"Orig Length: {len(orig_ids)} | New Length: {len(new_ids)}")
            return False

        # Subword (String) Parity
        if orig_tokens != new_tokens:
            logging.error(f"Subword tokens don't match")
            logging.error(f"Text: {text}")
            logging.error(f"Original Subwords: {orig_tokens}")
            logging.error(f"New Subwords:      {new_tokens}")
            import code; code.interact(local=dict(globals(), **locals()))
            return False

        # Mapping Integrity (ID Alignment)
        for i, (o_id, n_id) in enumerate(zip(orig_ids, new_ids)):
            expected_super_id = internal_to_super_map.get(o_id)
            
            if expected_super_id is None:
                logging.warning(f"Original ID {o_id} ('{orig_tokens[i]}') not found in mapping!")
                continue # Or return False if your mapping is strictly required to be complete

            if n_id != expected_super_id:
                logging.error(f"Mapping failure at position {i}")
                logging.error(f"Token: '{orig_tokens[i]}'")
                logging.error(f"Original ID: {o_id} -> Expected Super ID: {expected_super_id}")
                logging.error(f"Actual Super ID returned by tokenizer: {n_id}")
                return False
            
    return True

def _update_added_tokens(temp_path:Path,  mapping:dict):
    # necessary for Qwen
    """Updates added_tokens.json with re-indexed IDs."""
    added_tokens_path = temp_path / "added_tokens.json"
    if not added_tokens_path.exists():
        return
    
    with open(added_tokens_path, 'r') as f:
        added_tokens = json.load(f)

    if type(added_tokens) is dict:
        new_tokens = {}
        for key, token_id in added_tokens.items():
            if token_id in mapping:
                new_tokens[key] = mapping[token_id]
    else:
        return
    
    with open(added_tokens_path, 'w') as f:
        json.dump(new_tokens, f, indent=2)
    logging.info(f"Updated added_tokens at {added_tokens_path}")

def _update_decoder_tokens(temp_path:Path, mapping:dict):
    # necessary for Qwen
    """Updates added_tokens_decoder.json with re-indexed IDs."""
    path = temp_path / "tokenizer_config.json"
    if not path.exists():
        return
    
    with open(path, 'r') as f:
        added_tokens_decoder = json.load(f)

    if "added_tokens_decoder" not in added_tokens_decoder:
        return
    new_tokens_decoder = {mapping[k]: v for k, v in added_tokens_decoder["added_tokens_decoder"].items() if k in mapping}
    added_tokens_decoder["added_tokens_decoder"] = new_tokens_decoder
    
    with open(path, 'w') as f:
        json.dump(added_tokens_decoder, f, indent=2)
    logging.info(f"Updated added_tokens_decoder at {path}")

def update_auxiliary_configs(temp_path: Path, mapping: dict):
    """Updates sidecar files like special_tokens_map.json and tokenizer_config.json."""
    _update_added_tokens(temp_path, mapping)
    _update_decoder_tokens(temp_path, mapping)
    targets = ["special_tokens_map.json", "tokenizer_config.json"]
    
    for target in targets:
        file_path = temp_path / target
        if not file_path.exists():
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Recursive function to find and replace 'id' fields in nested dictionaries
        def remap_ids(obj):
            if isinstance(obj, dict):
                # If we find a dict like {"content": "<bos>", "id": 1}
                if "id" in obj and isinstance(obj["id"], int):
                    old_id = obj["id"]
                    if old_id in mapping:
                        obj["id"] = mapping[old_id]
                for k, v in obj.items():
                    remap_ids(v)
            elif isinstance(obj, list):
                for item in obj:
                    remap_ids(item)
        # import code; code.interact(local=dict(globals(), **locals()))
        # remap_ids(data)


        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

# def create_lingua_write_lingua_config(temp_path, )
def publish_tokenizer(original_name: str, original_path: str, args):
    sanitized_name = original_path.replace('/', '--')
    repo_name = f"{args.hf_upload_repo_prefix}{original_path.replace('/', '-')}"
    temp_path = Path(args.temp_dir) / sanitized_name
    temp_path.mkdir(parents=True, exist_ok=True)

    # Load Mappings
    mapping_path = Path(args.input_dir) / f"{sanitized_name}_super_mapping.json"
    internal_to_super_map = load_super_mapping(mapping_path)
    super_vocab = load_super_vocab(Path(args.input_dir) / "super_vocab.json")
    new_vocab_size = get_max_super_id(super_vocab)

    if original_name == "tiktoken":
        handle_tiktoken_export(original_path, internal_to_super_map, temp_path)
    elif original_name == "tokenmonster":
        handle_tokenmonster_export(original_path, internal_to_super_map, temp_path)
    elif original_name == "tekken":
        handle_tekken_export( internal_to_super_map, temp_path)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", use_fast=True)
        tokenizer.save_pretrained(temp_path)
        reindex_hf_json(temp_path / "tokenizer.json", internal_to_super_map)
    else:
        tokenizer = AutoTokenizer.from_pretrained(original_path, use_fast=True)
        tokenizer.save_pretrained(temp_path)
        reindex_hf_json(temp_path / "tokenizer.json", internal_to_super_map)
        update_auxiliary_configs(temp_path, internal_to_super_map)
        verify_parity(tokenizer, AutoTokenizer.from_pretrained(temp_path, use_fast=True), internal_to_super_map)

    # Finalize config & Upload
    if args.publish_to_hf:
        finalize_and_upload(repo_name, temp_path, new_vocab_size)


def reindex_hf_json(json_path, internal_to_super_map):
    """Helper to rewrite the vocab inside a standard tokenizer.json."""
    if not json_path.exists(): return
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Update vocab in model
    vocab = config["model"]["vocab"]
    # if unigram, the format is token, likelihood, ['̊', -17.45818519592285], ['础', -17.47956657409668], ['ẩ', -17.666366577148438], ['ઞ', -17.81585121154785], ['垃', -18.125295639038086], ['礎', -18.125324249267578], ['肪', -18.575355529785156], ['雰', -18.575380325317383]
    if config.get("model", {}).get("type") == "Unigram":
        pass
    if isinstance(vocab, dict):
        new_vocab = {k: internal_to_super_map[v] for k, v in vocab.items() if v in internal_to_super_map}
    else: # List format [["tok", id], ...]
        new_vocab = [[item[0], internal_to_super_map[item[1]]] for item in vocab if item[1] in internal_to_super_map]
        len([[internal_to_super_map[item[1]]] for item in vocab if item[1] in internal_to_super_map])
    config["model"]["vocab"] = new_vocab

    if "added_tokens_decoder" in config:
        new_added_tokens_decoder = {}
        for k, v in config["added_tokens_decoder"].items():
            old_id = int(k)
            if old_id in internal_to_super_map:
                new_id = internal_to_super_map[old_id]
                new_item = dict(v)
                new_item["id"] = new_id
                new_added_tokens_decoder[str(new_id)] = new_item
        config["added_tokens_decoder"] = new_added_tokens_decoder
    if "added_tokens" in config:
        # Qwen: "added_tokens": [{ "id": 462581, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        # XGLM: [{'id': 0, 'content': '<s>', 'single_word': False, 'lstrip': False, 'rstrip': False, 'normalized': False, 'special': True},]
        new_added_tokens = []
        for item in config["added_tokens"]:
            if item["id"] in internal_to_super_map:
                new_item = item.copy()
                new_item["id"] = internal_to_super_map[item["id"]]
                new_added_tokens.append(new_item)
                ## put added tokens to vocab, otherwise they get reindexed 
                if "xglm" in json_path.as_posix().lower():
                    import code; code.interact(local=locals()|globals())
                if isinstance(vocab, dict):
                    config["model"]["vocab"][new_item["content"]] = new_item["id"]
                else:
                    config["model"]["vocab"].append([new_item["content"], new_item["id"]])
                
        config["added_tokens"] = new_added_tokens

    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    logging.info(f"Re-indexed tokenizer JSON saved to {json_path}")
    

def handle_tiktoken_export(model_id, mapping, output_dir):
    """Converts tiktoken BPE to HF format with re-indexed IDs."""
    from tiktoken import encoding_for_model
    from transformers.integrations.tiktoken import convert_tiktoken_to_fast
    enc = encoding_for_model(model_id)
    logging.info(f"Converting tiktoken model '{model_id}' to HF format.")
    tokenizer = convert_tiktoken_to_fast(enc, output_dir)
    reindex_hf_json(output_dir / "tokenizer.json", mapping)

def handle_tekken_export(mapping, output_dir):
    """Converts Tekken tokenizer to HF format with re-indexed IDs."""
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer, Tekkenizer
    from transformers.integrations.tiktoken import convert_tiktoken_to_fast

    tokenizer = MistralTokenizer.v3(is_tekken=True)
    tokenizer = tokenizer.instruct_tokenizer.tokenizer
    enc = tokenizer._model
    ## maybe write it as a tekken, todo: problematic
    # tekken_file = tokenizer.file_path
    # with open(tekken_file, 'r') as f:
    #     tekken_data = json.load(f)
    # # Re-index tekken vocab
    # # "vocab": [{ "rank": 0, "token_bytes": "AA==", "token_str": "\u0000"},{ "rank": 1, "token_bytes": "AQ==", "token_str": "\u0001"}, ...
    # new_vocab = [entry | {"rank": mapping[entry["rank"]]} for entry in tekken_data["vocab"] if entry["rank"] in mapping]
    # tekken_data["vocab"] = new_vocab
    # tekken_data["config"]["num_vocab_tokens"] = len(new_vocab)
    # with open(output_dir / "tekken.json", 'w') as f:
    #     json.dump(tekken_data, f)
    # tokenizer = Tekkenizer.from_file((output_dir / "tekken.json").as_posix())

    # import code; code.interact(local=dict(globals(), **locals()))
    logging.info(f"Converting tekken model to HF format.")
    tokenizer = convert_tiktoken_to_fast(enc, output_dir)
    reindex_hf_json(output_dir / "tokenizer.json", mapping)

def handle_tokenmonster_export(vocab_name, mapping, output_dir):
    """Exports tokenmonster vocab to a HF compatible JSON structure."""
    import tokenmonster
    # # 31988: {'id': 31988, 'token': '****************************************', 'token_decoded': '****************************************', 'type': 'regular', 'score': 6.201580254128203e-05}, 
    # new_vocab = {mapping[k]: v   | {"id": mapping[k]} for k, v in vocab.items() if k in mapping}
    # keep it as tokenmonster format
    tm = tokenmonster.load(vocab_name)
    vocab = tm.get_dictionary()
    new_vocab = {mapping[k]: v   | {"id": mapping[k]} for k, v in vocab.items() if k in mapping}
    tm.dictionary = new_vocab
    tm.vocab_size = len(new_vocab)
    tm.save((output_dir / "tokenizer.vocab").as_posix())

def finalize_and_upload(repo_name, temp_path, vocab_size):
    # Update vocab size in tokenizer_config.json
    cfg_path = temp_path / "tokenizer_config.json"
    data = json.load(open(cfg_path)) if cfg_path.exists() else {}
    data["vocab_size"] = vocab_size
    with open(cfg_path, "w") as f: json.dump(data, f, indent=2)
    ## TODO: Add special tokens 
    ## todo: add config yamls
    ## todo: test that the outputs are the same (subwords)
    # HF Upload logic
    api = hf_hub.HfApi()
    api.create_repo(repo_id=repo_name, exist_ok=True)
    api.upload_folder(folder_path=str(temp_path), repo_id=repo_name)
    logging.info(f"Uploaded re-indexed tokenizer to HF repo: {repo_name}")

def main_publish(args):
    Path(args.temp_dir).mkdir(exist_ok=True)
    
    for tokenizer in args.tokenizers:
        logging.info(f"Publishing tokenizer: {tokenizer.name} from {tokenizer.path}")
        publish_tokenizer(tokenizer.name, tokenizer.path, args)


if __name__ == "__main__":
    args = load_args(ReindexingArgs)
    main_publish(args)