"""Create a shared vocabulary to keep models consistent.

Load tokenmonster with tokenmonster/...
Load tiktoken with tiktoken/${model-name}
"""

import argparse
import collections
import functools
import json
import logging
import operator as op
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import huggingface_hub as hf_hub
import numpy as np
import transformers
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torchgen import api
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from moto.utils.args import Args, HFArgs, dataclass_from_dict, load_args
from moto.utils.hf_utils import create_or_get_collection_slug

Vocab = dict[str, list[int]]


@dataclass
class TokenizerArgs:
    name: str = ""
    path: str = ""

@dataclass
class SuperVocabArgs(Args):
    config: Any = None
    output_dir: str = "vocabs"
    tokenizers: List[str] = field(default_factory=list)
    # HF repo id or local directory of the base model whose weights will initialise
    # the superset backbone + scatter base embeddings (non-overlapping positions are
    # randomly initialised).  Pass the same path to create_checkpoints.py via
    # --base_model.
    base_model: Optional[str] = None
    # Tokenizer used by the base model.  Its super_mapping is generated and it is
    # prepended to `tokenizers` if not already present.
    # Defaults to base_model when None (i.e. the model repo also hosts the tokenizer).
    base_tokenizer: Optional[str] = None
    pivot_tokenizers: List[str] = field(default_factory=list) # give in the order
    hf_upload_repo_prefix: str = ""
    hf: HFArgs = field(default_factory=HFArgs)
    upload_to_original_repo: bool = False # If True, uploads the mapping files to the original tokenizer repos instead of a new repo with hf_upload_repo_prefix. Only applies if hf.publish_to_hf is True. This is useful for ease of access when training tokenizers, but can lead to clutter in the original repos so it is optional.
    superset_code_name: str = "super_vocab" # The code name to use for the super vocab in the mapping files. This is just a string and does not have to match any actual tokenizer name, but it should be unique from the other superset tokenizers in the mapping files. It is used to identify the super vocab in the mapping files and should be consistent across all mapping files. Default is "super_vocab".
    temp_dir: str = f"{os.environ.get('TMPDIR', './temp_tokenizers')}"
    return_weights: bool = False # If True computes the overlap between tokenizers and returns a dict of dicts with the overlap percentages. This is expensive to compute, so it is optional. e.g. Llama: x tokens in all Z tokenizers, y tokens in (Z-1)/Z tokenizers etc.


logging.basicConfig(level=logging.INFO)

ALIGNED_BOS = "~SPECIAL~ALIGNED~BOS~SYMBOL~"


def hf_load_tokenizer(name: str):
    kwargs = dict()
    if "aya" in name.lower():
        kwargs["use_fast"] = True
    elif "xglm" in name.lower():
        kwargs["use_fast"] = True
    return transformers.AutoTokenizer.from_pretrained(name, **kwargs)


class Tokenizer:
    """Tokenizer wrapper that unifies interface."""

    def __init__(self, name: str, tokenizer):
        self._name = name
        self.tokenizer = tokenizer

    @property
    def name(self):
        return self._name

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_token(self, i):
        raise NotImplementedError

    def get_bos_str(self):
        raise NotImplementedError

    def info(self):
        raise NotImplementedError

    @classmethod
    def load(cls, name):
        try:
            if name.startswith("tokenmonster"):
                return TokenMonsterTokenizer.load(name)
            if name.startswith("tiktoken"):
                tok = TikTokenTokenizer.load(name)
                return tok
            if "tekken" in name:
                return MistralTokenizer.load(name)
            return HFTokenizer.load(name)
        except Exception as e:
            print(e)
            import code

            code.interact(local=locals() | globals())


class HFTokenizer(Tokenizer):
    def __init__(self, *args, bos_str: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bos_str = bos_str

    def info(self):
        return {"data": {"tokenizer": {"name": "huggingface", "path": self.name}}}

    def get_vocab_size(self):
        if "byt5" in self.name:
            return self.tokenizer.vocab_size
        return self.tokenizer.get_vocab_size()

    def get_token(self, i):
        if "byt5" in self.name:
            token = self.tokenizer.convert_ids_to_tokens(i)
            # We are a special value.
            if len(token) > 1:
                return token
            as_int = ord(token) 
            as_bytes = bytes([as_int])
            try:
                return as_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return as_int  # as_bytes
        t = self.tokenizer.id_to_token(i)
        if t == self.bos_str:
            return ALIGNED_BOS
        if isinstance(self.tokenizer.model, models.WordPiece):
            # If it is not a continuation character, then it is the start of a word. Other tokenizers start the word with a subword token that has a space to start.
            if not t.startswith("##"):
                return f" {t}"
            return re.sub(r"##([^#])", r"\1", t)
        if isinstance(self.tokenizer.model, models.Unigram) or any(
            n in self.name for n in ("gemma", "Phi-3", "Mistral-7B-Instruct-v0.3")
        ):
            # Replace whitespace handling with actual whitespace.
            return t.replace("▁", " ")
        # BPE models.
        return real_unicode(t)

    @classmethod
    def load(cls, name):
        tok = hf_load_tokenizer(name)
        sts = getattr(tok, "special_tokens_map", {})
        if "bert" in name:
            bos_str = sts.get("cls_token")
        elif "t5" in name:
            bos_str = sts.get("pad_token")
        else:
            bos_str = sts.get("bos_token")
        if hasattr(tok, "_tokenizer"):
            tok = tok._tokenizer
        return cls(name, tok, bos_str=bos_str)


# Note, GPT4 and GPT4o don't have BOS
class TikTokenTokenizer(Tokenizer):
    def info(self):
        return {
            "data": {"tokenizer": {"name": "tiktoken", "path": self.name.split("/")[1]}}
        }

    def get_token(self, i):
        try:
            b = self.tokenizer.decode_single_token_bytes(i)
        except KeyError:
            return f"~~~~~undefined {i}~~~~~~"
        return b.decode("latin-1")

    def get_vocab_size(self):
        return self.tokenizer.n_vocab

    @classmethod
    def load(cls, name):
        import tiktoken

        tok = tiktoken.encoding_for_model(name.split("/")[1])
        return cls(name, tok)


class TokenMonsterTokenizer(Tokenizer):
    def info(self):
        return {
            "data": {
                "tokenizer": {"name": "tokenmonster", "path": self.name.split("/")[1]}
            }
        }

    def get_token(self, i):
        return self.tokenizer.id_to_token(i)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    @classmethod
    def load(cls, name):
        import tokenmonster

        tok = tokenmonster.load(name.split("/")[1])
        return cls(name, tok)


class MistralTokenizer(Tokenizer):
    def info(self):
        return {"data": {"tokenizer": {"name": "tekken", "path": "tekken"}}}

    def get_token(self, i):
        if i == self.tokenizer.bos_id:
            return ALIGNED_BOS
        return self.tokenizer.id_to_piece(i)

    def get_vocab_size(self):
        return self.tokenizer.n_words

    @classmethod
    def load(cls, name):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tok = MistralTokenizer.v3(is_tekken=True)
        tok = tok.instruct_tokenizer.tokenizer
        return cls(name, tok)


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


BYTES_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTES = {v: k for k, v in BYTES_TO_UNICODE.items()}


def real_unicode(word: str) -> str:
    bytes_word = []
    for c in word:
        if c != " ":
            if c in UNICODE_TO_BYTES:
                c = chr(UNICODE_TO_BYTES[c])
        bytes_word.append(c.encode("utf-8"))
    return b"".join(bytes_word).decode("utf-8")


def make_vocab(tok: Tokenizer) -> Vocab:
    # returns a mapping from string to list of indices
    # Track multiple values because tekken and tokenmonster are weird
    vocab = collections.defaultdict(list)
    for i in range(tok.get_vocab_size()):
        vocab[tok.get_token(i)].append(i)
    if len(vocab) != tok.get_vocab_size():
        logging.error(
            "Built vocab size (%d) does not match declared vocab size (%d) for %s",
            len(vocab),
            tok.get_vocab_size(),
            tok.name,
        )
    return vocab


def to_bytes(s: bytes | str | int) -> bytes:
    if isinstance(s, str):
        s = s.encode("utf-8")
    if isinstance(s, int):
        s = bytes([s])
    # Now s is def bytes
    return s


def join_vocabs(vocabs: dict[str, Vocab]) -> Vocab:
    joint = functools.reduce(op.or_, [v.keys() for v in vocabs.values()])
    return {s: i for i, s in enumerate(sorted(joint, key=to_bytes))}


def align_to_super(super_vocab, model_vocab):
    alignment = {}
    for piece, idxs in model_vocab.items():
        super_idx = super_vocab[piece]
        for i in idxs:
            alignment[i] = super_idx
    return alignment


def main(args):
    # Resolve the tokenizer used by the base model (defaults to base_model itself).
    effective_base_tokenizer = args.base_tokenizer or args.base_model
    if effective_base_tokenizer is not None and effective_base_tokenizer not in args.tokenizers:
        logging.info(
            "base_tokenizer '%s' not in tokenizers list — prepending it so its "
            "super_mapping is generated.",
            effective_base_tokenizer,
        )
        args.tokenizers = [effective_base_tokenizer] + list(args.tokenizers)

    logging.info("Loading Tokenizers.")
    tokenizers: dict[str, Tokenizer] = {
        name: Tokenizer.load(name) for name in args.tokenizers
    }
  
    logging.info("Extracting Vocabularies.")
    tokenizer_vocabs: dict[str, Vocab] = {
        name: make_vocab(tokenizer) for name, tokenizer in tokenizers.items()
    }

    logging.info("Creating super set vocabulary")
    super_vocab = join_vocabs(tokenizer_vocabs)
    logging.info("Super set vocabulary has %d items", len(super_vocab))

    # Save the super vocab
    os.makedirs(args.output_dir, exist_ok=True)
    files_to_upload = []
    d = os.path.join(args.output_dir, "super_vocab.json")
    with open(d, "w") as wf:
        logging.info("Saving super set vocab to '%s'", d)
        json.dump(super_vocab, wf)
        files_to_upload.append(d)

    config_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_path, "w") as wf:
        logging.info("Saving config to '%s'", config_path)
        comment=f"""
# Config for super vocab built from tokenizers: {', '.join(args.tokenizers)}
## Training superset tokenizer with individual tokenizers trained on Fineweb-2-hq
# use with apps/main/configs/flexitok/llama_1b_base.yaml
##TODO: add router details, make sure name is correct
"""
        wf.write(comment)
        dct = {"model": {"vocab_size": len(super_vocab)}}
        dct["name"] = args.superset_code_name
        dct["dump_dir"] = f"/fsx/craffel/lingua_logs/{args.superset_code_name}"
        # for now default huggingface but could be tiktoken etc
        tokenizers_list = [{"name": "huggingface", "path": t, "load_supermapping": True} for t in tokenizers]
        dct["data"] = {"tokenizer": {"name": "supertokenizer", "seed": 42, 
                                     "superset_code_name": args.superset_code_name, "n_words": len(super_vocab), "tokenizers": tokenizers_list}}
        dct["checkpoint"] = {"path": f"/fsx/craffel/lingua_logs/checkpoints/{args.superset_code_name}"}
        yaml.dump(dct, wf, sort_keys=False)
        files_to_upload.append(config_path)

    readme_path = os.path.join(args.output_dir, "README.md")
    tokenizer_list = "\n".join(f"- `{t}`" for t in args.tokenizers)
    with open(readme_path, "w") as wf:
        logging.info("Saving README to '%s'", readme_path)
        wf.write(f"""# Super Vocabulary

A merged super-vocabulary built from {len(args.tokenizers)} tokenizer(s).

**Vocab size:** {len(super_vocab)}

## Tokenizers

{tokenizer_list}

## Files

- `super_vocab.json` — merged vocabulary mapping token string → super index
- `config.yaml` — model config with `vocab_size`
- `participating_tokenizers.json` — list of tokenizer names included
- `<tokenizer>_super_mapping.json` — per-tokenizer index → super index mapping
- `<tokenizer>_vocab.json` — per-tokenizer vocabulary
- `<tokenizer>_info.json` / `.yaml` — tokenizer metadata
""")
    files_to_upload.append(readme_path)
    
    participating_tokenizers = os.path.join(args.output_dir, f"participating_tokenizers.json")
    with open(participating_tokenizers, "w") as wf:
        logging.info("Saving list of participating tokenizers to '%s'", participating_tokenizers)
        json.dump(list(tokenizers.keys()), wf)
        files_to_upload.append(d)
# TODO: add code name to the config yaml
    if args.hf.publish_to_hf:
        if args.hf.hf_repo_id:
            repo_name = args.hf.hf_repo_id
        elif args.hf_upload_repo_prefix:
            repo_name = f"{args.hf_upload_repo_prefix}{args.superset_code_name.replace('/', '-')}"
        else:
            raise ValueError("Must provide either hf.hf_repo_id or hf_upload_repo_prefix to upload to HF")
        logging.info("Uploading super vocab to HF repo '%s'", repo_name)
        hf_hub.create_repo(
            repo_id=repo_name,
            repo_type="model",
            exist_ok=True,
        )
        for file in files_to_upload:
            hf_hub.upload_file(
                path_or_fileobj=file,
                path_in_repo=f"{os.path.basename(file)}",
                repo_id=repo_name,
                repo_type="model",
            )
        # hf_hub.upload_file(
        #     path_or_fileobj=d,
        #     path_in_repo="super_vocab.json",
        #     repo_id=repo_name,
        #     repo_type="model",
        # )
        if args.hf.collections:
            for collection in args.hf.collections:
                try:
                    collection_slug= create_or_get_collection_slug(collection)
                except Exception as e:
                    collection_slug = collection
                    continue
                    logging.warning(f"Error occurred while creating/getting collection slug for {collection}: {e}")
                api = hf_hub.HfApi()
                api.add_collection_item(
                    collection_slug=collection_slug,
                    item_id=repo_name,
                    item_type="model",
                    exists_ok=True
                )
                logging.info(f"Added {repo_name} to collection {collection_slug}")
    # Save each vocab mapping
    for name, vocab in tokenizer_vocabs.items():
        files_to_upload = []
        logging.info("Creating mapping for tokenizer '%s'", name)
        # Replace / with -- like the huggingface caching code does.
        d = os.path.join(
            args.output_dir, f"{name.replace('/', '--')}_super_mapping.json"
        )
        with open(
            d,
            "w",
        ) as wf:
            logging.info("Saving vocab mapping for %s to '%s'", name, d)
            json.dump(align_to_super(super_vocab, vocab), wf)

            files_to_upload.append(d)
        d = os.path.join(args.output_dir, f"{name.replace('/', '--')}_vocab.json")
        with open(
            d,
            "w",
        ) as wf:
            logging.info("Saving vocab for %s to '%s'", name, d)
            json.dump(vocab, wf)
            files_to_upload.append(d)

        d = os.path.join(args.output_dir, f"{name.replace('/', '--')}_info.json")
        with open(
            d,
            "w",
        ) as wf:
            logging.info("Saving tokenizer info for %s to '%s'", name, d)
            json.dump(tokenizers[name].info(), wf)
            files_to_upload.append(d)

        d = os.path.join(args.output_dir, f"{name.replace('/', '--')}.yaml")
        with open(d, "w") as wf:
            logging.info("Saving tokenizer info for %s to '%s'", name, d)
            yaml.dump(tokenizers[name].info(), wf)
            files_to_upload.append(d)
        d= os.path.join(args.output_dir, f"participating_tokenizers.json")
        with open(d, "w") as wf:
            logging.info("Saving list of participating tokenizers to '%s'", d)
            json.dump(list(tokenizers.keys()), wf)
            files_to_upload.append(d)
        if args.return_weights:
            overlap_counts = collections.Counter()
            for piece, idxs in vocab.items():
                count = sum(piece in v for v in tokenizer_vocabs.values())
                overlap_counts[count] += 1
            total_tokens = sum(overlap_counts.values())
            overlap_percentages = {
                count:{
                    "ratio_to_total_tokens": num_tokens/total_tokens,
                    "expected_training_ratio_in_superset": count/len(args.tokenizers) * num_tokens/total_tokens,
                    "num_tokens": num_tokens
                }
                for count, num_tokens in overlap_counts.items()
            }
            overlap_percentages = dict(sorted(overlap_percentages.items()))
            overlap_percentages = overlap_percentages | {"total_training_compared_to_full_model": sum([v["expected_training_ratio_in_superset"] for v in overlap_percentages.values()])}
            # assert np.isclose(sum(v["weight_in_super_vocab"] for v in overlap_percentages.values()), 1.0), f"Weights do not sum to 1: {overlap_percentages}"
            d = os.path.join(args.output_dir, f"{name.replace('/', '--')}_overlap.json")
            with open(d, "w") as wf:
                logging.info("Saving tokenizer overlap info for %s to '%s'", name, d)
                json.dump(overlap_percentages, wf)
                files_to_upload.append(d)
            if "llama" in name.lower():
                logging.info("LLaMA tokenizer overlap info: %s", overlap_percentages)

        if args.hf_upload_repo_prefix:
            path_prefix= f"{args.superset_code_name}/" if args.superset_code_name else ""
            if args.upload_to_original_repo:
                repo_name = name
            else:
                repo_name = f"{args.hf_upload_repo_prefix}{name.replace('/', '-')}"
                hf_hub.create_repo(
                    repo_id=repo_name,
                    repo_type="model",
                    exist_ok=True,
                )
            logging.info("Uploading tokenizer for %s to HF repo '%s'", name, repo_name)
            for file in files_to_upload:
                hf_hub.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=f"{path_prefix}{os.path.basename(file)}",
                    repo_id=repo_name,
                    repo_type="model",
                )
            if args.hf.collections:
                for collection in args.hf.collections:
                    try:
                        collection_slug= create_or_get_collection_slug(collection)
                    except Exception as e:
                        collection_slug = collection
                        logging.warning(f"Error occurred while creating/getting collection slug for {collection}: {e}")
                        continue
                    api = hf_hub.HfApi()
                    api.add_collection_item(
                        collection_slug=collection_slug,
                        item_id=repo_name,
                        item_type="model",
                        exists_ok=True
                    )
                    logging.info(f"Added {repo_name} to collection {collection_slug}")
            


if __name__ == "__main__":
    args = load_args(SuperVocabArgs)
    main(args)
