import base64
import json
import logging
import os
from pathlib import Path

import tiktoken
from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_folder
from tiktoken.load import dump_tiktoken_bpe, load_tiktoken_bpe
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoTokenizer, models
from transformers.integrations.tiktoken import convert_tiktoken_to_fast

logging.basicConfig(level=logging.INFO)
REPO_ID = "moe-dtoks/tiktoken_files"
# python moto/utils/convert_hf_to_tiktoken.py


def convert_hf_token_to_bytes(token_str, byte_decoder="gpt2"):
    # HF tokenizers map bytes to unicode (e.g., space -> Ġ). We must reverse this.
    if byte_decoder == "gpt2":
        byte_encoder = models.gpt2.tokenization_gpt2.bytes_to_unicode()
    else:
        raise ValueError(f"Unknown byte_decoder: {byte_decoder}")
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    decoded_bytes = bytearray()
    for char in token_str:
        if char in byte_decoder:
            decoded_bytes.append(byte_decoder[char])
        else:
            decoded_bytes.extend(char.encode("utf-8"))
    return bytes(decoded_bytes)

def handle_llama(path="meta-llama/Llama-3.2-1B", output_dir="/tmp/llama_tokenizers"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- 1. DOWNLOAD TOKENIZER.MODEL ---
    print(f"Downloading tokenizer.model from {path}...")
    tokenizer_model_path = hf_hub_download(
        repo_id=path,
        filename="original/tokenizer.model",
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )

    # --- 2. LOAD MERGEABLE RANKS ---
    print("Loading mergeable ranks...")
    mergeable_ranks = load_tiktoken_bpe(tokenizer_model_path)

    # 3. Save tokenizer.model (Ranks only)
    model_file = output_dir / "tokenizer.model"
    dump_tiktoken_bpe(mergeable_ranks, model_file)
    print(f"✅ Saved ranks to {model_file}")

    return mergeable_ranks

# 1. Define the Regex Patterns (Hardcoded for reliability)
#    Most modern models (Qwen, Llama 3, GPT-4) use this exact pattern.
GPT4_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

def convert_and_upload_qwen(
    hf_model_id="Qwen/Qwen3-8B", 
    target_repo=REPO_ID,
    target_subfolder="qwen_qwen3_8b"
):
    logging.info(f"Processing {hf_model_id}...")
    
    # --- A. Load the Source Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    
    # --- B. Prepare Config Data ---
    # 1. Get Special Tokens
    #    We use added_tokens_decoder to ensure we get the correct ID mapping
    special_tokens_map = {}
    for rank, token_obj in tokenizer.added_tokens_decoder.items():
        special_tokens_map[token_obj.content] = rank

    # 2. Build the config dictionary
    config_data = {
        "name": target_subfolder,
        "hf_repo_id": hf_model_id,
        "pat_str": GPT4_SPLIT_PATTERN,  # Qwen uses the standard cl100k pattern
        "explicit_n_vocab": tokenizer.vocab_size + len(special_tokens_map), # Optional metadata
        "special_tokens": special_tokens_map
    }

    # --- C. Prepare Ranks (tokenizer.model) ---
    vocab = tokenizer.get_vocab()
    special_ids = set(special_tokens_map.values())
    
    # Generate the lines for tokenizer.model
    # Format: base64_string <space> rank
    model_lines = []
    for token_str, rank in sorted(vocab.items(), key=lambda x: x[1]):
        if rank not in special_ids:
            token_bytes = convert_hf_token_to_bytes(token_str, "gpt2")
            token_b64 = base64.b64encode(token_bytes).decode("utf-8")
            model_lines.append(f"{token_b64} {rank}")

    for i in range(15):
        print(model_lines[i])

    # --- D. Write to disk temporarily ---
    os.makedirs(f"temp/{target_subfolder}", exist_ok=True)
    
    # Save config.json
    with open(f"temp/{target_subfolder}/config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
        
    # Save tokenizer.model
    with open(f"temp/{target_subfolder}/tokenizer.model", "w", encoding="utf-8") as f:
        f.write("\n".join(model_lines))

    logging.info(f"Converted locally to temp/{target_subfolder}")

    # --- E. Upload to Hugging Face ---
    api = HfApi()
    api.upload_folder(
        folder_path=f"temp/{target_subfolder}",
        path_in_repo=target_subfolder,
        repo_id=target_repo,
        repo_type="model",
        commit_message=f"Add unified tokenizer for {target_subfolder}"
    )
    logging.info(f"Uploaded to https://huggingface.co/{target_repo}/tree/main/{target_subfolder}")

# Example Usage
# convert_and_upload_qwen()
def main():
    create_repo(repo_id=REPO_ID, exist_ok=True)
    output_dir = "/tmp/qwen_tokenizers"

    convert_and_upload_qwen()
    # output_dir = "/tmp/llama_tokenizers"
    # handle_llama()
    # upload_folder(repo_id=REPO_ID, folder_path=output_dir, path_in_repo="meta-llama/Llama-3.2-1B")


if __name__ == "__main__":
    main()
