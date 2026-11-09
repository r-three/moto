import logging
from pathlib import Path
from typing import List

import huggingface_hub as hf_hub
from safetensors import safe_open

logging.basicConfig(level=logging.INFO)

def get_safetensors_files(repo_id: str) -> list[Path]:
    """Get local paths for all safetensors shards in a repo."""
    all_files = hf_hub.list_repo_files(repo_id)
    st_files = sorted([f for f in all_files if f.endswith(".safetensors")])
    return [Path(hf_hub.hf_hub_download(repo_id, f)) for f in st_files]

def get_supermapping_file(repo_id: str) -> str: 
    """Load the supermapping.json from the given repo."""
    all_files = hf_hub.list_repo_files(repo_id)
    mapping_file_name = [f for f in all_files if f.endswith("super_mapping.json")][0]
    mapping_path = hf_hub.hf_hub_download(repo_id, mapping_file_name)
    return mapping_path

def get_tensor_from_shards(files: list[Path], key: str):
    """Find and load a specific tensor from shards without loading the whole model."""
    for f in files:
        with safe_open(f.as_posix(), framework="pt", device="cpu") as f_open:
            if key in f_open.keys():
                return f_open.get_tensor(key)
    raise KeyError(f"Key {key} not found.")


def create_or_get_collection_slug(hf_collection: str) -> str:
    """Return the slug for a collection identified by 'namespace/title', creating it if needed.

    api.get_collection requires a slug, not a title, so we use api.list_collections
    to search for an existing collection with the matching title under the namespace.
    """
    api = hf_hub.HfApi()
    namespace, title = hf_collection.split("/", 1) if "/" in hf_collection else (None, hf_collection)

    existing = next(
        (col for col in api.list_collections(owner=namespace) if col.title == title),
        None,
    )
    if existing:
        logging.info("Found existing HF collection: %s", existing.slug)
        return existing.slug

    collection = api.create_collection(title=title, namespace=namespace, exists_ok=True)
    logging.info("Created new HF collection: %s", collection.slug)
    return collection.slug


def create_or_get_collection_slugs(hf_collections: List[str]) -> List[str]:
    collection_slugs = []
    for hf_collection in hf_collections:
        collection_slugs.append(create_or_get_collection_slug(hf_collection))
    return collection_slugs