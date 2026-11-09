# moto/lingua

MoTo-specific additions to the [Lingua](https://github.com/r-three/lingua) training framework. This package does not re-implement the trainer — it provides the model-side components and post-training tooling that plug into Lingua to support MoTo's superset tokenizer vocabulary.

## Files

### `transformer.py`

Core transformer primitives derived from Lingua's reference implementation:

- `BaseTransformer` / `BaseTransformerArgs` — configurable Llama-style transformer backbone
- `Attention` — grouped-query attention (GQA) with three pluggable backends: `sdpa`, `fmha` (xFormers), `flex_attention`
- `FeedForward` — SwiGLU MLP
- `RMSNorm`, `RotaryEmbedding` (RoPE), `TiedLinear`
- Document-aware causal masking for packed sequences (`generate_doc_mask_mod`)

### `lm_transformer.py`

Wraps `BaseTransformer` into a full causal language model:

- `LMTransformer` — adds token embeddings, output projection, optional weight tying, and causal/sliding-window masking
- `build_fsdp_grouping_plan` — FSDP sharding plan grouping embeddings, layers, and the output head separately
- `tp_parallelize` — tensor-parallelism setup for attention and feed-forward layers when `tp_size > 1`

### `modeling_llama_albert.py`

A Llama variant with ALBERT-style factorized embeddings — the core model architecture used in MoTo experiments with large superset vocabularies:

- `LlamaAlbertConfig` — extends `LlamaConfig` with an `embedding_dim` bottleneck smaller than `hidden_size`
- `LlamaAlbertForCausalLM` — both the embedding table (`vocab_size × embedding_dim → hidden_size`) and the LM head (`hidden_size → embedding_dim → vocab_size`) are factorized, keeping parameter cost manageable as vocabulary size grows

### `convert_lingua_to_huggingface.py`

Converts a trained Lingua checkpoint to HuggingFace format for evaluation and sharing:

- Handles sharded and unsharded `.pth` checkpoints, and DCP (distributed checkpoint) consolidation via `created_consolidated`
- Supports Llama versions 1, 2, 3, 3.1, 3.2, and Guard-3
- Converts tokenizers: SentencePiece for Llama 1/2, TikToken for Llama 3+
- Writes `safetensors` by default; optionally pushes the model and tokenizer to the HuggingFace Hub
- Handles factorized-embedding models (`LlamaAlbertForCausalLM`) automatically based on `params.json`

```bash
python -m moto.lingua.convert_lingua_to_huggingface \
    --input_dir /path/to/lingua/checkpoint \
    --output_dir /path/to/hf/output \
    --llama_version 3.2 \
    --num_shards 1
```

### `probe.py`

Activation statistics logger adapted from xFormers, used to inspect model internals during training without modifying model code:

- `AutoProbeD` — a `TorchDispatchMode` context manager that intercepts `mm`/`addmm` ops and attention kernels (PyTorch SDPA, Flash Attention, xFormers fmha) at the dispatch level
- Per-layer statistics logged for every linear layer: shape, mean, std, skew, kurtosis, and a fixed set of quantiles — for both forward tensors and gradients
- Attention logits and entropy logged for each attention module
- Writes a JSONL file; one entry appended per probed training step
- `log_stats(x, "name")` — inline hook to probe any arbitrary tensor anywhere in the model

```python
probe = AutoProbeD(model, "./probe.json")

for step, batch in enumerate(loader):
    with probe:
        probe.metadata = {"it": step}
        loss = model(batch)
        loss.backward()
    # stats for this step written to probe.json
```

Note: probing automatically disables `torch.compile` for the duration of the context.

## Training configs

Experiment configs for Lingua training runs live in `configs/lingua/` at the repo root, not inside this package.

## Relationship to Lingua

The [Lingua repo](https://github.com/r-three/lingua) provides the distributed training loop, optimizer, data pipeline, and evaluation harness. This package provides the model definitions and post-training tooling. Use these together: point Lingua at a config from `configs/lingua/` and it will import the model classes from this package.
