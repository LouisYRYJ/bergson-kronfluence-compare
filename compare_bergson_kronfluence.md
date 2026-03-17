# Bergson vs Kronfluence Comparison

Debugging notes from aligning influence scores and EKFAC covariance factors
between bergson and kronfluence.

## Differences Found and Fixed

### 1. SDPA Kernel Non-determinism (root cause of rank disagreement)

Flash and memory-efficient SDPA kernels produce subtly different **backward
gradients** depending on batch size, even when the forward outputs (losses) are
bit-identical. This was the dominant source of score disagreement (Spearman
rho ~0.73 before fix, 1.0 after).

**Fix** — disable non-deterministic kernels in both pipelines:

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
```

### 2. LM Head (embed_out) Tracking

Bergson registers hooks on `model.base_model` (the inner transformer), which
excludes the LM head (`embed_out`). For pythia-14m this is a 128x50304 = 6.4M
parameter module — 5.5x more gradient dimensions than the other 24 modules
combined.

Kronfluence was wrapping the full model, tracking all 25 `nn.Linear` modules.

**Fix** — dynamically discover modules from `model.base_model` to match bergson:

```python
prefix = model.base_model_prefix
tracked = [
    f"{prefix}.{n}" if prefix else n
    for n, m in model.base_model.named_modules()
    if isinstance(m, torch.nn.Linear)
]
task = LanguageModelingTask(tracked_modules=tracked)
```

### 3. Model Dtype

Bergson's pipeline loads the model in the precision specified by `--precision`
(fp32 in our comparison). Kronfluence's `construct_model` was defaulting to
bf16.

**Fix** — pass `torch_dtype=torch.float32`.

### 4. TF32 Matmuls

PyTorch defaults to using TF32 for fp32 matmuls on Ampere+ GPUs, which reduces
intermediate precision.

**Fix** — disable it:

```python
torch.backends.cuda.matmul.allow_tf32 = False
```

### 5. Query Gradient Aggregation (mean vs sum)

Bergson averages query gradients (`PreprocessConfig(aggregation="mean")`).
Kronfluence sums them (`aggregate_query_gradients=True`).

**Fix** — divide the loaded scores by query dataset size:

```python
if args.aggregate_query_gradients:
    scores = scores / len(query_dataset)
```

### 6. Attention Mask in Forward Call

Passing an explicit `attention_mask` (even all-1s) to the model forward call
triggers different SDPA kernel paths. Originally removed to match bergson
(which doesn't pass it).

**Current state** — restored now that flash/mem-efficient SDPA
are disabled. The math kernel handles the mask without changing behavior, and
the mask is needed for EKFAC covariance factor computation with variable-length
data.

### 7. Bias Augmentation in Covariance

Kronfluence appends a ones column to activations when a module has bias,
making activation covariance `[in_dim+1, in_dim+1]`. Bergson uses just
`weight_shape[1]` for `in_dim`, producing `[in_dim, in_dim]`.

**Fix** — set `include_bias=False` in kronfluence to match bergson.

### 8. Empirical vs True Fisher

Both libraries default to true Fisher (sampling labels from model output).
For consistent comparison, switched both to empirical Fisher (dataset labels).

**Fix** — set `use_empirical_fisher=True` in kronfluence's `FactorArguments`,
and `--use_dataset_labels` in bergson's hessian script.

---

## EKFAC Covariance Comparison

Tested with `Qwen/Qwen2.5-0.5B-Instruct` on variable-length medical
prompt/completion data (merged_medical.jsonl). Tracked layer 23 modules only
(7 modules: 4 attention + 3 MLP).

### 9. Gradient Masking at Invalid Positions

Kronfluence zeroed activations at masked (prompt/padding) positions via
`attention_mask` but did **not** zero gradients. Intermediate-layer gradients
at prompt positions are non-zero (due to attention backprop through the causal
mask), so they contaminated the gradient covariance.

Bergson physically selects valid positions: `g_bo = g[mask]` then
`g_bo.mT @ g_bo`, which naturally excludes invalid positions from both sides.

**Fix** — added gradient masking in `kronfluence/module/linear.py`
`get_flattened_gradient()`.

Before fix: k_proj grad_cov cosine = 0.838. After fix: 1.0.

### Covariance Match Results

After the gradient masking fix, all metrics match perfectly:
- All activation covariance cosines = 1.0
- All gradient covariance cosines = 1.0
- All lambda correlations = 1.0
- Eigenvector mismatches (e.g. mlp.down_proj mean|cos|=0.487) are explained
  by near-degenerate eigenvalues — not a bug, since lambdas match perfectly.

---

## EKFAC Influence Scores

### 10. Incomplete IVHP in apply_hessian.py

Bergson's `apply_hessian.py` was only computing the forward rotation into the
eigenbasis (`Q_S^T @ G @ Q_A / λ`) without rotating back to parameter space
(`Q_S @ result @ Q_A^T`). This produced near-zero correlation (-0.04) with
kronfluence.

**Fix** — added the back-rotation steps using `_transpose_matmul`.

After fix: Spearman rho = 0.9986.

---

## Performance Comparison

Profiled with identical PyTorch profiler settings.

### Steady-state timings (step 1, Qwen2.5-0.5B, batch_size=4)

|              | Bergson  | Kronfluence |
|--------------|----------|-------------|
| Forward      | 112 ms   | 46 ms       |
| Backward     | 176 ms   | 6 ms        |
| GPU memory   | ~25 GB   | ~18 GB      |

### Parameter Gradients Computed Unnecessarily (root cause: 30x backward)

Bergson freezes all parameters but re-enables `requires_grad` on input
embeddings. This forces the backward pass to compute parameter gradients
through **all** layers, then discards them with `model.zero_grad()`.

Kronfluence freezes everything and adds a tiny `_constant` tensor
(`requires_grad=True`) to each `TrackedModule`'s output. This keeps the
backward graph alive for hooks without computing any parameter gradients.

---

## Known Remaining Differences (minor)

### Padding Handling

The identity-strategy comparison uses a curated uniform-length dataset
to sidestep padding differences. Bergson uses custom `pad_and_tensor()` with
`padding_value=0`; kronfluence uses HF tokenizer `padding=True`.

---

## TODO / Ideas

### Freeze parameters during hessian collection (bergson)

Adopt kronfluence's `_constant` trick for full 30x backward speedup.

### Early backward pass termination

When only tracking a subset of layers, detach the hidden state at the input
to the earliest tracked block to stop gradient flow.
