<div align="center">

![GPT Prototype — animated project overview](./docs/assets/readme/gpt-prototype-hero.svg)

</div>

[![PyTorch](https://img.shields.io/badge/PyTorch-transformer-ee4c2c?logo=pytorch&logoColor=white)](./model.py)
[![Parameters](https://img.shields.io/badge/scale-≈125M-4d8dff)](./model.py)
[![Context](https://img.shields.io/badge/context-1024_tokens-16a085)](./model.py)
[![Optimizer](https://img.shields.io/badge/optimizer-Muon_%2B_AdamW-b689ff)](./train.py)

**A decoder-only language model implemented to study the details that disappear behind high-level training frameworks.**

The repository contains the transformer, tokenizer/data preparation, memory-mapped training loop, hybrid optimizer split, checkpointing, generation, and experiment visualizations.

## Model card

| Component | Configuration |
|---|---|
| Architecture | Decoder-only causal transformer |
| Width / depth | 768 hidden size · 12 layers · 12 heads |
| Context | 1,024 tokens |
| Tokenizer | 4,096-token BPE configuration |
| Position | Rotary positional embeddings (RoPE) |
| Normalization | RMSNorm plus QK normalization |
| Feed-forward | Gated SiLU / SwiGLU-style MLP |
| Attention | PyTorch scaled dot-product causal attention |
| Stability | Zero-initialized attention output projection and logit soft-capping |
| Optimization | Muon for eligible 2D matrices; AdamW for embeddings, norms, and output head |

## Data and training path

```mermaid
flowchart LR
  MIX["FineWeb-Edu + Python code + math"] --> TOK["BPE tokenizer"]
  TOK --> BIN["memory-mapped train.bin"]
  BIN --> GPT["125M GPT"]
  GPT --> MUON["Muon matrix updates"]
  GPT --> ADAM["AdamW remaining params"]
  MUON --> CKPT["checkpoint + history"]
  ADAM --> CKPT
  CKPT --> GEN["top-k generation"]
```

The data scripts stream a 50/30/20 text/code/math mixture from Hugging Face datasets. Review each dataset’s license and access conditions before running them.

## Reproduce the pipeline

```bash
git clone https://github.com/ReaperXD67/GPT-Prototype.git
cd GPT-Prototype
python -m venv .venv
pip install -r requirements.txt

python train_tokenizer.py
python prepare_mixed_data.py
python train.py
```

Generation expects the tokenizer artifacts and a compatible checkpoint path configured in `generate.py`.

## Experiment artifacts

| Training status | Phase comparison |
|---|---|
| ![Phase 2 training status](./phase2_live_status.png) | ![Optimizer and architecture comparison](./comparison_final.png) |

## Reproducibility notes

- Training defaults are tuned for a memory-constrained laptop (`batch_size=1`), not throughput records.
- Large datasets and checkpoints are intentionally not committed.
- Report parameter count from `model.get_num_params()` for the exact tokenizer/configuration used.
- The included images are experiment artifacts; they are not independent benchmark results.
