# Assignment 4 — English to Hindi Neural Machine Translation

**Task:** Sequence-to-sequence translation using a Transformer model, with hyperparameter tuning via Ray Tune + Optuna.

## Model on Hugging Face

[Hugging Face Model](https://huggingface.co/bappaiitj/Assingment-4_mlops)



## Model Architecture

A standard Transformer encoder-decoder built from scratch in PyTorch.

| Component | Details |
|---|---|
| Architecture | Transformer (Encoder–Decoder) |
| Embedding dim (`d_model`) | 512 |
| Encoder/Decoder layers | 6 |
| Attention heads (baseline) | 8 |
| Feed-forward dim (baseline) | 1024 |
| Positional Encoding | Sinusoidal |
| Loss Function | CrossEntropyLoss (ignores padding) |

---

## Baseline Results

Trained for **100 epochs** with fixed hyperparameters.

| Metric | Value |
|---|---|
| Learning Rate | 1e-4 |
| Batch Size | 64 |
| Dropout | 0.1 |
| Optimizer | Adam |
| Final Loss | 0.0917 |
| BLEU Score | 52.47 |
| Training Time | ~85.17 minutes |

---

## Hyperparameter Tuning

Used **Ray Tune** with **Optuna** (TPE sampler) and **ASHA** early-stopping scheduler over 15 trials.

| Hyperparameter | Search Type | Range / Choices |
|---|---|---|
| Learning Rate | Log-Uniform | 1e-5 to 1e-1 |
| Batch Size | Categorical | {16, 32, 64} |
| Attention Heads | Categorical | {4, 8, 16, 32} |
| Feed-Forward Dim | Categorical | {1024, 2048} |
| Dropout | Uniform | 0.1 to 0.5 |
| Optimizer | Categorical | {adam, adamw} |
| Weight Decay | Log-Uniform | 1e-5 to 1e-2 |

### Best Configuration Found

```json
{
  "lr": 0.0001468,
  "batch_size": 32,
  "num_heads": 32,
  "d_ff": 2048,
  "dropout": 0.154,
  "optimizer": "adam",
  "weight_decay": 1.96e-05,
  "epochs": 25
}
```

---

## Tuned Model Results

Retrained from scratch using the best config for **25 epochs**.

| Metric | Baseline (100 ep) | Tuned (25 ep) | Improvement |
|---|---|---|---|
| Final Loss | 0.0917 | 0.1946 | — |
| **BLEU Score** | **52.47** | **69.39** | **+16.92 pts** |
| Training Time | ~85 min | ~15 min | ~5.7× faster |
| Epochs needed | 100 | 25 | 75% fewer |

---


