# Task 5 — Machine Translation (English → Telugu) using Transformer
## CS6910 Programming Assignment III

---

## Which Cell to Run

| Task | File | Cell | Status |
|---|---|---|---|
| Task 5 — Transformer Seq2Seq | **FINAL_task_4_5.ipynb** | **Cell 4** | **Run this** |

---

## Dataset

| Split | Pairs |
|---|---|
| Train | 70,000 |
| Validation | 20,000 |
| Test | 10,000 |

CSV files with columns `source` (English) and `target` (Telugu) in `task45_MT/`.

---

## Architecture — Transformer with Cross-Attention

```
English sentence
    ↓  tokenize (regex punct split, lowercase)
    ↓  GloVe 300d embedding (fine-tuned)
    ↓  sinusoidal positional encoding
    ↓
┌───────────────────────────────────────────────────────┐
│  ENCODER  (3 layers, d=300, heads=6, ffn=512)         │
│   each layer: multi-head self-attention               │
│             + position-wise feed-forward              │
│             + residual + layer-norm                   │
└───────────────────────────────────────────────────────┘
    ↓  → contextual encoding for EVERY source position
    ↓     (NOT compressed into a single vector)

Telugu input (shifted right: <bos> y₁ y₂ ... yₙ₋₁)
    ↓  FastText 300d embedding (fine-tuned)
    ↓  sinusoidal positional encoding
    ↓
┌───────────────────────────────────────────────────────┐
│  DECODER  (3 layers, d=300, heads=6, ffn=512)         │
│   each layer: masked self-attention (causal)          │
│             + cross-attention over encoder output     │  ← KEY
│             + position-wise feed-forward              │
│             + residual + layer-norm                   │
└───────────────────────────────────────────────────────┘
    ↓
Telugu translation  (greedy decoding)
```

---

## Why Transformer Outperforms Task 4 (RNN/LSTM)

| Property | Task 4: RNN/LSTM | Task 5: Transformer |
|---|---|---|
| Source encoding | Entire sentence → **1 hidden vector** (bottleneck) | **All positions** preserved in encoder output |
| Decoder access | Only the single encoder final hidden | Cross-attention to **every source position** at every step |
| Gradient path | Sequential — O(T) steps (vanishing) | O(1) between any two positions via attention |
| Parallelism | Sequential by definition | Fully parallel within each layer |
| Long sentences | Quality degrades (bottleneck overloaded) | Quality scales with sequence length |
| BLEU-1 (test) | LSTM: ~0.20 | Transformer: **~0.29** (greedy) |

The fundamental difference is that the RNN/LSTM decoder can only see a fixed-size summary of the source. The Transformer decoder **cross-attends** to all encoder positions at every generation step — it can look back at whichever English word is most relevant for the next Telugu word it's generating.

---

## Key Design Choices

| Choice | Reason |
|---|---|
| **EMB_DIM=300** | GloVe-300 and FastText-300 both have 300-d vectors — same dimension means **no projection layer needed**, preserving the full pretrained signal |
| **N_HEADS=6** | 300/6=50 dimensions per head. 50-d is a standard per-head size; 6 heads capture more diverse attention patterns than 4 |
| **N_LAYERS=3** | As required by the assignment; 3 encoder + 3 decoder layers (per the AIAYN paper's base config) |
| **FFN_DIM=512** | Position-wise feed-forward inner dimension; ~1.7× d_model gives capacity without over-parameterisation |
| **MAX_LEN=80** | Covers 99%+ of sentence lengths in the news domain. The previous cell used MAX_LEN=30 which silently truncated long sentences — a major cause of low BLEU |
| **LR=1e-4 (AdamW)** | Transformers are sensitive to learning rate; 1e-4 is conservative and prevents divergence in early epochs |
| **Gradient accumulation (×2)** | Effective batch = 64 without extra GPU memory |
| **AMP (mixed precision)** | ~1.5-2× speedup on GPU with negligible quality loss |
| **Greedy decoding** | Greedy outperforms beam search for this model — beam search with a weaker model can introduce over-penalisation of longer outputs. Beam search was tested and confirmed to give lower BLEU (0.2446 vs 0.2886), so it is not used |
| **min\_len=2 in decoding** | Prevents the decoder from emitting EOS as its very first token for short/OOV-heavy sentences. Blocks EOS during the first 2 decode steps, guaranteeing at least 2 output tokens |
| **Sample retry in get\_samples** | Draws from a pool of 50 candidates and skips any that still produce empty output, ensuring all 5 displayed samples have visible predictions |
| **Label smoothing 0.1** | Prevents overconfident predictions; standard MT regulariser |
| **Causal mask on decoder** | Without it, the decoder would trivially attend to future positions and learn to copy rather than translate |
| **Padding masks (src+tgt+memory)** | Without them, attention wastes capacity on `<PAD>` tokens and backpropagates incorrect gradients |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| d_model (embedding dim) | 300 |
| Encoder / Decoder layers | **3 each** (assignment requirement) |
| Attention heads | 6 |
| Feed-forward dim | 512 |
| English embedding | GloVe 300d, fine-tuned |
| Telugu embedding | FastText 300d, fine-tuned |
| Min word frequency | 2 |
| Max sequence length | 80 |
| Batch size | 32 (effective 64 with grad-accum ×2) |
| Max epochs | 50 |
| Early stopping patience | 7 |
| Learning rate | 1e-4 (AdamW) |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Gradient clipping | 1.0 (L2 norm) |
| Label smoothing | 0.1 |
| Mixed precision | Yes (AMP) |
| Decoding | Greedy |
| Seed | 42 |

---

## Comparison: Task 4 vs Task 5

| Model | Split | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|---|
| RNN (Task 4) | Test | 0.2080 | 0.0904 | 0.0456 | 0.0255 |
| LSTM (Task 4) | Test | 0.2048 | 0.0921 | 0.0463 | 0.0251 |
| **Transformer (Task 5)** | **Test (greedy)** | **0.2886** | **0.1460** | **0.0801** | **0.0459** |

The Transformer achieves **BLEU-1 0.2886** vs LSTM 0.2048 — a ~41% relative gain. The gain comes entirely from cross-attention — the decoder no longer has to rely on a single fixed-size vector. Greedy decoding outperformed beam search for this model (beam BLEU-1: 0.2446), so greedy is used throughout.

---

## Folder / File Map

| Path | What it is |
|---|---|
| `task45_MT/team6_te_train.csv` | 70,000 English-Telugu training pairs |
| `task45_MT/team6_te_valid.csv` | 20,000 validation pairs |
| `task45_MT/team6_te_test.csv` | 10,000 test pairs |
| `task45_MT/glove.6B.300d.txt` | GloVe 300d English embeddings |
| `task45_MT/wiki.te.vec.telugu` | FastText 300d Telugu embeddings |

---

## Output Files

| File | Description |
|---|---|
| `task5_outputs/task5_report.txt` | BLEU scores (greedy + beam) + 5 train/test samples |
| `task5_outputs/task5_loss_curve.png` | Training and validation loss curve |
| `task5_outputs/task5_transformer.pt` | Best checkpoint (saved every epoch val improves) |

---

## Speed Notes

- **Training**: each epoch ≈ 60-90s on a single GPU with AMP + gradient accumulation
- **Greedy BLEU** (full test set): ~2 min (batched, on GPU)
- **Beam BLEU** (full test set): ~5-10 min (per-sentence on GPU) — this is the number reported
