# Task 3 — Sentiment Analysis by Fine-tuning BERT
## CS6910 Programming Assignment III

---

## Which Cell to Run

| Task | Cell | Status |
|---|---|---|
| Task 3 — BERT Sentiment | **Cell 16** |  **Run this** |

---

## Dataset

| Split | Size | Negative | Positive | Ratio |
|---|---|---|---|---|
| Train (after val split) | 5,040 | 4,687 | 353 | 13.3 : 1 |
| Validation (10% stratified) | 560 | 521 | 39 | 13.4 : 1 |
| Test | 1,400 | 1,292 | 108 | 12.0 : 1 |

Two classes: `Negative` (0) and `Positive` (1).  
Dataset is heavily imbalanced (~92% Negative / ~8% Positive).  
A trivial "predict-everything-Negative" classifier already scores **92.3% accuracy** — this is why accuracy alone is not the right metric.

---

## Architecture

```
tweet (raw text)
    ↓  light cleanup (URLs, @mentions, bare # removed)
    ↓  BertTokenizerFast (WordPiece)
    ↓  adds [CLS] ... tokens ... [SEP], pads/truncates to 96 tokens
token ids + attention mask
    ↓  BERT (bert-base-uncased)
       12 transformer layers, 768-d hidden, ~110M params
       fine-tuned end-to-end
    ↓  take [CLS] final hidden state  →  768-d sentence vector
    ↓  Linear(768 → 100) + Tanh          ← 100 nodes per assignment spec
    ↓  Dropout(0.3)
    ↓  Linear(100 → 2)                   ← Negative / Positive logits
```

---

## Key Design Choices

| Choice | Reason |
|---|---|
| `bert-base-uncased` | Canonical fine-tuning baseline; "uncased" lowercases input which suits noisy Twitter text |
| Fine-tune BERT end-to-end (not frozen) | Pretrained weights already encode English; fine-tuning nudges them toward sentiment without training from scratch |
| `[CLS]` hidden state (not `pooler_output`) | `pooler_output` adds an extra Tanh+Linear from pretraining that is unnecessary here; raw `[CLS]` is cleaner |
| Max sequence length 96 | Tweets average ~15 tokens; 96 covers >99% of inputs while keeping 12-layer attention cost manageable |
| Class-weighted cross-entropy | Without weights, plain CE collapses to predicting Negative always (92% "free" accuracy). Weights ($w_{Pos}=7.14$, $w_{Neg}=0.54$) make each Positive example contribute ~13× the gradient of a Negative one |
| Stratified train/val split | With 92/8 imbalance a random split may leave too few Positives in either half; stratification preserves the class ratio |
| Best model by validation macro F1 | Not accuracy — macro F1 treats both classes equally, which is correct when one class is rare |
| AdamW lr = 2e-5 | Standard BERT fine-tuning recipe; higher LRs catastrophically forget pretrained weights |
| Linear warmup (10% of steps) + decay | BERT fine-tuning is unstable in the first few hundred steps; warmup gives gradients time to settle |
| Weight decay 0.01 (not on bias/LayerNorm) | Standard AdamW recipe — bias and LayerNorm parameters are excluded from L2 penalty |
| Gradient clipping (L2 norm ≤ 1.0) | Prevents occasional large gradient spikes during BERT fine-tuning |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Base model | bert-base-uncased |
| Hidden layer (classification head) | 100 (per assignment) |
| Max sequence length | 96 |
| Batch size | 32 |
| Epochs | 4 |
| Optimizer | AdamW (lr = 2e-5, weight decay = 0.01) |
| LR schedule | Linear warmup (10%) → linear decay |
| Dropout | 0.3 |
| Gradient clip | 1.0 (L2 norm) |
| Class weights | Neg: 0.538, Pos: 7.139 |
| Validation split | 10% stratified |
| Seed | 42 |

---

## Why Not BLEU?

The assignment lists BLEU@k as the metric for all tasks. BLEU is an n-gram overlap metric for **generated text sequences** — it is undefined for a binary classification output (there is no token sequence to compare against a reference). For Task 3 we therefore report:

- **Macro F1** (primary) — treats both classes equally regardless of frequency
- Accuracy, per-class Precision / Recall / F1, Confusion Matrix

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 0.9679 |
| Test Macro F1 | 0.8857 |
| Test Weighted F1 | 0.9676 |
| Best Val Macro F1 | 0.8588 |

**Per-class (test):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.9815 | 0.9837 | 0.9826 | 1292 |
| Positive | 0.8000 | 0.7778 | 0.7887 | 108 |

**Confusion matrix (test):**
```
                 Pred Negative   Pred Positive
True Negative        1271             21
True Positive          24             84
```

21 false positives and 24 false negatives — roughly symmetric despite the 12:1 imbalance, showing the class-weighted loss worked as intended.

---

## Text Preprocessing

```python
URL_RE.sub(" ", t)        # remove URLs
HTML_RE.sub(" ", t)       # remove HTML tags
MENTION_RE.sub(" ", t)    # remove @user mentions
HASHTAG_RE.sub(" ", t)    # remove # symbol but keep the word
```

Stopwords and morphology are **not** removed — BERT's pretraining included function words and full word forms; stripping them would degrade the pretrained representations.

---

## Output Files

| File | Description |
|---|---|
| `task3_outputs/task3_report.txt` | Test metrics, per-class scores, confusion matrix |
| `task3_outputs/training_curves.png` | Loss, accuracy, macro F1 over 4 epochs (train + val) |
| `task3_outputs/confusion_matrix.png` | Confusion matrix heatmap on test set |
| `task3_outputs/task3_bert_model.pt` | Saved model weights + config + history |

---

## Folder / File Map

| Path | What it is |
|---|---|
| `task3_sentiment/train.csv` | Training tweets with sentiment labels |
| `task3_sentiment/test.csv` | Test tweets with sentiment labels |
| `task3_outputs/` | All outputs from Cell 16 |
