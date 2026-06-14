# Task 4 — Machine Translation (English → Telugu) using RNN / LSTM
## CS6910 Programming Assignment III

---

## Which Cell to Run

| Task | File | Cell | Status |
|---|---|---|---|
| Task 4 — RNN + LSTM Seq2Seq | **FINAL_task_4_5.ipynb** | **Cell 2** |  **Run this** |

Trains **both** a vanilla RNN seq2seq and an LSTM seq2seq back-to-back in the same cell. Results for both are printed and saved to `task4_outputs/`.

---

## Dataset

| Split | Pairs |
|---|---|
| Train | 70,000 |
| Validation | 20,000 |
| Test | 10,000 |

CSV files with columns `source` (English) and `target` (Telugu) in `task45_MT/`.

---

## Architecture — Pure Seq2Seq, NO Attention (intentional)

```
English sentence
    ↓  tokenize (lowercase, punct split)
    ↓  GloVe 300d embedding (fine-tuned)
    ↓
Bidirectional RNN / LSTM Encoder  (N_LAYERS=2, HIDDEN=512 per direction)
    ↓  pack_padded_sequence — final hidden taken at TRUE sentence end
    ↓  for each layer l: concat(fwd[l], bwd[l]) → Linear(1024→512) → tanh
    →  decoder init state: (N_LAYERS=2, B, 512)

          ← NO cross-attention, NO context vector →

Unidirectional RNN / LSTM Decoder  (N_LAYERS=2, HIDDEN=512)
    input at step t:  FastText_embed(prev_word)   [300-d only]
    output:           Linear(512 → |Telugu vocab|) → logits
    ↓
Telugu translation  (beam search, beam=5, length-norm α=0.7)
```

> **Why no attention here?**  
> Attention is Task 5's story (Transformer cross-attention). Task 4 deliberately
> uses only the encoder's **final hidden state** as a fixed-size bottleneck so the
> RNN vs LSTM comparison is clean and pedagogically meaningful.

---

## Why the Results Were Bad Before (V1 Analysis)

The V1 outputs showed catastrophic repetition (e.g. "ఇంకా" repeated 48 times
on a 7-word sentence). Three root causes:

| Problem | Root Cause | Fix Applied |
|---|---|---|
| Repetition loops | Greedy decoding — once a token is predicted it raises its own probability | **Beam search** (beam=5, length norm) |
| Low BLEU-3/4 | Model too small (256 hidden, 1 layer) — insufficient capacity | **HIDDEN=512, N_LAYERS=2** |
| Wrong final hidden | No `pack_padded_sequence` — RNN ran over padding zeros | **`pack_padded_sequence`** |
| Tokenisation mismatch | Telugu tokeniser was bare `split()` — missed punctuation tokens in vocab | **Regex-based tokeniser** |
| Overconfident predictions | No label smoothing | **label_smoothing=0.1** |

---

## RNN vs LSTM — What the Experiment Shows

Both models use the same architecture. The only difference is the recurrent cell.

| Property | Vanilla RNN | LSTM |
|---|---|---|
| Hidden state | Single vector $h_t$ | Cell state $c_t$ + hidden $h_t$ |
| Gradient path through time | Multiplicative chain → **vanishing gradients** for long sequences | Near-linear through $c_t$ (forget gate controls flow) |
| Source compression | Final $h_T$ forgets early words for longer sentences | $c_T$ retains more of the full sentence via gating |
| Expected result | Lower BLEU, especially BLEU-3/4 (long n-gram overlap) | Higher BLEU, particularly on longer sentences |

**Expected LSTM advantage:** The encoder's job is to compress the full source sentence into a fixed-size vector. With only $h_t$ (RNN), vanishing gradients cause early source words to be underweighted by the end of encoding. The LSTM's cell state provides a learnable highway for gradients, so later encoder layers retain more signal from early positions → better initialisation for the decoder → higher BLEU-3 and BLEU-4.

---

## Key Design Choices

| Choice | Reason |
|---|---|
| **Bidirectional encoder** | Forward pass sees left context; backward pass sees right context. The concatenated final hidden gives the decoder both directions of source context |
| **`pack_padded_sequence`** | Without packing, the forward RNN's last step is at `max_src_len` (past the real sentence end, over zero-padded tokens). Packing ensures the final hidden is captured at the TRUE sentence boundary — critical for correctness without attention |
| **No attention (deliberate)** | Forces all source information into a single fixed-size vector, making the RNN vs LSTM comparison clean. Task 5 adds cross-attention via Transformer to fix this bottleneck |
| **Beam search (beam=5, α=0.7)** | Greedy decoding causes repetition loops: the model's probability distribution at step t is conditioned on its own earlier outputs, so once a token is repeated it self-reinforces. Beam search explores multiple hypotheses simultaneously; length normalisation (÷len^0.7) prevents preferring short degenerate sequences |
| **GloVe 300d for English** | Pretrained on 840B tokens; captures rich semantic relationships between English words |
| **FastText 300d for Telugu** | Uses character n-gram subwords; essential for morphologically rich Telugu where the same root takes hundreds of inflectional forms that would be `<UNK>` with word-level embeddings |
| **Fine-tune both embeddings** | 70k pairs gives sufficient signal to adapt pretrained vectors to the news translation domain |
| **Teacher forcing decay 0.70→0.50** | At epoch 1, 70% ground-truth inputs stabilise early training. Gradual decay towards 50% reduces exposure bias (the gap between training-time teacher-forced inputs and test-time model-generated inputs) |
| **AdamW + weight decay 1e-4** | AdamW decouples weight decay from the adaptive gradient update, giving better regularisation than vanilla Adam |
| **Label smoothing 0.1** | Prevents the model from becoming overconfident on the training set; acts as a soft regulariser and typically improves BLEU by 0.3–0.8 points |
| **AMP (mixed precision)** | Uses FP16 for forward pass and FP32 for weight updates; ~1.5–2× faster on GPU with negligible quality loss |
| **Gradient accumulation (×2)** | Effective batch = 128 without doubling GPU memory; stabilises gradient estimates |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Encoder | Bidirectional RNN or LSTM, **2 layers**, hidden=512 |
| Decoder | Unidirectional RNN or LSTM, **2 layers**, hidden=512 |
| Encoder output used | Final hidden state only (no sequence output) |
| English embedding | GloVe 300d, fine-tuned |
| Telugu embedding | FastText 300d, fine-tuned |
| Min word frequency | 2 |
| Batch size | 64 (effective 128 with grad-accum ×2) |
| Max epochs | 50 |
| Early stopping patience | 7 |
| Learning rate | 3e-4 (AdamW) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Gradient clipping | 1.0 (L2 norm) |
| Teacher forcing | Decay from 0.70 → 0.50 linearly |
| Label smoothing | 0.1 |
| Mixed precision | Yes (AMP) |
| Beam size | 5 |
| Length normalisation α | 0.7 |
| Max generation length | 60 tokens |
| Seed | 42 |

---

## The Fixed-Size Bottleneck Problem

Without attention, the encoder must compress an entire source sentence into
a single vector of dimension 512. For short sentences (≤ 10 tokens) this works
reasonably well. For longer sentences the bottleneck limits quality:

- **RNN**: vanishing gradients through the recurrence mean the final hidden
  state is dominated by the last few source words; early words are underweighted.
- **LSTM**: the cell state $c_t$ acts as a protected memory lane — the forget
  gate learns to preserve important information across many timesteps. This
  mitigates (but does not eliminate) the bottleneck.

This is precisely why Task 5's Transformer (with full cross-attention over all
encoder positions at every decoder step) outperforms both.

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
| `task4_outputs/task4_report.txt` | BLEU scores + 5 train/test samples for RNN and LSTM |
| `task4_outputs/task4_loss_curves.png` | Training and validation loss curves for both models |
| `task4_outputs/task4_rnn.pt` | Saved RNN seq2seq weights |
| `task4_outputs/task4_lstm.pt` | Saved LSTM seq2seq weights |
