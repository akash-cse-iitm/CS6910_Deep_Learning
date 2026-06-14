# Tasks 1 & 2 — Image Captioning
## CS6910 Programming Assignment III

---

## Which Cell to Run in `FINAL_DL3.ipynb`

| Task | Cell | Dataset | Status |
|---|---|---|---|
| Task 1 — RNN | **Cell 5** | 4000-image team subset | **Run this** |
| Task 1 — RNN | Cell 1 | Full Flickr8k (8092 images) | Optional / exploratory |
| Task 2 — LSTM | **Cell 11** | 4000-image team subset | **Run this** |
| Task 2 — LSTM | Cell 7 | Full Flickr8k (8092 images) | Optional / exploratory |
| Preprocessing | Cell 4, Cell 10 | — |  Do NOT re-run |

### Folder / file map

| Path | What it is |
|---|---|
| `task12_IC/Images/` | All 8092 Flickr8k images |
| `task12_IC/Images_team/` | The 4000 images assigned to Team 6 (subset of `Images/`) |
| `task12_IC/team6_cap_map/captions.txt` | All 40 460 captions (8092 images × 5) |
| `task12_IC/team6_cap_map/captions_team.txt` | 20 000 captions for the 4000 team images (× 5) |
| `task12_IC/team6_cap_map/image_names.txt` | List of the 4000 team-assigned image filenames |
| `task12_IC/glove.6B.200d.txt` | GloVe 200-d pre-trained word vectors |

`Images_team/` and `captions_team.txt` were created by the preprocessing cells (Cell 4 / Cell 10) which have already been run — **do not run them again**.

---

## Architecture

### Stage 1 — VGG16 Feature Extraction (frozen, run once)

VGG16 (pretrained on ImageNet) is run with all weights frozen in `eval()` mode.
We extract the **fc7 layer output (4096-d)** from the VGG16 classifier — this is
richer than the raw convolutional features and is the standard choice for image
captioning.

The extraction happens once per image and the result is cached to disk
(`task1_outputs/vgg_features_4k.pkl`). Subsequent runs load from the cache
(~5 min the first time, instant afterwards).

```
Image (3×224×224)
    → VGG16 conv blocks (frozen)
    → VGG16 avgpool
    → flatten → 25 088-d
    → fc6 Linear(25088→4096) + ReLU
    → fc7 Linear(4096→4096)  + ReLU
    → 4096-d feature vector  (frozen, cached to disk)
```

### Stage 2 — Trainable Image Encoder

A small trainable network projects the 4096-d VGG features to a 256-d image
embedding that the decoder conditions on.

```
4096-d VGG feature
    → Linear(4096 → 256) + BatchNorm + ReLU + Dropout(0.3)
    → img_emb  (256-d, trained)
```

### Stage 3 — Decoder

#### Task 1 (Vanilla RNN)

```
h₀  =  tanh( Linear(256 → 50)(img_emb) )        # image → initial hidden state
img_proj  =  Linear(256 → 64)(img_emb)           # 64-d image signal, reused every step

For each timestep t:
  emb_t  =  GloVe_embed( word_t )                # 200-d word embedding (fine-tuned)
  x_t    =  concat( emb_t,  img_proj )           # 264-d  ← KEY: image at EVERY step
  h_t    =  RNN_cell( x_t,  h_{t-1} )            # hidden = 50
  y_t    =  Linear(50 → V)( Dropout(h_t) )       # logits over vocabulary
```

#### Task 2 (LSTM)

Identical to Task 1 except:
- `nn.RNN` → `nn.LSTM`
- **Two** hidden-state initialisers: `init_h` and `init_c` (LSTM needs both h and c)

```
h₀  =  tanh( Linear(256 → 50)(img_emb) )
c₀  =  tanh( Linear(256 → 50)(img_emb) )   ← extra cell-state init

For each timestep t:
  x_t  =  concat( embed(word_t),  img_proj )   # 264-d
  h_t, c_t  =  LSTM_cell( x_t,  (h_{t-1}, c_{t-1}) )
  y_t  =  Linear(50 → V)( Dropout(h_t) )
```

---

## Why Concatenation, Not Addition?

Previous attempts added a projected image vector to the word embedding:

```
x_t = embed(word_t) + img_inject(img_feat)   ← WRONG
```

This fails because the randomly-initialised `img_inject` weights produce noise in
the same 200-dimensional space as the GloVe vectors.  The model cannot distinguish
"image signal" from "word signal" and learns to ignore both — collapsing to the
most frequent word ("a") at every step.

**Concatenation keeps the two signals in separate dimensions:**

```
x_t = concat( embed(word_t)[200d],  img_proj(img_feat)[64d] )   # 264-d total
```

The RNN learns to use them independently.  The image dimensions retain their
meaning throughout training.

---

## Why Pre-Extract VGG Features?

Running the VGG16 forward pass inside every training batch (every epoch) wastes
~80% of training time on computation that produces identical results every run.

Pre-extracting once and caching:
- **~10× faster** training loop (encoder step eliminated from every batch)
- Features are deterministic — no randomness from dropout variations
- The 4 000-image cache file is only ~120 MB

---

## Word Embeddings

**GloVe 200-d** vectors initialise the word embedding table.  The embeddings are
**fine-tuned** during training (not frozen).

Why fine-tune?  In GloVe, the article "a" sits near the centroid of all English
words because it co-occurs with nearly everything.  A frozen "a" vector is
uninformative — the decoder cannot tell from it what kind of object is in the
image, so it keeps predicting "a" forever.  Fine-tuning lets the model shift the
"a" embedding away from the centroid and into the captioning domain.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Hidden size | 50 (fixed by assignment) |
| Embedding dim | 200 (GloVe) |
| Image embedding dim | 256 (after ImageEncoder) |
| Image per-step projection | 64-d (concatenated to word embed) |
| RNN/LSTM input size | 264 (200 word + 64 image) |
| Batch size | 64 |
| Epochs | 30 |
| Optimizer | Adam (lr = 1e-3) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Dropout | 0.3 (after RNN/LSTM output) |
| Label smoothing | 0.1 |
| Gradient clip | 5.0 (L2 norm) |
| Min word frequency | 5 |
| Max generation length | 22 tokens |
| Seed | 42 |
| Teacher-forcing schedule | 1.0 → 0.0 (linear over 30 epochs) |

### Scheduled Sampling

Teacher-forcing ratio decays linearly from 1.0 (full ground-truth) to 0.0
(fully autoregressive) over 30 epochs.

```
TF at epoch e  =  1.0  ×  (1 −  (e−1)/(EPOCHS−1))
```

By epoch 30 the decoder runs entirely on its own predictions, completely
eliminating the exposure-bias feedback loop that causes "a a a a …" collapse.

---

## Data

| Split | Images | Caption pairs |
|---|---|---|
| Train | 3 200 (80%) | ~16 000 |
| Test | 800 (20%) | ~4 000 |

Split is **by image** (not by caption) to prevent data leakage.
Vocabulary is built from training captions only (min frequency = 5).

---

## Inference

Greedy decoding with a **repetition penalty** (`REP_PENALTY = 1.3`):

```
for each step t:
    logits  ←  fc( dropout( h_t ) )
    logits[ already_seen_token_ids ] /= 1.3    # penalise repeats
    next_word  =  argmax( logits )
    generation stops at <eos> or after 22 tokens
```

The penalty divides the logit of any token that has already been emitted,
making it ~23 % less likely to be selected again.  This eliminates the
"a a a a …" tail that arises when the model becomes uncertain after a few
tokens and collapses to the most-frequent article.  The penalty is applied
at inference time only — no retraining required.

---

## Expected Results

| Metric | Task 1 (RNN) | Task 2 (LSTM) |
|---|---|---|
| BLEU-1 | ~0.50 – 0.58 | ~0.54 – 0.62 |
| BLEU-2 | ~0.22 – 0.30 | ~0.26 – 0.35 |
| BLEU-3 | ~0.09 – 0.16 | ~0.12 – 0.20 |
| BLEU-4 | ~0.04 – 0.08 | ~0.06 – 0.12 |

LSTM outperforms RNN most on BLEU-3/4 because the cell state retains context
across more timesteps — exactly what higher-order n-gram matches require.

---

## Key Design Choices

| Choice | Reason |
|---|---|
| Pre-extract VGG fc7 features | 10× faster training; deterministic features |
| Freeze VGG weights | Only ~3200 train images — fine-tuning 138 M params would overfit |
| Concatenate image to every decoder step | Prevents forgetting: with hidden=50, image info in h₀ decays within 3–4 steps |
| Fine-tune GloVe embeddings | Frozen "a" sits at centroid → collapses to "a a a"; fine-tuning breaks this |
| Scheduled sampling TF 1.0 → 0.0 | Fully eliminates exposure-bias feedback loop by epoch 30 |
| Dropout 0.3 | Prevents overconfidence on frequent tokens |
| Label smoothing 0.1 | Smooths the probability mass away from trivial predictions |
| ReduceLROnPlateau | Adaptive LR decay — only fires when loss plateaus |
| Split by image, not caption | Prevents the same image appearing in both train and test |
| Build vocab from train only | Prevents test label leakage |
| Repetition penalty 1.3 (inference) | Eliminates "a a a a" tails when model is uncertain; no retraining needed |

---

## Output Files

| File | Description |
|---|---|
| `task1_outputs/vgg_features_4k.pkl` | Cached VGG fc7 features (shared by Task 1 and Task 2) |
| `task1_outputs/task1_report.txt` | 5 train + 5 test samples with GT and predictions, BLEU scores |
| `task1_outputs/task1_loss_curve.png` | Training loss over epochs |
| `task1_outputs/task1_model.pt` | Saved weights (encoder + decoder + vocab) |
| `task2_outputs/task2_report.txt` | Same as above for Task 2 |
| `task2_outputs/task2_loss_curve.png` | Training loss over epochs |
| `task2_outputs/task2_model.pt` | Saved weights |
