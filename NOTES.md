# Engineering Notes

Decisions, data quirks, and the things that will cost time if you do not know them. Read before touching the code.

Authors: Mohammed Al Sheqaih, Abdulrahman Ammar, Naif Alenazi
Supervisor: Dr. Hamzah Luqman

---

## Layout

```
├── README.md
├── NOTES.md
├── report.pdf
├── requirements.txt
├── generate_figures.py            rebuilds every figure from the recorded epoch numbers
├── notebooks/
│   ├── 01-bilstm-and-conformer-ctc.ipynb
│   ├── 02-transformer-ctc.ipynb
│   └── 03-conformer-seq2seq.ipynb
├── figures/                       PNG, used by the README
└── report/
    ├── main.tex
    └── figures/                   PDF, used by the report
```

The corpus is **not** in this repository and is excluded by `.gitignore`, along with all model checkpoints.

---

## Dataset quirks

**500 training samples have no pose record.** `train.csv` lists 10,000 ids, but the pose pickle does not contain all of them. Notebook 01 filters the mismatches, leaving **9,500** usable training sentences. Any count you take from the CSV rather than from the intersection will be wrong by 500.

**Sequence lengths vary enormously.** Pose sequences run from tens to hundreds of frames, which is why batching needs a length-aware collate function and why padding masks matter throughout. A batch padded to its longest member routinely wastes most of its tensor.

**86 joints, 2 coordinates, no depth.** Keypoints are `(x, y)` pixel positions from MediaPipe Holistic covering hands, lips and body. There is no z-axis, so anything requiring true depth is unavailable.

**Velocity is added, not given.** The raw pose is `T × 86 × 2 = 172` values per frame. The feature builder appends per-frame differences, giving **344** features per frame. This is worth knowing before you try to reshape a feature vector back to joint coordinates.

**Vocabulary is 675 gloss tokens**, 677 including specials, and CTC adds a blank at index 677 for an output layer of 678. The Seq2Seq notebook builds its own vocabulary with SOS and EOS and lands at 679, so the two are not interchangeable.

---

## Architecture notes

### BiLSTM + CTC — the baseline

Flattens each frame to a vector and runs a bidirectional LSTM. Best dev WER **66.62%** at epoch 34.

Two structural limits, and they are the reason the Conformer wins:

- **Flattening destroys geometry.** Joint-to-joint relationships, such as hand-to-face distance, carry the linguistic signal, and a flat vector gives the model no notion that two of its inputs are spatially adjacent.
- **A single recurrent state is a bottleneck.** Relating a hand shape early in a sentence to a movement much later requires carrying information through every intervening step.

Training was **manually interrupted at epoch 37** once the curve flattened, to spend the remaining Colab GPU budget on stronger architectures. Treat 66.62% as a floor rather than a converged result.

### Conformer + CTC — the model that shipped

Self-attention relates any two frames directly; the convolutional module captures local motion between adjacent frames. 9,400,230 parameters. Trained 40 epochs with OneCycleLR, best dev WER **13.04%** at epoch 39, and the best checkpoint is explicitly reloaded before evaluation.

This is the only model run on the 3,800-sentence test set.

### Transformer + CTC

Self-attention without the convolutional module. Best dev WER **36.34%** with beam search, at **epoch 30 of 30** — the last epoch, so it was still improving when the budget ran out. The gap to the Conformer is real but overstated by this number.

Its own error analysis notes a high deletion rate: the model predicts fewer words than the reference.

### Conformer + Seq2Seq — lowest WER, not deployable

Same encoder, an autoregressive Transformer decoder with teacher forcing, cross-entropy with label smoothing 0.1. Best dev WER **8.76%** at epoch 23.

**Read the decoding path before believing that number.**

`evaluate()` truncates each prediction at its first EOS token before scoring, so any text generated after EOS is discarded and never reaches the WER computation. The qualitative error analysis measures the raw generation instead, and finds an **average length difference of +10.86 tokens**, failure to emit EOS, and repetitive loops.

Both observations are correct; they describe different decoding paths. The WER sees a cleaned-up sequence, the error analysis sees what the model really produces.

Two further details to keep in view:

- The notebook defines `decode_batch` twice. The first version, used during training, stops only when **every** sequence in the batch emits EOS, and otherwise runs the full 50 steps. A later cell redefines it with a repetition penalty and per-sequence EOS stopping. The error analysis ran after that redefinition, so it did not use the decoder that produced the training log.
- The training cell has stored output but **no execution count**, so its log predates the notebook's current state.

The project's conclusion stands on this: the deployed model is the Conformer with CTC at 13.04%, not the Seq2Seq at 8.76%.

---

## Why CTC suits this task

CTC assumes the output is monotonically aligned with the input, which is true of sign language: signs are produced in sequence and the gloss order follows the video order. That constraint is a useful prior here, and it is exactly what the autoregressive decoder lacks. Free to attend anywhere, on 9,500 sentences it learned to produce fluent-looking Arabic that had stopped depending on the pose input.

---

## Evaluation

WER is computed with `jiwer` against the reference gloss sequence:

```
WER = (S + D + I) / N
```

substitutions, deletions and insertions from the Levenshtein alignment, over the number of reference tokens. Every figure in the README is dev WER on 949 sentences unless stated otherwise.

---

## Reproducing the figures

```bash
python generate_figures.py
```

Writes PNG to `figures/` and PDF to `report/figures/`. The per-epoch numbers are transcribed from the notebook outputs rather than recomputed, so the script runs without the corpus or a GPU.

**Arabic labels need shaping.** Matplotlib draws a string codepoint by codepoint, left to right, with no OpenType shaping, so Arabic comes out as disconnected letters in reverse order. The `ar()` helper reshapes and bidi-reorders each label through `arabic-reshaper` and `python-bidi` before it reaches the plot. Those two packages are optional: without them the helper returns its input unchanged and the script still runs, but every Arabic label in figures 5 and 6 will be wrong. Install them before regenerating.

---

## Building the report

```bash
cd report && pdflatex main.tex && pdflatex main.tex
```

Twice, so the table of contents and cross-references resolve.
