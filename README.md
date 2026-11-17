# Ternary NCM: Next-Concept Modeling with BitNet b1.58 weights

Can a coarse-to-fine text generator run on ternary weights {−1, 0, +1}? This repository tests that on
[Next-Concept Modeling (NCM)](https://arxiv.org/abs/2610.14162) using the **official NCM code and Sudoku-hard
checkpoint** ([HazyResearch/ncm](https://github.com/HazyResearch/ncm) @ `c7027427`) and
[BitNet b1.58](https://arxiv.org/abs/2402.17764)-style quantization (ternary weights, 8-bit activations).

> Independent research project. Not affiliated with or endorsed by the NCM authors.
> Turkish summary: [docs/README_TR.md](docs/README_TR.md) · Research log (Turkish): [docs/BULGULAR.md](docs/BULGULAR.md)

## Main results (Sudoku-hard, 2,000 test puzzles, % boards solved)

| Model | Solved % | |
|---|---|---|
| Official NCM, full precision | **86.15** (A100) / 86.30 (T4) | paper: 86.1 |
| Post-training quantization, INT8 weights | 86.10 | lossless |
| Post-training quantization, INT4 weights | 81.90 | −4.4 pt |
| Post-training quantization, **ternary** (every variant) | **0.00** | complete collapse |
| **Ternary QAT fine-tuning** from the FP checkpoint, 5k steps | **71.15 ± 0.40** (3 seeds) | 82.6% of FP, 55 min on one A100 |

**Why post-training ternarization fails.** The collapse is set by the *size* of the weight error, not by its
ternary structure: Gaussian noise with the same relative error ‖ΔW‖/‖W‖ has the same effect. The model
tolerates ≈0.28 relative error (50% solved) and fails from ≈0.35, while even the best possible per-layer
ternary quantizer has ≈0.44 on Gaussian weights.

**What does not close the remaining gap.** Distillation from the FP teacher, coarse-level loss reweighting
(α=1), per-channel scales, 16-bit activations, and keeping the most sensitive block (block 0) in FP all stay
within noise of 71%. INT8 activations cost nothing. The gap is spread over all levels, not concentrated in the
coarsest code.

Full table with every experiment: [docs/README_TR.md](docs/README_TR.md). Paper draft: [paper/](paper/).

## Repository layout

```text
colab/                     Everything that runs (Google Colab notebooks + the modules they embed)
  ncm_ptq_asama01.ipynb    Stage 0–1: reproduce the official result, PTQ sweep      ← ncm_ptq.py
  ncm_asama2a_veri.ipynb   Stage 2a: regenerate NCM's Sudoku training data exactly
  ncm_asama2b_qat.ipynb    Stage 2b: QAT fine-tuning, from-scratch training, diagnostics  ← ncm_qat.py, ncm_diag.py
  ncm_ptq.py               Quantizers (INT8/4, ternary absmean/TWN), evaluator, PTQ sweep
  ncm_qat.py               TernaryLinear (STE), NCM training loop (teacher forcing, corruption, level weights)
  ncm_diag.py              PTQ mechanism diagnostic (matched-error noise, per-level and per-block measurements)
  build_notebook*.py       Regenerate the notebooks from the .py modules
results_official/          One JSON per verified result (every number in the paper comes from here)
paper/                     LaTeX draft (main.tex, refs.bib)
docs/                      Research log and Turkish summary
```

Notebook text and code comments are in Turkish; module and function names are in English.

## Reproducing the results: step by step

Everything runs on Google Colab. Each notebook saves its outputs to Google Drive under `MyDrive/ncm_ptq/`.
If the session drops, re-run the same cell and it resumes from the last checkpoint. External code is pinned to a commit.

### Step 0: prerequisites
- A Google account with Colab. A T4 is enough for steps 1–2. Step 3 training needs an A100 (≈55 min per 5k-step run).
- A few GB of free Drive space (training checkpoints are saved every 500 steps).

### Step 1: reproduce the official result and the PTQ sweep (T4)
Open `colab/ncm_ptq_asama01.ipynb` and run all cells. It clones NCM @ `c7027427`, downloads the
`sudoku/hard` checkpoint (≈160 MB), and checks the official evaluator (expect 86.30% on T4). It then confirms
our evaluator gives the same number and runs the PTQ sweep.
→ `results_official/asama1_ptq_sudoku_hard.json`

### Step 2: regenerate the training data (CPU, long; a many-core runtime helps)
Open `colab/ncm_asama2a_veri.ipynb`. It builds the S-FLM Sudoku generator @ `30def799` (hard, seed 42, 48k
train + 2k validation) and verifies that NCM's 2,000 test puzzles equal the validation split
(2000/2000, same order, zero overlap with train).
→ `MyDrive/ncm_ptq/sudoku_hard_sflm_seed42.npz`, `results_official/asama2a_veri_dogrulama.json`

### Step 3: training and diagnostics (A100)
Open `colab/ncm_asama2b_qat.ipynb`. **Always run sections 1–4 first** (GPU and Drive, official repo, module,
data). Section 5 runs sanity checks, and section 6 sets batch size and measures speed. After that, each section
is independent:

| Section | Experiment | Time (A100) | Result file |
|---|---|---|---|
| 7 | E3: ternary QAT fine-tune, 1.5k steps | 15 min | `asama2b_E3_ternary_ft.json` |
| 8, 9 | E1 / E2: FP vs ternary from scratch, 4k steps | 34 / 40 min | `asama2b_E1_fp_scratch.json`, `asama2b_E2_ternary_scratch.json` |
| 9b | E1 / E2 long, 12k steps | ≈2 h each | `*_12k.json` |
| 9c | Memorization check (train vs test) | 5 min | `asama2b_diag_E1_long_train_vs_test.json` |
| 9d | LM1B-style corruption | ≈2 h | `asama2b_E1_fp_scratch_12k_lm1bcorr.json` |
| 9e–9h | E3 variants: distillation, α=1, W1.58A16, per-channel scale | 55 min each | `asama2b_E3_ternary_ft_5k_{kd_partial,alpha1,a16,pc}.json` |
| 11 | PTQ collapse mechanism (measurement only) | 20–30 min | `asama2b_diag_ptq_mechanism.json` |
| 12 | E3-long, 5k steps, seeds 1 and 2 (seed 0: same cell with `SEED=0`) | 55 min each | `asama2b_E3_ternary_ft_5k.json`, `asama2b_E3_ternary_ft_5k_seeds.json` |
| 13 | E3 with block 0 kept FP | 55 min | `asama2b_E3_ternary_ft_5k_b0fp.json` |

The coarse-code decomposition (`asama2b_diag_coarse_code_decomposition.json`) was run as a standalone cell.
Its procedure is described in the paper (Section 6).

**Statistics.** Binomial SE on 2,000 puzzles ≈ 1 pt; seed std of the main recipe 0.40 pt. Compare results only
against the FP number from the same GPU type (A100 86.15 vs T4 86.30).

### Step 4: build the paper
Upload `paper/main.tex` and `paper/refs.bib` to Overleaf (pdfLaTeX + BibTeX). Figures are drawn with pgfplots from
numbers in the source, so there are no image files.

## Known limitations
- One task (Sudoku-hard) and one 25M-parameter model.
- NCM has no public training script. Our fine-tuning hyperparameters are our own choices, and our from-scratch FP
  recipe does not reach the paper's 86% (it memorizes the training set). See the paper's Section 7.
- Quantization is simulated in floating point, so there are no speed or energy claims.
- Only the main QAT recipe has multiple seeds.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT (see [LICENSE](LICENSE)). NCM code and checkpoints are downloaded from the official repository at run time
and are not redistributed here.
