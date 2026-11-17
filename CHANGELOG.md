# Changelog

## [1.0.0] - 2026-11-17

First archived release, accompanying the preprint "Can Coarse-to-Fine Text Generators Run on Ternary Weights?".

- Colab notebooks and the modules they embed: PTQ sweep (stage 0–1), exact regeneration of the Sudoku-hard data
  (stage 2a), ternary QAT fine-tuning, from-scratch training and diagnostics (stage 2b).
- One JSON file per reported result in `results_official/`.
- LaTeX source of the paper (v1.0) in `paper/`.
- External code is pinned: HazyResearch/ncm @ `c7027427` (downloaded at run time, not redistributed),
  S-FLM Sudoku generator @ `30def799`.
