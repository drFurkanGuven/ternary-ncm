"""Aşama 2a veri notebook'unu (ncm_asama2a_veri.ipynb) üretir. Çalıştır: python build_notebook_asama2a.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SFLM_COMMIT = "30def79976ca3b1cdcb5fcbfa0165524e44001de"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md(f"""
# Aşama 2a — NCM Sudoku eğitim verisini yeniden üret ve doğrula

**Amaç.** Ternary QAT (Aşama 2b) için gereken Sudoku eğitim verisini, NCM'nin kullandığı kurulumla aynı şekilde üretmek.

**Kaynak zinciri.**
- NCM makalesi (Ek B, D.3): Sudoku verisi ve değerlendirme S-FLM kurulumunu izliyor.
- S-FLM (Deschenaux & Gulcehre, arXiv 2605.11125) Ek C.3 ve `configs/data/sudoku.yaml`: zorluk başına ayrı veri seti,
  **48.000 eğitim + 2.000 doğrulama** bulmacası, `data_seed: 42`, tekil çözüm garantili; zor seviyede 30 ipucu.
- Üretici: S-FLM reposundaki `sudoku_generator.py` (Apache-2.0; çekirdeği Ali Alp'in üreticisinden uyarlanmış),
  commit `{SFLM_COMMIT[:12]}`.

**Hipotez.** NCM'nin 2000 test bulmacası, S-FLM'nin 2000 doğrulama bulmacasıyla aynıdır. Bu notebook bunu kontrol eder.
- Aynıysa: elimizde NCM'nin eğitim verisinin birebir aynısı var, ve test kümesiyle örtüşmediği garanti.
- Aynı değilse: yine aynı dağılımdan veri üretmiş oluruz, ama eğitim ve test kümelerinin örtüşmediğini ayrıca kontrol ederiz.

**Çalışma zamanı:** GPU gerekmez (üretim CPU'da yapılıyor). İlk parçadan sonra kalan süre tahmini yazdırılır.
Parçalar Drive'a kaydedildiği için oturum koparsa aynı hücre kaldığı yerden devam eder.
"""),
    md("## 1. Drive ve ayarlar"),
    code(f"""
from google.colab import drive
drive.mount('/content/drive')
import os
OUT_DIR = '/content/drive/MyDrive/ncm_ptq'
os.makedirs(OUT_DIR, exist_ok=True)

SFLM_COMMIT = '{SFLM_COMMIT}'
DIFF = 'hard'          # 'easy' (40 ipucu) / 'medium' (35) / 'hard' (30)
SEED = 42              # S-FLM configs/data/sudoku.yaml: data_seed
N_TRAIN, N_VALID = 48_000, 2_000
print('CPU çekirdeği:', os.cpu_count())
"""),
    md("## 2. S-FLM üreticisini sabit commit'ten indir"),
    code("""
!mkdir -p /content/sflm
!curl -sfL https://raw.githubusercontent.com/jdeschena/s-flm/{SFLM_COMMIT}/sudoku_generator.py -o /content/sflm/sudoku_generator.py
!sha256sum /content/sflm/sudoku_generator.py | tee {OUT_DIR}/sudoku_generator.sha256
import sys
sys.path.insert(0, '/content/sflm')
import sudoku_generator as sg
CLUES = sg.DIFFICULTY_TO_CLUES[DIFF]
print(DIFF, '->', CLUES, 'ipucu')
"""),
    md("""
## 3. Bulmacaları üret (parça parça, Drive'a kaydederek)

S-FLM'nin `_generate_raw_grids` fonksiyonu, `SEED, SEED+1, …` tohumlarıyla sırayla bulmaca üretir, aynı çözümü iki kez
almaz ve ilk `N_TRAIN + N_VALID` benzersiz bulmacada durur. Aşağıdaki döngü aynı tohum sırasını ve aynı tekilleştirmeyi
kullanır; tek fark, ilerlemeyi parçalar hâlinde Drive'a kaydetmesidir. `pool.map` sırayı koruduğu için sonuç
çekirdek sayısından bağımsızdır.
"""),
    code("""
import time, numpy as np
from multiprocessing import Pool

TOTAL = N_TRAIN + N_VALID
CHUNK = 2000
CACHE = f'{OUT_DIR}/sudoku_{DIFF}_seed{SEED}_chunks'
os.makedirs(CACHE, exist_ok=True)

puzzles, solutions, seen = [], [], set()
i, t0, fresh = 0, time.time(), 0
with Pool(os.cpu_count()) as pool:
    while len(puzzles) < TOTAL:
        path = f'{CACHE}/chunk_{i:04d}.npz'
        if os.path.exists(path):
            d = np.load(path); P, S = d['P'], d['S']
        else:
            tasks = [(SEED + i * CHUNK + j, CLUES) for j in range(CHUNK)]
            res = pool.map(sg._generate_one, tasks, chunksize=8)
            P = np.array([[v for row in pz for v in row] for pz, _ in res], dtype=np.int8)
            S = np.array([[v for row in so for v in row] for _, so in res], dtype=np.int8)
            np.savez_compressed(path, P=P, S=S)
            fresh += 1
        for p, s in zip(P, S):
            key = tuple(s.tolist())
            if key in seen:
                continue
            seen.add(key); puzzles.append(p); solutions.append(s)
            if len(puzzles) >= TOTAL:
                break
        i += 1
        el = time.time() - t0
        eta = (el / max(fresh, 1)) * max(0, (TOTAL - len(puzzles)) / CHUNK) if fresh else 0
        print(f'parça {i:3d}: {len(puzzles):6d}/{TOTAL} benzersiz | geçen {el/60:5.1f} dk | kalan ≈ {eta/60:5.1f} dk', flush=True)

puzzles, solutions = np.stack(puzzles), np.stack(solutions)
clue_counts = (puzzles != 0).sum(1)
print('ipucu sayısı min/max:', clue_counts.min(), clue_counts.max())
"""),
    md("""
## 4. S-FLM ile aynı karıştırma, bölme ve token formatı

Format (S-FLM `_tokenize_grids`, NCM `SudokuTokenizer` ile aynı): `[BOS] bulmaca(89) [BOS] çözüm(89)` = 180 token.
Satırlar arasında `|` (id 10), boş hücre 0, `BOS` = 11. NCM'de önek (prefix) ilk 91 token, hedef son 89 token.
"""),
    code("""
import random
BOS, SEP = 11, 10

def tokenize(p, s):
    ids = [BOS]
    for grid in (p.reshape(9, 9), s.reshape(9, 9)):
        for r in range(9):
            ids.extend(int(v) for v in grid[r])
            if r < 8:
                ids.append(SEP)
        ids.append(BOS)
    ids.pop()
    assert len(ids) == 180
    return ids

rng = random.Random(SEED)
idx = list(range(TOTAL)); rng.shuffle(idx)
seqs = np.array([tokenize(puzzles[k], solutions[k]) for k in idx], dtype=np.int8)
train_seqs, valid_seqs = seqs[:N_TRAIN], seqs[N_TRAIN:]
print(train_seqs.shape, valid_seqs.shape)
print('örnek önek:', ''.join('_123456789|B'[t] for t in valid_seqs[0][:91]))
"""),
    md("## 5. NCM'nin test bulmacalarıyla karşılaştır"),
    code("""
!pip install -q -U "huggingface_hub[cli]"
!hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/{DIFF}/*" --local-dir /content/ncm_data
import pickle
ncm_items = pickle.load(open(f'/content/ncm_data/sudoku/{DIFF}/puzzles.pickle', 'rb'))
ncm_refs = pickle.load(open(f'/content/ncm_data/sudoku/{DIFF}/solutions.pickle', 'rb'))
print('NCM test:', len(ncm_items), 'öğe; ilk öğe tipi/uzunluğu:', type(ncm_items[0]).__name__, len(ncm_items[0]))
print('NCM çözüm örneği tipi:', type(ncm_refs[0]).__name__, str(ncm_refs[0])[:60])

ncm_prefix = [tuple(int(t) for t in list(it)[:91]) for it in ncm_items]
valid_prefix = [tuple(int(t) for t in row[:91]) for row in valid_seqs]
train_prefix = {tuple(int(t) for t in row[:91]) for row in train_seqs}
vset = set(valid_prefix)

in_valid = sum(p in vset for p in ncm_prefix)
in_train = sum(p in train_prefix for p in ncm_prefix)
same_order = ncm_prefix == valid_prefix[:len(ncm_prefix)]
print(f'NCM testindeki bulmacalardan S-FLM doğrulama kümesinde olan: {in_valid}/{len(ncm_prefix)}')
print(f'NCM testindeki bulmacalardan bizim EĞİTİM kümemizde olan  : {in_train}/{len(ncm_prefix)}  (0 olmalı)')
print('Sıra da aynı mı:', same_order)
"""),
    md("""
**Yorum:**
- `in_valid = 2000/2000` ise hipotez doğrulandı: NCM'nin eğitim verisini birebir yeniden ürettik.
- `in_train` mutlaka **0** olmalı; değilse eğitim ve test kümesi örtüşüyor demektir ve bu veri kullanılmamalı.
- `in_valid` düşükse ama `in_train = 0` ise, veri aynı dağılımdan geliyor ve test kümesiyle örtüşmüyor; yine
  kullanılabilir, ama makalede "NCM'nin verisiyle aynı" denemez, "aynı üretici ve ayarlarla" denir.
"""),
    md("## 6. Kaydet"),
    code("""
out = f'{OUT_DIR}/sudoku_{DIFF}_sflm_seed{SEED}.npz'
np.savez_compressed(out, train=train_seqs, valid=valid_seqs,
                    meta=np.array(json.dumps({'difficulty': DIFF, 'clues': CLUES, 'seed': SEED,
                                              'n_train': N_TRAIN, 'n_valid': N_VALID,
                                              'sflm_commit': SFLM_COMMIT,
                                              'ncm_test_in_valid': int(in_valid),
                                              'ncm_test_in_train': int(in_train),
                                              'same_order': bool(same_order)})))
print('kaydedildi:', out, f'({os.path.getsize(out)/1e6:.1f} MB)')
"""),
]
cells[-1]["source"].insert(0, "import json\n")

nb = {
    "cells": cells,
    "metadata": {"colab": {"provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = HERE / "ncm_asama2a_veri.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("yazıldı:", out)
