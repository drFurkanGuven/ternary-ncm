"""ncm_ptq.py içeriğini gömerek ncm_ptq_asama01.ipynb dosyasını üretir. Çalıştır: python build_notebook.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
module_src = (HERE / "ncm_ptq.py").read_text(encoding="utf-8")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# NCM Sudoku — Aşama 0 ve 1: Yeniden üretim + PTQ taraması

**Amaç.** Resmî NCM Sudoku modelini (25M parametreli üreteç, 12 seviye) yükleyip makaledeki sonucu yeniden üretmek
(zor seviyede ≈%86.1 tamamen çözülen tahta), ardından üretecin ağırlıklarını **yeniden eğitmeden** INT8, INT4 ve
ternary (1.58-bit) değerlere çevirip kalitenin nasıl değiştiğini ölçmek.

**Araştırma sorusunun ilk yarısı:** *"Kabadan inceye üretimde eğitim sonrası ternary kuantalama çöker mi?"*

**Başlamadan önce:** Menüden `Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU` seç.

Notlar:
- Resmî repo dosyalarına dokunulmaz; ağırlıklar bellekte kuantalanır.
- Yalnızca üreteç kuantalanır, tokenizer (VQ-VAE) tam hassasiyette kalır.
- Sonuçlar Google Drive'a yazılır; oturum koparsa aynı hücreyi tekrar çalıştırınca kaldığı yerden devam eder.
"""),
    md("## 1. GPU ve Drive"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
import os
OUT_DIR = '/content/drive/MyDrive/ncm_ptq'
os.makedirs(OUT_DIR, exist_ok=True)
print('Sonuçlar:', OUT_DIR)
"""),
    md("## 2. Resmî repo, bağımlılıklar ve Sudoku checkpoint'i (yalnızca `sudoku/hard`, ≈160 MB)"),
    code("""
%cd /content
!test -d ncm || git clone https://github.com/HazyResearch/ncm.git
%cd /content/ncm
!git checkout -q c70274273f489abc73832b2b98584a8483b6b203
!git rev-parse HEAD > {OUT_DIR}/ncm_commit.txt && cat {OUT_DIR}/ncm_commit.txt
!pip install -q -U omegaconf hydra-core "transformers>=4.51" "huggingface_hub[cli]" regex rouge_score sacrebleu
!pip list 2>/dev/null | grep -iE "^(torch|transformers|numpy|omegaconf|hydra-core|huggingface.hub) " | tee /content/drive/MyDrive/ncm_ptq/env_versions.txt
!hf download hazyresearch/ncm-checkpoints --include "sudoku/hard/*" --local-dir artifacts
!hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/hard/*" --local-dir data
!ls -la artifacts/sudoku/hard data/sudoku/hard
"""),
    md("""
## 3. Aşama 0a — Resmî değerlendirme betiğiyle yeniden üretim

Hedef: `solved %` ≈ 86.1 (makale Tablo 3, zor seviye). Checkpoint'ler `artifacts/`, bulmacalar ve çözümler `data/`
klasöründe.
"""),
    code("""
%cd /content/ncm
!python evaluate.py --config config/eval/sudoku-hard.yaml --checkpoint-dir artifacts/sudoku/hard --data-dir data/sudoku/hard 2>&1 | tail -n 8
"""),
    md("## 4. PTQ modülünü yaz (`ncm_ptq.py`)"),
    code("%%writefile /content/ncm_ptq.py\n" + module_src),
    md("""
## 5. Aşama 0b — Kendi değerlendiricimiz resmî sonuçla aynı mı?

Aşağıdaki `fp` satırı, 3. adımdaki resmî çıktıyla **aynı** `solved %` değerini vermeli. Vermezse PTQ sonuçlarına
güvenme; önce farkın nedenini bul.
"""),
    code("""
import sys, importlib
sys.path.insert(0, '/content')
import ncm_ptq; importlib.reload(ncm_ptq)

ev = ncm_ptq.SudokuEvaluator(
    ncm_root='/content/ncm',
    config_path='/content/ncm/config/eval/sudoku-hard.yaml',
    ckpt_dir='/content/ncm/artifacts/sudoku/hard',
    data_dir='/content/ncm/data/sudoku/hard',
    n=0,            # 0 = config'teki tüm bulmacalar (2000). Hızlı deneme için 256 yaz.
)
print(len(ev.items), 'bulmaca, cihaz =', ev.device)
fp = ev.run('fp-check', None, 'all')
print(fp)
"""),
    md("## 6. Üreteçteki lineer katmanlar (kapsam filtrelerini kontrol etmek için)"),
    code("""
import pandas as pd
layers = pd.DataFrame(ncm_ptq.list_linear_layers(ev.net))
for sc, f in ncm_ptq.SCOPES.items():
    layers[sc] = layers['name'].map(f)
print('Toplam lineer parametre:', layers.params.sum())
print(layers.groupby(['body', 'attn', 'mlp', 'io_only']).params.agg(['count', 'sum']))
layers.head(12)
"""),
    md("""
## 7. Aşama 1 — PTQ taraması

| Şema | Açıklama |
|---|---|
| `int8_ch`, `int4_ch` | Simetrik absmax, çıkış kanalı başına ölçek |
| `tern_absmean_tensor` | BitNet b1.58: γ = mean\\|W\\| (tüm tensör), T = RoundClip(W/γ, −1, 1) |
| `tern_absmean_ch` | Aynı yöntem, kanal başına γ |
| `tern_twn_ch` | TWN eşiği 0.7·mean\\|W\\| + destek üzerinde yeniden hesaplanan ölçek (en güçlü basit PTQ) |
| `+a8` | Lineer girişlerine token başına INT8 aktivasyon kuantalaması |

| Kapsam | Hangi katmanlar |
|---|---|
| `all` | Tüm lineer katmanlar |
| `body` | Transformer blokları (giriş `wte` ve çıkış `lm_heads` hariç) |
| `attn` / `mlp` | Yalnızca dikkat / yalnızca MLP katmanları |
| `io_only` | Yalnızca giriş ve çıkış katmanları |

Her deney 2000 bulmacada birkaç dakika sürer. Sonuçlar her deneyden sonra Drive'a yazılır.
"""),
    code("""
OUT_JSON = f'{OUT_DIR}/sudoku_hard_ptq.json'
results = ncm_ptq.run_all(ev, OUT_JSON)
"""),
    md("## 8. Sonuç tablosu ve grafik"),
    code("""
import json, pandas as pd, matplotlib.pyplot as plt
df = pd.DataFrame(json.load(open(OUT_JSON)))
cols = ['name', 'solved_pct', 'cell_pct', 'valid_pct', 'rel_err', 'zero_frac', 'n_params', 'gen_seconds']
df = df[cols].round(3)
display(df)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(df['name'], df['solved_pct'], color='#4C72B0')
ax.axvline(float(df.loc[df.name == 'fp', 'solved_pct'].iloc[0]), color='k', ls='--', lw=1, label='FP')
ax.set_xlabel('Tamamen çözülen tahta (%) — Sudoku hard')
ax.invert_yaxis(); ax.legend(); plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sudoku_hard_ptq.png', dpi=150)
plt.show()
"""),
    md("""
## 9. Nasıl yorumlanır

- **`fp` ≈ `fp-check` ≈ resmî sonuç** olmalı. Değilse sonraki satırlar anlamsızdır.
- **`int8_ch` ≈ `fp`** beklenir. Belirgin bir düşüş, betikte bir hata olduğuna işaret eder.
- **Ternary satırları** araştırma sorusunun cevabıdır: ciddi düşüş, PTQ'nun yetmediğini ve QAT'nin (Aşama 2)
  gerektiğini gösterir. Düşüş küçükse, bu da yayımlanabilir ve ilginç bir bulgudur.
- **`attn` / `mlp` / `io_only`** satırları hangi katman grubunun en hassas olduğunu gösterir.
- `rel_err` = ‖W′ − W‖ / ‖W‖ ve `zero_frac` = ternary ağırlıklardaki sıfır oranı.
- Bu tek bir checkpoint ve tek bir veri bölümüdür; ilk sonuçlar ilginçse aynı taramayı `sudoku/easy` ve
  `sudoku/medium` için de çalıştır (2. adımdaki iki `--include` desenini ve yolları değiştirmen yeterli).

**Sonraki adım (Aşama 2):** Ternary PTQ çökerse, aynı üreteci BitLinear katmanlarıyla QAT ince ayarı yapmak.
Repoda eğitim kodu olmadığı için kayıp fonksiyonu ve eğitim döngüsü makaledeki Algoritma 1'e göre ayrıca yazılacak.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = HERE / "ncm_ptq_asama01.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("yazıldı:", out)
