"""ncm_qat.py'yi gömerek Aşama 2b notebook'unu (ncm_asama2b_qat.ipynb) üretir. Çalıştır: python build_notebook_asama2b.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
module_src = (HERE / "ncm_qat.py").read_text(encoding="utf-8")
diag_src = (HERE / "ncm_diag.py").read_text(encoding="utf-8")
NCM_COMMIT = "c70274273f489abc73832b2b98584a8483b6b203"   # Aşama 0/1'de kullanılan resmî repo commit'i


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Aşama 2b — NCM'yi BitNet gibi (ternary) eğitmek: %86'ya yaklaşıyor mu?

**Araştırma sorusu.** Eğitim sonrası ternary kuantalama (PTQ) NCM Sudoku-hard modelini %86.3'ten %0'a düşürdü
(Aşama 1). Aynı model **ternary ağırlıklarla eğitilirse** (QAT, BitNet b1.58 yöntemi) kalite ne kadar geri gelir?

**Deneyler** (hepsi aynı veri, aynı mimari, aynı değerlendirme):

| Ad | Başlangıç | Ağırlıklar | Ne ölçer |
|---|---|---|---|
| **E3** | Resmî FP checkpoint | Ternary (gövde), QAT ince ayar | En ucuz yol: FP modeli ternary'ye "alıştırmak" yetiyor mu? |
| **E1** | Rastgele | FP | Kendi eğitim döngümüzün kalitesi (kontrol). Aynı bütçede E2'nin referansı |
| **E2** | Rastgele | Ternary (gövde) | Asıl soru: baştan ternary eğitim (native QAT) |

**Önerilen sıra:** 1–6 (kontroller) → E3 → E1 ve E2. E3 en kısa süren ve tek başına en bilgilendirici deney.

**Neyi FP bırakıyoruz:** Giriş izdüşümü (`wte`) ve seviye çıkış katmanları (`lm_heads`), yani lineer parametrelerin
yaklaşık %0.9'u. BitNet'in gömme/çıkış katmanlarını FP tutma pratiğiyle uyumlu; PTQ teşhisinde giriş izdüşümü
(`wte`) en hassas katman çıktı; çıkış katmanları tek tek ternary yapılınca zararsız (≈%85). Ternary katmanlarda aktivasyonlar token başına INT8'e kuantalanır.

**Varsayımlarımız (makalede Sudoku için verilmeyen ayarlar):** koşullandırma bozma ortalama 0.5 (seviyeler boyunca
0.75'ten 0.25'e doğrusal azalan), kayıp ağırlığı α = 0. E1 bu varsayımların makul olup olmadığını gösterir.

**Çalışma zamanı:** GPU (A100 önerilir; T4'te çalışır ama çok yavaştır). Checkpoint'ler Drive'a yazılır; oturum koparsa aynı
hücreyi tekrar çalıştırınca kaldığı adımdan devam eder.
"""),
    md("## 1. GPU, Drive ve yollar"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
import os
OUT_DIR = '/content/drive/MyDrive/ncm_ptq'
QAT_DIR = f'{OUT_DIR}/qat'
os.makedirs(QAT_DIR, exist_ok=True)
DATA_NPZ = f'{OUT_DIR}/sudoku_hard_sflm_seed42.npz'    # Aşama 2a çıktısı
assert os.path.exists(DATA_NPZ), 'Önce Aşama 2a notebook\\'unu çalıştır'
"""),
    md("## 2. Resmî repo (Aşama 0/1 ile aynı commit), checkpoint ve test verisi"),
    code(f"""
%cd /content
!test -d ncm || git clone https://github.com/HazyResearch/ncm.git
%cd /content/ncm
!git checkout -q {NCM_COMMIT} && git rev-parse HEAD
!pip install -q omegaconf hydra-core "transformers>=4.51" regex
!pip install -q -U "huggingface_hub[cli]"
!pip list 2>/dev/null | grep -iE "^(torch|transformers|numpy|omegaconf|hydra-core|huggingface.hub) " | tee /content/drive/MyDrive/ncm_ptq/env_versions.txt
!hf download hazyresearch/ncm-checkpoints --include "sudoku/hard/*" --local-dir artifacts
!hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/hard/*" --local-dir data
"""),
    md("## 3. Eğitim modülünü yaz (`ncm_qat.py`)"),
    code("%%writefile /content/ncm_qat.py\n" + module_src),
    md("""
## 4. Veriyi yükle

Önemli: S-FLM'nin doğrulama kümesi NCM'nin **test** kümesinin ta kendisi (Aşama 2a'da 2000/2000 doğrulandı).
Bu yüzden eğitim sırasında izleme için doğrulama kümesini değil, eğitim kümesinden ayırdığımız 1000 bulmacayı
(`holdout`) kullanıyoruz. Test kümesine yalnızca ara alt-küme kontrolü (ilk 256) ve en sondaki tam değerlendirme
için bakılır; hiperparametre seçimi test kümesine bakılarak yapılmaz.
"""),
    code("""
import sys, json, pickle, importlib, numpy as np, torch
sys.path.insert(0, '/content')
import ncm_qat as Q; importlib.reload(Q)

d = np.load(DATA_NPZ)
meta = json.loads(str(d['meta']))
print(meta)
assert meta['ncm_test_in_train'] == 0 and meta['ncm_test_in_valid'] == 2000
all_train = d['train'].astype(np.int64)
train_seqs, holdout_seqs = all_train[:-1000], all_train[-1000:]
test_items = pickle.load(open('/content/ncm/data/sudoku/hard/puzzles.pickle', 'rb'))
test_refs = pickle.load(open('/content/ncm/data/sudoku/hard/solutions.pickle', 'rb'))
print('eğitim', train_seqs.shape, '| holdout', holdout_seqs.shape, '| test', len(test_items))

NCM = dict(ncm_root='/content/ncm', config_path='/content/ncm/config/eval/sudoku-hard.yaml',
           ckpt_dir='/content/ncm/artifacts/sudoku/hard')
"""),
    md("""
## 5. Kontroller (eğitimden önce mutlaka)

1. **Girdi kurucusu = resmî `generate()`**: resmî modelle greedy üret, aynı kodlardan teacher-forcing girdisi kur,
   tekrar tahmin et. `agree_all` ≈ 1.0 olmalı (küçük sayısal farklar dışında birebir).
2. **Resmî modelin kendi verisindeki teacher-forcing kaybı ve seviye doğrulukları**: düşük kayıp, yüksek doğruluk
   beklenir. Bu, verinin ve kod çıkarımının (tokenizer.encode) modelin eğitildiği veriyle uyumlu olduğunu gösterir.
3. **Değerlendiricimiz = resmî sonuç**: tam testte `solved_pct` = 86.30 olmalı.
"""),
    code("""
cfg_m, tk, net = Q.load_official(**NCM)
dev = 'cuda'
pre = torch.tensor([list(it)[:91] for it in test_items[:256]], device=dev)
print('Kontrol 1 (girdi kurucusu):', Q.consistency_check(net, tk, pre))

levels = list(tk.vqvae.vq.levels)
pos_w = Q.level_position_weights(levels, 0.0).to(dev)
with torch.no_grad():
    hb = torch.tensor(holdout_seqs[:512], device=dev)
    x, idx = Q.build_inputs(net, tk, hb, None)
    net.eval(); l, lg = Q.ncm_loss(net, x, idx, pos_w)
print('Kontrol 2 (resmî model, holdout TF kaybı):', round(float(l), 4))
print('          seviye doğrulukları:', [round(a, 3) for a in Q.per_level_accuracy(lg, idx, levels)])

r_off = Q.evaluate_test(net, tk, test_items, test_refs)
print('Kontrol 3 (resmî model, tam test):', r_off)
json.dump(r_off, open(f'{QAT_DIR}/official_fp_test.json', 'w'), indent=1)
"""),
    md("""
## 6. Bütçe ayarı ve hız ölçümü

Makalenin bütçesi (50.000 adım × 1024 batch) Colab'da pratik değil. Aşağıdaki varsayılanlar kısa bir karşılaştırma
içindir; önemli olan **aynı bütçede E1 ile E2 arasındaki fark** ve E3'ün PTQ'ya (%0) göre ne kadar geri kazandığı.
Hücre 20 adım ölçüp her deneyin tahmini süresini yazar; gerekirse `STEPS_*` değerlerini buna göre ayarla.
`MICRO` bellek yetmezse (CUDA out of memory) küçültülür; sonucu değiştirmez.
"""),
    code("""
STEPS_FT, STEPS_SCRATCH = 1500, 4000
BATCH = 256
# MICRO GPU belleğine göre otomatik seçilir (A100 40/80 GB → 256, L4 24 GB → 128, T4 16 GB → 64).
# Bellek hatası (CUDA out of memory) alırsan aşağıdaki satırı elle küçült, örn. MICRO = 64.
_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
MICRO = 256 if _gb >= 35 else (128 if _gb >= 20 else 64)
print(f'GPU: {torch.cuda.get_device_name(0)} ({_gb:.0f} GB)  →  BATCH={BATCH}, MICRO={MICRO}')

import time, shutil
shutil.rmtree('/content/_probe', ignore_errors=True)   # eski ölçüm checkpoint'i varsa sil (yoksa devam eder, süre 0 çıkar)
probe =Q.TrainConfig(run_name='_hiz_olcumu', steps=20, batch_size=BATCH, micro_batch=MICRO, warmup=5,
                      log_every=10, eval_every=10**9, ckpt_every=10**9, test_n_during=0)
_, _, tmp = Q.load_official(**NCM)
Q.ternarize_linears(tmp, 'body')
t = time.time(); Q.train(tmp, tk, train_seqs, holdout_seqs, probe, '/content/_probe'); dt = (time.time() - t) / 20
print(f'≈ {dt:.2f} s/adım  →  E3 ≈ {dt*STEPS_FT/60:.0f} dk,  E1 ≈ E2 ≈ {dt*STEPS_SCRATCH/60:.0f} dk (her biri)')
del tmp; torch.cuda.empty_cache()
"""),
    md("""
## 7. E3 — Resmî FP checkpoint'ten ternary QAT ince ayarı

Önce ternary'ye çevrilmiş ama henüz eğitilmemiş modelin sonucu ölçülür (PTQ ile aynı, ≈%0 beklenir), sonra ince ayar.
"""),
    code("""
torch.manual_seed(0)
_, tk, net = Q.load_official(**NCM)
changed = Q.ternarize_linears(net, 'body')
print(len(changed), 'katman ternary')
r0 = Q.evaluate_test(net, tk, test_items[:256], test_refs[:256])
print('İnce ayar öncesi (ilk 256 test):', r0)

cfg_e3 = Q.TrainConfig(run_name='E3_ternary_ft', steps=STEPS_FT, batch_size=BATCH, micro_batch=MICRO,
                       peak_lr=1e-4, floor_lr=1e-5, warmup=100, eval_every=250, ckpt_every=250)
Q.train(net, tk, train_seqs, holdout_seqs, cfg_e3, QAT_DIR, test_items, test_refs)
r_e3 = Q.evaluate_test(net, tk, test_items, test_refs)
print('E3 tam test:', r_e3)
json.dump({'before_ft_256': r0, 'test': r_e3, 'cfg': cfg_e3.__dict__}, open(f'{QAT_DIR}/E3_result.json', 'w'), indent=1)
del net; torch.cuda.empty_cache()
"""),
    md("## 8. E1 — FP, baştan eğitim (kontrol)"),
    code("""
torch.manual_seed(0)
_, tk, net = Q.load_official(**NCM, load_generator=False)
cfg_e1 = Q.TrainConfig(run_name='E1_fp_scratch', steps=STEPS_SCRATCH, batch_size=BATCH, micro_batch=MICRO)
Q.train(net, tk, train_seqs, holdout_seqs, cfg_e1, QAT_DIR, test_items, test_refs)
r_e1 = Q.evaluate_test(net, tk, test_items, test_refs)
print('E1 tam test:', r_e1)
json.dump({'test': r_e1, 'cfg': cfg_e1.__dict__}, open(f'{QAT_DIR}/E1_result.json', 'w'), indent=1)
del net; torch.cuda.empty_cache()
"""),
    md("## 9. E2 — Ternary, baştan eğitim (native QAT)"),
    code("""
torch.manual_seed(0)
_, tk, net = Q.load_official(**NCM, load_generator=False)
Q.ternarize_linears(net, 'body')
cfg_e2 = Q.TrainConfig(run_name='E2_ternary_scratch', steps=STEPS_SCRATCH, batch_size=BATCH, micro_batch=MICRO)
Q.train(net, tk, train_seqs, holdout_seqs, cfg_e2, QAT_DIR, test_items, test_refs)
r_e2 = Q.evaluate_test(net, tk, test_items, test_refs)
print('E2 tam test:', r_e2)
tern = {n: m.ternary_stats() for n, m in net.named_modules() if isinstance(m, Q.TernaryLinear)}
zero = np.mean([v['zero'] for v in tern.values()])
print(f'ortalama sıfır oranı: {zero:.3f}')
json.dump({'test': r_e2, 'mean_zero_frac': float(zero), 'cfg': cfg_e2.__dict__}, open(f'{QAT_DIR}/E2_result.json', 'w'), indent=1)
del net; torch.cuda.empty_cache()
"""),
    md("""
## 9b. Uzun bütçe — E1 ve E2'yi 3 kat daha uzun eğit (fark kapanıyor mu?)

4.000 adımda ternary ile FP arasındaki holdout kaybı farkı 0.053'ten 0.033'e daraldı. Bu bölüm aynı iki deneyi
`LONG_STEPS` (varsayılan 12.000) adımla, **yeni adlarla** sıfırdan çalıştırır; 4.000 adımlık sonuçlara dokunmaz.
Diğer bütün ayarlar E1/E2 ile aynıdır (aynı öğrenme oranı eğrisi, yalnızca daha uzun).

A100'de tahmini süre: E1-uzun ≈ 1 saat 50 dk, E2-uzun ≈ 2 saat 10 dk. **İki ayrı Colab oturumunda paralel**
çalıştırılabilir: ikisinde de 1–6. adımları çalıştır, sonra birinde yalnızca E1-uzun, diğerinde yalnızca E2-uzun
hücresini çalıştır. Oturum koparsa aynı adımlarla kaldığı yerden devam eder (checkpoint 500 adımda bir).
"""),
    code("""
LONG_STEPS = 12000
torch.manual_seed(0)
_, tk, net = Q.load_official(**NCM, load_generator=False)
cfg_e1l = Q.TrainConfig(run_name=f'E1_fp_scratch_{LONG_STEPS//1000}k', steps=LONG_STEPS, batch_size=BATCH,
                        micro_batch=MICRO, eval_every=2000, ckpt_every=500)
Q.train(net, tk, train_seqs, holdout_seqs, cfg_e1l, QAT_DIR, test_items, test_refs)
r_e1l = Q.evaluate_test(net, tk, test_items, test_refs)
print('E1-uzun tam test:', r_e1l)
json.dump({'test': r_e1l, 'cfg': cfg_e1l.__dict__}, open(f'{QAT_DIR}/E1_long_result.json', 'w'), indent=1)
del net; torch.cuda.empty_cache()
"""),
    code("""
LONG_STEPS = 12000
torch.manual_seed(0)
_, tk, net = Q.load_official(**NCM, load_generator=False)
Q.ternarize_linears(net, 'body')
cfg_e2l = Q.TrainConfig(run_name=f'E2_ternary_scratch_{LONG_STEPS//1000}k', steps=LONG_STEPS, batch_size=BATCH,
                        micro_batch=MICRO, eval_every=2000, ckpt_every=500)
Q.train(net, tk, train_seqs, holdout_seqs, cfg_e2l, QAT_DIR, test_items, test_refs)
r_e2l = Q.evaluate_test(net, tk, test_items, test_refs)
print('E2-uzun tam test:', r_e2l)
tern = {n: m.ternary_stats() for n, m in net.named_modules() if isinstance(m, Q.TernaryLinear)}
zero = np.mean([v['zero'] for v in tern.values()])
json.dump({'test': r_e2l, 'mean_zero_frac': float(zero), 'cfg': cfg_e2l.__dict__},
          open(f'{QAT_DIR}/E2_long_result.json', 'w'), indent=1)
del net; torch.cuda.empty_cache()
"""),
    md("""
## 9c. Teşhis — E1-uzun ezberliyor mu? (eğitimsiz, ≈5 dk)

E1-uzun'un son checkpoint'iyle aynı sayıda **eğitim** bulmacasını ve **test** bulmacasını çözdürür.
- Eğitimde çok yüksek, testte düşükse → ezber (aşırı öğrenme); daha güçlü düzenlileştirme (bozma) gerekir.
- İkisi de düşükse → ezber değil; tarif ya da kapasite sorunu.
"""),
    code("""
_, tk, net = Q.load_official(**NCM, load_generator=False)
ck = torch.load(f'{QAT_DIR}/E1_fp_scratch_12k.pt', map_location='cpu', weights_only=False)
net.load_state_dict(ck['model']); print('checkpoint adımı:', ck['step'])
N_DIAG = 500
tr_items = [list(map(int, s[:91])) for s in train_seqs[:N_DIAG]]
tr_refs = [list(map(int, s[91:])) for s in train_seqs[:N_DIAG]]
r_tr = Q.evaluate_test(net, tk, tr_items, tr_refs)
r_te = Q.evaluate_test(net, tk, test_items[:N_DIAG], test_refs[:N_DIAG])
print(f'EĞİTİM bulmacaları: {r_tr}')
print(f'TEST bulmacaları  : {r_te}')
json.dump({'checkpoint': 'E1_fp_scratch_12k', 'step': ck['step'], 'train': r_tr, 'test': r_te},
          open(f'{QAT_DIR}/diag_E1_long_train_vs_test.json', 'w'), indent=1)
del net; torch.cuda.empty_cache()
"""),
    md("""
## 9d. E1-LM1B-bozma — resmî LM1B config'indeki bozma ayarıyla FP, baştan, 12.000 adım

Resmî `config/eval/lm1b-*.yaml` dosyalarında, okuyan kodu yayımlanmamış bir blok var:
`code_corruption: {mode: uniform, random_rate: true, low: 0.0, high: 1.0, skip_level0: true}`
(bkz. HazyResearch/ncm issue #24). Bizim yorumumuz: her dizi için oran r ~ U(0, 1) çekilir ve koşul kodları bu
oranla rastgele kodlarla değiştirilir; en kaba seviye hiç bozulmaz.

**Tek değişiklik bozma tarifi.** Bütçe, öğrenme oranı eğrisi ve ısınma (500) E1-uzun ile aynı tutuldu ki fark
yalnızca bozmadan kaynaklansın. Karşılaştırma: E1-uzun (doğrusal 0.75→0.25, seviye 0 dahil) → %25.65.
Bakılacaklar: holdout kaybı 6.000. adımdan sonra yükselmeyi bırakıyor mu? Çözme oranı belirgin artıyor mu?
A100'de ≈1 saat 50 dk.
"""),
    code("""
torch.manual_seed(0)
_, tk, net = Q.load_official(**NCM, load_generator=False)
cfg_e1r = Q.TrainConfig(run_name='E1_fp_scratch_12k_lm1bcorr', steps=12000, batch_size=BATCH, micro_batch=MICRO,
                        eval_every=2000, ckpt_every=500,
                        corruption_mode='random_uniform', corruption_low=0.0, corruption_high=1.0,
                        corruption_skip_level0=True)
Q.train(net, tk, train_seqs, holdout_seqs, cfg_e1r, QAT_DIR, test_items, test_refs)
r_e1r = Q.evaluate_test(net, tk, test_items, test_refs)
print('E1-LM1B-bozma tam test:', r_e1r)
json.dump({'test': r_e1r, 'cfg': cfg_e1r.__dict__}, open(f'{QAT_DIR}/E1_lm1bcorr_result.json', 'w'), indent=1)
del net; torch.cuda.empty_cache()
"""),
    md("""## 9e. E3-KD — FP öğretmenden damıtma + kademeli kuantalama (5.000 adım)

E3-uzun (5.000 adım, yalnızca CE) ile aynı tarif; iki fark: kayıp = 0.5·CE + 0.5·KL(öğretmen‖öğrenci), öğretmen resmî
FP NCM (dondurulmuş), ve ilk 500 adımda kuantalama katsayısı λ 0→1 (HF 1.58-bit tarifi). Ara ölçümler λ=1 ile yapılır.
Karşılaştırma: E3-uzun tam test %71.60."""),
    code("""
torch.manual_seed(0)
_, tk, net = Q.load_official(**NCM)
Q.ternarize_linears(net, 'body')
_, _, teacher = Q.load_official(**NCM)          # dondurulmuş FP öğretmen (%86)
cfg_kd = Q.TrainConfig(run_name='E3_ternary_ft_5k_kd', steps=5000, batch_size=BATCH, micro_batch=MICRO,
                       peak_lr=1e-4, floor_lr=1e-5, warmup=100, eval_every=1000, ckpt_every=500,
                       kd_weight=0.5, kd_temperature=1.0, quant_warmup=500)
Q.train(net, tk, train_seqs, holdout_seqs, cfg_kd, QAT_DIR, test_items, test_refs, teacher=teacher)
r_kd = Q.evaluate_test(net, tk, test_items, test_refs)
print(f"E3-KD tam test: %{r_kd['solved_pct']:.2f}   (E3-uzun aynı adım: %71.60 | FP öğretmen: %86.15)")
json.dump({'test': r_kd, 'cfg': cfg_kd.__dict__}, open(f'{QAT_DIR}/E3_5k_kd_result.json', 'w'), indent=1)
del net, teacher; torch.cuda.empty_cache()
"""),
    md("""## 9f. E3-α1 — en kaba seviyeye daha çok ağırlık (α=1, her seviye eşit pay)

α=0'da tek kodluk en kaba seviyenin kayıptaki payı 1/322 ≈ %0,3; α=1'de her seviye 1/12 ≈ %8,3 alır (makale Denk. 10).
Hipotez: E3'ün takıldığı yer en kaba seviye (holdout TF doğruluğu ≈%81; testte en kaba kod doğruluğu E3 %76,7, FP %85,5). Diğer her şey 9e ile aynı; USE_KD=False ise damıtma
kapalı ve doğrudan E3-uzun (%71,60) ile karşılaştırılır."""),
    code("""
USE_KD = True    # 9e (E3-KD) işe yaradıysa True bırak; yaramadıysa False yap
torch.manual_seed(0)
_, tk, net = Q.load_official(**NCM)
Q.ternarize_linears(net, 'body')
teacher = Q.load_official(**NCM)[2] if USE_KD else None
run = 'E3_ternary_ft_5k_kd_a1' if USE_KD else 'E3_ternary_ft_5k_a1'
cfg_a1 = Q.TrainConfig(run_name=run, steps=5000, batch_size=BATCH, micro_batch=MICRO,
                       peak_lr=1e-4, floor_lr=1e-5, warmup=100, eval_every=1000, ckpt_every=500, alpha=1.0,
                       kd_weight=0.5 if USE_KD else 0.0, kd_temperature=1.0, quant_warmup=500 if USE_KD else 0)
w = Q.level_position_weights(list(tk.vqvae.vq.levels), 1.0)
print('en kaba seviyenin kayıptaki payı:', round(float(w[0]), 4))
Q.train(net, tk, train_seqs, holdout_seqs, cfg_a1, QAT_DIR, test_items, test_refs, teacher=teacher)
r_a1 = Q.evaluate_test(net, tk, test_items, test_refs)
print(f"{run} tam test: %{r_a1['solved_pct']:.2f}   (E3-uzun: %71.60 | FP: %86.15)")
json.dump({'test': r_a1, 'cfg': cfg_a1.__dict__}, open(f'{QAT_DIR}/{run}_result.json', 'w'), indent=1)
del net, teacher; torch.cuda.empty_cache()
"""),
    md("""## 9g. E3-A16 — ternary ağırlık, INT8 aktivasyon KAPALI (W1.58A16)

E3-uzun ile birebir aynı tarif (α=0, damıtma yok, 5.000 adım); tek fark act_int8=False. Amaç: en kaba seviyedeki
açığın (testte en kaba kod doğruluğu E3 %76,7, FP %85,5) ternary ağırlıktan mı INT8 aktivasyondan mı geldiğini ayırmak. Karşılaştırma: %71.60."""),
    code("""
torch.manual_seed(0)
_, tk, net = Q.load_official(**NCM)
changed = Q.ternarize_linears(net, 'body', act_int8=False)
assert all(not m.act_int8 for m in net.modules() if isinstance(m, Q.TernaryLinear))
print(len(changed), 'katman ternary, INT8 aktivasyon kapalı')
cfg_a16 = Q.TrainConfig(run_name='E3_ternary_ft_5k_a16', steps=5000, batch_size=BATCH, micro_batch=MICRO,
                        peak_lr=1e-4, floor_lr=1e-5, warmup=100, eval_every=1000, ckpt_every=500)
Q.train(net, tk, train_seqs, holdout_seqs, cfg_a16, QAT_DIR, test_items, test_refs)
r_a16 = Q.evaluate_test(net, tk, test_items, test_refs)
print(f"E3-A16 tam test: %{r_a16['solved_pct']:.2f}   (E3-uzun, INT8 aktivasyonlu: %71.60 | FP: %86.15)")
json.dump({'test': r_a16, 'cfg': cfg_a16.__dict__, 'act_int8': False},
          open(f'{QAT_DIR}/E3_ternary_ft_5k_a16_result.json', 'w'), indent=1)
del net; torch.cuda.empty_cache()
"""),
    md("""## 9h. E3-PC — kanal (satır) başına ternary ölçek, W ≈ diag(γ)·T

E3-uzun ile aynı tarif (α=0, INT8 aktivasyon açık, 5.000 adım); tek fark her çıkış satırına ayrı γ. Ek bellek
matris başına satır sayısı kadar sayı (≈%0,2); çıkarımda γ çıktıya bir kez uygulanır. Karşılaştırma: %71.60."""),
    code("""
torch.manual_seed(0)
_, tk, net = Q.load_official(**NCM)
changed = Q.ternarize_linears(net, 'body', act_int8=True, per_channel=True)
assert all(m.per_channel and m.act_int8 for m in net.modules() if isinstance(m, Q.TernaryLinear))
print(len(changed), 'katman ternary, kanal başına ölçek, INT8 aktivasyon açık')
cfg_pc = Q.TrainConfig(run_name='E3_ternary_ft_5k_pc', steps=5000, batch_size=BATCH, micro_batch=MICRO,
                       peak_lr=1e-4, floor_lr=1e-5, warmup=100, eval_every=1000, ckpt_every=500)
Q.train(net, tk, train_seqs, holdout_seqs, cfg_pc, QAT_DIR, test_items, test_refs)
r_pc = Q.evaluate_test(net, tk, test_items, test_refs)
print(f"E3-PC tam test: %{r_pc['solved_pct']:.2f}   (E3-uzun, tensör başına ölçek: %71.60 | FP: %86.15)")
json.dump({'test': r_pc, 'cfg': cfg_pc.__dict__, 'per_channel': True, 'act_int8': True},
          open(f'{QAT_DIR}/E3_ternary_ft_5k_pc_result.json', 'w'), indent=1)
del net; torch.cuda.empty_cache()
"""),
    md("""## 11. PTQ çöküş mekanizması (eğitim yok, yalnızca ölçüm, ≈20–30 dk)

**Gerekenler:** yalnızca 1–4. bölümler. Sonra bu bölümün iki hücresini sırayla çalıştır.

Üç soru:
- **A. Hata büyüklüğü mü, ternary'nin kendisi mi?** Gövdeye INT8/4/3, ternary ve *aynı büyüklükte rastgele Gauss
  gürültüsü* eklenir (ilk 500 test). Gürültü de aynı noktada çökertiyorsa sorun ternary değil, hatanın büyüklüğü.
- **B. Hangi katman/blok?** Her katman tek başına, her blok tek başına, ilk k ve son k blok ternary (ilk 256 test).
- **C. Hata seviyeler boyunca birikiyor mu?** Teacher-forcing doğruluğu ile üretim doğruluğu seviye seviye; FP'nin
  ürettiği koddan ilk ayrıldığı seviye; blok başına gizli durum sapması.

Sonuçlar `qat/ptq_mechanism.json` dosyasına adım adım yazılır; oturum koparsa hücre kaldığı yerden devam eder."""),
    code("%%writefile /content/ncm_diag.py\n" + diag_src),
    code("""
import re, pandas as pd
import ncm_diag as D; importlib.reload(D)
valid = np.load(DATA_NPZ)['valid'].astype(np.int64)        # = NCM test kümesi, aynı sıra (Aşama 2a)
_, tk, net = Q.load_official(**NCM); net.eval()
levels = list(tk.vqvae.vq.levels)
OUT = f'{QAT_DIR}/ptq_mechanism.json'
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, 'w'), indent=1)
snap = D.snapshot(net)
names = [n for n, _ in D.linears(net)]
body = [n for n in names if not Q._is_io(n)]; io = [n for n in names if Q._is_io(n)]
bidx = lambda n: int(re.search(r'blocks\\.(\\d+)\\.', n).group(1))
assert all(re.search(r'blocks\\.(\\d+)\\.', n) for n in body), [n for n in body if not re.search(r'blocks\\.', n)]
nblk = 1 + max(bidx(n) for n in body)
print(len(body), 'gövde katmanı,', nblk, 'blok,', len(io), 'giriş/çıkış katmanı')

N, NS = 500, 256
fp = D.fp_reference(net, tk, test_items[:N], test_refs[:N], valid[:N], levels)
print('FP (ilk 500):', fp['score']['solved_pct'])

# A. hata eğrisi (gövde, ilk 500, ayrıntılı)
A = [('fp', None, 0), ('int8', 'int8', 0), ('int4', 'int4', 0), ('int3', 'int3', 0),
     ('tern', 'tern', 0), ('tern_ch', 'tern_ch', 0)] + \\
    [(f'noise_{r}', 'noise', r) for r in (0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)]
for name, kind, rel in A:
    key = f'A/{name}'
    if key in R: continue
    info = D.perturb(net, body, kind, rel) if kind else {'rel_err': 0.0, 'n_layers': 0}
    R[key] = {**info, **D.measure(net, tk, test_items[:N], test_refs[:N], valid[:N], levels, fp)}
    D.restore(net, snap); save()
    print(f"{key:16s} rel_err={R[key]['rel_err']:.3f}  çözülen={R[key]['solved_pct']:6.2f}  FP ile aynı kod={R[key]['same_codes_as_fp_pct']:6.2f}")

# B. ternary tarama (ilk 256, hızlı)
G = {f'B/layer/{n}': [n] for n in names}
G.update({f'B/block/{i}': [n for n in body if bidx(n) == i] for i in range(nblk)})
G.update({f'B/first_k/{k}': [n for n in body if bidx(n) < k] for k in range(1, nblk)})
G.update({f'B/last_k/{k}': [n for n in body if bidx(n) >= nblk - k] for k in range(1, nblk)})
G['B/io_only'] = io
for i, (key, ns) in enumerate(G.items()):
    if key in R: continue
    info = D.perturb(net, ns, 'tern')
    R[key] = {**info, **D.measure(net, tk, test_items[:NS], test_refs[:NS], valid[:NS], levels, fp, full=False)}
    D.restore(net, snap); save()
    if i % 10 == 0: print(f'tarama {i+1}/{len(G)}')
print('bitti →', OUT)
"""),
    code("""
# Özet: bu hücrenin çıktısının tamamını kopyalayıp gönder
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print('=== A. Hata eğrisi (gövde 32 katman, ilk 500 test) ===')
print(pd.DataFrame([{'ad': k[2:], 'rel_err': round(v['rel_err'], 3), 'çözülen%': v['solved_pct'],
                     'hücre%': v['cell_pct'], 'FP_ile_aynı_kod%': v['same_codes_as_fp_pct']}
                    for k, v in R.items() if k.startswith('A/')]).to_string(index=False))
for k in ['A/fp', 'A/int4', 'A/int3', 'A/tern', 'A/noise_0.2', 'A/noise_0.3']:
    if k in R:
        v = R[k]
        print(f"\\n--- {k[2:]} ---")
        for f in ['tf_acc_per_level', 'gen_acc_per_level', 'kl_fp_q_per_level', 'first_div_level_hist', 'block_rel_drift']:
            print(f'{f:22s}', v.get(f))
print('\\n=== B. Ternary tarama (ilk 256 test), çözülen% ===')
print(pd.DataFrame([{'ad': k[2:], 'katman': v['n_layers'], 'rel_err': round(v['rel_err'], 3), 'çözülen%': v['solved_pct'],
                     'hücre%': v['cell_pct'], 'FP_ile_aynı_kod%': v['same_codes_as_fp_pct']}
                    for k, v in R.items() if k.startswith('B/')]).to_string(index=False))
"""),
    md("""## 12. Seed tekrarı — E3-uzun, seed 1 ve 2 (her biri ≈55 dk, A100)

**Gerekenler:** 1–4. bölümler, ardından 6. bölüm (`BATCH` ve `MICRO` değerlerini ayarlar). 5 ve 7–11 atlanabilir.
E3-uzun (seed 0, %71.60) ile birebir aynı tarif; değişen yalnızca veri sırası ve bozma için kullanılan seed.
FP checkpoint'ten başlandığı için başlangıç ağırlıkları her seed'de aynı.
Oturum koparsa hücreyi tekrar çalıştır: biten seed atlanır, yarım kalan checkpoint'ten devam eder."""),
    code("""
for SEED in (1, 2):
    res_path = f'{QAT_DIR}/E3_ternary_ft_5k_s{SEED}_result.json'
    if os.path.exists(res_path):
        print(f'seed {SEED} zaten bitmiş:', json.load(open(res_path))['test']['solved_pct']); continue
    torch.manual_seed(SEED)
    _, tk, net = Q.load_official(**NCM)
    Q.ternarize_linears(net, 'body')
    cfg_s = Q.TrainConfig(run_name=f'E3_ternary_ft_5k_s{SEED}', seed=SEED, steps=5000, batch_size=BATCH,
                          micro_batch=MICRO, peak_lr=1e-4, floor_lr=1e-5, warmup=100, eval_every=1000, ckpt_every=500)
    Q.train(net, tk, train_seqs, holdout_seqs, cfg_s, QAT_DIR, test_items, test_refs)
    r_s = Q.evaluate_test(net, tk, test_items, test_refs)
    print(f"E3-uzun seed {SEED} tam test: %{r_s['solved_pct']:.2f}   (seed 0: %71.60)")
    json.dump({'test': r_s, 'cfg': cfg_s.__dict__}, open(res_path, 'w'), indent=1)
    del net; torch.cuda.empty_cache()
"""),
    md("""## 13. E3-B0FP — blok 0 FP, kalan 28 gövde katmanı ternary (≈55 dk, A100)

**Gerekenler:** 1–4. bölümler. E3-uzun (seed 0) tarifi; tek fark blok 0'ın 4 doğrusal katmanının FP kalması
(PTQ teşhisinde en hassas blok). Sonuç: %71.30 (3 seed ortalaması 71.15 ± 0.40 ile fark gürültü içinde). Kayıtlı koşu 219 dk sürdü (GPU türü kaydedilmedi)."""),
    code("""
import os, json, torch, torch.nn as nn
if 'BATCH' not in globals():
    BATCH = 256
    _gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    MICRO = 256 if _gb >= 35 else (128 if _gb >= 20 else 64)

RUN = 'E3_ternary_ft_5k_b0fp'
res_path = f'{QAT_DIR}/{RUN}_result.json'
if os.path.exists(res_path):
    print('Zaten bitmiş:', json.load(open(res_path))['test'])
else:
    torch.manual_seed(0)
    _, tk, net = Q.load_official(**NCM)
    Q.ternarize_linears(net, 'body')
    # blok 0'daki TernaryLinear'ları aynı ağırlıklarla düz nn.Linear'a geri çevir
    kept = []
    for name, m in list(net.named_modules()):
        if isinstance(m, Q.TernaryLinear) and '.blocks.0.' in f'.{name}.':
            parent_name, _, attr = name.rpartition('.')
            parent = net.get_submodule(parent_name)
            lin = nn.Linear(m.in_features, m.out_features, bias=m.bias is not None,
                            device=m.weight.device, dtype=m.weight.dtype)
            with torch.no_grad():
                lin.weight.copy_(m.weight)
                if m.bias is not None:
                    lin.bias.copy_(m.bias)
            setattr(parent, attr, lin); kept.append(name)
    n_tern = sum(isinstance(m, Q.TernaryLinear) for m in net.modules())
    print('FP bırakılan:', kept)
    print('Ternary katman sayısı:', n_tern, '(beklenen 28)')
    assert len(kept) == 4 and n_tern == 28

    cfg_b = Q.TrainConfig(run_name=RUN, seed=0, steps=5000, batch_size=BATCH, micro_batch=MICRO,
                          peak_lr=1e-4, floor_lr=1e-5, warmup=100, eval_every=1000, ckpt_every=500)
    Q.train(net, tk, train_seqs, holdout_seqs, cfg_b, QAT_DIR, test_items, test_refs)
    r_b = Q.evaluate_test(net, tk, test_items, test_refs)
    json.dump({'test': r_b, 'cfg': cfg_b.__dict__, 'fp_layers': kept}, open(res_path, 'w'), indent=1)
    print('E3-B0FP tam test:', r_b)
"""),
    md("## 10. Özet"),
    code("""
import pandas as pd
rows = []
def add(name, path, key='test'):
    if os.path.exists(path):
        r = json.load(open(path)); r = r.get(key, r)
        rows.append({'deney': name, **{k: round(v, 2) for k, v in r.items() if k != 'n'}, 'n': r['n']})
add('Resmî FP (referans)', f'{QAT_DIR}/official_fp_test.json', key=None)
add('E3 ternary, FP\\'den ince ayar', f'{QAT_DIR}/E3_result.json')
add('E1 FP, baştan (kısa bütçe)', f'{QAT_DIR}/E1_result.json')
add('E2 ternary, baştan (kısa bütçe)', f'{QAT_DIR}/E2_result.json')
add('E1 FP, baştan (uzun bütçe)', f'{QAT_DIR}/E1_long_result.json')
add('E2 ternary, baştan (uzun bütçe)', f'{QAT_DIR}/E2_long_result.json')
add('E1 FP, baştan, LM1B tarzı bozma (uzun)', f'{QAT_DIR}/E1_lm1bcorr_result.json')
display(pd.DataFrame(rows))
"""),
    md("""
## Nasıl yorumlanır

- **Kontrol 1 ≈ 1.0 değilse** eğitim sonuçlarına güvenme: girdi kurucumuz resmî üretimle uyuşmuyor demektir.
- **E3 yüksekse** (örneğin %86'ya yakın): ternary NCM mümkün ve ucuz; FP modelden kısa bir ince ayarla
  ulaşılabiliyor. Bu, PTQ'nun %0'ına karşı çok güçlü bir sonuç.
- **E1 düşükse** (kısa bütçede %86'nın çok altında): ya bütçe yetersiz ya da varsayımlarımız (bozma oranı, α)
  makaledekinden farklı. Bu durumda E1 ile E2'yi birbirleriyle karşılaştır, resmî %86 ile değil.
- **E2 / E1 oranı** asıl sonuçtur: aynı bütçede ternary, FP'nin ne kadarına ulaşıyor?
- Tek seed ve kısaltılmış bütçe: sonuç umut vericiyse aynı deneyleri 2–3 seed ve daha uzun bütçeyle tekrarla.
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = HERE / "ncm_asama2b_qat.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("yazıldı:", out)
