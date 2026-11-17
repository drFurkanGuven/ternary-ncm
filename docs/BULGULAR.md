# Araştırma günlüğü — Ternary NCM

Son güncelleme: 17 Kasım 2026. Her bulgunun yanında kanıtın nerede olduğu yazılıdır. "Doğrulandı" yalnızca
resmî NCM kodu/checkpoint'i üzerinde, açıkça kontrol edilmiş sonuçlar için kullanılır.

## 1. Doğrulanan bulgular (resmî NCM, Sudoku-hard)

### 1.1 Yeniden üretim
- Resmî checkpoint, resmî `evaluate.py` ile **86.30%** çözüyor (Colab T4). Makale: 86.1%.
- Kendi değerlendiricimiz (`colab/ncm_ptq.py`) aynı checkpoint'te birebir **86.30%** veriyor.
- A100'de aynı checkpoint **86.15%** veriyor (2000'de 3 bulmaca fark; GPU'ya bağlı sayısal farklar).
  **Kural:** karşılaştırmalar aynı GPU tipinde yapılır.
- Kanıt: `results_official/asama1_ptq_sudoku_hard.json`, `results_official/asama2b_official_fp_A100.json`.

### 1.2 Eğitim sonrası kuantalama (PTQ) çöküyor
- INT8 (kanal başına) 86.10%, INT4 81.90%, **ternary'nin bütün varyantları 0.00%**:
  BitNet absmean (tensör ve kanal başına), TWN eşiği, INT8 aktivasyonlu ya da aktivasyonsuz.
- Yalnızca dikkat, yalnızca MLP ya da **yalnızca 13 giriş/çıkış katmanı** (parametrelerin %0.9'u) ternary
  yapıldığında da 0%. Ternary'yi tek başına kaldırabilen bir katman grubu yok.
- Ternary hücre doğruluğu %18–41: ipuçlarını kopyalayıp gerisini rastgele dolduran bir program bile yaklaşık %44
  alırdı.
- Kanıt: `results_official/asama1_ptq_sudoku_hard.json`.

**Çöküşün mekanizması (2026-11-15, yalnızca ölçüm; `results_official/asama2b_diag_ptq_mechanism.json`):**
- **Çöküş yalnızca hatanın büyüklüğüyle tutarlı; ternary yapısını ayrıca suçlamaya gerek yok** (gürültü eğrisi 0.35'te zaten %0 olduğu için ölçülen ternary hatalarında ikisini ayıramıyor; eşleşen kanıt tamsayı kuantalayıcılar). Gövdeye aynı göreli hatada (‖ΔW‖/‖W‖) rastgele
  Gauss gürültüsü eklemek kuantalamayla aynı etkiyi yapıyor: INT4 (0.19) %83.4 / gürültü 0.20 %80.8; INT3 (0.42) ve
  gürültü 0.40 ikisi de %0 (ilk 500 test).
- **Tolerans eşiği:** %50 çözüm ≈0.28 göreli hatada; ≥0.35'te ≈%0. Gauss ağırlıklar için *en iyi* ternary bile
  ≈0.44 hata veriyor (sayısal hesap; gerçek ağırlıklarda 0.465–0.568). Yani yalnızca ağırlık hatasını küçülten
  eleman bazlı ternary yuvarlama bu modelde başarılı olamaz: alt sınır eşiğin çok üstünde. Katman çıktısı hatasını
  küçülten kalibrasyonlu PTQ (GPTQ, PT²-LLM gibi) bu argümanla sınırlanmıyor ve denenmedi.
- **Çöküşü açıklamak için seviyeler arası çığa gerek yok.** Ternary'de doğru önceki kodlar verildiğinde bile (teacher forcing)
  seviye 0–10'un doğruluğu 0.08–0.18 (neredeyse belirleyici seviye 1 ve 4 hariç: ternary 0.91 / 1.0, FP 0.99 / 1.0), en ince seviyede 0.43; üretim doğruluğuna yakın. 500 bulmacanın 446'sı daha seviye 0'da FP'den ayrılıyor.
  Hasar transformer'ın içinde, derinlikle birikiyor: gizli durum sapması blok 0'dan sonra 0.37, blok 7'den sonra 1.01.
- **Hassasiyet girişte toplanıyor.** Yalnızca `wte` (giriş izdüşümü) ya da yalnızca blok 0'ın dikkat katmanları ternary
  olunca %0. Blok 0 tek başına %0, blok 1 %24.6, blok 2–7 tek tek %70–77. **Düzeltme:** yukarıdaki "yalnızca 13
  giriş/çıkış katmanı → %0" sonucunun nedeni tek bir katman, `wte`. 12 çıkış katmanının her biri tek başına
  ternary yapılınca ≈%85 (zararsız).
- **Üretilen kodlar tek değil:** INT8 %84.8 çözüyor ama FP ile aynı kodu yalnızca %41.8 bulmacada üretiyor;
  0.05 gürültüde %85.8 çözüm, %1.6 aynı kod. Tahtayı aynı kalırken kod yolu değişebiliyor (kaba kod bulgusuyla uyumlu).

### 1.3 Eğitim verisi birebir yeniden üretildi
- S-FLM üreticisi (commit `30def799`), zor seviye, tohum 42, 48.000 eğitim + 2.000 doğrulama bulmacası.
- **NCM'nin 2000 test bulmacası = S-FLM doğrulama kümesi** (2000/2000, aynı sırayla). Eğitim kümesiyle örtüşme 0.
- Kanıt: `results_official/asama2a_veri_dogrulama.json`.

### 1.4 Eğitim döngümüz resmî üretimle uyumlu
- Teacher-forcing girdi kurucusu (`colab/ncm_qat.py::build_inputs`): resmî greedy `generate()` ile kodlar
  üretildi, aynı kodlardan girdi kuruldu, tahminler **%100 aynı** çıktı (her seviyede, her dizide).
- Resmî modelin bizim eğitim verimizdeki teacher-forcing kaybı 0.105; seviye doğrulukları %94–100.

### 1.5 Ternary QAT ince ayarı (E3): 0% → 69.7%
- Resmî FP checkpoint'ten başlayıp gövde katmanları ternary (BitNet b1.58, INT8 aktivasyon), giriş/çıkış FP;
  1.500 adım × 256 batch, A100'de 15 dakika.
- Tam testte **69.70%** çözüldü (aynı GPU'daki FP referansı 86.15%, yani FP'nin yaklaşık %81'i).
- Eğri hâlâ yükseliyordu (ilk 256 test bulmacasında 250. adımda 49.6%, 1500. adımda 69.1%) → daha uzun eğitimle
  artması olası, ama **henüz ölçülmedi**.
- En çok etkilenen seviye en kaba seviye (l=1): teacher-forcing doğruluğu FP'de 95.5%, E3'te 78.9%.
  **Düzeltme (2026-11-15):** Bu karşılaştırma yanlı. `holdout_seqs` bizim eğitimimizden ayrılmış ama resmî FP
  modelin eğitim verisinin parçası; FP'nin %95.5'i kendi eğitim verisinde ölçüldü. Test kümesinde FP'nin en kaba kod
  doğruluğu %85.5, E3'ün %76.7 (aşağıdaki ayrıştırma).
- Kanıt: `results_official/asama2b_E3_ternary_ft.json`.
- **E3-uzun (5.000 adım, 55 dk A100): 71.60%** (FP'nin ≈%83'ü). 3,3 kat daha fazla adım yalnızca +1,9 puan getirdi:
  ilk 256 testte 2.000. adımdan sonra 69.5%'te düzleşti (5.000'de 71.1%), oysa holdout TF kaybı 0.225 → 0.204 düşmeye
  devam etti. En kaba seviyenin TF doğruluğu ≈%81'de takıldı (FP %95.5; **düzeltme:** bu FP değeri yanlı, bkz. yukarıdaki not; 3 seed ile kazanç +1.2–1.9 puan, ortalamada +1.45). Sonuç: yalnızca CE ile adım artırmak
  yetmiyor; tarif değişikliği gerekiyor (sıradaki: FP öğretmenden damıtma + kademeli kuantalama, §9e).
  Kanıt: `results_official/asama2b_E3_ternary_ft_5k.json`.
- **E3-KD (FP öğretmenden damıtma + kademeli kuantalama), 2.000. adımda durduruldu:** ilk 256 testte %66,8 / %68,8
  (E3-uzun aynı adımlarda %66,4 / %69,5) → fark yok. Kademeli kuantalama şoku önlemiyor, erteliyor (λ=1'e ulaşınca
  CE 0,52; uyarısız E3'te aynı adımda 0,48). Damıtma kaybı da α=0 ağırlıklarını kullandığı için en kaba seviyeye
  ≈%0,3 pay veriyordu. Kanıt: `results_official/asama2b_E3_ternary_ft_5k_kd_partial.json`.
- **E3-α1 (her seviyeye eşit kayıp payı, en kaba seviye 27 kat daha ağır): 69.05%** (α=0: 71.60%) → daha kötü.
  En kaba seviyenin TF doğruluğu yine ≈%80'de kaldı, ince seviyeler biraz kötüleşti. Sonuç: en kaba seviyedeki açık
  kayıp ağırlığından değil; ternary modelin o kararı verme kapasitesinden (ya da INT8 aktivasyondan) kaynaklanıyor
  olabilir. Kanıt: `results_official/asama2b_E3_ternary_ft_5k_alpha1.json`. Tek seed.
- **E3-A16 (INT8 aktivasyon kapalı, W1.58A16): 71.05%** (W1.58A8: 71.60%) → fark gürültü içinde; kayıp eğrileri
  üst üste. INT8 aktivasyonun bu modelde ölçülebilir bir bedeli yok; FP'ye kalan açık ternary ağırlıklardan geliyor. Sıradaki: kanal başına
  ternary ölçek (W ≈ diag(γ)·T, §9h) ve açığın en kaba koda ayrıştırılması (teşhis hücresi).
  Kanıt: `results_official/asama2b_E3_ternary_ft_5k_a16.json`.
- **Açığın en kaba koda ayrıştırılması (tam test, bulmaca bazında): hipotez çürüdü.**
  FP: kaba kod doğru %85.5; P(çözüldü | doğru) %88.1, P(çözüldü | yanlış) %74.8. E3: kaba kod doğru %76.7;
  P(çözüldü | doğru) %74.6, P(çözüldü | yanlış) %61.8. Yanlış kaba kod çoğu zaman yine doğru tahtaya götürüyor
  (kodlar çözümü tek başına belirlemiyor). Ternary'ye FP'nin kaba kod doğruluğunu verseydik ≈%72.7 olurdu (+1.1):
  14.5 puanlık açığın yalnızca ≈1 puanı en kaba koddan. Açığın geri kalanı en kaba koddan sonraki seviyelerden geliyor (daha ayrıntılı yerelleştirilmedi); bu, seviyelere yayılmış bir kapasite kaybıyla tutarlı.
  Bu, α=1 ve damıtma denemelerinin işe yaramamasıyla da tutarlı. Kanıt:
  `results_official/asama2b_diag_coarse_code_decomposition.json`.
- **E3-PC (kanal başına ternary ölçek, W ≈ diag(γ)·T): 71.80%** (tensör başına: 71.60%). Tek seed'de binom
  standart hatası ≈1 puan; fark gürültü içinde. Holdout TF kaybı biraz düşüyor (0.195 vs 0.204) ama çözülen tahta
  oranı değişmiyor (TF kaybı yine güvenilir bir vekil değil). Sonuç: E3 tarifinde ölçek ayrıntısı, α, damıtma ve
  INT8 aktivasyon açığı kapatmıyor (α=1 kötüleştiriyor); ince ayarla ulaşılan tavan ≈%71–72 (FP'nin ≈%83'ü). Kanıt:
  `results_official/asama2b_E3_ternary_ft_5k_pc.json`.
- **E3-uzun seed tekrarı (3 seed): 71.15 ± 0.40** (71.60 / 70.85 / 71.00), FP'nin %82.6'sı. Seed değişkenliği
  küçük. Tek seed'li varyantlar bu aralıkla okunmalı: kanal başına ölçek (71.80) ve A16 (71.05) fark yaratmıyor;
  α=1 (69.05) ortalamanın ≈5 std altında, gerçekten kötü. Kanıt: `results_official/asama2b_E3_ternary_ft_5k_seeds.json`.
- **E3-B0FP (blok 0'ın 4 katmanı FP, kalan 28 gövde katmanı ternary): 71.30%.** 3 seed ortalaması 71.15 ± 0.40 ile
  aynı; kazanç yok. Blok 0, PTQ'da ternary'ye en hassas blok olsa da QAT sonrası açığın belirgin kaynağı değil; kalan açık ağa
  yayılmış genel bir kapasite kaybıyla tutarlı (kaba kod ayrıştırması da öyle). Holdout TF kaybı 0.204 (E3-uzun ile aynı).
  Not: koşu 219 dk sürdü (önceki A100 koşuları 43–55 dk); GPU türü kaydedilmedi. Tek seed.
  Kanıt: `results_official/asama2b_E3_ternary_ft_5k_b0fp.json`.

### 1.6 FP baştan eğitim, kısa bütçe (E1): 21.75%
- Rastgele başlatma, FP, 4.000 adım × 256 batch (makalenin ≈51M dizilik bütçesinin ≈%2'si), A100'de 34 dakika.
- Tam testte **21.75%**. Holdout teacher-forcing kaybı 0.378 (resmî model 0.105).
- Eğri: ilk 256 test bulmacasında 1.000. adımda 0.8%, 2.000'de 9.8%, 3.000'de 22.3%, 4.000'de 20.3%. Öğrenme oranı
  tabana indikçe çözme oranı düzleşti, kayıp ise yavaşça düşmeye devam etti → bütçe sınırlı.
- En yavaş öğrenilen seviye en kaba seviye (l=1: 56.6%; resmî 95.5%).
- Bu bütçede E1, varsayımlarımızın (bozma 0.5, α=0) **modeli öğrenebildiğini** gösteriyor, ama %86'yı yeniden
  üretmiyor; bu beklenen bir durum. E2 ile karşılaştırma aynı bütçede yapılır.
- Kanıt: `results_official/asama2b_E1_fp_scratch.json`.

### 1.7 Ternary baştan eğitim, aynı kısa bütçe (E2): 16.45%
- E1 ile aynı ayarlar; tek fark gövde katmanlarının ternary (BitNet b1.58) ve INT8 aktivasyonlu olması. A100'de
  40 dakika.
- Tam testte **16.45%** (E1: 21.75%) → aynı bütçede FP'nin **≈%76'sı**. Ternary katmanlarda sıfır oranı %32.1.
- Holdout teacher-forcing kaybı, E2 ve E1 farkı: 1.000. adımda 0.605 / 0.552, 2.000'de 0.487 / 0.436,
  3.000'de 0.428 / 0.390, 4.000'de 0.411 / 0.378. Fark 0.053'ten 0.033'e daralıyor. E2'nin 4.000. adımdaki kaybı,
  E1'in kabaca 2.500–3.000. adımdaki kaybına denk; yani ternary aynı kayba yaklaşık 1.3–1.6 kat daha fazla adımda
  ulaşıyor (kaba tahmin, tek seed).
- Çözme oranı farkı (5.3 puan), 2000 bulmacalık test örnekleme gürültüsünün (≈±0.9 puan) belirgin şekilde üstünde;
  ama **seed'ler arası değişkenlik henüz bilinmiyor**.
- Kanıt: `results_official/asama2b_E2_ternary_scratch.json`.

### 1.8 FP baştan eğitim, 3 kat uzun bütçe (E1-uzun, 12.000 adım): 25.65% — yeni bir sorun
- 4.000 adıma göre yalnızca +3.9 puan (21.75 → 25.65). Ama asıl bulgu kayıp eğrisinde:
  - Eğitim kaybı (bozmalı girdiyle) sürekli düştü: 0.655 → 0.372.
  - **Holdout teacher-forcing kaybı (temiz girdiyle) 6.000. adımda en düşük değerine (0.3735) indi, sonra yükseldi:**
    8.000'de 0.395, 10.000'de 0.422, 12.000'de 0.436.
  - İlk 256 test bulmacasında çözme oranı 6.000–10.000. adımlar arasında %26–30 bandında, sonra %24.
  - Orta seviyelerin holdout doğruluğu 6.000. adımdan sonra hafifçe düştü; en kaba seviye yükselmeye devam etti.
- 12.000 × 256 ≈ 3.1M örnek, 47.000 eğitim bulmacası üzerinde ≈65 tur demek. Makale ise ≈51M örnekle (≈1.000 tur)
  %86'ya ulaşıyor. Bu yüzden basit bir "veri az, aşırı öğrenme" açıklaması tek başına yetmiyor. Olası açıklamalar
  (hiçbiri henüz test edilmedi):
  1. Bizim tarifimiz makalenin tarifinden farklı: batch 256 vs 1024, bozma oranları ve α tahmin edildi, belki veri
     artırma (Sudoku simetrileri) kullanılıyor.
  2. Holdout kaybı temiz girdiyle, eğitim bozmalı girdiyle ölçülüyor; yükselen kayıp, aşırı öğrenmeden çok
     kalibrasyon kaymasını (aşırı güven) gösteriyor olabilir.
- **Teşhis (9c) — ezber doğrulandı:** Aynı checkpoint ilk 500 eğitim bulmacasının **%98.6**'sını, ilk 500 test
  bulmacasının **%24.8**'ini çözüyor (hücre doğruluğu %99.98'e karşı %88.7). Model eğitim kümesini ezberliyor.
  Kanıt: `results_official/asama2b_diag_E1_long_train_vs_test.json`.
- Sonuç: **mevcut tarifle bütçeyi uzatmak %86'ya yaklaştırmıyor.** E1 ile E2 karşılaştırması aynı tarif içinde
  hâlâ geçerli.
- Kanıt: `results_official/asama2b_E1_fp_scratch_12k.json`.

### 1.9 Ternary baştan eğitim, 12.000 adım (E2-uzun): 22.75% — ternary daha az ezberliyor
- E1-uzun ile aynı tarif; tam testte **22.75%** (E1-uzun 25.65%) → FP'nin **≈%89'u** (4.000 adımda ≈%76'ydı).
- Holdout teacher-forcing kaybı (temiz girdi), E2 ve E1:
  | Adım | 2k | 4k | 6k | 8k | 10k | 12k |
  |---|---|---|---|---|---|---|
  | E2 (ternary) | 0.493 | 0.408 | 0.380 | **0.368** | **0.368** | **0.369** |
  | E1 (FP) | 0.434 | 0.377 | 0.374 | 0.395 | 0.422 | 0.436 |
- **E2'nin holdout kaybı yükselmiyor**, 8.000. adımdan sonra sabit kalıyor ve 8.000. adımdan itibaren E1'den daha
  düşük. Eğitim kaybı da E2'de daha yüksek (12k: 0.474 / 0.372). Yorum: ternary kısıtı bir düzenlileştirici gibi
  davranıyor ve ezberi yavaşlatıyor. Ama çözme oranında hâlâ FP'nin biraz altında (−2.9 puan, test gürültüsü ≈±1 puan,
  tek seed).
- İki model de bu tarifle %86'nın çok altında kalıyor; sınır tarif (bkz. 1.8 ve teşhis).
- Kanıt: `results_official/asama2b_E2_ternary_scratch_12k.json`.

### 1.10 LM1B tarzı rastgele oranlı bozma (12.000 adım, FP): 18.35% — işe yaramadı
- Tek değişiklik bozma tarifi: her dizi için r ~ U(0,1), seviye 0 bozulmuyor (resmî LM1B config'indeki
  `code_corruption` bloğunun bizim yorumumuz). Sonuç **18.35%** (doğrusal bozmalı E1-uzun: 25.65%).
- **Ezber sürüyor:** eğitim %99.8, test %20.4 (ilk 500).
- Temiz girdili holdout TF kaybı çok daha düşük (6.000. adımda 0.222, doğrusalda 0.374), ama çözme oranı daha düşük.
  **Ders:** teacher-forcing kaybı, üretim kalitesinin güvenilir bir göstergesi değil. Muhtemel neden: düşük oranlı
  örnekler modeli temiz koşula alıştırıyor; üretimde kendi hatalı kodlarıyla karşılaşınca hata büyüyor. Karar için
  çözme oranı kullanılmalı.
- LM1B bloğu, issue #24'te sorulduğu gibi, belki üretecin değil tokenizer'ın eğitiminde kullanılıyordu.
- Kanıt: `results_official/asama2b_E1_fp_scratch_12k_lm1bcorr.json`.

## 2. Geri çekilen önceki iddialar

Bu projeden önce yapılan, yayımlanmamış yerel bir prototipten gelen sonuçlar (Sudoku çözme kazancı, hız, enerji ve
"çarpımsız model" iddiaları) doğrulanamadığı için geri çekildi. Bu repodaki hiçbir sonuç o prototipe dayanmıyor; tüm
sayılar resmî NCM kodu ve kontrol noktasıyla üretildi (`results_official/`).

## 3. Bizim varsaymamız gereken ayarlar (makalede Sudoku için yok)

- Koşullandırma bozma: seviyeler boyunca doğrusal azalan, ortalama 0.5 (0.75 → 0.25).
- Kayıp ağırlığı α = 0 (ELBO ile uyumlu).
- Rastgele başlatmada resmî `_init_weights` (N(0, 0.02)); resmî sınıf bunu `__init__` içinde çağırmıyor.
- Ternary kapsamı: transformer blokları; giriş (`wte`) ve seviye çıkış katmanları (`lm_heads`) FP.

E1 (FP, baştan) bu varsayımlarla modelin öğrenebildiğini ama %86'yı yeniden üretmediğini gösterdi (ezber; bkz. 1.8).

## 4. Açık işler

0. **Tarif sorunu (öncelikli):** Mevcut tarifle FP baştan eğitim %26 civarında düzleşiyor. Resmî LM1B config'inde
   okuyan kodu yayımlanmamış bir `code_corruption` bloğu var (`mode: uniform, random_rate: true, low: 0, high: 1,
   skip_level0: true`; bkz. HazyResearch/ncm issue #24, yazarlardan henüz cevap yok). Bu tarif (9d)
   test edildi ve kötüleştirdi (%18.35); 9c teşhisi ezberi doğruladı (bkz. 1.8, 1.10). İlgili bağımsız çalışma: Red-fuchs/ncm-modular
   (MLX; resmî Sudoku sonucunu yeniden üretmemiş, küçük bir görevde sıfırdan NCM, AR'a kaybetmiş).
1. **Daha uzun bütçe (E1 ve E2):** 12.000 adımda yapıldı (bkz. 1.8–1.9); tam bütçede ternary ön eğitim hâlâ açık.
2. **E3 tavanı:** 5.000 adımda %71.6–71.8; uzun CE, damıtma, α=1, A16 ve kanal başına ölçek açığı kapatmadı.
   Anlaşılan durma kuralı gereği makale yazımına geçiliyor; seed tekrarı (madde 3) makale için gerekli.
3. **Seed tekrarı:** E3-uzun için 3 seed tamam (71.15 ± 0.40). Varyantlar (PC, A16, α1, B0FP) ve baştan eğitimler hâlâ tek seed.
4. **Kolay/orta seviye** ve **metin görevleri** (resmî checkpoint'ler mevcut): bulgu Sudoku'ya özgü mü?
5. **Enerji/bellek:** Ağırlık belleği makalede hesaplandı (gövde 50.3 MB → 5.0–6.3 MB; çıkarımda gereken VQ-VAE kod çözücüsü, kod kitapları ve kodlayıcının token gömmesi hariç); enerji iddiası yalnızca dikkat hesabı da dahil edilerek ve
   "analitik tahmin" olarak verilmeli. Gerçek hız için özel ternary çekirdek gerekir (bu çalışmanın kapsamı dışında).

## 5. Literatür konumu (özet)

- Kabadan inceye görüntü modellerinde (VAR) yalnızca PTQ çalışılmış; 4 bitte bile, hassas katmanlar yüksek hassasiyette tutulmazsa kalite belirgin bozuluyor (LiteVAR,
  Shift-and-Sum, HGA-Quant, PTQ4ARVG). Diffusion LLM'lerde de yalnızca PTQ var, 2 bitin altı çözülmemiş.
- Ternary QAT, difüzyon görüntü modellerinde çalışıyor (TerDiT). Native ternary kabadan inceye **metin** üretimi
  için yapılmış bir çalışma bulunamadı (sistematik olmayan tarama; NCM'ye atıf veren çalışmalar ayrıca
  kontrol edilmeli).
- Bilinen ve araç olarak kullanılabilecek yöntemler: Hadamard döndürmesi (BitNet v2), LFQ/BSQ implicit kod-kitapları.
