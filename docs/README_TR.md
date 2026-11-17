# Ternary NCM — Next-Concept Modeling'i BitNet b1.58 ile eğitmek (Türkçe özet)

İngilizce ana sayfa ve yeniden üretim rehberi: [../README.md](../README.md).

**Araştırma sorusu.** Kabadan inceye paralel metin üretimi yapan Next-Concept Modeling (NCM,
[arXiv 2610.14162](https://arxiv.org/abs/2610.14162)), doğrusal ağırlıkları {−1, 0, +1} olacak şekilde
(BitNet b1.58, [arXiv 2402.17764](https://arxiv.org/abs/2402.17764)) eğitildiğinde kalitesini koruyabilir mi?
Eğitim sonrası kuantalama (PTQ) bunu başaramıyorsa, kuantalama farkındalıklı eğitim (QAT) başarabilir mi?

Bütün deneyler **resmî NCM kodu ve checkpoint'leri** üzerinde yapılır
([HazyResearch/ncm](https://github.com/HazyResearch/ncm), commit `c7027427`). Görev: Sudoku-hard (30 ipucu),
NCM makalesindeki kurulumla birebir aynı.

## Şu ana kadar doğrulanan sonuçlar (Sudoku-hard, 2000 test bulmacası)

| Deney | Çözülen tahta % | Not |
|---|---|---|
| Resmî NCM, FP | **86.30** (T4) / **86.15** (A100) | Makale: 86.1 |
| PTQ, INT8 ağırlık | 86.10 | Kayıpsız |
| PTQ, INT4 ağırlık | 81.90 | −4.4 puan |
| PTQ, ternary (denenen her varyant) | **0.00** | Tam çöküş; hata büyüklüğüyle tutarlı: model ≈0.28 göreli ağırlık hatasında %50'ye, ≈0.35'te %0'a düşüyor; Gauss ağırlıklarda en iyi eleman bazlı ternary yuvarlama bile ≈0.44 hata veriyor (kalibrasyonlu ternary PTQ denenmedi) |
| **QAT ince ayar (E3)**, ternary gövde, FP checkpoint'ten, 1.500 adım | **69.70** | 15 dk (A100); hâlâ yükseliyordu |
| QAT ince ayar (E3-uzun), aynı tarif, 5.000 adım | **71.15 ± 0.40** (3 seed: 71.60 / 70.85 / 71.00) | 55 dk (A100); 2.000. adımdan sonra düzleşiyor; FP'nin %82.6'sı |
| E3, 5.000 adım, α=1 (en kaba seviyeye daha çok kayıp ağırlığı) | 69.05 | α=0'dan kötü (−2.1 puan; gürültü aralığının dışında) |
| E3, 5.000 adım, INT8 aktivasyon kapalı (W1.58A16) | 71.05 | A8 ile fark gürültü içinde: INT8 aktivasyonun ölçülebilir bir bedeli yok |
| E3, 5.000 adım, kanal başına ternary ölçek | 71.80 | Tensör başına ölçekle aynı (fark gürültü içinde) |
| E3, FP öğretmenden damıtma + kademeli kuantalama | — | 2.000. adımda durduruldu; 1.000/2.000. adımda E3-uzun ile aynı (ilk 256 test) |
| E3, 5.000 adım, blok 0 FP (28 gövde katmanı ternary) | 71.30 | Fark gürültü içinde: PTQ'da en hassas blok, QAT sonrası açığın belirgin kaynağı değil |
| Baştan eğitim, FP (E1), 4.000 adım (makale bütçesinin ≈%2'si) | 21.75 | 34 dk (A100); kısa bütçe referansı |
| Baştan eğitim, ternary (E2), aynı bütçe | 16.45 | 40 dk (A100); aynı bütçede FP'nin ≈%76'sı |
| Baştan eğitim, FP, 12.000 adım (E1-uzun) | 25.65 | 110 dk; holdout kaybı 6.000. adımdan sonra yükseliyor |
| Baştan eğitim, ternary, 12.000 adım (E2-uzun) | 22.75 | 132 dk; FP'nin ≈%89'u; holdout kaybı yükselmiyor |
| Baştan eğitim, FP, 12.000 adım, LM1B tarzı rastgele oranlı bozma | 18.35 | Doğrusal bozmadan kötü; ezber sürüyor |

Kalan açık en kaba kodda toplanmıyor: ternary modele FP'nin en kaba kod doğruluğu verilseydi ≈%72.7 olurdu
(+1.1 puan; 14.6 puanlık açığın ≈1 puanı). Açığın geri kalanı sonraki seviyelerden geliyor ve daha ayrıntılı
yerelleştirilmedi.

Ayrıntılar, varsayımlar ve sınırlar: [BULGULAR.md](BULGULAR.md). Ham sonuçlar: [results_official/](../results_official/).

## Yeniden üretim

Her notebook Colab'da baştan sona çalışır, sonuçlarını Google Drive'daki `MyDrive/ncm_ptq/` klasörüne yazar ve
kopan oturumda kaldığı yerden devam eder. Sıra: Aşama 0–1 → 2a → 2b. Kullanılan dış kaynaklar commit veya sürüm
düzeyinde sabitlenmiştir:
- NCM: `c70274273f489abc73832b2b98584a8483b6b203`
- S-FLM Sudoku üreticisi: `30def79976ca3b1cdcb5fcbfa0165524e44001de`
