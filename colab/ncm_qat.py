"""
NCM Sudoku — Aşama 2b: resmî NCM kodu üzerinde FP ve ternary (BitNet b1.58) eğitim döngüsü.

Resmî repoda (HazyResearch/ncm) eğitim kodu yok; bu modül makaledeki Algoritma 1'i ve resmî
`generate()` / `quant.py` mantığını izleyerek teacher-forcing eğitimini kurar.

Girdi (resmî `generate()` ile aynı yapı, tek ileri geçişte tüm seviyeler):
    x = [ önek gömmesi (P=91) ; seviye-0 sorgusu (l0=1) ; idx_to_ncm_input(kodlar) (Σ l_k, k≥1) ]
    - önek gömmesi  = tokenizer.encoder.wte(önek token'ları)       (config: raw_token_prefix=true)
    - seviye-0      = net.l0_queries                                (config: query_level0=true)
    - seviye k≥1    = önceki seviyelerin birikmiş rekonstrüksiyonu, l_k uzunluğuna 'area' ile küçültülmüş
Kayıp (makale Denk. 10):  L = − Σ_k w_k Σ_i log p(r_k^i | z̃^(k−1)),   w_k = l_k^(−α) / Σ_j l_j^(1−α)
Koşullandırma bozma (makale §3.4, Ek F.3): koşul kodlarının bir kısmı aynı seviyenin kod-kitabından
    düzgün rastgele kodlarla değiştirilir; oran seviyeler boyunca doğrusal azalır. Hedefler temiz kalır.
    NOT: Sudoku için kesin oranlar ve α makalede verilmiyor; varsayılanlar (ortalama 0.5, α=0) bizim
    seçimimizdir ve FP kontrol deneyi (resmî %86'ya yaklaşılabiliyor mu?) ile sınanır.

Ternary: BitNet b1.58 — ağırlık absmean ile {−1,0,+1}·γ, aktivasyon token başına absmax INT8,
    her ikisi de STE ile. Bias ve normalizasyon katmanları tam hassasiyette kalır.
    Varsayılan kapsam 'body' (transformer blokları); giriş (wte) ve seviye çıkış katmanları (lm_heads)
    FP kalır (BitNet'in gömme/çıkış katmanlarını FP tutma pratiği; PTQ taramasında bu katmanlar en hassas çıktı).
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Ternary katman (BitNet b1.58) — repo bağımsız, yerelde birim testli
# ---------------------------------------------------------------------------

def ste_ternary_absmean(w: torch.Tensor, eps: float = 1e-5, lam: float = 1.0, per_channel: bool = False) -> torch.Tensor:
    """lam=1: tam ternary (BitNet b1.58). 0<lam<1: kademeli geçiş W + lam·(W_t − W) (HF 1.58-bit tarifi).
    per_channel=True: her çıkış satırına ayrı γ (W ≈ diag(γ)·T); çıkarımda γ çıktıya bir kez uygulanır, çarpmasızlık korunur."""
    gamma = (w.abs().mean(dim=1, keepdim=True) if per_channel else w.abs().mean()).clamp(min=eps)
    wq = (w / gamma).round().clamp(-1, 1) * gamma
    return w + lam * (wq - w).detach()


def ste_int8_absmax_per_token(x: torch.Tensor, eps: float = 1e-5, lam: float = 1.0) -> torch.Tensor:
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps) / 127.0
    xq = (x / scale).round().clamp(-127, 127) * scale
    return x + lam * (xq - x).detach()


class TernaryLinear(nn.Linear):
    """nn.Linear ile aynı arayüz; ileri geçişte ternary ağırlık + (isteğe bağlı) INT8 aktivasyon, STE ile."""

    def __init__(self, in_features, out_features, bias=True, act_int8=True, per_channel=False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.act_int8 = act_int8
        self.per_channel = per_channel
        self.quant_lambda = 1.0   # 1.0 = tam kuantalı; yalnızca eğitimin başındaki kademeli geçişte <1

    def forward(self, x):
        lam = self.quant_lambda
        w = ste_ternary_absmean(self.weight.float(), lam=lam, per_channel=self.per_channel).to(x.dtype)
        if self.act_int8:
            x = ste_int8_absmax_per_token(x, lam=lam)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

    @torch.no_grad()
    def ternary_stats(self) -> Dict[str, float]:
        g = (self.weight.abs().mean(dim=1, keepdim=True) if self.per_channel else self.weight.abs().mean()).clamp(min=1e-5)
        t = (self.weight / g).round().clamp(-1, 1)
        n = t.numel()
        return {"neg": float((t == -1).sum()) / n, "zero": float((t == 0).sum()) / n, "pos": float((t == 1).sum()) / n}


def _is_io(name: str) -> bool:
    return ("wte" in name) or ("lm_head" in name) or ("ctx_" in name)


SCOPES = {
    "body": lambda n: not _is_io(n),
    "all": lambda n: True,
}


def ternarize_linears(net: nn.Module, scope: str = "body", act_int8: bool = True, per_channel: bool = False) -> List[str]:
    """Kapsamdaki nn.Linear katmanlarını aynı ağırlıklarla TernaryLinear'a çevirir (yerinde). Değişen adları döndürür."""
    sel = SCOPES[scope]
    targets = [(n, m) for n, m in net.named_modules()
               if type(m) is nn.Linear and sel(n)]
    for name, lin in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = net.get_submodule(parent_name) if parent_name else net
        new = TernaryLinear(lin.in_features, lin.out_features, bias=lin.bias is not None, act_int8=act_int8,
                            per_channel=per_channel, device=lin.weight.device, dtype=lin.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(lin.weight)
            if lin.bias is not None:
                new.bias.copy_(lin.bias)
        setattr(parent, attr, new)
    return [n for n, _ in targets]


# ---------------------------------------------------------------------------
# 2. Kayıp ağırlıkları ve koşullandırma bozma — repo bağımsız, yerelde birim testli
# ---------------------------------------------------------------------------

def set_quant_lambda(net: nn.Module, lam: float) -> int:
    """Tüm TernaryLinear katmanlarında kademeli kuantalama katsayısını ayarlar; kaç katman etkilendiğini döndürür."""
    n = 0
    for m in net.modules():
        if isinstance(m, TernaryLinear):
            m.quant_lambda = float(lam); n += 1
    return n


def level_position_weights(levels: List[int], alpha: float) -> torch.Tensor:
    """Makale Denk. 10: her kod pozisyonu için w_k = l_k^(−α) / Σ_j l_j^(1−α). Toplamları her zaman 1'dir."""
    denom = sum(l ** (1.0 - alpha) for l in levels)
    return torch.cat([torch.full((l,), (l ** (-alpha)) / denom) for l in levels])


def linear_corruption_rates(n_cond_levels: int, mean: float = 0.5, spread: float = 0.5) -> List[float]:
    """Koşul olarak kullanılan seviyeler (0..K−2) için doğrusal azalan oranlar; ortalama = mean.
    spread = ilk ve son seviye oranları arasındaki fark."""
    if n_cond_levels == 1:
        return [mean]
    hi, lo = mean + spread / 2, mean - spread / 2
    assert 0.0 <= lo and hi <= 1.0, "oranlar [0,1] dışında"
    return [hi + (lo - hi) * k / (n_cond_levels - 1) for k in range(n_cond_levels)]


def corrupt_codes_random_rate(idx: torch.Tensor, levels: List[int], vocab_sizes: List[int],
                              low: float = 0.0, high: float = 1.0, skip_level0: bool = True,
                              generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Resmî LM1B config'indeki `code_corruption` bloğunun yorumu (mode: uniform, random_rate: true, low, high,
    skip_level0). Bu bloğu okuyan kod yayımlanmadığı için yorum bizimdir:
      - her dizi için bir oran r ~ U(low, high) çekilir (ortalama (low+high)/2; low=0, high=1 → 0.5),
      - koşul olarak kullanılan seviyelerdeki (0..K−2) her kod r olasılıkla aynı seviyenin kod-kitabından düzgün
        rastgele bir kodla değiştirilir,
      - skip_level0=True ise en kaba seviye (seviye 0) hiç bozulmaz; son seviye zaten koşul olarak kullanılmaz.
    """
    B = idx.size(0)
    r = low + (high - low) * torch.rand(B, 1, device=idx.device, generator=generator)
    out = idx.clone()
    off = 0
    for k, l in enumerate(levels):
        if k < len(levels) - 1 and not (skip_level0 and k == 0):
            seg = out[:, off:off + l]
            mask = torch.rand(seg.shape, device=idx.device, generator=generator) < r
            rnd = torch.randint(0, vocab_sizes[k], seg.shape, device=idx.device, generator=generator)
            out[:, off:off + l] = torch.where(mask, rnd, seg)
        off += l
    return out


def corrupt_codes(idx: torch.Tensor, levels: List[int], vocab_sizes: List[int], rates: List[float],
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """idx: (B, Σ l_k). Seviye k<K−1 kodlarını rates[k] olasılıkla aynı seviyenin kod-kitabından düzgün rastgele
    kodlarla değiştirir. Son seviye hiçbir zaman koşul olarak kullanılmadığından değiştirilmez."""
    out = idx.clone()
    off = 0
    for k, l in enumerate(levels):
        if k < len(levels) - 1 and rates[k] > 0:
            seg = out[:, off:off + l]
            mask = torch.rand(seg.shape, device=idx.device, generator=generator) < rates[k]
            rnd = torch.randint(0, vocab_sizes[k], seg.shape, device=idx.device, generator=generator)
            out[:, off:off + l] = torch.where(mask, rnd, seg)
        off += l
    return out


# ---------------------------------------------------------------------------
# 3. Resmî kodla yükleme, girdi kurma, değerlendirme
# ---------------------------------------------------------------------------

def load_official(ncm_root: str, config_path: str, ckpt_dir: str, device: str = "cuda",
                  load_generator: bool = True):
    """Resmî tokenizer'ı (dondurulmuş) ve üreteci yükler. load_generator=False ise üreteç rastgele başlatılır."""
    if ncm_root not in sys.path:
        sys.path.insert(0, ncm_root)
    from omegaconf import OmegaConf
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    cfg = OmegaConf.load(config_path)
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ckpt_dir, cfg.checkpoints.tokenizer)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    for p in tk.vqvae.parameters():
        p.requires_grad_(False)
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    if load_generator:
        sd = torch.load(os.path.join(ckpt_dir, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
        net.load_state_dict(sd["model"], strict=True)
    else:
        # Resmî sınıf kendi _init_weights'ini (nanoGPT tarzı N(0, 0.02)) __init__ içinde çağırmıyor; yayımlanmamış
        # eğitim betiğinin çağırdığını varsayıyoruz. Aksi hâlde nn.Embedding varsayılanı N(0,1) olurdu.
        # Bu, sınıfın __init__'te açıkça başlattığı parametreleri (l0_queries, wpe) etkilemez.
        net.apply(net._init_weights)
    assert bool(cfg.model.get("raw_token_prefix", False)) and not bool(cfg.model.get("contextual_prefix", False)), \
        "Bu modül raw_token_prefix=true, contextual_prefix=false kurulumunu varsayar (Sudoku config'i)"
    assert bool(cfg.model.get("query_level0", False)), "Bu modül query_level0=true varsayar"
    assert not bool(getattr(net, "predict_level0", False)), "predict_level0 desteklenmiyor"
    return cfg, tk, net


def build_inputs(net, tk, seqs: torch.Tensor, rates=None,
                 generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seqs: (B, 180) = [BOS] bulmaca(89) [BOS] çözüm(89).  Döndürür: x (B, P+Σl, D), hedef kodlar idx (B, Σl).
    Önek ve kod girdileri dondurulmuş tokenizer'dan gelir (gradyansız); yalnızca l0_queries eğitilebilir.
    rates: None (bozma yok) | seviye başına oran listesi (doğrusal mod) |
           {"mode": "random_uniform", "low", "high", "skip_level0"} (LM1B tarzı mod)
    """
    vq = tk.vqvae.vq
    P = int(net.prefix_len)
    prefix_tokens, solution = seqs[:, :P], seqs[:, P:]
    with torch.no_grad():
        idx = tk.vqvae.encode(solution)                       # (B, Σl)
        if rates is None:
            cond_idx = idx
        elif isinstance(rates, dict):
            cond_idx = corrupt_codes_random_rate(idx, list(vq.levels), list(vq.vqvae_vocab_sizes),
                                                 rates["low"], rates["high"], rates["skip_level0"], generator)
        else:
            cond_idx = corrupt_codes(idx, list(vq.levels), list(vq.vqvae_vocab_sizes), rates, generator)
        x_cond = vq.idx_to_ncm_input(cond_idx)                # (B, Σ_{k≥1} l_k, D)
        z_prefix = tk.vqvae.encoder.wte(prefix_tokens)[:, :P, :]
    q0 = net.build_l0_query_input(seqs.size(0), z_prefix.dtype)  # (B, l0, D), gradyanlı
    return torch.cat([z_prefix, q0, x_cond], dim=1), idx


def ncm_loss(net, x: torch.Tensor, idx: torch.Tensor, pos_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ağırlıklı kod çapraz entropisi (dizi başına toplam, batch ortalaması) ve logitler."""
    logits = net(x)                                           # (B, Σl, V)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), idx.reshape(-1), reduction="none")
    loss = (ce.view(idx.shape) * pos_w.to(ce.device)[None, :]).sum(dim=1).mean()
    return loss, logits


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, pos_w: torch.Tensor,
            temperature: float = 1.0) -> torch.Tensor:
    """Konum başına KL(p_öğretmen || p_öğrenci)·T², CE ile aynı seviye ağırlıklarıyla; dizi başına toplam, batch ort."""
    T = float(temperature)
    log_ps = F.log_softmax(student_logits.float() / T, dim=-1)
    log_pt = F.log_softmax(teacher_logits.float() / T, dim=-1)
    kl = (log_pt.exp() * (log_pt - log_ps)).sum(-1)          # (B, Σl)
    return (kl * pos_w.to(kl.device)[None, :]).sum(dim=1).mean() * (T * T)


def teacher_inputs(teacher, student_x: torch.Tensor, P: int) -> torch.Tensor:
    """Öğrencinin girdisinde yalnızca öğrenilen l0 sorgularını öğretmeninkiyle değiştirir (önek ve kodlar ortak)."""
    q0_t = teacher.build_l0_query_input(student_x.size(0), student_x.dtype)
    l0 = q0_t.size(1)
    return torch.cat([student_x[:, :P], q0_t, student_x[:, P + l0:]], dim=1).detach()


@torch.no_grad()
def per_level_accuracy(logits: torch.Tensor, idx: torch.Tensor, levels: List[int]) -> List[float]:
    pred = logits.argmax(-1)
    out, off = [], 0
    for l in levels:
        out.append(float((pred[:, off:off + l] == idx[:, off:off + l]).float().mean()))
        off += l
    return out


@torch.no_grad()
def consistency_check(net, tk, prefixes: torch.Tensor) -> Dict[str, float]:
    """
    Girdi kurucumuzun resmî generate() ile birebir uyumlu olduğunu sınar:
    greedy generate → kodlar g;  g'den teacher-forcing girdisi kur → forward → argmax  ==  g  olmalı
    (her seviyenin tahmini yalnızca önceki seviyelerin kodlarına bağlı olduğu için).
    """
    net.eval()
    g = net.generate(None, prefixes, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=None)
    vq = tk.vqvae.vq
    P = int(net.prefix_len)
    x_cond = vq.idx_to_ncm_input(g)
    z_prefix = tk.vqvae.encoder.wte(prefixes)[:, :P, :]
    x = torch.cat([z_prefix, net.build_l0_query_input(prefixes.size(0), z_prefix.dtype), x_cond], dim=1)
    pred = net(x).argmax(-1)
    agree = (pred == g).float()
    off, per = 0, []
    for l in vq.levels:
        per.append(float(agree[:, off:off + l].mean()))
        off += l
    return {"agree_all": float(agree.mean()), "agree_per_level": per,
            "exact_sequences": float((pred == g).all(dim=1).float().mean())}


def _decode(tk, ids):
    eos, pad = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    ids = list(ids)
    if eos in ids:
        ids = ids[:ids.index(eos)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (eos, pad)], skip_special_tokens=True)


def score_sudoku(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    """Resmî scripts/eval_merge.py::exact() ile aynı skorlama."""
    def cells(t):
        return t.replace("|", "")

    def valid(t):
        g = cells(t)
        if len(g) != 81 or not g.isdigit() or "0" in g:
            return False
        rows_ = [g[i * 9:(i + 1) * 9] for i in range(9)]
        cols_ = ["".join(r[j] for r in rows_) for j in range(9)]
        boxes = ["".join(rows_[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3))
                 for br in (0, 3, 6) for bc in (0, 3, 6)]
        return all(sorted(x) == list("123456789") for x in rows_ + cols_ + boxes)

    n = len(hyps)
    return {"solved_pct": 100.0 * sum(h == r for h, r in zip(hyps, refs)) / n,
            "cell_pct": 100.0 * sum(sum(a == b for a, b in zip(cells(h), cells(r))) / max(1, len(cells(r)))
                                    for h, r in zip(hyps, refs)) / n,
            "valid_pct": 100.0 * sum(valid(h) for h in hyps) / n, "n": n}


@torch.no_grad()
def evaluate_test(net, tk, items: List, refs: List, batch_size: int = 128, device: str = "cuda",
                  amp_dtype: Optional[torch.dtype] = None) -> Dict[str, float]:
    """Resmî eval_shard.py (Sudoku, greedy) ile aynı üretim + resmî skorlama."""
    net.eval()
    P = int(net.prefix_len)
    pad = tk.pre_tokenizer.pad_token_id
    ref_txt = [r if isinstance(r, str) else _decode(tk, r) for r in refs]
    hyps = []
    for s in range(0, len(items), batch_size):
        x = torch.tensor([(list(it)[:P] + [pad] * P)[:P] for it in items[s:s + batch_size]], device=device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None and device == "cuda"):
            g = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=None)
            _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            ids = tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
        hyps += [_decode(tk, r) for r in ids]
    return score_sudoku(hyps, ref_txt)


# ---------------------------------------------------------------------------
# 4. Eğitim döngüsü (makale Tablo 6: AdamW β=(0.9,0.99), wd 0.01, tepe lr 3e-4, taban 3e-5, ısınma, clip 1)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    run_name: str
    steps: int = 5000
    batch_size: int = 256
    micro_batch: int = 128            # bellek yetmezse küçült; batch_size'ın böleni olmalı
    peak_lr: float = 3e-4
    floor_lr: float = 3e-5
    warmup: int = 500
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.99)
    grad_clip: float = 1.0
    alpha: float = 0.0
    corruption_mode: str = "linear"   # "linear" (seviye başına sabit, doğrusal azalan) | "random_uniform" (LM1B tarzı)
    corruption_mean: float = 0.5      # yalnızca linear modda
    corruption_spread: float = 0.5    # yalnızca linear modda
    corruption_low: float = 0.0       # yalnızca random_uniform modda
    corruption_high: float = 1.0      # yalnızca random_uniform modda
    corruption_skip_level0: bool = True  # yalnızca random_uniform modda
    seed: int = 0
    log_every: int = 50
    eval_every: int = 1000            # tutulan eğitim dilimi üzerinde teacher-forcing kaybı/doğruluğu
    ckpt_every: int = 500
    test_n_during: int = 256          # ara test değerlendirmesinde kaç test bulmacası (0 = kapalı)
    quant_warmup: int = 0             # >0: ilk bu kadar adımda kuantalama katsayısı λ 0→1 doğrusal artar
    kd_weight: float = 0.0            # >0: kayıp = (1−w)·CE + w·KD; train(teacher=...) gerekir
    kd_temperature: float = 1.0


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.peak_lr * (step + 1) / cfg.warmup
    prog = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.floor_lr + 0.5 * (cfg.peak_lr - cfg.floor_lr) * (1 + math.cos(math.pi * prog))


def make_optimizer(net: nn.Module, cfg: TrainConfig):
    decay, no_decay = [], []
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg.peak_lr, betas=cfg.betas)


def pick_amp_dtype() -> Optional[torch.dtype]:
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def train(net, tk, train_seqs: np.ndarray, holdout_seqs: np.ndarray, cfg: TrainConfig, out_dir: str,
          test_items=None, test_refs=None, device: str = "cuda", teacher=None) -> List[Dict]:
    """Drive'a düzenli checkpoint yazar; aynı run_name ile yeniden çağrılırsa kaldığı adımdan devam eder.
    teacher: kd_weight>0 ise dondurulmuş FP öğretmen (aynı tokenizer'la yüklenmiş NextConceptTransformer)."""
    if cfg.kd_weight > 0:
        assert teacher is not None, "kd_weight>0 için teacher verilmeli"
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    P = int(net.prefix_len)
    use_lambda = cfg.quant_warmup > 0
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{cfg.run_name}.pt")
    log_path = os.path.join(out_dir, f"{cfg.run_name}.log.json")
    vq = tk.vqvae.vq
    levels = list(vq.levels)
    pos_w = level_position_weights(levels, cfg.alpha).to(device)
    if cfg.corruption_mode == "random_uniform":
        rates = {"mode": "random_uniform", "low": cfg.corruption_low, "high": cfg.corruption_high,
                 "skip_level0": cfg.corruption_skip_level0}
    elif cfg.corruption_mode == "linear":
        rates = linear_corruption_rates(len(levels) - 1, cfg.corruption_mean, cfg.corruption_spread) \
            if cfg.corruption_mean > 0 else None
    else:
        raise ValueError(f"bilinmeyen corruption_mode: {cfg.corruption_mode}")
    amp = pick_amp_dtype()
    scaler = torch.cuda.amp.GradScaler(enabled=(amp == torch.float16))
    opt = make_optimizer(net, cfg)
    step, logs = 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        logs = ck.get("logs", [])
        print(f"[devam] {cfg.run_name}: adım {step}'ten sürüyor")

    rng = np.random.default_rng(cfg.seed + step)
    gen = torch.Generator(device=device); gen.manual_seed(cfg.seed * 1000003 + step)
    hold = torch.tensor(holdout_seqs[:1024], dtype=torch.long, device=device)
    assert cfg.batch_size % cfg.micro_batch == 0
    n_micro = cfg.batch_size // cfg.micro_batch
    t0 = time.time()
    net.train()
    while step < cfg.steps:
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, cfg)
        lam = min(1.0, (step + 1) / cfg.quant_warmup) if use_lambda else 1.0
        if use_lambda:
            set_quant_lambda(net, lam)
        opt.zero_grad(set_to_none=True)
        bidx = rng.integers(0, len(train_seqs), size=cfg.batch_size)
        batch = torch.tensor(train_seqs[bidx], dtype=torch.long, device=device)
        tot, tot_ce, tot_kd = 0.0, 0.0, 0.0
        for m in range(n_micro):
            mb = batch[m * cfg.micro_batch:(m + 1) * cfg.micro_batch]
            x, idx = build_inputs(net, tk, mb, rates, gen)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                ce, s_logits = ncm_loss(net, x, idx, pos_w)
                if cfg.kd_weight > 0:
                    with torch.no_grad():
                        t_logits = teacher(teacher_inputs(teacher, x, P))
                    kd = kd_loss(s_logits, t_logits, pos_w, cfg.kd_temperature)
                    loss = (1.0 - cfg.kd_weight) * ce + cfg.kd_weight * kd
                    tot_kd += float(kd) / n_micro
                else:
                    loss = ce
            scaler.scale(loss / n_micro).backward()
            tot += float(loss) / n_micro
            tot_ce += float(ce) / n_micro
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        scaler.step(opt); scaler.update()
        step += 1

        if step % cfg.log_every == 0:
            rec = {"step": step, "loss": round(tot, 5), "lr": lr_at(step - 1, cfg), "grad_norm": float(gn),
                   "elapsed_min": round((time.time() - t0) / 60, 2)}
            extra = ""
            if cfg.kd_weight > 0:
                rec["ce"] = round(tot_ce, 5); rec["kd"] = round(tot_kd, 5)
                extra += f"  ce={tot_ce:.4f}  kd={tot_kd:.4f}"
            if use_lambda:
                rec["quant_lambda"] = round(lam, 4)
                extra += f"  λ={lam:.2f}"
            logs.append(rec)
            print(f"[{cfg.run_name}] adım {step:6d}/{cfg.steps}  kayıp={tot:.4f}{extra}  lr={rec['lr']:.2e}  "
                  f"|g|={float(gn):.2f}  ({rec['elapsed_min']:.1f} dk)", flush=True)

        if step % cfg.eval_every == 0 or step == cfg.steps:
            net.eval()
            if use_lambda:
                set_quant_lambda(net, 1.0)   # ara ölçümler her zaman gerçek ternary modelle
            with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                xs, ids = build_inputs(net, tk, hold[:512], None)
                hl, hlog = ncm_loss(net, xs, ids, pos_w)
            rec = {"step": step, "holdout_tf_loss": float(hl),
                   "holdout_tf_acc_per_level": [round(a, 4) for a in per_level_accuracy(hlog, ids, levels)]}
            if cfg.test_n_during and test_items is not None:
                # Üretim FP32'de yapılır (resmî değerlendirmeyle aynı); yalnızca eğitim karma hassasiyetlidir.
                rec["test_subset"] = evaluate_test(net, tk, test_items[:cfg.test_n_during],
                                                   test_refs[:cfg.test_n_during], device=device, amp_dtype=None)
            logs.append(rec)
            print(f"  [eval] {json.dumps(rec)}", flush=True)
            if use_lambda:
                set_quant_lambda(net, lam)
            net.train()

        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            ck = {"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "logs": logs,
                  "cfg": asdict(cfg)}
            if scaler.is_enabled():
                ck["scaler"] = scaler.state_dict()
            torch.save(ck, ckpt_path + ".tmp"); os.replace(ckpt_path + ".tmp", ckpt_path)
            json.dump(logs, open(log_path, "w"), indent=1)
    if use_lambda:
        set_quant_lambda(net, 1.0)
    net.eval()
    return logs
