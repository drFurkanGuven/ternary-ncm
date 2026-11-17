"""
NCM Sudoku — Aşama 0/1: resmî checkpoint'i yeniden üret ve eğitim sonrası kuantalama (PTQ) taraması.

Resmî HazyResearch/ncm reposundaki scripts/eval_shard.py (Sudoku dalı) ve
scripts/eval_merge.py::exact() mantığını tek süreçte, değiştirmeden taklit eder.
Repo dosyalarına dokunmaz; ağırlıklar yüklendikten sonra bellekte kuantalanır.

Yalnızca üreteç (NextConceptTransformer) kuantalanır. Tokenizer (VQ-VAE) tam hassasiyette kalır,
çünkü üreteç onu `vqvae_local` adlı bir tuple içinde tutar ve net.modules() onu kapsamaz.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Ağırlık kuantalayıcıları (fake-quant: değerler kuantalanır, dtype float kalır)
#    Ağırlık şekli nn.Linear için [out_features, in_features]; "kanal" = çıkış satırı.
# ---------------------------------------------------------------------------

def q_int_per_channel(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Simetrik absmax, satır başına ölçek. bits=8 -> [-127,127], bits=4 -> [-7,7]."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return (w / scale).round().clamp(-qmax, qmax) * scale


def q_ternary_absmean_tensor(w: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58: gamma = mean|W| (tüm tensör), T = RoundClip(W/gamma, -1, 1), W' = gamma*T."""
    gamma = w.abs().mean().clamp(min=1e-8)
    return (w / gamma).round().clamp(-1, 1) * gamma


def q_ternary_absmean_channel(w: torch.Tensor) -> torch.Tensor:
    """Absmean ternary, satır başına gamma."""
    gamma = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return (w / gamma).round().clamp(-1, 1) * gamma


def q_ternary_twn_channel(w: torch.Tensor) -> torch.Tensor:
    """
    Ternary Weight Networks (Li & Liu, 2016) eşiği, satır başına:
        delta = 0.7 * mean|w|,  T = sign(w) * [|w| > delta]
        alpha = mean_{|w|>delta} |w|   (destek üzerinde L2-optimal ölçek)
    Absmean'den farkı: ölçek sıfır olmayan ağırlıklara göre yeniden hesaplanır.
    """
    a = w.abs()
    delta = 0.7 * a.mean(dim=1, keepdim=True)
    mask = (a > delta).to(w.dtype)
    alpha = (a * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return torch.sign(w) * mask * alpha


WEIGHT_QUANTIZERS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "int8_ch": lambda w: q_int_per_channel(w, 8),
    "int4_ch": lambda w: q_int_per_channel(w, 4),
    "tern_absmean_tensor": q_ternary_absmean_tensor,
    "tern_absmean_ch": q_ternary_absmean_channel,
    "tern_twn_ch": q_ternary_twn_channel,
}


def act_quant_int8_per_token(x: torch.Tensor) -> torch.Tensor:
    """BitNet tarzı aktivasyon kuantalaması: token başına absmax, [-127,127]."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
    return (x / scale).round().clamp(-127, 127) * scale


# ---------------------------------------------------------------------------
# 2. Kapsam (scope): hangi lineer katmanlar kuantalanacak
#    Üreteçteki adlar: transformer.wte (giriş izdüşümü), transformer.blocks.{i}.attn.c_attn / c_proj,
#    transformer.blocks.{i}.mlp.c_fc / c_proj, lm_heads.{k} (seviye başına çıkış katmanı).
#    Gerçek adlar notebook'ta list_linear_layers() ile yazdırılır; buradaki filtreler alt-dize eşleşmesidir.
# ---------------------------------------------------------------------------

def _is_io(name: str) -> bool:
    return ("wte" in name) or ("lm_head" in name) or ("ctx_" in name)


SCOPES: Dict[str, Callable[[str], bool]] = {
    "all": lambda n: True,
    "body": lambda n: not _is_io(n),             # giriş/çıkış katmanları hariç
    "attn": lambda n: (".attn." in n) and not _is_io(n),
    "mlp": lambda n: (".mlp." in n) and not _is_io(n),
    "io_only": _is_io,                            # yalnızca giriş/çıkış katmanları
}


def list_linear_layers(net: nn.Module) -> List[Dict[str, object]]:
    rows = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Linear):
            rows.append({"name": name, "shape": tuple(m.weight.shape), "params": m.weight.numel(),
                         "bias": m.bias is not None})
    return rows


@dataclass
class QuantReport:
    n_layers: int
    n_params: int
    rel_err: float          # ||W'-W||_F / ||W||_F, kuantalanan katmanlar üzerinden toplam
    zero_frac: float        # kuantalanan ağırlıklarda sıfır oranı (ternary için seyreklik)


def quantize_net(net: nn.Module, weight_scheme: Optional[str], scope: str,
                 act_int8: bool = False) -> QuantReport:
    """Ağırlıkları yerinde kuantalar; act_int8=True ise girişlere forward_pre_hook ile INT8 uygular."""
    sel = SCOPES[scope]
    fn = WEIGHT_QUANTIZERS[weight_scheme] if weight_scheme else None
    n_layers = n_params = 0
    err2 = ref2 = 0.0
    zeros = 0
    for name, m in net.named_modules():
        if not isinstance(m, nn.Linear) or not sel(name):
            continue
        n_layers += 1
        n_params += m.weight.numel()
        if fn is not None:
            with torch.no_grad():
                w = m.weight.data.float()
                wq = fn(w)
                err2 += float((wq - w).pow(2).sum())
                ref2 += float(w.pow(2).sum())
                zeros += int((wq == 0).sum())
                m.weight.data.copy_(wq.to(m.weight.dtype))
        if act_int8:
            m.register_forward_pre_hook(lambda _mod, inp: (act_quant_int8_per_token(inp[0]),) + tuple(inp[1:]))
    return QuantReport(
        n_layers=n_layers,
        n_params=n_params,
        rel_err=(err2 / ref2) ** 0.5 if ref2 > 0 else 0.0,
        zero_frac=zeros / n_params if (fn is not None and n_params) else 0.0,
    )


# ---------------------------------------------------------------------------
# 3. Resmî kodla yükleme ve değerlendirme (eval_shard.py Sudoku dalının birebir karşılığı)
# ---------------------------------------------------------------------------

class SudokuEvaluator:
    def __init__(self, ncm_root: str, config_path: str, ckpt_dir: str, data_dir: str,
                 n: int = 0, batch_size: int = 0, device: Optional[str] = None):
        sys.path.insert(0, ncm_root)
        from omegaconf import OmegaConf
        from models.next_concept import NextConceptTransformer
        from tokenization.vqvae_tokenizer import VQVAETokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = OmegaConf.load(config_path)
        assert str(cfg.eval.metric) == "exact", "Bu betik yalnızca Sudoku (metric: exact) içindir"
        self.cfg = cfg
        G = cfg.generation
        self.temp = float(G.get("temperature", 1.0))
        self.top_k = int(G.get("top_k", 1))
        self.top_p = float(G.get("top_p", 0.0))
        self.cfg_scale = float(G.get("cfg_scale", 0.0))
        assert not G.get("frequency_penalty") and not G.get("adjacency_penalty"), \
            "Bu config ek örnekleme cezaları kullanıyor; betik bunları desteklemiyor"

        cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ckpt_dir, cfg.checkpoints.tokenizer)
        self.tk = VQVAETokenizer.from_config(cfg.tokenizer)
        self.tk.vqvae = self.tk.vqvae.to(self.device).eval()
        self.net = NextConceptTransformer(vqvae_local=self.tk.vqvae, config=cfg.model).to(self.device)
        sd = torch.load(os.path.join(ckpt_dir, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
        self.net.load_state_dict(sd["model"], strict=True)
        self.net.eval()
        self.pristine = copy.deepcopy(self.net.state_dict())  # her deney öncesi geri yüklemek için

        self.P = int(cfg.model.prefix_len)
        self.EOS = self.tk.pre_tokenizer.eos_token_id
        self.PAD = self.tk.pre_tokenizer.pad_token_id
        split = cfg.eval.get("split", "test")
        n = n or int(cfg.eval.get("n_docs", 0))
        self.bs = batch_size or int(cfg.eval.batch_size)
        self.items = pickle.load(open(os.path.join(data_dir, cfg.data[split]), "rb"))[:n]
        refs = pickle.load(open(os.path.join(data_dir, cfg.data[f"{split}_refs"]), "rb"))[:n]
        assert len(self.items) == len(refs)
        self.refs = [r if isinstance(r, str) else self._dec(r) for r in refs]

    # --- eval_shard.py ile aynı yardımcılar ---
    def _prefix(self, item):
        ids = list(item)
        return (ids[:self.P] + [self.PAD] * self.P)[:self.P]

    def _dec(self, ids):
        ids = list(ids)
        if self.EOS in ids:
            ids = ids[:ids.index(self.EOS)]
        return self.tk.pre_tokenizer.decode([t for t in ids if t not in (self.EOS, self.PAD)],
                                            skip_special_tokens=True)

    def reset(self):
        """Ağırlıkları orijinaline döndürür ve tüm forward hook'larını temizler."""
        self.net.load_state_dict(self.pristine, strict=True)
        for m in self.net.modules():
            m._forward_pre_hooks.clear()
        self.net.eval()

    @torch.no_grad()
    def generate_all(self) -> List[str]:
        hyps: List[str] = []
        for s in range(0, len(self.items), self.bs):
            x = torch.tensor([self._prefix(it) for it in self.items[s:s + self.bs]], device=self.device)
            g = self.net.generate(None, x, temperature=self.temp, top_k=self.top_k, top_p=self.top_p,
                                  cfg_scale=self.cfg_scale, cfg_null_eos=self.EOS)
            _, zl = self.tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            ids = self.tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
            hyps += [self._dec(r) for r in ids]
        return hyps

    # --- eval_merge.py::exact() ile aynı skorlama ---
    @staticmethod
    def score(hyps: List[str], refs: List[str]) -> Dict[str, float]:
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
        exact_n = sum(h == r for h, r in zip(hyps, refs))
        cell = sum(sum(a == b for a, b in zip(cells(h), cells(r))) / max(1, len(cells(r)))
                   for h, r in zip(hyps, refs)) / n
        valid_n = sum(valid(h) for h in hyps)
        return {"solved_pct": 100.0 * exact_n / n, "cell_pct": 100.0 * cell,
                "valid_pct": 100.0 * valid_n / n, "n": n}

    def run(self, name: str, weight_scheme: Optional[str], scope: str, act_int8: bool = False) -> Dict:
        self.reset()
        rep = quantize_net(self.net, weight_scheme, scope, act_int8=act_int8)
        t0 = time.perf_counter()
        hyps = self.generate_all()
        if self.device == "cuda":
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        out = {"name": name, "weight_scheme": weight_scheme or "fp", "scope": scope,
               "act_int8": act_int8, **asdict(rep), **self.score(hyps, self.refs), "gen_seconds": sec}
        self.reset()
        return out


# ---------------------------------------------------------------------------
# 4. Deney listesi (Aşama 1)
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # (ad,                         ağırlık şeması,          kapsam,  INT8 aktivasyon)
    ("fp",                          None,                    "all",   False),
    ("fp+a8",                       None,                    "all",   True),
    ("int8_ch/all",                 "int8_ch",               "all",   False),
    ("int4_ch/all",                 "int4_ch",               "all",   False),
    ("int4_ch/body",                "int4_ch",               "body",  False),
    ("tern_absmean_tensor/all",     "tern_absmean_tensor",   "all",   False),
    ("tern_absmean_tensor/body",    "tern_absmean_tensor",   "body",  False),
    ("tern_absmean_tensor/body+a8", "tern_absmean_tensor",   "body",  True),
    ("tern_absmean_ch/body",        "tern_absmean_ch",       "body",  False),
    ("tern_twn_ch/body",            "tern_twn_ch",           "body",  False),
    ("tern_absmean_tensor/attn",    "tern_absmean_tensor",   "attn",  False),
    ("tern_absmean_tensor/mlp",     "tern_absmean_tensor",   "mlp",   False),
    ("tern_absmean_tensor/io_only", "tern_absmean_tensor",   "io_only", False),
]


def run_all(ev: SudokuEvaluator, out_json: str, experiments=EXPERIMENTS) -> List[Dict]:
    """Sonuçları her deneyden sonra diske yazar; dosyada olan deneyleri atlar (oturum koparsa devam)."""
    done: Dict[str, Dict] = {}
    if os.path.exists(out_json):
        for r in json.load(open(out_json)):
            done[r["name"]] = r
    for name, scheme, scope, a8 in experiments:
        if name in done:
            print(f"[atla] {name} (zaten var)")
            continue
        r = ev.run(name, scheme, scope, act_int8=a8)
        done[name] = r
        json.dump(list(done.values()), open(out_json, "w"), indent=2)
        print(f"[{name:30s}] solved={r['solved_pct']:6.2f}%  cell={r['cell_pct']:6.2f}%  "
              f"valid={r['valid_pct']:6.2f}%  rel_err={r['rel_err']:.3f}  zero={r['zero_frac']:.3f}  "
              f"({r['gen_seconds']:.1f}s)")
    return list(done.values())
