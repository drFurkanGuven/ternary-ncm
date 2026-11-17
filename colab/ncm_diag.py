"""
PTQ çöküş mekanizması teşhisi (yalnızca ölçüm, eğitim yok).

Sorular:
  1. Hata büyüklüğü mü, ternary'nin kendisi mi?  -> Gauss gürültüsü ile hata eğrisi (rel_err ekseninde).
  2. Hangi katman / blok çökertiyor?             -> tek katman, tek blok ve kümülatif blok taramaları.
  3. Hata seviyeler boyunca birikiyor mu?        -> teacher-forcing (TF) ile üretim doğruluğu seviye seviye,
                                                    FP'nin üretiminden ilk ayrıldığı seviye, blok başına sapma.
Ağırlıklar yerinde değiştirilir ve her ölçümden sonra anlık görüntüden geri yüklenir.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ncm_qat as Q

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- ağırlık işlemleri

def linears(net: nn.Module):
    return [(n, m) for n, m in net.named_modules() if isinstance(m, nn.Linear)]


def snapshot(net: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: m.weight.detach().clone() for n, m in linears(net)}


@torch.no_grad()
def restore(net: nn.Module, snap: Dict[str, torch.Tensor]) -> None:
    for n, m in linears(net):
        m.weight.copy_(snap[n])


def _q_int_ch(w, bits):
    qmax = 2 ** (bits - 1) - 1
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return (w / s).round().clamp(-qmax, qmax) * s


def _q_tern(w, per_channel=False):
    g = (w.abs().mean(dim=1, keepdim=True) if per_channel else w.abs().mean()).clamp(min=1e-8)
    return (w / g).round().clamp(-1, 1) * g


@torch.no_grad()
def perturb(net: nn.Module, names: List[str], kind: str, rel: float = 0.0, seed: int = 0) -> Dict[str, float]:
    """kind: tern | tern_ch | int8 | int4 | int3 | int2 | noise (Gauss, katman başına ||ΔW||/||W|| = rel)."""
    sel = set(names)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    err2 = ref2 = 0.0
    for n, m in linears(net):
        if n not in sel:
            continue
        w = m.weight.float()
        if kind == "tern":
            wq = _q_tern(w)
        elif kind == "tern_ch":
            wq = _q_tern(w, per_channel=True)
        elif kind.startswith("int"):
            wq = _q_int_ch(w, int(kind[3:]))
        elif kind == "noise":
            z = torch.randn(w.shape, generator=gen).to(w.device)
            wq = w + z * (rel * w.norm() / z.norm())
        else:
            raise ValueError(kind)
        err2 += float((wq - w).pow(2).sum()); ref2 += float(w.pow(2).sum())
        m.weight.copy_(wq.to(m.weight.dtype))
    return {"rel_err": (err2 / ref2) ** 0.5 if ref2 else 0.0, "n_layers": len(sel)}


# --------------------------------------------------------------------------- ölçümler

def _lvl_mean(a: torch.Tensor, levels: List[int]) -> List[float]:
    out, off = [], 0
    for l in levels:
        out.append(round(float(a[:, off:off + l].float().mean()), 4)); off += l
    return out


def _first_level(mask: torch.Tensor, levels: List[int]) -> torch.Tensor:
    """mask (B, Σl) bool: her örnek için mask'ın ilk True olduğu seviye; hiç yoksa len(levels)."""
    lvl_of_pos = torch.cat([torch.full((l,), i) for i, l in enumerate(levels)]).to(mask.device)
    big = torch.full_like(lvl_of_pos, len(levels))
    return torch.where(mask, lvl_of_pos[None, :], big[None, :]).min(dim=1).values


@torch.no_grad()
def generate_codes(net, tk, items, bs: int = 128, device: str = None):
    device = device or DEVICE
    """Q.evaluate_test ile aynı üretim; kodları ve çözülen metinleri döndürür."""
    net.eval()
    P = int(net.prefix_len)
    pad = tk.pre_tokenizer.pad_token_id
    codes, hyps = [], []
    for s in range(0, len(items), bs):
        x = torch.tensor([(list(it)[:P] + [pad] * P)[:P] for it in items[s:s + bs]], device=device)
        g = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=None)
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        ids = tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
        codes.append(g); hyps += [Q._decode(tk, r) for r in ids]
    return torch.cat(codes), hyps


@torch.no_grad()
def tf_logits(net, tk, seqs: np.ndarray, bs: int = 128, device: str = None):
    device = device or DEVICE
    net.eval()
    lg, ix = [], []
    for s in range(0, len(seqs), bs):
        x, idx = Q.build_inputs(net, tk, torch.as_tensor(seqs[s:s + bs], device=device), None)
        lg.append(net(x).float()); ix.append(idx)
    return torch.cat(lg), torch.cat(ix)


@torch.no_grad()
def block_outputs(net, tk, seqs: np.ndarray, device: str = None) -> List[torch.Tensor]:
    device = device or DEVICE
    outs = []
    hooks = [b.register_forward_hook(lambda _m, _i, o: outs.append((o[0] if isinstance(o, tuple) else o).float()))
             for b in net.transformer.blocks]
    try:
        x, _ = Q.build_inputs(net, tk, torch.as_tensor(seqs, device=device), None)
        net.eval(); net(x)
    finally:
        for h in hooks:
            h.remove()
    return outs


@torch.no_grad()
def fp_reference(net, tk, items, refs, seqs, levels, n_drift: int = 64) -> Dict:
    codes, hyps = generate_codes(net, tk, items)
    lg, idx = tf_logits(net, tk, seqs)
    return {"codes": codes, "logits": lg, "idx": idx, "blocks": block_outputs(net, tk, seqs[:n_drift]),
            "score": Q.score_sudoku(hyps, list(refs) if isinstance(refs[0], str) else [Q._decode(tk, r) for r in refs]),
            "n_drift": n_drift}


@torch.no_grad()
def measure(net, tk, items, refs, seqs, levels, fp: Optional[Dict] = None, full: bool = True) -> Dict:
    """full=False: yalnızca çözülen % (taramalar için hızlı). full=True: seviye ve blok düzeyinde teşhis."""
    ref_txt = [r if isinstance(r, str) else Q._decode(tk, r) for r in refs]
    codes, hyps = generate_codes(net, tk, items)
    sc = Q.score_sudoku(hyps, ref_txt)
    out = {"solved_pct": round(sc["solved_pct"], 2), "cell_pct": round(sc["cell_pct"], 2), "n": sc["n"]}
    if fp is not None:
        fl = _first_level(codes != fp["codes"][:len(codes)], levels)
        out["same_codes_as_fp_pct"] = round(100 * float((fl == len(levels)).float().mean()), 2)
        if not full:
            out["first_div_level_hist"] = torch.bincount(fl, minlength=len(levels) + 1).tolist()
    if not full:
        return out
    lg, idx = tf_logits(net, tk, seqs)
    out["tf_acc_per_level"] = _lvl_mean(lg.argmax(-1) == idx, levels)
    out["gen_acc_per_level"] = _lvl_mean(codes == idx, levels)          # üretilen kod = gerçek çözümün kodu
    ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), idx.reshape(-1), reduction="none").view(idx.shape)
    out["tf_ce_per_level"] = _lvl_mean(ce, levels)
    if fp is not None:
        lp, lq = F.log_softmax(fp["logits"], -1), F.log_softmax(lg, -1)
        out["kl_fp_q_per_level"] = _lvl_mean((lp.exp() * (lp - lq)).sum(-1), levels)
        out["first_div_level_hist"] = torch.bincount(fl, minlength=len(levels) + 1).tolist()
        bo = block_outputs(net, tk, seqs[:fp["n_drift"]])
        out["block_rel_drift"] = [round(float((a - b).norm() / b.norm()), 4) for a, b in zip(bo, fp["blocks"])]
    return out
