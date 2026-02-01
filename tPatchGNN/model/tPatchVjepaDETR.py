# -*- coding: utf-8 -*-
"""
Python 3.8 compatible

Pipeline:
1) patchify_by_len using patch_index = per-patch lengths
2) Intra-patch GAT: aggregate timesteps -> patch tokens
3) Self-Attn #1 (imputation stage) -> loss1 computed here (teacher vs masked)
4) Self-Attn #2 -> keep your pipeline
5) 5-query cross-attn classifier -> BCE loss2
Total = w_recon*loss1 + w_bce*loss2
"""
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Basic modules
# =========================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for (B, M, D)."""
    def __init__(self, d_model: int, max_len: int = 2048):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class IntraPatchGraphAggregator(nn.Module):
    """
    Intra-patch attention with time-diff bias + learnable readout query.
    Input: timestep tokens (N,L,D), timestamps (N,L,1), mask (N,L,1)
    Output: patch vector (N,D)
    """
    def __init__(self, d_model: int, nhead: int = 4, tau_seconds: float = 300.0):
        super(IntraPatchGraphAggregator, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.tau = float(tau_seconds)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.readout = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, tokens: torch.Tensor, ts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        tokens: (N,L,D)
        ts:     (N,L,1) seconds (unix ok, we make relative)
        mask:   (N,L,1) 1=valid,0=pad
        """
        N, L, D = tokens.shape
        assert D == self.d_model

        # relative time to avoid huge unix scale
        t = (ts - ts[:, :1, :]).squeeze(-1)  # (N,L)
        dt = (t[:, :, None] - t[:, None, :]).abs()  # (N,L,L)
        w = torch.exp(-dt / max(self.tau, 1e-6))
        bias = torch.log(w + 1e-12)  # (N,L,L)

        q = self.q_proj(tokens).view(N, L, self.nhead, self.d_head).transpose(1, 2)  # (N,h,L,dh)
        k = self.k_proj(tokens).view(N, L, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(tokens).view(N, L, self.nhead, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (N,h,L,L)
        attn = attn + bias[:, None, :, :]

        key_pad = (mask.squeeze(-1) <= 0)  # (N,L)
        attn = attn.masked_fill(key_pad[:, None, None, :], -1e9)

        A = torch.softmax(attn, dim=-1)
        out = torch.matmul(A, v).transpose(1, 2).contiguous().view(N, L, D)
        out = self.out_proj(out)
        out = out * mask

        # readout pooling
        q0 = self.readout.expand(N, 1, D)  # (N,1,D)
        qh = q0.view(N, 1, self.nhead, self.d_head).transpose(1, 2)          # (N,h,1,dh)
        kh = out.view(N, L, self.nhead, self.d_head).transpose(1, 2)         # (N,h,L,dh)
        vh = out.view(N, L, self.nhead, self.d_head).transpose(1, 2)         # (N,h,L,dh)

        pool = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)  # (N,h,1,L)
        pool = pool.masked_fill(key_pad[:, None, None, :], -1e9)
        P = torch.softmax(pool, dim=-1)
        pooled = torch.matmul(P, vh).transpose(1, 2).contiguous().view(N, 1, D)
        return pooled.squeeze(1)  # (N,D)


class QueryClassifier5(nn.Module):
    """5 learnable queries cross-attend to patch tokens -> logits (B,5)."""
    def __init__(self, d_model: int, nhead: int = 4):
        super(QueryClassifier5, self).__init__()
        self.num_queries = 5
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.query_to_logit = nn.Linear(d_model, 1)

    def forward(self, patch_tokens: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, P, D = patch_tokens.shape
        q = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,5,D)

        key_padding_mask = None
        if patch_mask is not None:
            key_padding_mask = (patch_mask <= 0)

        z, _ = self.cross_attn(q, patch_tokens, patch_tokens, key_padding_mask=key_padding_mask)
        logits = self.query_to_logit(z).squeeze(-1)  # (B,5)
        return logits


# =========================
# Model
# =========================

@dataclass
class Args:
    device: str = "cpu"
    in_dim: int = 5
    hid_dim: int = 64
    te_dim: int = 8
    npatch: int = 12
    patch_len: int = 16
    nhead: int = 4
    tf_layer: int = 1
    tau_seconds_intra: float = 300.0


class tPatchGNN_Classifier(nn.Module):
    """
    patch_index is LENGTHS per patch (B,P), data_sequence is flattened timesteps (B,Lmax,D).
    """
    def __init__(self, args: Args):
        super(tPatchGNN_Classifier, self).__init__()
        self.device_name = args.device
        self.in_dim = int(args.in_dim)
        self.hid_dim = int(args.hid_dim)
        self.te_dim = int(args.te_dim)
        self.npatch = int(args.npatch)
        self.patch_len = int(args.patch_len)
        self.nhead = int(args.nhead)
        self.tf_layer = int(args.tf_layer)

        assert self.hid_dim % self.nhead == 0, "hid_dim must be divisible by nhead"

        # Learnable time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        # timestep token -> hid_dim
        self.point_proj = nn.Sequential(
            nn.Linear(self.in_dim + self.te_dim, self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, self.hid_dim),
        )

        # intra patch aggregation operates on hid_dim (FIXED)
        self.intra = IntraPatchGraphAggregator(
            d_model=self.hid_dim,
            nhead=self.nhead,
            tau_seconds=args.tau_seconds_intra
        )

        # use patch_exists as a gate (not concatenation)
        self.exists_gate = nn.Sequential(
            nn.Linear(1, self.hid_dim),
            nn.Sigmoid()
        )

        # mask token for masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hid_dim))

        # patch self-attn blocks
        self.ADD_PE = PositionalEncoding(self.hid_dim)
        self.self_attn1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=self.nhead, batch_first=True),
            num_layers=self.tf_layer,
        )
        self.self_attn2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=self.nhead, batch_first=True),
            num_layers=self.tf_layer,
        )

        self.classifier = QueryClassifier5(d_model=self.hid_dim, nhead=self.nhead)

    def LearnableTE(self, tt: torch.Tensor) -> torch.Tensor:
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], dim=-1)

    def patchify_by_len(
        self,
        data_sequence: torch.Tensor,
        ts_sequence: torch.Tensor,
        patch_mask: torch.Tensor,
        patch_index: torch.Tensor,
        num_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patch_index: (B,P) length per patch
        patch_mask:  (B,P) 1 valid patch else 0
        num_batch:   (B,) valid length in sequence
        returns:
          X_patch (B,P,L,D), ts_patch (B,P,L,1), mask_pt (B,P,L,1), mask_patch (B,P)
        """
        B, Lmax, D = data_sequence.shape
        P = patch_mask.shape[1]
        L = self.patch_len
        device = data_sequence.device

        X_patch = torch.zeros((B, P, L, D), device=device, dtype=data_sequence.dtype)
        ts_patch = torch.zeros((B, P, L, 1), device=device, dtype=torch.float32)
        mask_pt = torch.zeros((B, P, L, 1), device=device, dtype=torch.float32)

        mask_patch = patch_mask.float()
        patch_len = patch_index.long().clamp(min=0)

        for b in range(B):
            valid_len = int(num_batch[b].item())
            valid_len = max(0, min(valid_len, Lmax))

            cursor = 0
            for p in range(P):
                if mask_patch[b, p].item() <= 0:
                    continue

                seg_len = int(patch_len[b, p].item())
                if seg_len <= 0 or cursor >= valid_len:
                    continue

                end = min(cursor + seg_len, valid_len)
                real_len = end - cursor
                if real_len <= 0:
                    continue

                take = min(real_len, L)
                X_patch[b, p, :take, :] = data_sequence[b, cursor:cursor + take, :]
                ts_patch[b, p, :take, 0] = ts_sequence[b, cursor:cursor + take].float()
                mask_pt[b, p, :take, 0] = 1.0

                cursor = end

        return X_patch, ts_patch, mask_pt, mask_patch

    def _build_patch_tokens0(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        data_sequence = batch["data_sequence"]
        ts_sequence = batch["ts_sequence"]
        patch_mask = batch["patch_mask"]
        patch_index = batch["patch_index"]
        num_batch = batch["num_batch"]

        B, Lmax, D = data_sequence.shape
        assert D == self.in_dim, "Expected in_dim={}, got D={}".format(self.in_dim, D)
        assert patch_mask.shape[1] == self.npatch, "Expected npatch={}, got P={}".format(self.npatch, patch_mask.shape[1])

        X_patch, ts_patch, mask_pt, mask_patch = self.patchify_by_len(
            data_sequence, ts_sequence, patch_mask, patch_index, num_batch
        )  # (B,P,L,D), (B,P,L,1), (B,P,L,1), (B,P)

        te = self.LearnableTE(ts_patch)                    # (B,P,L,te_dim)
        x_step = torch.cat([X_patch, te], dim=-1)          # (B,P,L,in_dim+te_dim)

        x_step = x_step.view(B * self.npatch, self.patch_len, self.in_dim + self.te_dim)
        m_step = mask_pt.view(B * self.npatch, self.patch_len, 1)
        t_step = ts_patch.view(B * self.npatch, self.patch_len, 1)

        tok = self.point_proj(x_step) * m_step             # (B*P,L,hid_dim)
        patch_feat = self.intra(tok, t_step, m_step)        # (B*P,hid_dim)

        # exists gate
        patch_exists = (m_step.sum(dim=1) > 0).float()      # (B*P,1)
        gate = self.exists_gate(patch_exists)               # (B*P,hid_dim)
        patch_feat = patch_feat * gate                      # gate invalid / weak patches

        x0 = patch_feat.view(B, self.npatch, self.hid_dim)  # (B,P,H)
        x0 = x0 * mask_patch.unsqueeze(-1)
        return x0, mask_patch

    def encode_after_impute_attn(
        self,
        batch: Dict[str, Any],
        patch_drop_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, patch_mask = self._build_patch_tokens0(batch)

        if patch_drop_mask is not None:
            drop = (patch_drop_mask > 0) & (patch_mask > 0)
            x0 = x0.clone()
            x0[drop] = self.mask_token.expand_as(x0)[drop]

        x = self.ADD_PE(x0)
        x1 = self.self_attn1(x)
        x1 = x1 * patch_mask.unsqueeze(-1)
        return x1, patch_mask

    def forward_from_x1(self, x1: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        x = self.ADD_PE(x1)
        x2 = self.self_attn2(x)
        x2 = x2 * patch_mask.unsqueeze(-1)
        logits = self.classifier(x2, patch_mask)
        return logits

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x1, patch_mask = self.encode_after_impute_attn(batch, patch_drop_mask=None)
        logits = self.forward_from_x1(x1, patch_mask)
        return logits


# =========================
# Training utilities
# =========================

def sample_patch_drop_mask(patch_mask: torch.Tensor, drop_ratio: float = 0.15) -> torch.Tensor:
    B, P = patch_mask.shape
    r = torch.rand(B, P, device=patch_mask.device)
    drop = (r < drop_ratio) & (patch_mask > 0)

    for b in range(B):
        if patch_mask[b].sum() > 0 and drop[b].sum() == 0:
            valid = torch.nonzero(patch_mask[b] > 0).squeeze(-1)
            j = valid[torch.randint(0, valid.numel(), (1,), device=patch_mask.device)]
            drop[b, j] = True
    return drop


def loss_completion_mse(x1_full: torch.Tensor, x1_masked: torch.Tensor, drop_mask: torch.Tensor) -> torch.Tensor:
    if drop_mask.sum() == 0:
        return x1_full.new_tensor(0.0)
    diff = x1_masked[drop_mask] - x1_full[drop_mask]
    return (diff * diff).mean()


def compute_total_loss(
    model: tPatchGNN_Classifier,
    batch: Dict[str, Any],
    drop_ratio: float = 0.15,
    w_recon: float = 1.0,
    w_bce: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    patch_mask = batch["patch_mask"].float()
    drop_mask = sample_patch_drop_mask(patch_mask, drop_ratio=drop_ratio)

    x1_full, patch_mask2 = model.encode_after_impute_attn(batch, patch_drop_mask=None)
    x1_masked, _ = model.encode_after_impute_attn(batch, patch_drop_mask=drop_mask)

    loss1 = loss_completion_mse(x1_full.detach(), x1_masked, drop_mask)

    logits = model.forward_from_x1(x1_masked, patch_mask2)  # (B,5)

    label = batch["label"]
    if label.dim() == 1:
        label = F.one_hot(label.long(), num_classes=logits.size(-1)).float()
    else:
        label = label.float()

    loss2 = F.binary_cross_entropy_with_logits(logits, label)

    total = w_recon * loss1 + w_bce * loss2
    return total, loss1, loss2, logits


def train_one_step(
    model: tPatchGNN_Classifier,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Any],
    drop_ratio: float = 0.15,
    w_recon: float = 1.0,
    w_bce: float = 1.0,
    grad_clip: Optional[float] = None
) -> Tuple[float, float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total, loss1, loss2, _ = compute_total_loss(
        model, batch, drop_ratio=drop_ratio, w_recon=w_recon, w_bce=w_bce
    )
    total.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return float(total.item()), float(loss1.item()), float(loss2.item())


# =========================
# Test example
# =========================

def make_random_patch_lengths(total_len: int, P: int) -> torch.Tensor:
    if total_len <= 0:
        return torch.zeros(P, dtype=torch.long)
    if P == 1:
        return torch.tensor([total_len], dtype=torch.long)
    k = min(P - 1, total_len - 1)
    cuts = sorted(random.sample(range(1, total_len), k=k)) if k > 0 else []
    points = [0] + cuts + [total_len]
    lens = [points[i + 1] - points[i] for i in range(len(points) - 1)]
    if len(lens) < P:
        lens += [0] * (P - len(lens))
    return torch.tensor(lens[:P], dtype=torch.long)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    args = Args(
        device="cpu",     # change to "cuda" if you want
        in_dim=5,
        hid_dim=64,
        te_dim=8,
        npatch=12,
        patch_len=16,
        nhead=4,          # IMPORTANT: hid_dim % nhead == 0
        tf_layer=1,
        tau_seconds_intra=300.0,
    )

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = tPatchGNN_Classifier(args).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = 4
    Lmax = 80
    D = args.in_dim
    P = args.npatch

    num_batch = torch.tensor([80, 67, 52, 35], dtype=torch.long, device=device)

    patch_mask = torch.zeros(B, P, device=device)
    patch_index = torch.zeros(B, P, dtype=torch.long, device=device)
    for b in range(B):
        valid_p = random.randint(6, P)
        patch_mask[b, :valid_p] = 1.0
        patch_index[b, :valid_p] = make_random_patch_lengths(int(num_batch[b].item()), valid_p).to(device)

    data_sequence = torch.randn(B, Lmax, D, device=device)

    base = 1_700_000_000.0
    ts_sequence = torch.zeros(B, Lmax, device=device)
    for b in range(B):
        ts_sequence[b] = base + torch.arange(Lmax, device=device).float() * 60.0

    label = torch.zeros(B, 5, device=device)
    cls = torch.randint(0, 5, (B,), device=device)
    label[torch.arange(B, device=device), cls] = 1.0

    batch = {
        "data_sequence": data_sequence,
        "ts_sequence": ts_sequence,
        "delta_sequence": torch.ones(B, Lmax, device=device) * 60.0,
        "label": label,
        "patch_mask": patch_mask,
        "patch_index": patch_index,
        "num_batch": num_batch,
    }

    with torch.no_grad():
        logits = model(batch)
        print("logits:", tuple(logits.shape))  # (B,5)

    total, l1, l2 = train_one_step(
        model, optim, batch,
        drop_ratio=0.2,
        w_recon=1.0,
        w_bce=1.0,
        grad_clip=1.0
    )
    print("train step -> total={:.6f}, loss1(recon)={:.6f}, loss2(BCE)={:.6f}".format(total, l1, l2))
