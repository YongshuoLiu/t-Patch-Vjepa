import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.evaluation import *

# -------------------------
# Basic modules
# -------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for (B, M, D)."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # x: (B, M, D)
        return x + self.pe[:, :x.size(1), :]


class PatchGraphAttention(nn.Module):
    """
    Patch-to-patch attention with time-diff weight bias.
    time diff is fixed by patch index distance: |i-j| * delta_seconds
    """
    def __init__(self, d_model, nhead=4, tau_seconds=300.0, delta_minutes=5.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.tau = float(tau_seconds)
        self.delta = float(delta_minutes) * 60.0

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _time_bias(self, M, device):
        idx = torch.arange(M, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().float() * self.delta  # (M, M)
        w = torch.exp(-dist / max(self.tau, 1e-6))                       # (M, M) in (0,1]
        bias = torch.log(w + 1e-12)                                      # (M, M)
        return bias

    def forward(self, H, patch_mask=None):
        """
        H: (B, M, D)
        patch_mask: (B, M) float/bool, 1=valid patch, 0=invalid patch
        """
        B, M, D = H.shape
        device = H.device

        q = self.q_proj(H).view(B, M, self.nhead, self.d_head).transpose(1, 2)  # (B,h,M,dh)
        k = self.k_proj(H).view(B, M, self.nhead, self.d_head).transpose(1, 2)  # (B,h,M,dh)
        v = self.v_proj(H).view(B, M, self.nhead, self.d_head).transpose(1, 2)  # (B,h,M,dh)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)     # (B,h,M,M)
        attn = attn + self._time_bias(M, device)[None, None, :, :]

        if patch_mask is not None:
            if patch_mask.dtype == torch.bool:
                key_pad = ~patch_mask
            else:
                key_pad = (patch_mask <= 0)
            attn = attn.masked_fill(key_pad[:, None, None, :], -1e9)

        A = torch.softmax(attn, dim=-1)
        out = torch.matmul(A, v)  # (B,h,M,dh)
        out = out.transpose(1, 2).contiguous().view(B, M, D)
        out = self.out_proj(out)

        if patch_mask is not None:
            out = out * patch_mask.unsqueeze(-1).float()

        return out


class QueryClassifier5(nn.Module):
    """5 learnable queries cross-attend to patch tokens, produce 5-class logits."""
    def __init__(self, d_model, nhead=4, num_classes=5):
        super().__init__()
        assert num_classes == 5
        self.num_queries = 5
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.query_to_logit = nn.Linear(d_model, 1)

    def forward(self, patch_tokens, patch_mask=None):
        """
        patch_tokens: (B, M, D)
        patch_mask: (B, M) 1=valid, 0=invalid
        return logits: (B, 5)
        """
        B, M, D = patch_tokens.shape
        q = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,5,D)

        key_padding_mask = None
        if patch_mask is not None:
            key_padding_mask = (patch_mask <= 0)  # True = pad

        z, _ = self.cross_attn(q, patch_tokens, patch_tokens, key_padding_mask=key_padding_mask)
        logits = self.query_to_logit(z).squeeze(-1)  # (B,5)
        return logits


# -------------------------
# Main model
# -------------------------

class tPatchGNN_Classifier(nn.Module):
    """
    - patch_index: (B,P) means *length* (#timesteps) per patch (NOT start indices)
    - data_sequence: (B,Lmax,D) flattened timesteps
    """

    def __init__(self, args):
        super().__init__()
        self.device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")

        self.hid_dim = int(args.hid_dim)
        self.te_dim = int(args.te_dim)
        self.npatch = int(args.npatch)
        self.patch_len = int(args.patch_len)
        self.n_layer = int(args.nlayer)
        self.nhead = int(args.nhead)
        self.tf_layer = int(args.tf_layer)
        self.in_dim = int(getattr(args, "in_dim", 4))

        # Learnable TE
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        # TTCN
        self.point_dim = self.in_dim + self.te_dim
        self.ttcn_dim = self.hid_dim - 1
        assert self.ttcn_dim > 0, "hid_dim must be >= 2"

        self.Filter_Generators = nn.Sequential(
            nn.Linear(self.point_dim, self.ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.point_dim * self.ttcn_dim, bias=True),
        )
        self.T_bias = nn.Parameter(torch.randn(1, self.ttcn_dim))

        # patch graph attn + self-attn
        tau_seconds = float(getattr(args, "tau_seconds", 300.0))
        self.patch_gattn = nn.ModuleList([
            PatchGraphAttention(self.hid_dim, nhead=self.nhead, tau_seconds=tau_seconds, delta_minutes=5.0)
            for _ in range(self.n_layer)
        ])
        self.ADD_PE = PositionalEncoding(self.hid_dim)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(self.n_layer):
            enc_layer = nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=self.nhead, batch_first=True)
            self.transformer_encoder.append(nn.TransformerEncoder(enc_layer, num_layers=self.tf_layer))

        # classifier
        self.classifier = QueryClassifier5(d_model=self.hid_dim, nhead=self.nhead, num_classes=5)

    # -------- time embedding --------
    def LearnableTE(self, tt):
        """
        tt: (..., 1) float tensor
        """
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], dim=-1)

    # -------- TTCN aggregation --------
    def TTCN(self, X_int, mask_X):
        """
        X_int: (B*P, L, point_dim)
        mask_X: (B*P, L, 1)
        """
        N, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int)                         # (N, Lx, point_dim*ttcn_dim)
        Filter_mask = Filter * mask_X + (1.0 - mask_X) * (-1e8)
        Filter_seqnorm = torch.softmax(Filter_mask, dim=-2)            # over L

        Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.ttcn_dim, self.point_dim)
        X_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)

        ttcn_out = torch.sum(torch.sum(X_broad * Filter_seqnorm, dim=-3), dim=-1)  # (N, ttcn_dim)
        h_t = torch.relu(ttcn_out + self.T_bias)                                   # (N, ttcn_dim)
        return h_t

    # -------- NEW: patchify by lengths --------
    def patchify_by_len(self, data_sequence, ts_sequence, patch_mask, patch_index, num_batch):
        """
        patch_index: (B,P) length per patch
        patch_mask:  (B,P) 1=valid patch else 0
        num_batch:   (B,) total valid timesteps in data_sequence

        returns:
          X_patch: (B,P,L,D)
          ts_patch:(B,P,L,1)
          mask_pt: (B,P,L,1)
          mask_patch:(B,P) float
        """
        B, Lmax, D = data_sequence.shape
        P = patch_mask.shape[1]
        L = self.patch_len
        device = data_sequence.device

        X_patch = torch.zeros((B, P, L, D), device=device, dtype=data_sequence.dtype)
        ts_patch = torch.zeros((B, P, L, 1), device=device, dtype=torch.float32)
        mask_pt = torch.zeros((B, P, L, 1), device=device, dtype=torch.float32)

        patch_mask_f = patch_mask.float()
        patch_len = patch_index.long().clamp(min=0)

        for b in range(B):
            valid_len = int(num_batch[b].item())
            valid_len = max(0, min(valid_len, Lmax))

            cursor = 0
            for p in range(P):
                if patch_mask_f[b, p].item() <= 0:
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

            # (optional) consistency check: uncomment to debug
            # total = int((patch_len[b] * (patch_mask_f[b] > 0).long()).sum().item())
            # if total != valid_len:
            #     print(f"[warn] b={b}: sum(patch_len valid)={total}, num_batch={valid_len}")

        return X_patch, ts_patch, mask_pt, patch_mask_f

    # -------- forward --------
    def forward(self, batch):
        data_sequence = batch["data_sequence"]   # (B,Lmax,D)
        ts_sequence = batch["ts_sequence"]       # (B,Lmax)
        patch_mask = batch["patch_mask"]         # (B,P)
        patch_index = batch["patch_index"]       # (B,P) lengths
        num_batch = batch["num_batch"]           # (B,)

        B, Lmax, D = data_sequence.shape
        assert D == self.in_dim, f"Expected in_dim={self.in_dim}, got D={D}"
        assert patch_mask.shape[1] == self.npatch, f"Expected npatch={self.npatch}, got P={patch_mask.shape[1]}"

        # 1) patchify (BY LENGTHS!)
        X_patch, ts_patch, mask_pt, mask_patch = self.patchify_by_len(
            data_sequence, ts_sequence, patch_mask, patch_index, num_batch
        )
        # X_patch: (B,P,L,D)
        # ts_patch:(B,P,L,1)
        # mask_pt: (B,P,L,1)
        # mask_patch:(B,P)

        # 2) time embedding per point
        # NOTE: unix seconds is large; consider normalization if training unstable.
        te = self.LearnableTE(ts_patch)                       # (B,P,L,te_dim)

        # 3) build point features
        X_int = torch.cat([X_patch, te], dim=-1)              # (B,P,L,point_dim)
        X_int = X_int.view(B * self.npatch, self.patch_len, self.point_dim)
        mask_pt2 = mask_pt.view(B * self.npatch, self.patch_len, 1)

        # 4) TTCN => patch token
        patch_feat = self.TTCN(X_int, mask_pt2)               # (B*P, hid_dim-1)

        patch_exists = (mask_pt2.sum(dim=1) > 0).float()      # (B*P,1)
        patch_tok = torch.cat([patch_feat, patch_exists], dim=-1)  # (B*P,hid_dim)
        x = patch_tok.view(B, self.npatch, self.hid_dim)           # (B,P,hid_dim)

        patch_mask2 = mask_patch.clone()                           # (B,P) float 0/1

        # 5) patch graph attn + patch self-attn
        for layer in range(self.n_layer):
            x_last = x
            x = x + self.patch_gattn[layer](x, patch_mask2)        # (B,P,D)
            x = self.ADD_PE(x)
            x = self.transformer_encoder[layer](x)                 # (B,P,D)
            x = x_last + x                                         # residual
            x = x * patch_mask2.unsqueeze(-1)                      # zero invalid patches

        # 6) 5-query cross-attn classification
        logits = self.classifier(x, patch_mask2)                   # (B,5)
        return logits

    def compute_loss(self, batch):
        logits = self.forward(batch)
        label = batch["label"].long()      # (B,) in [0..4]
        loss = F.cross_entropy(logits, label)
        return loss, logits


# -------------------------
# Example args container
# -------------------------
class Args:
    def __init__(self, cfg: dict):
        """
        cfg: dict-like config, e.g. loaded from json/yaml
        """
        # --- model ---
        self.device = cfg["device"]
        self.hid_dim = cfg["hid_dim"]
        self.te_dim = cfg["te_dim"]
        self.npatch = cfg["npatch"]
        self.patch_len = cfg["patch_len"]
        self.nlayer = cfg["nlayer"]
        self.nhead = cfg["nhead"]
        self.tf_layer = cfg["tf_layer"]
        self.in_dim = cfg["in_dim"]

        # --- data / time ---
        self.tau_seconds = cfg["tau_seconds"]

# -------------------------
# Quick sanity check
# -------------------------
if __name__ == "__main__":
    args = Args(device="cpu", in_dim=3, npatch=15, patch_len=16, hid_dim=64, te_dim=8, nlayer=2, nhead=4, tf_layer=1)
    model = tPatchGNN_Classifier(args)

    B, Lmax, D = 2, 50, args.in_dim
    P = args.npatch

    # patch_index is LENGTHS now:
    # sample 0: patches lengths sum=50
    # sample 1: patches lengths sum=50
    patch_lens0 = torch.tensor([6, 6, 6, 6, 6, 6, 7, 7])  # sum=50
    patch_lens1 = torch.tensor([10, 5, 5, 5, 5, 5, 5, 10]) # sum=50

    batch = {
        "data_sequence": torch.randn(B, Lmax, D),
        "ts_sequence": torch.arange(Lmax).unsqueeze(0).repeat(B, 1).float() * 60.0 + 1_700_000_000.0,
        "delta_sequence": torch.ones(B, Lmax) * 60.0,
        "label": torch.randint(0, 5, (B,)),
        "patch_mask": torch.ones(B, P),
        "patch_index": torch.stack([patch_lens0, patch_lens1], dim=0),
        "num_batch": torch.tensor([Lmax, Lmax]),
    }

    loss, logits = model.compute_loss(batch)
    print("loss:", float(loss.item()), "logits:", logits.shape)
