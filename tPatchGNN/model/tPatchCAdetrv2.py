import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from lib.evaluation import *



# -------------------------
# Basic modules
# -------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for (B, M, D)."""
    def __init__(self, d_model, max_len=2048):
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
        w = torch.exp(-dist / max(self.tau, 1e-6))                       # (M, M)
        return torch.log(w + 1e-12)                                      # (M, M)

    def forward(self, H, patch_mask=None):
        """
        H: (B, M, D)
        patch_mask: (B, M) 1=valid patch, 0=invalid patch
        """
        B, M, D = H.shape
        device = H.device

        q = self.q_proj(H).view(B, M, self.nhead, self.d_head).transpose(1, 2)  # (B,h,M,dh)
        k = self.k_proj(H).view(B, M, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(H).view(B, M, self.nhead, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)     # (B,h,M,M)
        attn = attn + self._time_bias(M, device)[None, None, :, :]

        if patch_mask is not None:
            key_pad = (patch_mask <= 0) if patch_mask.dtype != torch.bool else (~patch_mask)
            attn = attn.masked_fill(key_pad[:, None, None, :], -1e9)

        A = torch.softmax(attn, dim=-1)
        out = torch.matmul(A, v)  # (B,h,M,dh)
        out = out.transpose(1, 2).contiguous().view(B, M, D)
        out = self.out_proj(out)

        if patch_mask is not None:
            out = out * patch_mask.unsqueeze(-1).float()

        return out


class IntraPatchGraphAggregator(nn.Module):
    """
    NEW: use graph-attention inside each patch to aggregate timesteps -> patch token.
    - tokens: (B*P, L, Dtok)
    - ts:     (B*P, L, 1) unix seconds (or any seconds scale)
    - mask:   (B*P, L, 1) 1=valid timestep, 0=pad
    Output:
      patch_vec: (B*P, Dtok)
    """
    def __init__(self, d_model, nhead=4, tau_seconds=300.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.tau = float(tau_seconds)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)

        # learnable readout query for pooling to one vector
        self.readout = nn.Parameter(torch.randn(1, 1, d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, tokens, ts, mask):
        """
        tokens: (N, L, D)
        ts:     (N, L, 1)
        mask:   (N, L, 1)
        """
        N, L, D = tokens.shape
        device = tokens.device

        # build time-bias: |t_i - t_j| -> exp(-dt/tau) -> log bias
        # ts can be huge (unix). use relative time inside patch for stability:
        ts_rel = ts - ts[:, :1, :]                          # (N,L,1)
        dt = (ts_rel - ts_rel.transpose(1, 0).transpose(0, 1))  # WRONG (keep simple below)

        # correct dt computation (N, L, L)
        t = ts_rel.squeeze(-1)                              # (N,L)
        dt = (t[:, :, None] - t[:, None, :]).abs()          # (N,L,L)
        w = torch.exp(-dt / max(self.tau, 1e-6))            # (N,L,L)
        bias = torch.log(w + 1e-12)                         # (N,L,L)

        # project
        q = self.q_proj(tokens).view(N, L, self.nhead, self.d_head).transpose(1, 2)  # (N,h,L,dh)
        k = self.k_proj(tokens).view(N, L, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(tokens).view(N, L, self.nhead, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (N,h,L,L)
        attn = attn + bias[:, None, :, :]                                     # add time bias

        # mask invalid keys
        key_pad = (mask.squeeze(-1) <= 0)  # (N,L) True=pad
        attn = attn.masked_fill(key_pad[:, None, None, :], -1e9)

        A = torch.softmax(attn, dim=-1)
        out = torch.matmul(A, v)  # (N,h,L,dh)
        out = out.transpose(1, 2).contiguous().view(N, L, D)
        out = self.out_proj(out)

        # zero invalid positions
        out = out * mask

        # readout pooling: a learnable query attends to timestep tokens
        q0 = self.readout.expand(N, 1, D)                                      # (N,1,D)
        # scaled dot-product attention
        qh = q0.view(N, 1, self.nhead, self.d_head).transpose(1, 2)            # (N,h,1,dh)
        kh = out.view(N, L, self.nhead, self.d_head).transpose(1, 2)           # (N,h,L,dh)
        vh = out.view(N, L, self.nhead, self.d_head).transpose(1, 2)           # (N,h,L,dh)

        pool_attn = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)  # (N,h,1,L)
        pool_attn = pool_attn.masked_fill(key_pad[:, None, None, :], -1e9)
        P = torch.softmax(pool_attn, dim=-1)                                   # (N,h,1,L)
        pooled = torch.matmul(P, vh).transpose(1, 2).contiguous().view(N, 1, D) # (N,1,D)

        return pooled.squeeze(1)  # (N,D)


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
        B, M, D = patch_tokens.shape
        q = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,5,D)

        key_padding_mask = None
        if patch_mask is not None:
            key_padding_mask = (patch_mask <= 0)

        z, _ = self.cross_attn(q, patch_tokens, patch_tokens, key_padding_mask=key_padding_mask)
        logits = self.query_to_logit(z).squeeze(-1)  # (B,5)
        return logits


# -------------------------
# Main model
# -------------------------

class tPatchGNN_Classifier(nn.Module):
    """
    NEW structure:
    1) patchify_by_len -> (B,P,L,D), ts_patch, mask_pt
    2) patch-internal graph attention aggregation (timesteps->patch token)
    3) self-attention over patches (TransformerEncoder) to update patch tokens
    4) patch-to-patch graph attention (time-diff by patch distance) + self-attn stack (unchanged)
    5) 5-query cross attention -> logits (B,5)
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
        self.in_dim = int(getattr(args, "in_dim", 5))

        # Learnable TE
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        # point token projection: (in_dim + te_dim) -> (hid_dim-1)
        self.point_dim = self.in_dim + self.te_dim
        self.ttcn_dim = self.hid_dim - 1
        assert self.ttcn_dim > 0, "hid_dim must be >= 2"

        self.point_proj = nn.Sequential(
            nn.Linear(self.point_dim, self.ttcn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.ttcn_dim),
        )

        tau_intra = float(getattr(args, "tau_seconds_intra", 300.0))
        self.intra_patch_gattn = IntraPatchGraphAggregator(
            d_model=self.ttcn_dim, nhead=self.nhead, tau_seconds=tau_intra
        )

        # one self-attention immediately after patch generation (your requirement)
        self.ADD_PE = PositionalEncoding(self.hid_dim)
        self.patch_self_attn_once = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=self.nhead, batch_first=True),
            num_layers=self.tf_layer
        )

        # patch-to-patch graph attn + self-attn stack (unchanged)
        tau_seconds = float(getattr(args, "tau_seconds", 300.0))
        self.patch_gattn = nn.ModuleList([
            PatchGraphAttention(self.hid_dim, nhead=self.nhead, tau_seconds=tau_seconds, delta_minutes=5.0)
            for _ in range(self.n_layer)
        ])

        self.transformer_encoder = nn.ModuleList()
        for _ in range(self.n_layer):
            enc_layer = nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=self.nhead, batch_first=True)
            self.transformer_encoder.append(nn.TransformerEncoder(enc_layer, num_layers=self.tf_layer))

        self.classifier = QueryClassifier5(d_model=self.hid_dim, nhead=self.nhead, num_classes=5)

    def LearnableTE(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], dim=-1)

    def patchify_by_len(self, data_sequence, ts_sequence, patch_mask, patch_index, num_batch):
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

        return X_patch, ts_patch, mask_pt, patch_mask_f

    def forward(self, batch):
        data_sequence = batch["data_sequence"]   # (B,Lmax,D)
        ts_sequence = batch["ts_sequence"]       # (B,Lmax)
        patch_mask = batch["patch_mask"]         # (B,P)
        patch_index = batch["patch_index"]       # (B,P) lengths
        num_batch = batch["num_batch"]           # (B,)

        B, Lmax, D = data_sequence.shape
        assert D == self.in_dim, f"Expected in_dim={self.in_dim}, got D={D}"
        assert patch_mask.shape[1] == self.npatch, f"Expected npatch={self.npatch}, got P={patch_mask.shape[1]}"

        # 1) patchify
        X_patch, ts_patch, mask_pt, mask_patch = self.patchify_by_len(
            data_sequence, ts_sequence, patch_mask, patch_index, num_batch
        )  # X_patch: (B,P,L,D), ts_patch:(B,P,L,1), mask_pt:(B,P,L,1)

        # 2) timestep tokens inside patch: concat TE then project
        te = self.LearnableTE(ts_patch)                       # (B,P,L,te_dim)
        X_int = torch.cat([X_patch, te], dim=-1)              # (B,P,L,in_dim+te_dim)

        # flatten to (B*P, L, dim)
        X_int = X_int.view(B * self.npatch, self.patch_len, self.point_dim)
        mask_pt2 = mask_pt.view(B * self.npatch, self.patch_len, 1)
        ts2 = ts_patch.view(B * self.npatch, self.patch_len, 1)

        step_tok = self.point_proj(X_int)                     # (B*P, L, hid_dim-1)
        step_tok = step_tok * mask_pt2

        # 3) NEW: intra-patch graph attention aggregation => patch token (hid_dim-1)
        patch_feat = self.intra_patch_gattn(step_tok, ts2, mask_pt2)   # (B*P, hid_dim-1)

        patch_exists = (mask_pt2.sum(dim=1) > 0).float()               # (B*P,1)
        patch_tok = torch.cat([patch_feat, patch_exists], dim=-1)      # (B*P, hid_dim)
        x = patch_tok.view(B, self.npatch, self.hid_dim)               # (B,P,hid_dim)
        patch_mask2 = mask_patch.clone()                                # (B,P) float

        # 4) your requested: immediately self-attention over all patches (one block)
        x = x * patch_mask2.unsqueeze(-1)
        x = self.ADD_PE(x)
        x = self.patch_self_attn_once(x)
        x = x * patch_mask2.unsqueeze(-1)

        # 5) unchanged: patch-to-patch graph attn + patch self-attn stack
        for layer in range(self.n_layer):
            x_last = x
            x = x + self.patch_gattn[layer](x, patch_mask2)
            x = self.ADD_PE(x)
            x = self.transformer_encoder[layer](x)
            x = x_last + x
            x = x * patch_mask2.unsqueeze(-1)

        # 6) 5-query cross-attn classification
        logits = self.classifier(x, patch_mask2)              # (B,5)
        return logits

    def compute_loss(self, batch):
        logits = self.forward(batch)
        label = batch["label"].long()      # (B,) in [0..4]
        loss = F.cross_entropy(logits, label)
        return loss, logits

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

def make_patch_lens(B: int, P: int, Lmax: int, device="cpu") -> torch.Tensor:
    """
    生成 shape=(B, P) 的 patch 长度，每个样本的 P 个 patch 长度之和严格等于 Lmax。
    这里保证每个 patch 至少长度为 1。
    """
    assert P <= Lmax, f"P(={P}) must be <= Lmax(={Lmax}) to keep each patch_len >= 1"
    # 先给每个 patch 分 1 个长度，剩余的长度再随机分配
    base = torch.ones(B, P, dtype=torch.long, device=device)
    remaining = Lmax - P
    if remaining == 0:
        return base

    # 用 multinomial 随机分配 remaining 个“单位长度”到 P 个 patch 上
    probs = torch.ones(B, P, device=device)  # uniform
    extra = torch.multinomial(probs, num_samples=remaining, replacement=True)  # (B, remaining)
    for b in range(B):
        base[b].scatter_add_(0, extra[b], torch.ones(remaining, dtype=torch.long, device=device))
    return base


if __name__ == "__main__":
    # --------- 1) Args：按你原本 Args(cfg) 的方式 ---------
    cfg = {
        "device": "cpu",
        "in_dim": 4,
        "npatch": 15,
        "patch_len": 16,
        "hid_dim": 64,
        "te_dim": 8,
        "nlayer": 2,
        "nhead": 1,          # ✅ 关键：避免 d_model % nhead 断言失败
        "tf_layer": 1,
        "tau_seconds": 1.0,
    }
    args = Args(cfg)

    # --------- 2) 构建模型 ---------
    model = tPatchGNN_Classifier(args)
    model.eval()

    # --------- 3) 构造 batch ---------
    B, Lmax, D = 2, 50, args.in_dim
    P = args.npatch

    device = torch.device(args.device)
    data_sequence = torch.randn(B, Lmax, D, device=device)

    # 时间戳（float），间隔 60s
    ts_sequence = (
        torch.arange(Lmax, device=device).unsqueeze(0).repeat(B, 1).float() * 60.0
        + 1_700_000_000.0
    )
    delta_sequence = torch.ones(B, Lmax, device=device) * 60.0

    # patch_index：这里按你注释“patch_index is LENGTHS now”来喂长度 (B,P)，每行 sum=Lmax
    patch_index = make_patch_lens(B=B, P=P, Lmax=Lmax, device=device)  # long tensor

    # patch_mask：一般是 (B,P)，全 1 表示全有效（如果你的实现需要 padding mask，可再改）
    patch_mask = torch.ones(B, P, device=device)

    # label：假设 5 类（你自己按模型输出类别数改）
    label = torch.randint(0, 5, (B,), device=device)

    # num_batch：你原来给的是 [Lmax, Lmax]，保持一致
    num_batch = torch.tensor([Lmax] * B, device=device)

    batch = {
        "data_sequence": data_sequence,
        "ts_sequence": ts_sequence,
        "delta_sequence": delta_sequence,
        "label": label,
        "patch_mask": patch_mask,
        "patch_index": patch_index,
        "num_batch": num_batch,
    }

    # --------- 4) forward / loss ---------
    with torch.no_grad():
        loss, logits = model.compute_loss(batch)

    print("loss:", float(loss.item()))
    print("logits shape:", tuple(logits.shape))
    print("patch_index shape:", tuple(patch_index.shape), "sum per sample:", patch_index.sum(dim=1).tolist())