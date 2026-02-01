import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Transformer_EncDec import Encoder, EncoderLayer
from model.SelfAttention_Family import FullAttention, AttentionLayer

import lib.utils as utils
from lib.evaluation import *
from model.patchTransformer import PatchTransformer 

class nconv(nn.Module):
	def __init__(self):
		super(nconv,self).__init__()

	def forward(self, x, A):
		# x (B, F, N, M)
		# A (B, M, N, N)
		x = torch.einsum('bfnm,bmnv->bfvm',(x,A)) # used
		# print(x.shape)
		return x.contiguous() # (B, F, N, M)
		

class linear(nn.Module):
	def __init__(self, c_in, c_out):
		super(linear,self).__init__()
		# self.mlp = nn.Linear(c_in, c_out)
		self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)

	def forward(self, x):
		# x (B, F, N, M)

		# return self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
		return self.mlp(x)
		
class gcn(nn.Module):
	def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
		super(gcn,self).__init__()
		self.nconv = nconv()
		c_in = (order*support_len+1)*c_in
		# c_in = (order*support_len)*c_in
		self.mlp = linear(c_in, c_out)
		self.dropout = dropout
		self.order = order

	def forward(self, x, support):
		# x (B, F, N, M)
		# a (B, M, N, N)
		out = [x]
		for a in support:
			x1 = self.nconv(x,a)
			out.append(x1)
			for k in range(2, self.order + 1):
				x2 = self.nconv(x1,a)
				out.append(x2)
				x1 = x2

		h = torch.cat(out, dim=1) # concat x and x_conv
		h = self.mlp(h)
		return F.relu(h)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        """
        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
	

class tPatchGNN(nn.Module):
	def __init__(self, args, supports = None, dropout = 0):
	
		super(tPatchGNN, self).__init__()
		self.device = args.device
		self.hid_dim = args.hid_dim
		self.N = args.ndim
		self.M = args.npatch
		self.batch_size = None
		self.supports = supports
		self.n_layer = args.nlayer

		### Intra-time series modeling ## 
		## Time embedding
		self.te_scale = nn.Linear(1, 1)
		self.te_periodic = nn.Linear(1, args.te_dim-1)

		## TTCN
		input_dim = 1 + args.te_dim
		ttcn_dim = args.hid_dim - 1
		self.ttcn_dim = ttcn_dim
		self.Filter_Generators = nn.Sequential(
				nn.Linear(input_dim, ttcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(ttcn_dim, ttcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(ttcn_dim, input_dim*ttcn_dim, bias=True))
		self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))
		
		d_model = args.hid_dim
		## Transformer
		self.ADD_PE = PositionalEncoding(d_model) 

		#===========================================  refill the explicit masked patch modeling module 
		# self.mask_ratio = getattr(args, "mask_ratio", 0.15)         
		# self.mp_lambda  = getattr(args, "mp_lambda", 1.0)           
		# self.explicit_mp = nn.ModuleList()
		# for _ in range(self.n_layer):
		# 	self.explicit_mp.append(
		# 		PatchTransformer(
		# 			d_model=args.hid_dim,
		# 			nhead=args.nhead,
		# 			num_layers=args.tf_layer,
		# 			pred_dim=args.hid_dim - 1,       
		# 			mask_ratio=self.mask_ratio
		# 		)
		# 	)
		#===========================================

		self.transformer_encoder = nn.ModuleList()
		for _ in range(self.n_layer):
			encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=args.nhead, batch_first=True)
			self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=args.tf_layer))			

		### Inter-time series modeling ###
		self.supports_len = 0
		if supports is not None:
			self.supports_len += len(supports)

		nodevec_dim = args.node_dim
		self.nodevec_dim = nodevec_dim
		if supports is None:
			self.supports = []

		self.nodevec1 = nn.Parameter(torch.randn(self.N, nodevec_dim).cuda(), requires_grad=True)
		self.nodevec2 = nn.Parameter(torch.randn(nodevec_dim, self.N).cuda(), requires_grad=True)

		self.nodevec_linear1 = nn.ModuleList()
		self.nodevec_linear2 = nn.ModuleList()
		self.nodevec_gate1 = nn.ModuleList()
		self.nodevec_gate2 = nn.ModuleList()
		for _ in range(self.n_layer):
			self.nodevec_linear1.append(nn.Linear(args.hid_dim, nodevec_dim))
			self.nodevec_linear2.append(nn.Linear(args.hid_dim, nodevec_dim))
			self.nodevec_gate1.append(nn.Sequential(
				nn.Linear(args.hid_dim+nodevec_dim, 1),
				nn.Tanh(),
				nn.ReLU()))
			self.nodevec_gate2.append(nn.Sequential(
				nn.Linear(args.hid_dim+nodevec_dim, 1),
				nn.Tanh(),
				nn.ReLU()))
			
		self.supports_len +=1

		self.gconv = nn.ModuleList() # gragh conv
		for _ in range(self.n_layer):
			self.gconv.append(gcn(d_model, d_model, dropout, support_len=self.supports_len, order=args.hop))

		### Encoder output layer ###
		self.outlayer = args.outlayer
		enc_dim = args.hid_dim
		if(self.outlayer == "Linear"):
			self.temporal_agg = nn.Sequential(
				nn.Linear(args.hid_dim*self.M, enc_dim))
		
		elif(self.outlayer == "CNN"):
			self.temporal_agg = nn.Sequential(
				nn.Conv1d(d_model, enc_dim, kernel_size=self.M))
		#===========================================  subsitute query-based attention for dncoder
		# self.num_queries = 4
		# self.query_embed = nn.Embedding(self.num_queries, enc_dim)  

		# self.query_attn = nn.MultiheadAttention(
		# 	embed_dim=enc_dim,
		# 	num_heads=args.nhead,
		# 	batch_first=True
		# )

		# self.query_classifier = nn.Linear(enc_dim, 2)
		#===========================================

		self.decoder = nn.Sequential(
			nn.Linear(enc_dim+args.te_dim, args.hid_dim),
			nn.ReLU(inplace=True),
			nn.Linear(args.hid_dim, args.hid_dim),
			nn.ReLU(inplace=True),
			nn.Linear(args.hid_dim, 1)
			)
		self.patch_gattn = nn.ModuleList([
		PatchGraphAttention(d_model=args.hid_dim, nhead=args.nhead, tau=getattr(args, "tau", 60.0), delta_minutes=5.0)
		for _ in range(self.n_layer)
		])
				

		# self-attention over patches (you already have transformer_encoder[layer])
		# keep self.transformer_encoder as patch-level SA

		# 5-query classifier
		self.classifier = QueryClassifier5(d_model=args.hid_dim, nhead=args.nhead, num_classes=5)


	def LearnableTE(self, tt):
		# tt: (N*M*B, L, 1)
		out1 = self.te_scale(tt)
		out2 = torch.sin(self.te_periodic(tt))
		return torch.cat([out1, out2], -1)
	
	def TTCN(self, X_int, mask_X):
		# X_int: shape (B*N*M, L, F)
		# mask_X: shape (B*N*M, L, 1)

		N, Lx, _ = mask_X.shape
		Filter = self.Filter_Generators(X_int) # (N, Lx, F_in*ttcn_dim)
		Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
		# normalize along with sequence dimension
		Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N, Lx, F_in*ttcn_dim)
		Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.ttcn_dim, -1) # (N, Lx, ttcn_dim, F_in)
		X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
		ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) # (N, ttcn_dim)
		h_t = torch.relu(ttcn_out + self.T_bias) # (N, ttcn_dim)
		return h_t

	def IMTS_Model(self, x, mask_X):
		# x (B*N*M, L, F), mask_X (B*N*M, L, 1)
		mask_patch = (mask_X.sum(dim=1) > 0)          # (B*N*M, 1) bool

		# ---- patch internal aggregation (keep TTCN) ----
		x_patch = self.TTCN(x, mask_X)                # (B*N*M, hid_dim-1)
		x_patch = torch.cat([x_patch, mask_patch], dim=-1)  # (B*N*M, hid_dim)

		x_patch = x_patch.view(self.batch_size, self.N, self.M, -1)  # (B, N, M, D)
		B, N, M, D = x_patch.shape
		patch_mask = mask_patch.view(B, N, M)         # (B, N, M) bool

		x = x_patch

		for layer in range(self.n_layer):
			if layer > 0:
				x_last = x

			# ---- (1) patch graph attention (time-diff weighted) ----
			# apply per node N: reshape to (B*N, M, D)
			x_bn = x.reshape(B * N, M, D)
			pm_bn = patch_mask.reshape(B * N, M).float()
			x_bn = x_bn + self.patch_gattn[layer](x_bn, pm_bn)        # residual-style
			x = x_bn.view(B, N, M, D)

			# ---- (2) patch self-attention (Transformer) ----
			x_bn = x.reshape(B * N, M, D)
			x_bn = self.ADD_PE(x_bn)
			x_bn = self.transformer_encoder[layer](x_bn)             # (B*N, M, D)
			x = x_bn.view(B, N, M, D)

			# ---- (3) keep / remove your inter-series GNN part? ----
			# 你这次描述里没有要 node-to-node 的 inter-series gconv，
			# 如果你要“只做 patch 之间”，这里就应该删掉下面整段 gconv 图学习。
			# 如果仍想保留 inter-series（每个 patch 的 N 节点之间），就保留原来的 gconv。
			# 我先按你描述：删除 gconv（否则是另一个图）。

			if layer > 0:
				x = x_last + x

		# ---- classification head ----
		# 你要 “整体 patch” 做分类：对每个 node N 做？还是对整条序列（把N汇聚）做？
		# 通常 anomaly 是对一个对象整体：建议先在 N 维做 pooling，再用 query 分类。
		x_seq = x.mean(dim=1)                         # (B, M, D)  在 N 上平均池化
		pm_seq = patch_mask.any(dim=1).float()        # (B, M)     N里任一有效就算有效 patch

		logits = self.classifier(x_seq, pm_seq)       # (B, 5)
		return logits

	def forward(self, batch):
		# batch:
		# data_sequence: (B, Lmax, D)
		# ts_sequence: (B, Lmax)
		# delta_sequence: (B, Lmax)
		# label: (B,)
		# patch_mask: (B, P)
		# patch_index: (B, P)
		# num_batch: (B,)

		X = batch["data_sequence"]         # (B, Lmax, D)
		ts = batch["ts_sequence"]          # (B, Lmax)
		# 你现在的 TTCN 路径需要 (B, M, L, 1) 这种“已切 patch”的形式
		# 所以你必须在这里先把 (B, Lmax, D) 按 patch_index/patch_mask 切成 (B, M, L, D)
		# 然后再 reshape 成你现有 IMTS_Model 需要的 (B*N*M, L, F)

		# ---- 这里先给接口：你实现 patchify() 后接入 ----
		X_patch, ts_patch, mask_patch_points = self.patchify(X, ts, batch["patch_index"], batch["patch_mask"])
		# X_patch: (B, M, L, 1)  (如果是单变量)
		# ts_patch: (B, M, L, 1)
		# mask_patch_points: (B, M, L, 1)

		B, M, L, _ = X_patch.shape
		self.batch_size = B

		X_patch = X_patch.permute(0, 2, 1, 3)  # (B, L, M, 1) 只是示例，按你 N/M 约定调整

class PatchGraphAttention(nn.Module):
    def __init__(self, d_model, nhead=4, tau=1.0, delta_minutes=5.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.tau = tau
        self.delta = delta_minutes * 60.0  # seconds

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _time_weight_bias(self, M, device):
        # dist(i,j) = |i-j| * delta
        idx = torch.arange(M, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().float() * self.delta
        w = torch.exp(-dist / self.tau)                 # (M, M), in (0,1]
        bias = torch.log(w + 1e-12)                     # log-weight as additive bias
        return bias                                     # (M, M)

    def forward(self, H, patch_mask):
        """
        H: (B, M, D) patch tokens
        patch_mask: (B, M) bool/0-1, 1 means valid patch
        """
        B, M, D = H.shape
        device = H.device

        q = self.q_proj(H).view(B, M, self.nhead, self.d_head).transpose(1, 2)  # (B, h, M, dh)
        k = self.k_proj(H).view(B, M, self.nhead, self.d_head).transpose(1, 2)  # (B, h, M, dh)
        v = self.v_proj(H).view(B, M, self.nhead, self.d_head).transpose(1, 2)  # (B, h, M, dh)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)    # (B, h, M, M)

        # add time-diff bias (shared across batch & heads)
        bias = self._time_weight_bias(M, device)                                 # (M, M)
        attn = attn + bias[None, None, :, :]

        # mask invalid patches (as keys)
        if patch_mask is not None:
            key_mask = (patch_mask == 0)                                         # (B, M) True=invalid
            attn = attn.masked_fill(key_mask[:, None, None, :], -1e9)

        A = torch.softmax(attn, dim=-1)                                          # (B, h, M, M)
        out = torch.matmul(A, v)                                                 # (B, h, M, dh)

        out = out.transpose(1, 2).contiguous().view(B, M, D)                     # (B, M, D)
        out = self.out_proj(out)

        # optionally zero-out invalid query positions too
        if patch_mask is not None:
            out = out * patch_mask.unsqueeze(-1).float()

        return out

class QueryClassifier5(nn.Module):
    def __init__(self, d_model, nhead=4, num_classes=5):
        super().__init__()
        assert num_classes == 5
        self.num_queries = 5
        self.query_embed = nn.Embedding(self.num_queries, d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )

        # each query -> one class logit
        self.query_to_logit = nn.Linear(d_model, 1)

    def forward(self, patch_tokens, patch_mask=None):
        """
        patch_tokens: (B, M, D)
        patch_mask: (B, M) 1=valid, 0=invalid
        return logits: (B, 5)
        """
        B, M, D = patch_tokens.shape
        q = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, 5, D)

        key_padding_mask = None
        if patch_mask is not None:
            key_padding_mask = (patch_mask == 0)                 # True=pad

        z, _ = self.cross_attn(q, patch_tokens, patch_tokens, key_padding_mask=key_padding_mask)
        logits = self.query_to_logit(z).squeeze(-1)              # (B, 5)
        return logits
