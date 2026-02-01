# t-Patch-VJEPA

PyTorch implementation of a transformable patch-based trajectory modeling framework
with patch completion (V-JEPA-style) and query-based anomaly detection.

This repository focuses on modeling irregular multivariate time series / trajectories
(e.g., vessel trajectories) by:
- transformable temporal patching,
- intra-patch graph attention,
- patch completion via masked representation learning,
- and patch-query cross-attention for downstream classification.


[ Figure 1. Overall framework / model architecture ]
<img width="2251" height="1190" alt="Weixin Image_2026-02-01_020503_445" src="https://github.com/user-attachments/assets/fea31ff4-be21-4606-a807-5be097ed8fd0" />


---

## Key Features

- Transformable Patching
  - Converts irregularly sampled trajectories into variable-length patches with consistent temporal semantics.
- Intra-Patch Graph Attention
  - Models fine-grained temporal relations inside each patch using time-difference-aware GAT.
- Patch Completion (V-JEPA-style)
  - Learns to recover missing patch representations via mask-denoising in representation space.
- Query-Based Decoder
  - Uses DETR-style learnable queries for anomaly classification.
- Fully End-to-End PyTorch Implementation


[ Figure 2. Patch completion / V-JEPA-style pretraining ]
<img width="1320" height="893" alt="Weixin Image_2026-02-01_020521_799" src="https://github.com/user-attachments/assets/e43fbc38-8cb3-49a5-a463-7ceecc403b88" />



---

## Repository Structure

```text
t-Patch-Vjepa/
├── tPatchGNN/
│   ├── model/
│   │   ├── tPatchCAdetrv2.py        # main model & loss
│   │   └── ...
│   ├── utils/
│   │   └── ...
│   └── ...
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.8
- NumPy

Install dependencies:

pip install -r requirements.txt

---

## Quick Test

Run a minimal synthetic test to verify model forward & loss computation:

python tPatchGNN/model/tPatchCAdetrv2.py

Expected output (example):

loss: 1.23
logits shape: (B, num_classes)

---

## Input Batch Format

The model expects a batch dictionary with the following keys:

batch = {
    "data_sequence":   (B, L, D)  # observation features
    "ts_sequence":     (B, L)     # timestamps (float)
    "delta_sequence":  (B, L)     # time gaps
    "patch_index":     (B, P)     # patch lengths (sum = L)
    "patch_mask":      (B, P)     # valid patch mask
    "label":           (B,)       # class labels
    "num_batch":       (B,)       # original sequence lengths
}

---

## Model Configuration

Model hyperparameters are passed via a config dictionary:

cfg = {
    "device": "cpu",
    "in_dim": 3,
    "hid_dim": 64,
    "te_dim": 8,
    "npatch": 15,
    "patch_len": 16,
    "nlayer": 2,
    "nhead": 1,
    "tf_layer": 1,
    "tau_seconds": 1.0,
}

---

## Notes

- patch_index represents patch lengths, not indices.
- Each row of patch_index must sum to the sequence length L.
- For debugging or custom data, setting nhead=1 avoids attention dimension constraints.

---

## Citation

If you find this code useful in your research, please consider citing the related work.

(BibTeX to be added)

---

## Contact

Yongshuo Liu
GitHub: https://github.com/YongshuoLiu

---

## License

This project is for research and academic use.
Please check upstream licenses if you integrate it into other projects.
