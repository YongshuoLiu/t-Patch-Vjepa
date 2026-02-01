import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from sklearn import model_selection


# ============================================================
# 0) args
# ============================================================
@dataclass
class Args:
    device: torch.device = torch.device("cpu")
    batch_size: int = 2

    patch_minutes: int = 5      # 每个 patch 5 分钟
    max_patches: int = 15       # 最多 15 个 patch（时间窗口 = 75min）
    max_total_steps: int = 15   # 最终拼接后的总时间步数最多 15
    json_path: str = "/home/UNT/yl0826/QAU/t-PatchGNN/lib/test.json"
    dataset: str = "ship_json"



# ============================================================
# 1) Stage A：从 JSON sample 合并三源（不裁剪、不 padding）
# ============================================================
def merge_one_sample(sample: dict) -> pd.DataFrame:
    rows = []
    sample_id = sample["sample_id"]
    ship_id = sample.get("ship_id", None)
    label = sample["label"]

    for source_name, source_obj in sample["sources"].items():
        for p in source_obj.get("data", []):
            t, lat, lon, speed, heading = p
            rows.append({
                "sample_id": sample_id,
                "ship_id": ship_id,
                "source": source_name,
                "t": pd.to_datetime(t, utc=True),
                "lat": lat,
                "lon": lon,
                "speed": speed,
                "heading": heading,
                "label": label
            })

    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)

    # dt（秒），第一步 dt=0
    df["dt"] = df["t"].diff().dt.total_seconds().fillna(0.0)

    return df


def load_and_merge_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    merged_samples = []
    for sample in dataset["samples"]:
        df = merge_one_sample(sample)
        merged_samples.append({
            "record_id": sample["sample_id"],
            "label": sample["label"],
            "df": df
        })
    return merged_samples


# ============================================================
# 2) Dataset：输出 (record_id, tt, vals, label)
# ============================================================
class MergedTrajectoryDataset(Dataset):
    def __init__(self, merged_samples, device=torch.device("cpu"), use_dt_as_feature=True):
        self.items = merged_samples
        self.device = device
        self.feature_cols = ["lat", "lon", "speed", "heading"]


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        record_id = item["record_id"]
        label = int(item["label"])
        df = item["df"].copy()

        D = len(self.feature_cols)

        if df.empty:
            tt_rel = torch.zeros(1, device=self.device)
            ts_abs = torch.zeros(1, device=self.device)
            vals = torch.zeros(1, D, device=self.device)
            return record_id, tt_rel, ts_abs, vals, torch.tensor(label, dtype=torch.long, device=self.device)

        # 绝对时间戳：Unix seconds（可还原成 YYYY-mm-ddTHH:MM:SSZ）
        ts_abs_np = (df["t"].astype("int64") // 10**9).astype(np.int64).to_numpy()

        # 相对时间（秒）：用于 patch 划分
        t0 = df["t"].iloc[0]
        tt_rel_np = (df["t"] - t0).dt.total_seconds().astype(np.float32).to_numpy()

        vals = df[self.feature_cols].astype(np.float32).to_numpy()
        vals = np.nan_to_num(vals, nan=0.0)

        return (
            record_id,
            torch.from_numpy(tt_rel_np).to(self.device),                 # (T,)
            torch.from_numpy(ts_abs_np).to(self.device),                 # (T,) int64
            torch.from_numpy(vals).to(self.device),                       # (T,D)
            torch.tensor(label, dtype=torch.long).to(self.device)         # ()
        )



# ============================================================
# 这里的collate_fn是分类任务 + 固定间隔 patch 我这里设置的是5分钟
#    输出：
#      data_sequence: (B, P, Lmax, D)
#      label: (B,)
#      patch_mask: (B, P)         1=有真实点，0=空patch
#      patch_index: (B, P)        每个patch真实点数；空patch=1(因为dummy占位)
# ============================================================
def patch_classification_collate_fn_flat(batch, args, device=torch.device("cpu"), **kwargs):
    patch_sec = args.patch_minutes * 60.0
    P = args.max_patches
    max_L = args.max_total_steps

    seq_list = []
    ts_list = []
    delta_list = []
    patch_mask_list = []
    patch_index_list = []
    label_list = []
    num_batch_list = []

    for record_id, tt_rel, ts_abs, vals, y in batch:
        label_list.append(y.to(device))

        # 用相对时间分 patch（0~75min）
        patch_id = torch.floor(tt_rel / patch_sec).to(torch.long)
        keep = patch_id < P

        patch_id_kept = patch_id[keep]
        vals_kept = vals[keep]
        ts_kept = ts_abs[keep]  # 绝对时间戳（unix秒）

        flat_val_chunks = []
        flat_ts_chunks = []

        patch_index = torch.zeros(P, dtype=torch.long, device=device)
        patch_mask = torch.zeros(P, dtype=torch.float32, device=device)

        for p in range(P):
            idxs = torch.where(patch_id_kept == p)[0]
            n = int(idxs.numel())
            patch_index[p] = n
            patch_mask[p] = 1.0 if n > 0 else 0.0
            if n > 0:
                flat_val_chunks.append(vals_kept[idxs])
                flat_ts_chunks.append(ts_kept[idxs])

        if len(flat_val_chunks) == 0:
            flat_seq = torch.zeros((1, vals.size(-1)), device=device, dtype=vals.dtype)
            flat_ts = torch.zeros((1,), device=device, dtype=torch.long)
            flat_delta = torch.zeros((1,), device=device, dtype=torch.float32)
            # patch_index 全0，patch_mask 全0
        else:
            flat_seq = torch.cat(flat_val_chunks, dim=0)          # (L,D)
            flat_ts = torch.cat(flat_ts_chunks, dim=0).to(device) # (L,) int64

            # 裁剪总步数
            if flat_seq.size(0) > max_L:
                flat_seq = flat_seq[:max_L]
                flat_ts = flat_ts[:max_L]

                remaining = max_L
                new_index = torch.zeros_like(patch_index)
                new_mask = torch.zeros_like(patch_mask)
                for p in range(P):
                    n = int(patch_index[p].item())
                    if n == 0:
                        continue
                    take = min(n, remaining)
                    if take > 0:
                        new_index[p] = take
                        new_mask[p] = 1.0
                        remaining -= take
                    if remaining == 0:
                        break
                patch_index, patch_mask = new_index, new_mask

            # delta：相邻时间戳差（秒）
            flat_delta = torch.zeros((flat_ts.size(0),), device=device, dtype=torch.float32)
            if flat_ts.size(0) > 1:
                flat_delta[1:] = (flat_ts[1:] - flat_ts[:-1]).to(torch.float32)

        seq_list.append(flat_seq)
        ts_list.append(flat_ts.to(torch.long))
        delta_list.append(flat_delta)
        patch_index_list.append(patch_index)
        patch_mask_list.append(patch_mask)
        num_batch_list.append(flat_seq.size(0))

    data_sequence = pad_sequence(seq_list, batch_first=True)        # (B,Lmax,D)
    ts_sequence = pad_sequence(ts_list, batch_first=True)           # (B,Lmax) int64
    delta_sequence = pad_sequence(delta_list, batch_first=True)     # (B,Lmax) float32

    label = torch.stack(label_list, dim=0)
    patch_index = torch.stack(patch_index_list, dim=0)
    patch_mask = torch.stack(patch_mask_list, dim=0)
    num_batch = torch.tensor(num_batch_list, dtype=torch.long, device=device)

    return {
        "data_sequence": data_sequence.to(device),      # (B,Lmax,D)
        "ts_sequence": ts_sequence.to(device),          # (B,Lmax) 绝对时间戳 unix秒
        "delta_sequence": delta_sequence.to(device),    # (B,Lmax) 相邻时间间隔(秒)
        "label": label.to(device),                      # (B,)
        "patch_mask": patch_mask.to(device),            # (B,P)
        "patch_index": patch_index.to(device),          # (B,P)
        "num_batch": num_batch,                         # (B,)
    }


# ============================================================
# 4) build data_objects（纯测试）
# ============================================================
def build_data_objects_for_test(args):
    merged_samples = load_and_merge_json(args.json_path)
    dataset = MergedTrajectoryDataset(merged_samples, device=args.device, use_dt_as_feature=True)

    seen, test = model_selection.train_test_split(dataset, train_size=0.8, random_state=42, shuffle=True)
    train, val = model_selection.train_test_split(seen, train_size=0.75, random_state=42, shuffle=False)

    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: patch_classification_collate_fn_flat(batch, args, args.device)
    )
    
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: patch_classification_collate_fn_flat(batch, args, args.device)
    )
    test_loader = DataLoader(
        test, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: patch_classification_collate_fn_flat(batch, args, args.device)
    )

    # input_dim = D
    _, tt0, ts0, vals0, y0 = train[0]
    input_dim = vals0.size(-1)

    data_objects = {
        "train_dataloader": iter(train_loader),
        "val_dataloader": iter(val_loader),
        "test_dataloader": iter(test_loader),
        "input_dim": input_dim,
        "n_train_batches": len(train_loader),
        "n_val_batches": len(val_loader),
        "n_test_batches": len(test_loader),
    }
    return data_objects


# ============================================================
# 5) quick test
# ============================================================
if __name__ == "__main__":
    args = Args(device=torch.device("cpu"), batch_size=2, patch_minutes=5, max_patches=15, json_path = "/home/UNT/yl0826/QAU/t-PatchGNN/lib/test.json")

    data_obj = build_data_objects_for_test(args)

    batch = next(data_obj["train_dataloader"])
    print("keys:", list(batch.keys()))
    for k, v in batch.items():
        print(k, v, tuple(v.shape), v.dtype)
