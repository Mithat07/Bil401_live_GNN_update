"""Elliptic Bitcoin veri setini streaming-GNN deneyine hazırlar.

Girdi (Kaggle: ellipticco/elliptic-data-set):
    elliptic_txs_features.csv   header YOK. col0=txId, col1=time_step(1..49),
                                sonraki sütunlar özellikler (93 yerel + 72 agregat).
    elliptic_txs_classes.csv    header: txId,class  ('1'=illicit, '2'=licit, 'unknown')
    elliptic_txs_edgelist.csv   header: txId1,txId2 (yönlü; aynı time step içinde)

Çıktı (out_dir):
    data.pt            x, edge_index (undirected), y, time_step, maskeler, meta
    stream_edges.csv   src,dst,time_step — kronolojik; Kafka producer/replay girdisi
    node_id_map.csv    txId <-> dahili index eşlemesi
    meta.json          split sınırları, boyutlar, sınıf sayıları

Temporal split (literatür standardı, 34/49 sınırı):
    train: ts 1..train_end   (varsayılan 29)
    val:   ts train_end+1..val_end (varsayılan 34)
    test:  ts val_end+1..49  — streaming deneyinin replay edeceği dönem

Notlar:
- time_step ÖZELLİK OLARAK KULLANILMAZ (temporal leakage); ayrı tensörde tutulur.
- Özellikler yalnızca train dönemindeki düğümlerin ortalama/std'siyle standardize
  edilir (leakage önlemi); mean/std data.pt içine kaydedilir ki streaming tarafı
  aynı dönüşümü uygulayabilsin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import to_undirected

LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}  # illicit=1, licit=0, etiketsiz=-1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="Elliptic CSV'lerinin bulunduğu klasör")
    p.add_argument("--out-dir", required=True, help="İşlenmiş çıktıların yazılacağı klasör")
    p.add_argument("--train-end", type=int, default=29, help="Train son time step (dahil)")
    p.add_argument("--val-end", type=int, default=34, help="Val son time step (dahil)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Oku ----------
    feats = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    print(f"[oku] features={feats.shape}  classes={classes.shape}  edges={edges.shape}")

    # ---------- 2) txId -> ardışık index ----------
    tx_ids = feats[0].astype(np.int64).to_numpy()
    id_map = {tx: i for i, tx in enumerate(tx_ids)}
    n = len(tx_ids)

    time_step = feats[1].astype(np.int64).to_numpy().copy()
    x = feats.iloc[:, 2:].to_numpy(dtype=np.float32)  # time_step özellik değil!

    # ---------- 3) Etiketler ----------
    y = np.full(n, -1, dtype=np.int64)
    cls = classes.copy()
    cls["idx"] = cls["txId"].map(id_map)
    cls = cls.dropna(subset=["idx"])
    y[cls["idx"].astype(int).to_numpy()] = cls["class"].astype(str).map(LABEL_MAP).to_numpy()

    # ---------- 4) Temporal maskeler ----------
    labeled = y >= 0
    train_mask = labeled & (time_step <= args.train_end)
    val_mask = labeled & (time_step > args.train_end) & (time_step <= args.val_end)
    test_mask = labeled & (time_step > args.val_end)

    # ---------- 5) Özellik standardizasyonu (yalnızca train istatistikleriyle) ----------
    train_rows = time_step <= args.train_end
    mu = x[train_rows].mean(axis=0)
    sd = x[train_rows].std(axis=0)
    sd[sd < 1e-6] = 1.0
    x = (x - mu) / sd

    # ---------- 6) Kenarlar ----------
    src = edges["txId1"].map(id_map).to_numpy(dtype=np.int64)
    dst = edges["txId2"].map(id_map).to_numpy(dtype=np.int64)
    edge_index_directed = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_index = to_undirected(edge_index_directed, num_nodes=n)  # GraphSAGE için

    # Elliptic garantisi: kenarlar aynı time step içinde. Doğrula, sonra replay dosyası yaz.
    ts_src, ts_dst = time_step[src], time_step[dst]
    cross = int((ts_src != ts_dst).sum())
    if cross:
        print(f"[uyarı] {cross} kenar farklı time step'leri bağlıyor (beklenmiyordu).")
    stream = pd.DataFrame({"src": src, "dst": dst, "time_step": np.maximum(ts_src, ts_dst)})
    stream = stream.sort_values(["time_step"]).reset_index(drop=True)
    stream.to_csv(out_dir / "stream_edges.csv", index=False)

    # ---------- 7) Kaydet ----------
    meta = {
        "num_nodes": int(n),
        "num_features": int(x.shape[1]),
        "num_edges_undirected": int(edge_index.shape[1]),
        "train_end": args.train_end,
        "val_end": args.val_end,
        "max_ts": int(time_step.max()),
        "counts": {
            "train_labeled": int(train_mask.sum()),
            "val_labeled": int(val_mask.sum()),
            "test_labeled": int(test_mask.sum()),
            "illicit_total": int((y == 1).sum()),
            "licit_total": int((y == 0).sum()),
            "unknown_total": int((y == -1).sum()),
        },
    }
    torch.save(
        {
            "x": torch.from_numpy(x),
            "edge_index": edge_index,
            "y": torch.from_numpy(y),
            "time_step": torch.from_numpy(time_step),
            "train_mask": torch.from_numpy(train_mask),
            "val_mask": torch.from_numpy(val_mask),
            "test_mask": torch.from_numpy(test_mask),
            "feat_mean": torch.from_numpy(mu),
            "feat_std": torch.from_numpy(sd),
            "meta": meta,
        },
        out_dir / "data.pt",
    )
    pd.DataFrame({"txId": tx_ids, "idx": np.arange(n)}).to_csv(out_dir / "node_id_map.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))
    print(f"[tamam] Çıktılar: {out_dir}/data.pt, stream_edges.csv, node_id_map.csv, meta.json")


if __name__ == "__main__":
    main()
