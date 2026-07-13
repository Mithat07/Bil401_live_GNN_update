"""Offline modelin test dönemindeki time-step bazlı performansı.

Amaç: val (0.95) -> test (0.43) düşüşünün Elliptic'teki bilinen rejim
değişiminden (ts ~43 dark market kapatması, Weber et al. 2019) geldiğini
doğrulamak. Beklenti: ts 35-42 arasında makul skorlar, ts 43 sonrası çöküş.

Çıktılar (model klasörüne):
  per_ts_metrics.csv   her test time step için AUPRC / F1 / illicit sayısı
  per_ts_metrics.png   rapora girecek figür

Kullanım:
  python src/analyze_offline_by_ts.py --data processed/data.pt --model runs/sage_v1/model_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import GraphSAGEModel
from utils import compute_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    out_dir = Path(args.model).parent

    d = torch.load(args.data, map_location="cpu", weights_only=True)
    model, extra = GraphSAGEModel.load(args.model)
    thr = extra["threshold"]

    with torch.no_grad():
        scores = F.softmax(model(d["x"], d["edge_index"]), dim=-1)[:, 1].numpy()

    y, ts = d["y"].numpy(), d["time_step"].numpy()
    test_mask = d["test_mask"].numpy()

    rows = []
    for t in sorted(np.unique(ts[test_mask])):
        m = test_mask & (ts == t)
        yt, st = y[m], scores[m]
        if yt.sum() == 0 or (yt == 0).sum() == 0:
            rows.append({"time_step": int(t), "auprc": np.nan, "f1": np.nan,
                         "n_labeled": int(m.sum()), "n_illicit": int(yt.sum())})
            continue
        met = compute_metrics(yt, st, thr)
        rows.append({"time_step": int(t), "auprc": met["auprc"], "f1": met["f1"],
                     "n_labeled": int(m.sum()), "n_illicit": int(yt.sum())})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_ts_metrics.csv", index=False)
    print(df.to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(df.time_step, df.auprc, "o-", label="AUPRC", color="tab:blue")
    ax1.plot(df.time_step, df.f1, "s--", label="F1", color="tab:orange")
    ax1.set_xlabel("Time step"); ax1.set_ylabel("Skor"); ax1.set_ylim(0, 1)
    ax1.axvline(43, color="red", ls=":", label="ts 43 (dark market kapatması)")
    ax2 = ax1.twinx()
    ax2.bar(df.time_step, df.n_illicit, alpha=0.15, color="gray", label="illicit sayısı")
    ax2.set_ylabel("illicit düğüm sayısı")
    ax1.legend(loc="upper right"); ax1.set_title("Offline GraphSAGE — test dönemi time-step bazlı performans")
    fig.tight_layout(); fig.savefig(out_dir / "per_ts_metrics.png", dpi=150)
    print(f"[tamam] {out_dir}/per_ts_metrics.csv, per_ts_metrics.png")


if __name__ == "__main__":
    main()
