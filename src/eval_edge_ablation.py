"""Kenar ablasyonu — 'bayat gömü neden kazandı' hipotezinin kontrollü testi.

Aynı eğitilmiş modeli test kümesinde iki kez değerlendirir:
  full_graph : gerçek kenarlarla (normal GNN çıkarımı)
  no_edges   : BOŞ kenar kümesiyle (SAGEConv komşusuz -> yalnızca-özellik / MLP yolu;
               replay'deki 'hiç tazelenmemiş' gömülerin birebir karşılığı)

Hipotez: Elliptic test döneminde (özellikle ts>=43 drift sonrası) no_edges,
full_graph'ı geçer -> komşuluk agregasyonu bu rejimde net zarar veriyor ve
replay'de bayat politikaların önde görünmesi bir bug değil, bu mekanizmadır.

Çıktılar (model klasörüne): edge_ablation.csv, edge_ablation.png

Kullanım:
  python src/eval_edge_ablation.py --data processed/data.pt --model runs/sage_v1/model_best.pt
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


@torch.no_grad()
def scores_with(model, x, edge_index) -> np.ndarray:
    return F.softmax(model(x, edge_index), dim=-1)[:, 1].numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    out_dir = Path(args.model).parent

    d = torch.load(args.data, map_location="cpu", weights_only=True)
    model, extra = GraphSAGEModel.load(args.model)
    thr = extra["threshold"]
    x = d["x"]
    y, ts = d["y"].numpy(), d["time_step"].numpy()
    test_mask = d["test_mask"].numpy()

    variants = {
        "full_graph": scores_with(model, x, d["edge_index"]),
        "no_edges": scores_with(model, x, torch.empty((2, 0), dtype=torch.long)),
    }

    # ---------- Genel test metrikleri ----------
    print(f"{'variant':<12} {'AUPRC':>8} {'F1':>8} {'P':>8} {'R':>8}")
    overall = {}
    for name, s in variants.items():
        m = compute_metrics(y[test_mask], s[test_mask], thr)
        overall[name] = m
        print(f"{name:<12} {m['auprc']:>8.4f} {m['f1']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f}")

    # ---------- Time-step bazlı ----------
    rows = []
    for t in sorted(np.unique(ts[test_mask])):
        msk = test_mask & (ts == t)
        yt = y[msk]
        if yt.sum() == 0 or (yt == 0).sum() == 0:
            continue
        row = {"time_step": int(t), "n_illicit": int(yt.sum())}
        for name, s in variants.items():
            row[f"auprc_{name}"] = compute_metrics(yt, s[msk], thr)["auprc"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "edge_ablation.csv", index=False)
    print("\n" + df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df.time_step, df.auprc_full_graph, "o-", label="tam graf (GNN)")
    ax.plot(df.time_step, df.auprc_no_edges, "s--", label="kenarsız (yalnızca özellik)")
    ax.axvline(43, color="red", ls=":", label="ts 43 (dark market kapatması)")
    ax.set_xlabel("Time step"); ax.set_ylabel("AUPRC"); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Komşuluk agregasyonunun değeri zaman içinde — kenar ablasyonu")
    fig.tight_layout(); fig.savefig(out_dir / "edge_ablation.png", dpi=150)
    print(f"\n[tamam] {out_dir}/edge_ablation.csv, edge_ablation.png")
    diff = overall["no_edges"]["auprc"] - overall["full_graph"]["auprc"]
    print(f"[yorum] no_edges - full_graph = {diff:+.4f} AUPRC "
          f"({'hipotez DOĞRULANDI: agregasyon net zarar' if diff > 0 else 'hipotez reddedildi: replay tarafını incele'})")


if __name__ == "__main__":
    main()
