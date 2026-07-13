"""Politika bazlı 'zaman içinde kalite' figürü (proposal'daki streaming ana grafiği).

Her replay koşusunun predictions.csv'sinden time-step bazlı AUPRC hesaplar ve
politikaları aynı eksende karşılaştırır. Beklenen okuma:
  * ts<43: taze politikalar ile bayat politikalar arasındaki gerçek fark
  * ts>=43: drift altında hangi politikanın nasıl davrandığı

Kullanım:
  python src/plot_quality_over_time.py --runs runs/replay/* runs/replay/adaptive_t* \
      --out runs/replay/quality_over_time.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def label_of(run_dir: Path) -> str:
    f = run_dir / "summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        par = s.get("params") or {}
        short = ",".join(f"{k}={v}" for k, v in par.items() if k in ("period", "tau"))
        return s["policy"] + (f" ({short})" if short else "")
    return run_dir.name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="quality_over_time.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))
    table = []
    for r in args.runs:
        run = Path(r)
        f = run / "predictions.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        xs, ys = [], []
        for t, g in df.groupby("time_step"):
            if g.label.sum() == 0 or (g.label == 0).sum() == 0:
                continue
            xs.append(t)
            ys.append(average_precision_score(g.label, g.score))
        ax.plot(xs, ys, "o-", ms=4, label=label_of(run))
        table.append(pd.DataFrame({"time_step": xs, "auprc": ys, "run": label_of(run)}))

    if not table:
        raise SystemExit("predictions.csv bulunamadı; --runs yollarını kontrol et.")
    ax.axvline(43, color="red", ls=":", lw=1.5, label="ts 43 (drift)")
    ax.set_xlabel("Time step"); ax.set_ylabel("AUPRC"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Refresh politikaları — zaman içinde tahmin kalitesi (arrival-time scoring)")
    fig.tight_layout(); fig.savefig(args.out, dpi=150)
    pd.concat(table).to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(f"[tamam] {args.out} ve .csv")


if __name__ == "__main__":
    main()
