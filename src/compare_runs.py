"""Replay koşularını karşılaştır: tablo + accuracy-cost Pareto figürü (proposal çıktısı).

Kullanım:
  python src/compare_runs.py --runs runs/replay/* --out runs/replay/pareto.png
(--runs, summary.json içeren klasörleri alır; glob genişletmesini shell yapar.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="pareto.png")
    args = p.parse_args()

    rows = []
    for r in args.runs:
        f = Path(r) / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        if s.get("status", "ok") != "ok" or s.get("n_batches", 1) == 0:
            print(f"[ATLANDI] {r}: status={s.get('status')} n_batches={s.get('n_batches')} "
                  f"fallback={s.get('n_fallback')}")
            continue
        rows.append({
            "policy": s["policy"], "params": json.dumps(s["params"]),
            "auprc": s["quality"]["auprc"], "f1": s["quality"]["f1"],
            "updates": s["cost"]["total_node_updates"],
            "cpu_s": s["cost"]["total_cpu_refresh_s"],
            "p50_ms": s["cost"]["wall_ms_p50"], "p99_ms": s["cost"]["wall_ms_p99"],
        })
    if not rows:
        raise SystemExit("summary.json bulunamadı; --runs yollarını kontrol et.")
    df = pd.DataFrame(rows).sort_values("updates")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, xcol, xlabel in [(axes[0], "updates", "Toplam güncellenen düğüm (maliyet)"),
                             (axes[1], "p99_ms", "p99 batch süresi (ms)")]:
        for _, row in df.iterrows():
            ax.scatter(row[xcol], row.auprc, s=90)
            ax.annotate(row.policy, (row[xcol], row.auprc),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("AUPRC")
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy–cost Pareto: refresh politikaları")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(f"[tamam] {args.out} ve .csv")


if __name__ == "__main__":
    main()
