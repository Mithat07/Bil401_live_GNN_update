"""Çoklu seed koşularını topla: ortalama ± std tablosu + hata çubuklu figür.

Girdi düzeni (run_multiseed.sh üretir):
    runs/multiseed/seed<S>/replay/<policy>/summary.json
    runs/multiseed/seed<S>/replay/adaptive_t<TAU>/summary.json
    runs/multiseed/seed<S>/edge_ablation.csv        (eval_edge_ablation.py çıktısı)
    runs/multiseed/seed<S>/metrics.json             (train_graphsage.py çıktısı)

Çıktı:
    <out-dir>/multiseed_summary.csv     her koşu tek satır (uzun format)
    <out-dir>/multiseed_table.csv       ortalama ± std, rapora giren tablo
    <out-dir>/freshness_multiseed.png/.pdf

Kullanım:
    python src/aggregate_seeds.py --root runs/multiseed --out-dir figures
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

# rapordaki gösterim adları
LABEL = {
    "no_refresh": "NoRefresh",
    "full_always": "FullAlways",
    "local_always": "LocalAlways",
    "local_periodic": "LocalPeriodic N=5",
    "local_adaptive": "LocalAdaptive",
}
ORDER = ["FullAlways", "LocalAlways", "LocalAdaptive t=0.05", "LocalAdaptive t=0.15",
         "LocalAdaptive t=0.25", "LocalAdaptive t=0.35", "LocalAdaptive t=0.5",
         "LocalPeriodic N=5", "NoRefresh"]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """scipy'siz Spearman (bağ yoksa tam, bağ varsa yaklaşık)."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, abl = [], []
    seed_dirs = sorted([d for d in root.glob("seed*") if d.is_dir()],
                       key=lambda p: int(p.name.replace("seed", "")))
    if not seed_dirs:
        raise SystemExit(f"[hata] {root} altinda seed* klasoru yok.")

    for sd in seed_dirs:
        seed = int(sd.name.replace("seed", ""))

        # --- politika koşuları ---
        for f in sorted(sd.glob("replay/**/summary.json")):
            s = json.loads(f.read_text())
            if s.get("status", "ok") != "ok" or s.get("n_batches", 1) == 0:
                print(f"[atlandi] {f}: status={s.get('status')} n_batches={s.get('n_batches')}")
                continue
            name = LABEL.get(s["policy"], s["policy"])
            tau = s.get("params", {}).get("tau")
            if s["policy"] == "local_adaptive" and tau is not None:
                name = f"LocalAdaptive t={tau:g}"
            rows.append({
                "seed": seed, "run": name, "policy": s["policy"], "tau": tau,
                "fresh": s.get("fresh_at_arrival_rate"),
                "arrival_auprc": s["quality"]["auprc"],
                "arrival_f1": s["quality"]["f1"],
                "eos_auprc": s["quality_end_of_stream"]["auprc"],
                "updates": s["cost"]["total_node_updates"],
                "cpu_s": s["cost"]["total_cpu_refresh_s"],
                "p50_ms": s["cost"]["wall_ms_p50"],
                "p99_ms": s["cost"]["wall_ms_p99"],
                "n_batches": s.get("n_batches"),
            })

        # --- kenar ablasyonu (varsa) ---
        af = sd / "edge_ablation.csv"
        mf = sd / "metrics.json"
        if af.exists():
            df = pd.read_csv(af)
            rec = {"seed": seed,
                   "per_ts_mean_gain": float((df.auprc_no_edges - df.auprc_full_graph).mean())}
            pre = df[df.time_step <= 42]
            rec["pre_drift_gain"] = float((pre.auprc_no_edges - pre.auprc_full_graph).mean())
            rec["pre_drift_wins"] = int((pre.auprc_no_edges > pre.auprc_full_graph).sum())
            rec["pre_drift_n"] = int(len(pre))
            if mf.exists():
                m = json.loads(mf.read_text())
                rec["val_auprc"] = m["val"]["auprc"]
                rec["threshold"] = m["val"]["threshold"]
                rec["test_full_graph_auprc"] = m["test_offline_upper_bound"]["auprc"]
            abl.append(rec)

    long = pd.DataFrame(rows)
    # --policy all zaten local_adaptive'i varsayilan tau ile kosuyor; ayni tau
    # taramada da varsa cift sayilmasin (std yapay olarak kuculurdu).
    dup = long.duplicated(subset=["seed", "run"], keep="first")
    if dup.any():
        print(f"[uyari] {int(dup.sum())} yinelenen (seed, run) satiri atlandi "
              f"-- ayni tau hem --policy all hem taramada kosulmus.")
        long = long[~dup].reset_index(drop=True)
    return long, pd.DataFrame(abl)


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("run")
    out = pd.DataFrame({
        "n_seeds": g.size(),
        "fresh_mean": g.fresh.mean(), "fresh_std": g.fresh.std(ddof=1),
        "arrival_mean": g.arrival_auprc.mean(), "arrival_std": g.arrival_auprc.std(ddof=1),
        "f1_mean": g.arrival_f1.mean(), "f1_std": g.arrival_f1.std(ddof=1),
        "eos_mean": g.eos_auprc.mean(), "eos_std": g.eos_auprc.std(ddof=1),
        "updates_mean": g.updates.mean(),
        "p99_mean": g.p99_ms.mean(), "p99_std": g.p99_ms.std(ddof=1),
    })
    out = out.reindex([r for r in ORDER if r in out.index]
                      + [r for r in out.index if r not in ORDER])
    return out.reset_index()


def make_figure(long: pd.DataFrame, tab: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    C = {"NoRefresh": "#1b6e3c", "FullAlways": "#1f4e79", "LocalAlways": "#4a7fb5",
         "LocalPeriodic N=5": "#e0a800"}
    ad = "#c0504d"

    for ax, mean_c, std_c, ylab, title in [
        (axes[0], "arrival_mean", "arrival_std", "Arrival-time AUPRC",
         "(a) Less refresh $\\rightarrow$ higher arrival-time quality"),
        (axes[1], "eos_mean", "eos_std", "End-of-stream AUPRC",
         "(b) Partially refreshed caches are worst"),
    ]:
        for _, r in tab.iterrows():
            col = C.get(r["run"], ad)
            mk = "D" if r["run"] == "NoRefresh" else ("^" if "Periodic" in r["run"] else
                 ("s" if r["run"] == "FullAlways" else ("P" if r["run"] == "LocalAlways" else "o")))
            ax.errorbar(r.fresh_mean, r[mean_c],
                        yerr=(0 if np.isnan(r[std_c]) else r[std_c]),
                        xerr=(0 if np.isnan(r.fresh_std) else r.fresh_std),
                        fmt=mk, color=col, ms=8, capsize=3, lw=1.2,
                        markeredgecolor="white", markeredgewidth=0.9, zorder=3)
        ax.set_xlabel("Fresh-at-arrival rate", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=10.5, pad=9)
        ax.grid(alpha=0.28)

    # panel (a): trend + korelasyon (seed bazında hesaplanıp ortalanır)
    ax = axes[0]
    f, a = tab.fresh_mean.values, tab.arrival_mean.values
    m, b = np.polyfit(f, a, 1)
    xs = np.linspace(min(f) - 0.05, max(f) + 0.05, 40)
    ax.plot(xs, m * xs + b, "--", color="#8a8f96", lw=1.6, zorder=1)
    rhos = [spearman(d.fresh.values, d.arrival_auprc.values)
            for _, d in long.groupby("seed") if len(d) >= 3]
    rho_txt = (f"Spearman $\\rho$ = {np.mean(rhos):+.2f} $\\pm$ {np.std(rhos, ddof=1):.2f}"
               if len(rhos) > 1 else f"Spearman $\\rho$ = {np.mean(rhos):+.2f}")
    ax.text(0.97, 0.97, f"{rho_txt}\n{len(rhos)} seeds $\\times$ {len(tab)} configurations\n"
                        "error bars: $\\pm$1 s.d. across seeds",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.4,
            bbox=dict(fc="#f4f6f8", ec="#d4d8dd", boxstyle="round,pad=0.45"))

    for _, r in tab.iterrows():
        if r["run"] in ("NoRefresh", "LocalPeriodic N=5"):
            axes[0].annotate(r["run"], (r.fresh_mean, r.arrival_mean),
                             textcoords="offset points", xytext=(11, 3), fontsize=8,
                             color=C.get(r["run"], ad))

    fig.suptitle("Refresh freshness vs. classification quality — mean $\\pm$ s.d. over "
                 f"{int(tab.n_seeds.max())} training seeds", fontsize=10.8, y=1.005)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "freshness_multiseed.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "freshness_multiseed.pdf", bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="runs/multiseed")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()
    root, out_dir = Path(args.root), Path(args.out_dir)

    long, abl = collect(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_dir / "multiseed_summary.csv", index=False)

    tab = summarise(long)
    tab.to_csv(out_dir / "multiseed_table.csv", index=False)

    print(f"\n=== {long.seed.nunique()} seed x {long.run.nunique()} kosu ===\n")
    show = tab.assign(
        fresh=lambda d: d.fresh_mean.map("{:.3f}".format) + " ± " + d.fresh_std.map("{:.3f}".format),
        arrival=lambda d: d.arrival_mean.map("{:.4f}".format) + " ± " + d.arrival_std.map("{:.4f}".format),
        EOS=lambda d: d.eos_mean.map("{:.4f}".format) + " ± " + d.eos_std.map("{:.4f}".format),
        p99=lambda d: d.p99_mean.map("{:.1f}".format) + " ± " + d.p99_std.map("{:.1f}".format),
    )[["run", "n_seeds", "fresh", "arrival", "EOS", "p99"]]
    print(show.to_string(index=False))

    rhos = [spearman(d.fresh.values, d.arrival_auprc.values)
            for _, d in long.groupby("seed") if len(d) >= 3]
    if rhos:
        print(f"\nSpearman(fresh, arrival) seed bazinda: "
              f"{np.mean(rhos):+.4f} ± {np.std(rhos, ddof=1) if len(rhos)>1 else 0:.4f}  "
              f"(tekil: {', '.join(f'{r:+.2f}' for r in rhos)})")

    if len(abl):
        abl.to_csv(out_dir / "multiseed_ablation.csv", index=False)
        print("\n=== KENAR ABLASYONU (no_edges - full_graph) ===")
        print(abl.to_string(index=False))
        print(f"\ndrift oncesi (ts<=42) ortalama kazanc: "
              f"{abl.pre_drift_gain.mean():+.4f} ± "
              f"{abl.pre_drift_gain.std(ddof=1) if len(abl)>1 else 0:.4f}")
        if "test_full_graph_auprc" in abl:
            print(f"test full-graph AUPRC: {abl.test_full_graph_auprc.mean():.4f} ± "
                  f"{abl.test_full_graph_auprc.std(ddof=1) if len(abl)>1 else 0:.4f}")
            print(f"karar esigi          : {abl.threshold.mean():.4f} ± "
                  f"{abl.threshold.std(ddof=1) if len(abl)>1 else 0:.4f}")

    make_figure(long, tab, out_dir)
    print(f"\n[tamam] {out_dir}/multiseed_summary.csv, multiseed_table.csv, "
          f"freshness_multiseed.png/.pdf")


if __name__ == "__main__":
    main()
