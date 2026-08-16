"""Statik replay motoru — Kafka'sız uçtan uca politika deneyi.

Test dönemi kenarları (ts > val_end) kronolojik mikro-batch'ler halinde beslenir;
her batch'te proposal'daki döngü aynen işler:

  1. grafı güncelle            (GraphStore.add_edges)
  2. affected set çıkar        (endpoints ∪ L-hop, denklem 1)
  3. politika kararı           (Policy.decide)
  4. yerel/tam gömü refresh    (model.embed, alt graf veya tam graf)
  5. tahmin: bu batch'te İLK KEZ görünen etiketli düğümler cache'teki gömüyle
     skorlanır (streaming serving hikayesinin birebir karşılığı)

Loglanan maliyet/kalite metrikleri: güncellenen düğüm sayısı, refresh CPU süresi,
batch duvar süresi (p50/p99 buradan), AUPRC/F1 (eğitimde seçilen sabit eşikle).

Kullanım (tek politika):
  python src/replay.py --data processed/data.pt --model runs/sage_v1/model_best.pt \
      --initial-state runs/sage_v1/initial_state.pt --policy local_adaptive --tau 0.5 \
      --out-dir runs/replay/adaptive_t05

Dört politikayı varsayılanlarla arka arkaya koşmak için: --policy all
Not: full_always CPU'da en yavaş politikadır (her batch tam graf forward).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from embedding_store import EmbeddingStore
from graph_store import GraphStore
from model import GraphSAGEModel
from policies import FULL_GRAPH, make_policy
from utils import compute_metrics, edges_up_to, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--initial-state", required=True)
    p.add_argument("--stream", default=None, help="stream_edges.csv (varsayılan: data.pt klasörü)")
    p.add_argument("--policy", required=True,
                   choices=["no_refresh", "full_always", "local_always", "local_periodic", "local_adaptive", "all"])
    p.add_argument("--period", type=int, default=5, help="local_periodic: N")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--staleness-norm", default="exp", choices=["exp", "pool_max"],
                   help="local_adaptive skor normalizasyonu (exp: mutlak, önerilen)")
    p.add_argument("--batches-per-ts", type=int, default=4,
                   help="her time step kaç mikro-batch'e bölünsün")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_one(policy_name: str, args, d, model, init, stream_df) -> dict:
    set_seed(args.seed)
    out_dir = Path(args.out_dir) / (policy_name if args.policy == "all" else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, ts = d["x"], d["y"].numpy(), d["time_step"].numpy()
    val_end, max_ts = init["val_end"], int(ts.max())
    thr = init["threshold"]
    L = model.hparams["num_layers"]

    # ---------- Başlangıç durumu: graf ts<=val_end, gömüler initial_state ----------
    graph = GraphStore(num_nodes=x.shape[0])
    e_init = edges_up_to(d["edge_index"], d["time_step"], val_end)
    graph.add_edges(e_init[0].tolist(), e_init[1].tolist())
    store = EmbeddingStore(init["embeddings"], start_batch_id=0.0)
    policy = make_policy(policy_name, period=args.period, tau=args.tau,
                         alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                         norm=args.staleness_norm)

    predicted: dict[int, tuple[float, int, int]] = {}  # node -> (score, ts, batch)
    rows, batch_id = [], 0
    print(f"\n=== {policy_name} {policy.params()} | L={L} thr={thr:.3f} "
          f"| ts {val_end + 1}..{max_ts} x {args.batches_per_ts} batch ===")

    for t in range(val_end + 1, max_ts + 1):
        chunk = stream_df[stream_df.time_step == t]
        if len(chunk) == 0:
            continue
        for part in np.array_split(np.arange(len(chunk)), args.batches_per_ts):
            if len(part) == 0:
                continue
            batch_id += 1
            b = chunk.iloc[part]
            src, dst = b["src"].to_numpy(), b["dst"].to_numpy()
            t_wall = time.perf_counter()

            # 1-2) graf + affected set
            graph.add_edges(src, dst)
            store.on_edges_added(src, dst)
            endpoints = set(map(int, src)) | set(map(int, dst))
            affected = graph.affected_set(endpoints, L)

            # 3) politika kararı
            decision = policy.decide(affected, store, batch_id)

            # 4) refresh
            t_cpu = time.process_time()
            if decision == FULL_GRAPH:
                emb = model.embed(x, graph.edge_index_full())
                store.emb = emb
                store.last_update.fill_(float(batch_id))
                store.degree_change.zero_()
                store.neighbor_change.zero_()
                store.total_node_updates += graph.num_nodes
                n_refresh = graph.num_nodes
            elif decision:
                targets = sorted(decision)
                sub_nodes, e_sub, target_pos = graph.subgraph(targets, L)
                emb_sub = model.embed(x[sub_nodes], e_sub)
                neigh = set()
                for u in targets:
                    neigh |= graph.neighbors(u)
                store.update(targets, emb_sub[target_pos], batch_id, neigh)
                n_refresh = len(targets)
            else:
                n_refresh = 0
            cpu_refresh = time.process_time() - t_cpu

            # 5) tahmin: ilk kez görünen etiketli test düğümleri
            new_nodes = [n for n in endpoints
                         if y[n] >= 0 and ts[n] > val_end and n not in predicted]
            if new_nodes:
                scores = model.score_illicit(store.emb[new_nodes]).tolist()
                for n, s in zip(new_nodes, scores):
                    fresh = bool(store.last_update[n].item() == float(batch_id))
                    predicted[n] = (s, int(ts[n]), batch_id, fresh)

            rows.append({"batch": batch_id, "time_step": t, "n_edges": len(b),
                         "n_affected": len(affected), "n_refreshed": n_refresh,
                         "cpu_refresh_s": cpu_refresh,
                         "wall_ms": (time.perf_counter() - t_wall) * 1e3,
                         "n_new_preds": len(new_nodes)})
        print(f"  ts {t}: batch {batch_id}, toplam güncelleme {store.total_node_updates}")

    # ---------- Akışta hiç görünmeyen etiketli test düğümleri (fallback) ----------
    all_test = np.where((y >= 0) & (ts > val_end))[0]
    missing = [int(n) for n in all_test if n not in predicted]
    if missing:
        scores = model.score_illicit(store.emb[missing]).tolist()
        for n, s in zip(missing, scores):
            predicted[n] = (s, int(ts[n]), -1, False)

    # ---------- Metrikler ----------
    nodes = sorted(predicted)
    y_true = y[nodes]
    y_score = np.array([predicted[n][0] for n in nodes])
    fresh_flags = np.array([predicted[n][3] for n in nodes])
    quality = compute_metrics(y_true, y_score, thr)
    # İkincil metrik: akış sonunda TÜM düğümleri son gömülerle yeniden skorla
    # (query'lerin her an gelebildiği serving senaryosunun karşılığı)
    y_score_end = model.score_illicit(store.emb[nodes]).numpy()
    quality_end = compute_metrics(y_true, y_score_end, thr)
    per_batch = pd.DataFrame(rows)
    summary = {
        "policy": policy_name, "params": policy.params(),
        "quality": quality,                       # birincil: arrival-time scoring
        "quality_end_of_stream": quality_end,     # ikincil: akış sonu re-scoring
        "fresh_at_arrival_rate": float(fresh_flags.mean()),
        "cost": {
            "total_node_updates": int(store.total_node_updates),
            "total_cpu_refresh_s": float(per_batch.cpu_refresh_s.sum()),
            "wall_ms_p50": float(np.percentile(per_batch.wall_ms, 50)),
            "wall_ms_p99": float(np.percentile(per_batch.wall_ms, 99)),
            "wall_ms_mean": float(per_batch.wall_ms.mean()),
        },
        "n_batches": int(batch_id), "n_predicted": len(nodes),
        "n_fallback": len(missing), "batches_per_ts": args.batches_per_ts,
        "seed": args.seed, "threshold": float(thr),
    }
    per_batch.to_csv(out_dir / "per_batch.csv", index=False)
    pd.DataFrame({"node": nodes, "score": y_score, "label": y_true,
                  "time_step": [predicted[n][1] for n in nodes],
                  "pred_batch": [predicted[n][2] for n in nodes],
                  "fresh_at_arrival": fresh_flags}
                 ).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{policy_name}] arrival AUPRC={quality['auprc']:.4f} F1={quality['f1']:.4f} | "
          f"end AUPRC={quality_end['auprc']:.4f} | "
          f"fresh@arrival={summary['fresh_at_arrival_rate']:.1%} | "
          f"güncelleme={summary['cost']['total_node_updates']} | "
          f"p99={summary['cost']['wall_ms_p99']:.1f} ms -> {out_dir}")
    return summary


def main() -> None:
    args = parse_args()
    d = torch.load(args.data, map_location="cpu", weights_only=True)
    model, _ = GraphSAGEModel.load(args.model)
    init = torch.load(args.initial_state, map_location="cpu", weights_only=True)
    stream_path = args.stream or (Path(args.data).parent / "stream_edges.csv")
    stream_df = pd.read_csv(stream_path)

    names = (["no_refresh", "full_always", "local_always", "local_periodic", "local_adaptive"]
             if args.policy == "all" else [args.policy])
    results = [run_one(n, args, d, model, init, stream_df) for n in names]

    if len(results) > 1:
        print("\n=== KARŞILAŞTIRMA ===")
        for r in results:
            print(f"{r['policy']:<16} AUPRC={r['quality']['auprc']:.4f} fresh={r['fresh_at_arrival_rate']:.0%} "
                  f"F1={r['quality']['f1']:.4f} "
                  f"updates={r['cost']['total_node_updates']:>10} "
                  f"p99={r['cost']['wall_ms_p99']:>8.1f} ms")


if __name__ == "__main__":
    main()
