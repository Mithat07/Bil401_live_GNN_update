"""StreamingEngine — replay döngüsünün Spark foreachBatch'e takılabilir hali.

Replay.py'deki batch gövdesinin sınıf haline getirilmiş, kaynaktan bağımsız
versiyonu: batch'in pandas DataFrame olarak NEREDEN geldiği (statik replay,
Kafka+Spark, test kodu) motoru ilgilendirmez.

Kullanım (Spark tarafı, bkz. spark_consumer.py):
    engine = StreamingEngine(data=..., model=..., initial_state=...,
                             policy="local_adaptive", out_dir=..., tau=0.6)
    def handle(df, epoch_id):
        pdf = df.toPandas()          # kolonlar: src, dst[, event_ts]
        if len(pdf):
            engine.process_batch(pdf)
    ...
    engine.finalize()                # akış durunca metrik/dosyaları yazar

event_ts kolonu (producer'ın epoch-saniye damgası) varsa uçtan uca gecikme
ölçülür: e2e_ms = (şimdi - batch'teki en eski event_ts). Kafka+Spark kuyruk
beklemesini de içerdiği için RQ1'in GERÇEK gecikme ayağı budur; statik
replay'deki wall_ms yalnızca işleme süresidir.
"""
from __future__ import annotations

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
from utils import compute_metrics, edges_up_to


class StreamingEngine:
    def __init__(self, data: str, model: str, initial_state: str, policy: str,
                 out_dir: str, **policy_kw):
        d = torch.load(data, map_location="cpu", weights_only=True)
        self.model, _ = GraphSAGEModel.load(model)
        init = torch.load(initial_state, map_location="cpu", weights_only=True)

        self.x = d["x"]
        self.y = d["y"].numpy()
        self.ts = d["time_step"].numpy()
        self.val_end = init["val_end"]
        self.threshold = float(init["threshold"])
        self.L = self.model.hparams["num_layers"]

        self.graph = GraphStore(num_nodes=self.x.shape[0])
        e_init = edges_up_to(d["edge_index"], d["time_step"], self.val_end)
        self.graph.add_edges(e_init[0].tolist(), e_init[1].tolist())
        self.store = EmbeddingStore(init["embeddings"], start_batch_id=0.0)
        self.policy = make_policy(policy, **policy_kw)
        self.policy_name = policy

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.predicted: dict[int, tuple[float, int, int, bool]] = {}
        self.rows: list[dict] = []
        self.batch_id = 0

    # ------------------------------------------------------------------
    def process_batch(self, pdf: pd.DataFrame, batch_id: int | None = None) -> dict:
        """Bir mikro-batch işle. pdf kolonları: src, dst, opsiyonel event_ts."""
        self.batch_id = self.batch_id + 1 if batch_id is None else int(batch_id) + 1
        bid = self.batch_id
        t_wall = time.perf_counter()
        src = pdf["src"].to_numpy(dtype=np.int64)
        dst = pdf["dst"].to_numpy(dtype=np.int64)

        # 1-2) graf + affected set
        self.graph.add_edges(src, dst)
        self.store.on_edges_added(src, dst)
        endpoints = set(map(int, src)) | set(map(int, dst))
        affected = self.graph.affected_set(endpoints, self.L)

        # 3) politika kararı
        decision = self.policy.decide(affected, self.store, bid)

        # 4) refresh
        t_cpu = time.process_time()
        if decision == FULL_GRAPH:
            self.store.emb = self.model.embed(self.x, self.graph.edge_index_full())
            self.store.last_update.fill_(float(bid))
            self.store.degree_change.zero_()
            self.store.neighbor_change.zero_()
            self.store.total_node_updates += self.graph.num_nodes
            n_refresh = self.graph.num_nodes
        elif decision:
            targets = sorted(decision)
            sub_nodes, e_sub, target_pos = self.graph.subgraph(targets, self.L)
            emb_sub = self.model.embed(self.x[sub_nodes], e_sub)
            neigh: set[int] = set()
            for u in targets:
                neigh |= self.graph.neighbors(u)
            self.store.update(targets, emb_sub[target_pos], bid, neigh)
            n_refresh = len(targets)
        else:
            n_refresh = 0
        cpu_refresh = time.process_time() - t_cpu

        # 5) tahmin: ilk kez görünen etiketli test düğümleri
        new_nodes = [n for n in endpoints
                     if self.y[n] >= 0 and self.ts[n] > self.val_end
                     and n not in self.predicted]
        if new_nodes:
            scores = self.model.score_illicit(self.store.emb[new_nodes]).tolist()
            for n, s in zip(new_nodes, scores):
                fresh = bool(self.store.last_update[n].item() == float(bid))
                self.predicted[n] = (s, int(self.ts[n]), bid, fresh)

        # Uçtan uca gecikme (Kafka producer damgası varsa)
        e2e_ms = None
        if "event_ts" in pdf.columns and len(pdf):
            e2e_ms = (time.time() - float(pdf["event_ts"].min())) * 1e3

        row = {"batch": bid, "n_edges": len(pdf), "n_affected": len(affected),
               "n_refreshed": n_refresh, "cpu_refresh_s": cpu_refresh,
               "wall_ms": (time.perf_counter() - t_wall) * 1e3,
               "e2e_ms": e2e_ms, "n_new_preds": len(new_nodes)}
        self.rows.append(row)
        return row

    # ------------------------------------------------------------------
    def finalize(self) -> dict:
        """Akış bitince metrikleri hesapla ve dosyaları yaz (replay ile aynı format)."""
        # Akışta hiç görünmeyen etiketli test düğümleri (fallback)
        all_test = np.where((self.y >= 0) & (self.ts > self.val_end))[0]
        missing = [int(n) for n in all_test if n not in self.predicted]
        if missing:
            scores = self.model.score_illicit(self.store.emb[missing]).tolist()
            for n, s in zip(missing, scores):
                self.predicted[n] = (s, int(self.ts[n]), -1, False)

        nodes = sorted(self.predicted)
        y_true = self.y[nodes]
        y_score = np.array([self.predicted[n][0] for n in nodes])
        fresh = np.array([self.predicted[n][3] for n in nodes])
        quality = compute_metrics(y_true, y_score, self.threshold)
        y_end = self.model.score_illicit(self.store.emb[nodes]).numpy()
        quality_end = compute_metrics(y_true, y_end, self.threshold)

        pb = pd.DataFrame(self.rows)
        e2e = pb["e2e_ms"].dropna() if "e2e_ms" in pb else pd.Series(dtype=float)
        summary = {
            "policy": self.policy_name, "params": self.policy.params(),
            "quality": quality, "quality_end_of_stream": quality_end,
            "fresh_at_arrival_rate": float(fresh.mean()) if len(fresh) else None,
            "cost": {
                "total_node_updates": int(self.store.total_node_updates),
                "total_cpu_refresh_s": float(pb.cpu_refresh_s.sum()) if len(pb) else 0.0,
                "wall_ms_p50": float(np.percentile(pb.wall_ms, 50)) if len(pb) else None,
                "wall_ms_p99": float(np.percentile(pb.wall_ms, 99)) if len(pb) else None,
                "e2e_ms_p50": float(np.percentile(e2e, 50)) if len(e2e) else None,
                "e2e_ms_p99": float(np.percentile(e2e, 99)) if len(e2e) else None,
            },
            "n_batches": int(self.batch_id), "n_predicted": len(nodes),
            "n_fallback": len(missing), "threshold": self.threshold,
        }
        pb.to_csv(self.out_dir / "per_batch.csv", index=False)
        pd.DataFrame({"node": nodes, "score": y_score, "label": y_true,
                      "time_step": [self.predicted[n][1] for n in nodes],
                      "pred_batch": [self.predicted[n][2] for n in nodes],
                      "fresh_at_arrival": fresh}
                     ).to_csv(self.out_dir / "predictions.csv", index=False)
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[engine] arrival AUPRC={quality['auprc']:.4f} "
              f"end AUPRC={quality_end['auprc']:.4f} "
              f"güncelleme={summary['cost']['total_node_updates']} -> {self.out_dir}")
        return summary
