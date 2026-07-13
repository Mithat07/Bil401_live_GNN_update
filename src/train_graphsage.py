"""GraphSAGE offline eğitimi (streaming sırasında ağırlıklar SABİT kalacak).

Deney tasarımı gereksinimleri:
  * Temporal hijyen: train forward pass yalnızca ts<=train_end kenarlarını,
    val değerlendirmesi ts<=val_end kenarlarını görür. (Elliptic'te kenarlar
    zaten time step içinde kaldığından bu filtre kesin sınır çizer.)
  * Dengesiz sınıf: CrossEntropyLoss'a train etiket oranından sınıf ağırlığı.
  * Model seçimi: val AUPRC üzerinde early stopping.
  * Karar eşiği: val üzerinde F1'i maksimize eden threshold — test'te sabit.

Çıktılar (out_dir):
  model_best.pt   encoder+head ağırlıkları + hiperparametreler + threshold
  metrics.json    val/test AUPRC, F1, precision, recall
  train_log.csv   epoch bazlı loss ve val metrikleri

Kullanım:
  python src/train_graphsage.py --data processed/data.pt --out-dir runs/sage_v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import GraphSAGEModel
from utils import best_f1_threshold, compute_metrics, edges_up_to, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="preprocess çıktısı data.pt yolu")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2, help="L; affected-set yarıçapıyla aynı olmalı")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=30, help="val AUPRC iyileşmezse erken durdurma")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def illicit_scores(model: GraphSAGEModel, x, edge_index) -> torch.Tensor:
    model.eval()
    return F.softmax(model(x, edge_index), dim=-1)[:, 1]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    # ---------- Veri ----------
    d = torch.load(args.data, map_location="cpu", weights_only=True)
    x, edge_index, y, ts = d["x"].to(dev), d["edge_index"].to(dev), d["y"].to(dev), d["time_step"].to(dev)
    train_mask, val_mask, test_mask = (d[k].to(dev) for k in ("train_mask", "val_mask", "test_mask"))
    meta = d["meta"]
    train_end, val_end, max_ts = meta["train_end"], meta["val_end"], meta["max_ts"]

    # Temporal kenar kümeleri (bir kez hesapla)
    e_train = edges_up_to(edge_index, ts, train_end)
    e_val = edges_up_to(edge_index, ts, val_end)
    e_test = edge_index  # tam graf = ts<=max_ts

    # ---------- Sınıf ağırlığı (yalnızca train etiketlerinden) ----------
    y_tr = y[train_mask]
    n_licit, n_illicit = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    w = torch.tensor([1.0, n_licit / max(n_illicit, 1)], dtype=torch.float32, device=dev)
    print(f"[veri] train {len(y_tr)} etiketli (licit={n_licit}, illicit={n_illicit}) "
          f"-> sınıf ağırlığı illicit x{w[1]:.1f}")

    # ---------- Model ----------
    model = GraphSAGEModel(in_dim=x.shape[1], hidden_dim=args.hidden_dim,
                           emb_dim=args.emb_dim, num_layers=args.num_layers,
                           dropout=args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------- Eğitim döngüsü ----------
    best_val_auprc, best_state, best_epoch = -1.0, None, -1
    log_rows = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, e_train)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=w)
        loss.backward()
        opt.step()

        # Val değerlendirmesi
        scores_val = illicit_scores(model, x, e_val)[val_mask].cpu().numpy()
        yv = y[val_mask].cpu().numpy()
        val_auprc = compute_metrics(yv, scores_val, 0.5)["auprc"]
        loss_val = loss.detach().item()
        log_rows.append({"epoch": epoch, "loss": loss_val, "val_auprc": val_auprc})

        if val_auprc > best_val_auprc:
            best_val_auprc, best_epoch = val_auprc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss={loss_val:.4f}  val_AUPRC={val_auprc:.4f}  "
                  f"(best {best_val_auprc:.4f} @ {best_epoch})")
        if epoch - best_epoch >= args.patience:
            print(f"[erken durdurma] {args.patience} epoch'tur iyileşme yok.")
            break
    print(f"[eğitim] {time.time() - t0:.1f} sn, en iyi epoch {best_epoch}")

    # ---------- En iyi modeli geri yükle, eşiği val'de seç ----------
    model.load_state_dict(best_state)
    scores_val = illicit_scores(model, x, e_val)[val_mask].cpu().numpy()
    yv = y[val_mask].cpu().numpy()
    thr, f1_val = best_f1_threshold(yv, scores_val)
    val_metrics = compute_metrics(yv, scores_val, thr)
    print(f"[val] AUPRC={val_metrics['auprc']:.4f}  F1@{thr:.3f}={val_metrics['f1']:.4f}")

    # ---------- Test (streaming döneminin offline üst sınırı: her gömü taze) ----------
    scores_test = illicit_scores(model, x, e_test)[test_mask].cpu().numpy()
    yt = y[test_mask].cpu().numpy()
    test_metrics = compute_metrics(yt, scores_test, thr)
    print(f"[test] AUPRC={test_metrics['auprc']:.4f}  F1={test_metrics['f1']:.4f}  "
          f"P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}")
    print("  (Bu test skoru = 2. fazdaki Full-Always politikasının hedef üst sınırı.)")

    # ---------- Kaydet ----------
    model.save(out_dir / "model_best.pt",
               extra={"threshold": thr, "best_epoch": best_epoch,
                      "train_end": train_end, "val_end": val_end, "max_ts": max_ts,
                      "seed": args.seed})
    (out_dir / "metrics.json").write_text(json.dumps(
        {"val": val_metrics, "test_offline_upper_bound": test_metrics,
         "class_weight_illicit": float(w[1]), "args": vars(args) | {"device": str(dev)}},
        indent=2))
    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    print(f"[tamam] {out_dir}/model_best.pt, metrics.json, train_log.csv")


if __name__ == "__main__":
    main()
