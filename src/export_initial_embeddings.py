"""2. faza köprü: streaming başlamadan önceki gömü cache'ini üretir.

Streaming deneyi test dönemini (ts > val_end) replay eder. Replay başladığı anda
EmbeddingStore'da "o ana kadarki grafla" (ts <= val_end) hesaplanmış gömüler
bulunmalıdır. Bu script tam olarak o başlangıç durumunu üretir:

  initial_state.pt:
    embeddings      [num_nodes, emb_dim]  — ts<=val_end kenarlarıyla encoder çıktısı
    last_update     [num_nodes]           — hepsi val_end (batch-id cinsinden başlangıç)
    threshold       eğitimde val'de seçilen karar eşiği

Kullanım:
  python src/export_initial_embeddings.py \
      --data processed/data.pt --model runs/sage_v1/model_best.pt \
      --out runs/sage_v1/initial_state.pt
"""
from __future__ import annotations

import argparse

import torch

from model import GraphSAGEModel
from utils import edges_up_to


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = torch.load(args.data, map_location="cpu", weights_only=True)
    model, extra = GraphSAGEModel.load(args.model)
    val_end = extra["val_end"]

    e_init = edges_up_to(d["edge_index"], d["time_step"], val_end)
    emb = model.embed(d["x"], e_init)

    torch.save({
        "embeddings": emb,
        "last_update": torch.full((emb.shape[0],), float(val_end)),
        "threshold": extra["threshold"],
        "val_end": val_end,
    }, args.out)
    print(f"[tamam] {args.out}  embeddings={tuple(emb.shape)}  threshold={extra['threshold']:.3f}")


if __name__ == "__main__":
    main()
