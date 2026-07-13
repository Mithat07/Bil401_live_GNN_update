"""GraphSAGE modeli — encoder ve classifier head BİLEREK ayrılmıştır.

Streaming mimarisinin temeli bu ayrım:
  * encoder.embed(x, edge_index) -> 128-d düğüm gömüsü  (maliyetli, L-hop komşuluk okur)
  * head(embedding)              -> illicit skoru        (ucuz, tek matris çarpımı)

2. fazda EmbeddingStore encoder çıktısını cache'ler; refresh politikası yalnızca
encoder'ın hangi düğümler için yeniden çalıştırılacağına karar verir. Head her
tahminde cache'teki gömü üzerinde çalışır.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SageEncoder(nn.Module):
    """L katmanlı GraphSAGE encoder. L = reseptif alan = affected-set yarıçapı."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, emb_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [emb_dim]
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1], aggr="mean") for i in range(num_layers)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # [num_nodes, emb_dim]


class ClassifierHead(nn.Module):
    """Gömüden 2 sınıflı logit üretir (illicit=1, licit=0)."""

    def __init__(self, emb_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.lin = nn.Linear(emb_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.lin(F.dropout(F.relu(emb), p=self.dropout, training=self.training))


class GraphSAGEModel(nn.Module):
    """Encoder + head sarmalayıcı; offline eğitimde uçtan uca kullanılır."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, emb_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hparams = dict(in_dim=in_dim, hidden_dim=hidden_dim, emb_dim=emb_dim,
                            num_layers=num_layers, dropout=dropout)
        self.encoder = SageEncoder(in_dim, hidden_dim, emb_dim, num_layers, dropout)
        self.head = ClassifierHead(emb_dim, 2, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, edge_index))

    @torch.no_grad()
    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Streaming tarafında refresh anında çağrılacak fonksiyon."""
        self.eval()
        return self.encoder(x, edge_index)

    @torch.no_grad()
    def score_illicit(self, emb: torch.Tensor) -> torch.Tensor:
        """Cache'teki gömüden illicit olasılığı (softmax[:,1])."""
        self.eval()
        return F.softmax(self.head(emb), dim=-1)[:, 1]

    # ---------- kalıcılık ----------
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save({"hparams": self.hparams,
                    "state_dict": self.state_dict(),
                    "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> tuple["GraphSAGEModel", dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt.get("extra", {})
