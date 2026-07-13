"""EmbeddingStore — gömü cache'i + bayatlık (staleness) muhasebesi.

Proposal denklem (2)'nin üç bileşeni burada düğüm başına sayaç olarak tutulur:
  time_since_update(v)     = batch_id - last_update[v]
  degree_change(v)         = v'nin son refresh'inden beri v'ye eklenen kenar sayısı
  neighbor_change_rate(v)  = v'nin son refresh'inden beri gömüsü güncellenen komşu sayısı

Kritik ilke: TAHMİN her zaman bu cache'ten okunur; politika yalnızca cache'in
hangi satırlarının ne zaman tazeleneceğine karar verir.
"""
from __future__ import annotations

import torch


class EmbeddingStore:
    def __init__(self, embeddings: torch.Tensor, start_batch_id: float = 0.0):
        self.emb = embeddings.clone()                                  # [N, d]
        n = embeddings.shape[0]
        self.last_update = torch.full((n,), float(start_batch_id))
        self.degree_change = torch.zeros(n)
        self.neighbor_change = torch.zeros(n)
        self.total_node_updates = 0   # maliyet metriği: toplam güncellenen düğüm sayısı

    # ---------- politikaların okuduğu durum ----------
    def staleness_components(self, nodes: list[int], batch_id: int
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.tensor(nodes, dtype=torch.long)
        tsu = batch_id - self.last_update[idx]
        return tsu, self.degree_change[idx], self.neighbor_change[idx]

    # ---------- akış olayları ----------
    def on_edges_added(self, src, dst) -> None:
        """Yeni kenarlar geldiğinde uç düğümlerin degree_change'i artar."""
        for u, v in zip(map(int, src), map(int, dst)):
            self.degree_change[u] += 1
            self.degree_change[v] += 1

    def update(self, nodes: list[int], new_emb: torch.Tensor,
               batch_id: int, neighbors_of_updated: set[int]) -> None:
        """Refresh: gömüleri yaz, sayaçları sıfırla, komşuları 'dirty' işaretle."""
        idx = torch.tensor(nodes, dtype=torch.long)
        self.emb[idx] = new_emb
        self.last_update[idx] = float(batch_id)
        self.degree_change[idx] = 0.0
        self.neighbor_change[idx] = 0.0
        self.total_node_updates += len(nodes)
        # StreamTGN'deki dirty-flag propagation'ın hafif karşılığı:
        dirty = list(neighbors_of_updated - set(map(int, nodes)))
        if dirty:
            didx = torch.tensor(dirty, dtype=torch.long)
            self.neighbor_change[didx] += 1.0
