"""GraphStore — akış sırasında büyüyen grafın durumu.

Sorumluluklar:
  * add_edges: mikro-batch kenarlarını komşuluk yapısına ekle (undirected, dedupe)
  * affected_set: endpoints(E_t) ∪ N_L(endpoints)  (proposal denklem 1)
  * subgraph: refresh edilecek kümenin L-hop indüklenmiş alt grafı
    (L-hop komşuluğun indüklenmiş alt grafı, hedef kümenin L katmanlı
    GraphSAGE çıktısını TAM grafla birebir aynı verir — yaklaşım yoktur;
    mesafesi < L olan her düğümün tüm komşuları alt grafın içindedir.)
  * edge_index_full: Full-Always politikası için mevcut tam kenar tensörü
    (artımlı cache; her batch'te sıfırdan kurulmaz)

Not: Bu sınıf 2. fazda statik replay, 3. fazda Spark foreachBatch tarafından
aynen kullanılır — Spark'a taşınırken değişmez.
"""
from __future__ import annotations

import torch


class GraphStore:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.adj: list[set[int]] = [set() for _ in range(num_nodes)]
        self._pair_buffer: list[tuple[int, int]] = []  # yeni benzersiz undirected çiftler
        self._edge_cache: torch.Tensor | None = None   # [2, E*2] her iki yön

    # ---------- güncelleme ----------
    def add_edges(self, src, dst) -> int:
        """Kenarları ekle; kaç YENİ benzersiz undirected kenar eklendiğini döndür."""
        new = 0
        for u, v in zip(map(int, src), map(int, dst)):
            if u == v or v in self.adj[u]:
                continue
            self.adj[u].add(v)
            self.adj[v].add(u)
            self._pair_buffer.append((u, v))
            new += 1
        return new

    # ---------- sorgular ----------
    def neighbors(self, node: int) -> set[int]:
        return self.adj[node]

    def affected_set(self, seeds, L: int) -> set[int]:
        """seeds ∪ L-hop komşuluk (StreamTGN-esinli affected set)."""
        visited = set(map(int, seeds))
        frontier = set(visited)
        for _ in range(L):
            nxt = set()
            for u in frontier:
                nxt |= self.adj[u]
            nxt -= visited
            if not nxt:
                break
            visited |= nxt
            frontier = nxt
        return visited

    def subgraph(self, targets: list[int], L: int
                 ) -> tuple[list[int], torch.Tensor, list[int]]:
        """targets'ın L katmanlı TAM doğrulukta gömüsü için gereken alt graf.

        targets LİSTE olarak verilir; target_pos aynı sırayı korur (gömüler
        çağıran tarafta bu sırayla yazılır — sıra bozulursa cache bozulur).

        Returns:
          sub_nodes    alt graf düğümleri (global id, sıralı liste)
          edge_index   yerel indekslerle [2, e] kenar tensörü
          target_pos   targets'ın sub_nodes içindeki pozisyonları (aynı sırayla)
        """
        sub = self.affected_set(targets, L)          # L-hop kapsama
        sub_nodes = sorted(sub)
        local = {g: i for i, g in enumerate(sub_nodes)}
        rows, cols = [], []
        for u in sub_nodes:
            lu = local[u]
            for v in self.adj[u]:
                if v in sub:                          # indüklenmiş alt graf
                    rows.append(lu)
                    cols.append(local[v])
        edge_index = (torch.tensor([rows, cols], dtype=torch.long)
                      if rows else torch.empty((2, 0), dtype=torch.long))
        target_pos = [local[int(t)] for t in targets]
        return sub_nodes, edge_index, target_pos

    def edge_index_full(self) -> torch.Tensor:
        """Mevcut tam graf kenar tensörü (undirected -> iki yön), artımlı cache."""
        if self._pair_buffer:
            pairs = torch.tensor(self._pair_buffer, dtype=torch.long).t()  # [2, k]
            both = torch.cat([pairs, pairs.flip(0)], dim=1)
            self._edge_cache = (both if self._edge_cache is None
                                else torch.cat([self._edge_cache, both], dim=1))
            self._pair_buffer = []
        if self._edge_cache is None:
            return torch.empty((2, 0), dtype=torch.long)
        return self._edge_cache
