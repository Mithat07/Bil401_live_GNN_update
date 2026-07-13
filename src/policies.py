"""Refresh politikaları — deneyin bağımsız değişkeni.

Dört politika tek arayüzün arkasında (proposal Tablo 1):
  FullAlways     her batch, TÜM düğümler                (en pahalı baseline)
  LocalAlways    her batch, affected L-hop kümesi       (yerellik etkisi)
  LocalPeriodic  her N batch, biriken affected kümesi   (ucuz baseline)
  LocalAdaptive  stale(v) > tau olan affected düğümler  (önerilen politika)

decide(affected, store, batch_id) -> set[node] | FULL_GRAPH
Aynı akış + aynı model + aynı seed altında yalnızca decide değişir; ölçülen
her fark politikaya atfedilir.
"""
from __future__ import annotations

import torch

from embedding_store import EmbeddingStore

FULL_GRAPH = "FULL_GRAPH"  # sentinel: tüm graf yeniden hesaplanacak


class Policy:
    name = "base"

    def decide(self, affected: set[int], store: EmbeddingStore, batch_id: int):
        raise NotImplementedError

    def params(self) -> dict:
        return {}


class FullAlways(Policy):
    name = "full_always"

    def decide(self, affected, store, batch_id):
        return FULL_GRAPH


class LocalAlways(Policy):
    name = "local_always"

    def decide(self, affected, store, batch_id):
        return set(affected)


class LocalPeriodic(Policy):
    """Her N batch'te bir, aradaki batch'lerde biriken affected kümesini tazeler.

    Biriktirme şart: tetik anında yalnızca o batch'in affected'ı tazelense,
    aradaki batch'lerden etkilenen düğümler sonsuza dek bayat kalırdı.
    """
    name = "local_periodic"

    def __init__(self, period: int = 5):
        self.period = period
        self.pending: set[int] = set()

    def decide(self, affected, store, batch_id):
        self.pending |= affected
        if batch_id % self.period == 0:
            out, self.pending = self.pending, set()
            return out
        return set()

    def params(self):
        return {"period": self.period}


class LocalAdaptive(Policy):
    """stale(v) = a*n(time_since_update) + b*n(degree_change) + g*n(neighbor_change)

    Aday havuzu = henüz tazelenmemiş tüm affected düğümler (batch'ler arası taşınır).
    Her batch havuz üzerinde skor hesaplanır; stale(v) > tau olanlar tazelenir,
    kalanlar havuzda bekler.

    İki normalizasyon modu:
      pool_max : bileşenler havuz içi max'a bölünür (göreli; tau'nun anlamı havuz
                 içeriğine bağlı -> tau taramasında keskin uçurumlar yaratabilir)
      exp      : n(t) = 1 - exp(-t/scale) (mutlak ve sınırlı; tau'nun anlamı
                 sabit -> pürüzsüz tau taraması, ÖNERİLEN)
    """
    name = "local_adaptive"

    def __init__(self, tau: float = 0.5, alpha: float = 0.5,
                 beta: float = 0.25, gamma: float = 0.25, norm: str = "exp",
                 tsu_scale: float = 20.0, deg_scale: float = 5.0, nch_scale: float = 5.0):
        self.tau, self.alpha, self.beta, self.gamma = tau, alpha, beta, gamma
        self.norm = norm
        self.tsu_scale, self.deg_scale, self.nch_scale = tsu_scale, deg_scale, nch_scale
        self.pool: set[int] = set()

    def decide(self, affected, store, batch_id):
        self.pool |= affected
        if not self.pool:
            return set()
        nodes = sorted(self.pool)
        tsu, deg, nch = store.staleness_components(nodes, batch_id)

        if self.norm == "exp":
            n_tsu = 1.0 - torch.exp(-tsu / self.tsu_scale)
            n_deg = 1.0 - torch.exp(-deg / self.deg_scale)
            n_nch = 1.0 - torch.exp(-nch / self.nch_scale)
        else:  # pool_max
            def _pm(t: torch.Tensor) -> torch.Tensor:
                m = float(t.max())
                return t / m if m > 0 else t
            n_tsu, n_deg, n_nch = _pm(tsu), _pm(deg), _pm(nch)

        score = self.alpha * n_tsu + self.beta * n_deg + self.gamma * n_nch
        chosen = {n for n, s in zip(nodes, score.tolist()) if s > self.tau}
        self.pool -= chosen
        return chosen

    def params(self):
        return {"tau": self.tau, "alpha": self.alpha, "beta": self.beta,
                "gamma": self.gamma, "norm": self.norm}


def make_policy(name: str, **kw) -> Policy:
    table = {
        "full_always": lambda: FullAlways(),
        "local_always": lambda: LocalAlways(),
        "local_periodic": lambda: LocalPeriodic(period=kw.get("period", 5)),
        "local_adaptive": lambda: LocalAdaptive(
            tau=kw.get("tau", 0.5), alpha=kw.get("alpha", 0.5),
            beta=kw.get("beta", 0.25), gamma=kw.get("gamma", 0.25),
            norm=kw.get("norm", "exp")),
    }
    if name not in table:
        raise ValueError(f"bilinmeyen politika: {name}; seçenekler: {list(table)}")
    return table[name]()
