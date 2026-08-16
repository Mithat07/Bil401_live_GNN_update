#!/usr/bin/env bash
# Çoklu seed deneyi — varyans eğitimden gelir, replay'den değil.
#
# ÖNEMLİ: replay.py --seed DEĞİŞTİRMEK HİÇBİR ŞEYİ DEĞİŞTİRMEZ.
#   replay tamamen deterministik: model eval modunda (dropout kapalı),
#   batch bölme np.array_split ile sabit, hiçbir örnekleme yok.
#   Rastgelelik yalnızca train_graphsage.py'de: ağırlık ilklendirme + dropout.
#   Bu yüzden her seed için MODELİ YENİDEN EĞİTİYORUZ.
#
# Kullanım:  bash run_multiseed.sh            (varsayılan 5 seed)
#            SEEDS="0 1 2" bash run_multiseed.sh   (3 seed, daha hızlı)
#
# Süre: seed başına ~8-10 dk (eğitim ~1.5 dk + ablasyon + 5 politika + 4 tau)

set -euo pipefail

SEEDS="${SEEDS:-0 1 2 3 4}"
# NOT: tau=0.50'yi buraya KOYMAYIN -- "--policy all" zaten varsayilan tau=0.5 ile
# local_adaptive'i kosuyor. Tekrar koyarsaniz ayni kosu iki kez sayilir.
TAUS="${TAUS:-0.05 0.15 0.25 0.35}"
DATA="processed/data.pt"
ROOT="runs/multiseed"

mkdir -p "$ROOT"
echo "=== Çoklu seed deneyi: SEEDS=[$SEEDS] TAUS=[$TAUS] ==="
date

for S in $SEEDS; do
  D="$ROOT/seed$S"
  echo ""
  echo "################  SEED $S  ################"
  mkdir -p "$D"

  echo "--- [1/4] eğitim (seed $S) ---"
  python src/train_graphsage.py --data "$DATA" --out-dir "$D" --seed "$S"

  echo "--- [2/4] başlangıç embedding'leri ---"
  python src/export_initial_embeddings.py --data "$DATA" \
      --model "$D/model_best.pt" --out "$D/initial_state.pt"

  echo "--- [3/4] kenar ablasyonu ---"
  python src/eval_edge_ablation.py --data "$DATA" --model "$D/model_best.pt"

  echo "--- [4/4] beş politika + tau taraması ---"
  python src/replay.py --data "$DATA" --model "$D/model_best.pt" \
      --initial-state "$D/initial_state.pt" --policy all --out-dir "$D/replay"

  for T in $TAUS; do
    python src/replay.py --data "$DATA" --model "$D/model_best.pt" \
        --initial-state "$D/initial_state.pt" --policy local_adaptive \
        --tau "$T" --staleness-norm exp --out-dir "$D/replay/adaptive_t$T"
  done
done

echo ""
echo "=== Toplama ve figür ==="
python src/aggregate_seeds.py --root "$ROOT" --out-dir figures
date
echo "[tamam] figures/multiseed_summary.csv ve figures/freshness_multiseed.pdf"
