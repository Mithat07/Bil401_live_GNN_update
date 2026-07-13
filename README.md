# Elliptic + GraphSAGE — Faz 1: Ön İşleme ve Offline Eğitim

BIL 401 projesi (StreamTGN-esinli adaptif yerel gömü güncelleme) — Üye 2 tarafı,
1. adım. Bu repo parçası üç şey üretir:

1. **İşlenmiş veri** (`data.pt`) — PyG tensörleri + temporal split maskeleri
2. **Replay dosyası** (`stream_edges.csv`) — Kafka producer'ın okuyacağı kronolojik kenar akışı *(takım arkadaşına teslim edilecek arayüz)*
3. **Eğitilmiş model** (`model_best.pt`) — streaming sırasında ağırlıkları sabit kalacak GraphSAGE + başlangıç gömü cache'i (`initial_state.pt`)

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Veri

Kaggle'dan indir: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Üç CSV'yi `data/elliptic/` altına koy:
`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`

## Çalıştırma

```bash
# 1) Ön işleme (≈1-2 dk)
python src/preprocess_elliptic.py --data-dir data/elliptic --out-dir processed

# 2) Eğitim (CPU'da ~dakikalar; full-batch)
python src/train_graphsage.py --data processed/data.pt --out-dir runs/sage_v1

# 3) Streaming başlangıç durumu (faz 2'ye köprü)
python src/export_initial_embeddings.py \
    --data processed/data.pt --model runs/sage_v1/model_best.pt \
    --out runs/sage_v1/initial_state.pt
```

## Tasarım kararları (rapora girecek)

- **Temporal split 34/49** (train ts 1–29, val 30–34, test 35–49): literatür standardı;
  rastgele split temporal leakage yaratırdı. Test dönemi = streaming replay dönemi.
- **`time_step` özellik olarak modele verilmez** (leakage). Standardizasyon yalnızca
  train dönemi istatistikleriyle yapılır; mean/std `data.pt` içinde saklanır.
- **Encoder / head ayrımı** (`model.py`): streaming fazında refresh politikası yalnızca
  encoder'ı (pahalı, L-hop) yeniden çalıştırır; tahmin her zaman cache'teki gömü +
  head (ucuz) ile yapılır. Staleness kavramının teknik karşılığı bu ayrımdır.
- **L=2 katman**: affected-set yarıçapı = modelin reseptif alanı. L büyürse affected-set
  patlar ve Local/Full farkı ölçülemez hale gelir.
- **Sınıf dengesizliği**: CrossEntropyLoss'a train oranından illicit ağırlığı; birincil
  metrik AUPRC (threshold'suz), F1 eşiği val'de seçilir ve test'te sabitlenir.
- **Temporal kenar hijyeni**: her forward pass yalnızca ilgili dönemin kenarlarını görür
  (`utils.edges_up_to`). Elliptic'te kenarlar zaten time step içinde kalır; filtre bunu
  garantiye alır.
- **Test skoru = Full-Always üst sınırı**: offline test değerlendirmesi "her gömü her an
  taze" senaryosudur; faz 2'deki politika karşılaştırması bu üst sınıra göre okunur.

## Faz 2 — Statik replay ile politika deneyi (Kafka'sız uçtan uca)

```bash
# Offline modelin ts bazlı analizi (ts-43 çöküşünü doğrular; rapor figürü)
python src/analyze_offline_by_ts.py --data processed/data.pt --model runs/sage_v1/model_best.pt

# Dört politikayı aynı akışta koş (CPU'da ~5-15 dk; full_always en yavaşı)
python src/replay.py --data processed/data.pt --model runs/sage_v1/model_best.pt \
    --initial-state runs/sage_v1/initial_state.pt --policy all --out-dir runs/replay

# Adaptif politikanın tau taraması (exp normalizasyonla; Pareto'nun adaptif kolu)
for TAU in 0.2 0.3 0.4 0.5 0.6; do
  python src/replay.py --data processed/data.pt --model runs/sage_v1/model_best.pt \
      --initial-state runs/sage_v1/initial_state.pt --policy local_adaptive \
      --tau $TAU --staleness-norm exp --out-dir runs/replay/adaptive_t$TAU
done

# Tablo + Pareto figürü
python src/compare_runs.py --runs runs/replay/* runs/replay/adaptive_t* \
    --out runs/replay/pareto.png
```

## Analiz araçları (mekanizma kanıtı + rapor figürleri)

```bash
# Kenar ablasyonu: komşuluk agregasyonunun değeri (bayat-kazandı hipotezinin testi)
python src/eval_edge_ablation.py --data processed/data.pt --model runs/sage_v1/model_best.pt

# Politika bazlı zaman-içinde-kalite figürü (streaming ana grafiği)
python src/plot_quality_over_time.py --runs runs/replay/* runs/replay/adaptive_t* \
    --out runs/replay/quality_over_time.png
```

`summary.json` alanları: `quality` (birincil, arrival-time scoring — düğüm ilk
göründüğü anda skorlanır), `quality_end_of_stream` (ikincil — akış sonunda son
gömülerle yeniden skorlama), `fresh_at_arrival_rate` (skorlama anında gömüsü
taze olan düğüm oranı; politikalar arasındaki kalite farkının mekanik açıklaması).

Sanity check: `local_always` AUPRC'si `full_always` ile (hemen hemen) aynı olmalı —
L-hop refresh kayıpsızdır; fark yalnızca maliyet sütunlarında görülmelidir.
Kalite farkı periodic/adaptive'in *ertelediği* güncellemelerden doğar (RQ2/RQ3).

## Takım arkadaşıyla arayüz

- Kafka producer `processed/stream_edges.csv`'yi okur: `src,dst,time_step`
  (dahili index'ler; `node_id_map.csv` ile orijinal txId'ye çevrilebilir).
- Spark `foreachBatch` tarafı faz 3'te `replay.py`'deki batch döngüsünün
  gövdesini aynen çağırır: `GraphStore` + `EmbeddingStore` + `Policy` sınıfları
  Spark'a taşınırken değişmez; değişen tek şey batch'in nereden geldiğidir.

## Beklenen sonuç aralığı

Elliptic + GraphSAGE literatürde test AUPRC ~0.4–0.6, illicit F1 ~0.6–0.8 bandında
seyreder (split ve seed'e duyarlı). Bunun belirgin altındaysa önce sınıf ağırlığını ve
erken durdurmayı kontrol et.
