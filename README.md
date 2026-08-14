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

# 2b) Dağıtık offline eğitim (örnek: 2 GPU)
python -m torch.distributed.run --nproc_per_node=2 src/train_graphsage.py \
    --distributed --device cuda --data processed/data.pt --out-dir runs/sage_v1_ddp

# 3) Streaming başlangıç durumu (faz 2'ye köprü)
python src/export_initial_embeddings.py \
    --data processed/data.pt --model runs/sage_v1/model_best.pt \
    --out runs/sage_v1/initial_state.pt

## CPU tabanlı yerel Docker dağıtımı (2 worker)
Aşağıdaki adımlar, tek GPU yerine CPU üzerinde iki containerlı yerel dağıtık çalışma sağlar.

```bash
# 1) Docker image oluştur
docker compose build

# 2) Kafka ve iki consumer container'ı ayağa kaldır
docker compose up -d

# 3) Consumer loglarını takip et
docker compose logs -f consumer1
# veya
docker compose logs -f consumer2
```

Her consumer aynı Kafka consumer group içinde çalışır; bu sayede topic partition'ları
iki container arasında paylaşılır. Consumer çıktıları `runs/kafka/consumer1` ve
`runs/kafka/consumer2` klasörlerine yazılır.

# 4) Kafka producer başlat
python src/kafka_producer.py --stream processed/stream_edges.csv `
    --bootstrap localhost:9092 --topic edges --events-per-sec 500 --from-ts 35
````
# 5) Çalışmayı durdur ve çıktı kontrol et
docker compose down
```

## Spark cluster + MinIO ile genişletilmiş yerel dağıtım
Aşağıdaki `docker-compose.cluster.yml` daha gerçekçi bir dağıtık test hattı kurar:
- `zookeeper` + `kafka`
- `spark-master`
- `spark-worker1`, `spark-worker2`
- `minio` (opsiyonel paylaşılan depolama)
- `gnn-app` (Spark submit için Python çalışma alanı)

```bash

# 0a) Once hafif smoke image'ini olustur (Torch kurulmaz)
docker build --target spark-base --memory=2g `
  -f Dockerfile.spark-run `
  -t elliptic-spark-smoke:3.5.1 .

# 0b) Smoke testi basarili olduktan sonra CPU-only model runner'ini olustur
docker build --memory=4g --memory-swap=4g `
  -f Dockerfile.spark-run `
  -t elliptic-spark-runner:light .

# 1) Cluster bileşenlerini ayağa kaldır
docker compose -f docker-compose.cluster.yml up -d

# 2) Topic oluşturulur
docker compose -f docker-compose.cluster.yml exec kafka `
  kafka-topics --bootstrap-server kafka:29092 `
  --create --if-not-exists --topic edges `
  --partitions 2 --replication-factor 1

# 3) Consumer job'unu Spark bulunan runner container'inda baslat.
# Dusuk RAM icin politikalari tek tek calistir; POLICY ve OUT_DIR'i her
# calistirmada full_always/local_always/local_periodic/local_adaptive olarak degistir.
$POLICY = "local_adaptive"
$OUT_DIR = "runs/kafka/local_adaptive_t0.6"

docker compose -f docker-compose.cluster.yml run --rm spark-runner `
  /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 `
  --driver-memory 2g `
  --executor-memory 768m `
  --executor-cores 1 `
  --jars /opt/spark/jars/spark-sql-kafka-0-10_2.12-3.5.1.jar,/opt/spark/jars/spark-token-provider-kafka-0-10_2.12-3.5.1.jar,/opt/spark/jars/kafka-clients-3.5.1.jar,/opt/spark/jars/commons-pool2-2.11.1.jar `
  src/spark_consumer.py `
  --data processed/data.pt `
  --model runs/sage_v1/model_best.pt `
  --initial-state runs/sage_v1/initial_state.pt `
  --policy local_adaptive `
  --tau 0.6 `
  --bootstrap kafka:29092 `
  --topic edges `
  --trigger-sec 2 `
  --idle-timeout-sec 60 `
  --out-dir runs/kafka/local_adaptive_fixed

Bu komut Spark tarafındaki consumer job'unu `spark-runner` içinde başlatır.
Mevcut `foreachBatch` + `toPandas()` tasarımında `StreamingEngine` ve model
hesabı driver'da yapılır; Spark worker'lar Kafka/DataFrame görevlerini yürütür.
`--policy all` dört engine'i aynı anda bellekte tuttuğu için düşük RAM'li
sistemlerde yukarıdaki gibi politikaları ayrı ayrı çalıştırın.

# 4) Kafka producer host üzerinde başlatılır
python src/kafka_producer.py `
  --stream processed/stream_edges.csv `
  --bootstrap localhost:9092 `
  --topic edges `
  --events-per-sec 500 `
  --from-ts 35

# 4) Cluster'ı kapat
docker compose -f docker-compose.cluster.yml down
```

Bu yapıdaki `checkpointLocation` ve `out-dir` için MinIO veya paylaşılan HDFS'e geçmek istersen:
- Spark ayarlarını `spark.hadoop.fs.s3a.endpoint` / AWS S3 uyumlu endpoint ile genişlet
- `checkpointLocation` ve model dosyalarını `s3a://...` yoluna taşı
- `minio` ile `http://minio:9000` kullanarak yerel S3 benzeri depolama sağlayabilirsin
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
