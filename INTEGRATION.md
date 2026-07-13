# INTEGRATION.md — Kafka–Spark hattı ile model tarafının arayüzü

Bu belge iki tarafın (Üye 1: Kafka/Spark/Docker; Üye 2: model/politika/metrik)
buluştuğu sözleşmeyi tanımlar.

## Mesaj sözleşmesi (Kafka topic: `edges`)

JSON, olay başına bir mesaj; key = `src` (partition dengesi için):

```json
{"src": 12345, "dst": 67890, "time_step": 35, "event_ts": 1721390400.123}
```

- `src`, `dst`: `node_id_map.csv`'deki DAHİLİ index'ler (int)
- `time_step`: 35–49 (test dönemi)
- `event_ts`: producer'ın gönderim anı (epoch saniye, float) — uçtan uca
  gecikme bundan ölçülür

Kaynak dosya: `processed/stream_edges.csv` (ön işleme çıktısı). Üye 1 kendi
producer'ını yazabilir ya da `src/kafka_producer.py`'yi kullanabilir
(`--events-per-sec` ile throughput deneyleri).

## Çalıştırma sırası

```bash
# 1) Kafka ayakta (Üye 1: docker compose)
# 2) Consumer'ı başlat (motor + politika; Üye 2 tarafı)
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
  src/spark_consumer.py --data processed/data.pt \
  --model runs/sage_v1/model_best.pt --initial-state runs/sage_v1/initial_state.pt \
  --policy local_adaptive --tau 0.6 --bootstrap localhost:9092 --topic edges \
  --trigger-sec 2 --out-dir runs/kafka/adaptive_t0.6
# 3) Producer'ı başlat
python src/kafka_producer.py --stream processed/stream_edges.csv \
  --bootstrap localhost:9092 --topic edges --events-per-sec 500 --from-ts 35
# 4) Akış bitince consumer idle-timeout ile durur ve summary.json yazar
```

Her politika/parametre için topic'i temiz başlatın (yeni topic adı veya
`--checkpointLocation` klasörünü silip offsets=earliest) — dört politika AYNI
akışı baştan görmeli.

## Sorumluluk sınırı

- **Üye 1:** Kafka/ZooKeeper (veya KRaft) docker compose, topic yönetimi,
  producer'ın çalıştırılması, istenirse Spark'ın dağıtık (Hadoop/YARN) kurulumu.
  `spark_consumer.py` içindeki okuma kısmı (readStream/kafka options) onun alanı;
  değiştirebilir.
- **Üye 2:** `foreachBatch` içindeki her şey — `StreamingEngine` ve altındaki
  `GraphStore`/`EmbeddingStore`/`policies`/model. Bu sınıflar statik replay'de
  doğrulandı; Spark'a taşınırken DEĞİŞMEZ.

## Hangi metrik nereden gelir

| Metrik | Statik replay | Kafka+Spark |
|---|---|---|
| AUPRC/F1, fresh@arrival, güncelleme sayısı, CPU | ✔ (nihai değerler) | ✔ (aynı çıkmalı — doğrulama) |
| wall_ms (işleme süresi) | ✔ | ✔ |
| **e2e p50/p99 gecikme** (olay→tahmin) | ✘ | ✔ `summary.json: cost.e2e_ms_p50/p99` |
| **maks. sürdürülebilir throughput** | ✘ | ✔ producer hızını artırarak (500→1000→2000/sn) e2e p99'un patladığı nokta |

Kalite metrikleri Kafka'da statik replay ile (batch sınırları farklı olacağından
küçük sapmayla) tutarlı çıkmalı — çıkmıyorsa entegrasyon hatası var demektir;
önce bunu doğrulayın, throughput deneylerine sonra geçin.
