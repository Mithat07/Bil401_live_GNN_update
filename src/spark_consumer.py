"""Spark Structured Streaming consumer — Kafka'dan oku, StreamingEngine'e ver.

Mimari (rapor terimiyle 'Seçenek A + B karışımı'):
  Kafka topic -> Spark readStream -> mikro-batch -> foreachBatch (DRIVER'da
  çalışır) -> engine.process_batch(pandas_df) -> GraphStore/EmbeddingStore/
  Policy/GNN (replay ile birebir aynı sınıflar).

Çalıştırma (Kafka connector JAR'ı için --packages şart; Spark sürümüne göre):
  spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
    src/spark_consumer.py \
      --data processed/data.pt --model runs/sage_v1/model_best.pt \
      --initial-state runs/sage_v1/initial_state.pt \
      --policy local_adaptive --tau 0.6 \
      --bootstrap localhost:9092 --topic edges \
      --trigger-sec 2 --idle-timeout-sec 30 \
      --out-dir runs/kafka/adaptive_t0.6

Notlar:
  * foreachBatch fonksiyonu driver'da koşar; motorun tüm durumu (graf, gömü
    cache) tek süreçte yaşar. Executor'lara durum dağıtılmaz — bkz. rapor
    'tasarım kararları' (mapInPandas'ın neden seçilmediği).
  * Akış Ctrl+C ile veya idle-timeout ile durunca finalize() metrikleri yazar.
  * Batch boyutunu producer hızı + trigger süresi belirler; ayrıca
    maxOffsetsPerTrigger ile üstten sınırlanabilir.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from streaming_engine import StreamingEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--initial-state", required=True)
    p.add_argument("--policy", required=True,
                   choices=["no_refresh", "full_always", "local_always", "local_periodic", "local_adaptive", "all"])
    p.add_argument("--period", type=int, default=5)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--staleness-norm", default="exp", choices=["exp", "pool_max"])
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="edges")
    p.add_argument("--trigger-sec", type=int, default=2)
    p.add_argument("--max-offsets-per-trigger", type=int, default=None)
    p.add_argument("--idle-timeout-sec", type=int, default=30,
                   help="bu süre boyunca yeni batch gelmezse akışı durdur ve finalize et")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    policy_names = (["no_refresh", "full_always", "local_always", "local_periodic", "local_adaptive"]
                    if args.policy == "all" else [args.policy])
    engines = {}
    for name in policy_names:
        out_dir = Path(args.out_dir) / name if args.policy == "all" else Path(args.out_dir)
        engines[name] = StreamingEngine(
            data=args.data, model=args.model, initial_state=args.initial_state,
            policy=name, out_dir=str(out_dir),
            period=args.period, tau=args.tau, alpha=args.alpha,
            beta=args.beta, gamma=args.gamma, norm=args.staleness_norm)

    spark = (SparkSession.builder.appName("gnn-refresh-serving").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([
        StructField("src", IntegerType()), StructField("dst", IntegerType()),
        StructField("time_step", IntegerType()), StructField("event_ts", DoubleType()),
    ])
    reader = (spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", args.bootstrap)
              .option("subscribe", args.topic)
              .option("startingOffsets", "earliest"))
    if args.max_offsets_per_trigger:
        reader = reader.option("maxOffsetsPerTrigger", args.max_offsets_per_trigger)
    events = (reader.load()
              .select(from_json(col("value").cast("string"), schema).alias("e"))
              .select("e.*"))

    last_data = {"t": time.time()}

    def handle(df, epoch_id):
        pdf = df.toPandas()
        if len(pdf) == 0:
            return
        last_data["t"] = time.time()
        for policy_name, engine in engines.items():
            row = engine.process_batch(pdf)
            e2e = f"{row['e2e_ms']:.0f}ms" if row["e2e_ms"] is not None else "n/a"
            print(f"[{policy_name}] [batch {row['batch']}] edges={row['n_edges']} "
                  f"refreshed={row['n_refreshed']} wall={row['wall_ms']:.0f}ms e2e={e2e}")

    query = (events.writeStream.foreachBatch(handle)
             .trigger(processingTime=f"{args.trigger_sec} seconds")
             .option("checkpointLocation", f"{args.out_dir}/_checkpoint")
             .start())
    try:
        while query.isActive:
            query.awaitTermination(5)
            if time.time() - last_data["t"] > args.idle_timeout_sec:
                print(f"[consumer] {args.idle_timeout_sec} sn'dir veri yok; durduruluyor.")
                query.stop()
    finally:
        for engine in engines.values():
            engine.finalize()


if __name__ == "__main__":
    main()
