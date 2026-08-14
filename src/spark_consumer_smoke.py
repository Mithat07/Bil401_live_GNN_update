"""Lightweight Spark Structured Streaming smoke test that reads Kafka and logs batch sizes.

Does not load the heavy ML stack; just verifies Spark+Kafka connectivity and basic foreachBatch handling.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import IntegerType, StructField, StructType


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="edges")
    p.add_argument("--trigger-sec", type=int, default=2)
    p.add_argument("--idle-timeout-sec", type=int, default=30)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("gnn-smoke-consumer").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([StructField("src", IntegerType()), StructField("dst", IntegerType()), StructField("time_step", IntegerType())])

    reader = (spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", args.bootstrap)
              .option("subscribe", args.topic)
              .option("startingOffsets", "earliest"))
    events = (reader.load().select(from_json(col("value").cast("string"), schema).alias("e")).select("e.*"))

    last_data = {"t": time.time()}

    def handle(df, epoch_id):
        n = df.count()
        if n == 0:
            return
        last_data["t"] = time.time()
        print(f"[smoke] [batch {epoch_id}] rows={n}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query = (events.writeStream.foreachBatch(handle)
             .trigger(processingTime=f"{args.trigger_sec} seconds")
             .option("checkpointLocation", f"{str(out_dir)}/_checkpoint")
             .start())
    try:
        while query.isActive:
            query.awaitTermination(5)
            if time.time() - last_data["t"] > args.idle_timeout_sec:
                print(f"[smoke] {args.idle_timeout_sec}s no data; stopping")
                query.stop()
    finally:
        pass


if __name__ == "__main__":
    main()
