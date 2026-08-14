"""CPU-only Kafka stream worker for local distributed run.

Bu çalışma şekli, Kafka topic'ini bir consumer group ile iki worker'a
bölerek yerel bir dağıtık pipeline sağlar. Her worker kendi partition'larını
okur ve batch'ler halinde `StreamingEngine`'e gönderir.

Not: Bu script, mevcut kodun state paylaşımı olmadan sadece Kafka tabanlı
parallel tüketimi sağlar. Local CPU üzerinde dağıtık çalıştırmak için uygun.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer

from streaming_engine import StreamingEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="processed/data.pt yolu")
    p.add_argument("--model", required=True, help="model_best.pt yolu")
    p.add_argument("--initial-state", required=True, help="initial_state.pt yolu")
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="edges")
    p.add_argument("--group-id", default="gnn-refresh")
    p.add_argument("--batch-size", type=int, default=2000,
                   help="Kafka'dan kaç olay biriktirilince işleneceği")
    p.add_argument("--batch-timeout", type=float, default=1.0,
                   help="Batch işleme için maksimum bekleme süresi (s)")
    p.add_argument("--idle-timeout", type=int, default=30,
                   help="Yeni veri gelmezse kaç saniye sonra sonlandırılacağı")
    p.add_argument("--out-dir", required=True,
                   help="Çıktı klasörü (log/summary yazmak için)")
    p.add_argument("--policy", default="local_adaptive", choices=["local_adaptive", "global_adaptive", "fixed"])
    return p.parse_args()


def make_consumer(args: argparse.Namespace) -> KafkaConsumer:
    return KafkaConsumer(
        args.topic,
        bootstrap_servers=[args.bootstrap],
        group_id=args.group_id,
        key_deserializer=lambda k: int(k.decode()) if k else None,
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=1000,
    )


def process_buffer(engine: StreamingEngine, buffer: list[dict]) -> None:
    if not buffer:
        return
    pdf = pd.DataFrame(buffer)
    if len(pdf) == 0:
        return
    engine.process_batch(pdf)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = StreamingEngine(
        data=args.data,
        model=args.model,
        initial_state=args.initial_state,
        policy=args.policy,
        out_dir=str(out_dir),
        tau=0.5,
        alpha=0.5,
        beta=0.25,
        gamma=0.25,
        norm="exp",
    )

    consumer = make_consumer(args)
    buffer: list[dict] = []
    last_activity = time.time()
    running = True

    def stop_handler(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    print(f"[worker] bootstrap={args.bootstrap} topic={args.topic} group={args.group_id}")

    try:
        while running:
            records = consumer.poll(timeout_ms=1000, max_records=args.batch_size)
            now = time.time()
            if records:
                for tp_records in records.values():
                    for rec in tp_records:
                        value = rec.value
                        if not value:
                            continue
                        buffer.append({
                            "src": int(value["src"]),
                            "dst": int(value["dst"]),
                            "time_step": int(value["time_step"]),
                            "event_ts": float(value["event_ts"]),
                        })
                last_activity = now

            if buffer and (len(buffer) >= args.batch_size or now - last_activity >= args.batch_timeout):
                print(f"[worker] processing batch {len(buffer)} events")
                process_buffer(engine, buffer)
                buffer = []
                last_activity = now

            if now - last_activity > args.idle_timeout and not buffer:
                print(f"[worker] idle timeout {args.idle_timeout}s reached, exiting")
                break
    finally:
        if buffer:
            print(f"[worker] flushing remaining {len(buffer)} events")
            process_buffer(engine, buffer)
        consumer.close()
        engine.finalize()


if __name__ == "__main__":
    main()
