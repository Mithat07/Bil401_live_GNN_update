"""Kafka producer/consumer yardımcı scripti.

Bu script iki mod sunar:
  1) producer: processed/stream_edges.csv içeriğini Kafka topic'ine gönderir
  2) consumer: Kafka topic'inden mesaj okur ve konsola yazdırır

Kullanım:
  pip install kafka-python pandas
  python src/kafka_client.py produce --stream processed/stream_edges.csv \
      --bootstrap localhost:9092 --topic edges --events-per-sec 500

  python src/kafka_client.py consume --bootstrap localhost:9092 \
      --topic edges --group-id edge-consumer --timeout-sec 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    subparsers = p.add_subparsers(dest="command", required=True)

    prod = subparsers.add_parser("produce", help="Kafka topic'e event gönderir")
    prod.add_argument("--stream", required=True,
                      help="Preprocess edilmiş stream_edges.csv yolu")
    prod.add_argument("--bootstrap", default="localhost:9092",
                      help="Kafka bootstrap sunucusu")
    prod.add_argument("--topic", default="edges",
                      help="Gönderilecek topic")
    prod.add_argument("--events-per-sec", type=float, default=500.0,
                      help="Hedef throughput")
    prod.add_argument("--from-ts", type=int, default=None,
                      help="Yalnızca bu time_step ve sonrası gönderilir")

    cons = subparsers.add_parser("consume", help="Kafka topic'inden mesaj okur")
    cons.add_argument("--bootstrap", default="localhost:9092",
                      help="Kafka bootstrap sunucusu")
    cons.add_argument("--topic", default="edges",
                      help="Okunacak topic")
    cons.add_argument("--group-id", default="elliptic-edge-consumer",
                      help="Consumer group id")
    cons.add_argument("--auto-offset-reset", choices=["earliest", "latest"],
                      default="earliest",
                      help="Offset başlangıcı")
    cons.add_argument("--timeout-sec", type=int, default=30,
                      help="Mesaj gelmezse kapanma süresi")
    cons.add_argument("--max-messages", type=int, default=None,
                      help="Maksimum okunacak mesaj sayısı")
    cons.add_argument("--out-file", default=None,
                      help="Alınan olayları JSONL olarak kaydet")

    return p.parse_args()


def run_producer(args: argparse.Namespace) -> None:
    stream_path = Path(args.stream)
    if not stream_path.exists():
        raise FileNotFoundError(f"Stream dosyası bulunamadı: {stream_path}")

    df = pd.read_csv(stream_path)
    if args.from_ts is not None:
        df = df[df["time_step"] >= args.from_ts]
    df = df.sort_values("time_step").reset_index(drop=True)

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        linger_ms=5,
    )

    interval = 1.0 / args.events_per_sec
    t0, sent = time.time(), 0
    print(f"[producer] {len(df)} olay gönderilecek, hedef={args.events_per_sec}/sn, topic={args.topic}")
    for r in df.itertuples(index=False):
        producer.send(args.topic, key=int(r.src),
                      value={
                          "src": int(r.src),
                          "dst": int(r.dst),
                          "time_step": int(r.time_step),
                          "event_ts": time.time(),
                      })
        sent += 1
        target = t0 + sent * interval
        lag = target - time.time()
        if lag > 0:
            time.sleep(lag)
        if sent % 10000 == 0:
            throughput = sent / max(1.0, time.time() - t0)
            print(f"  {sent}/{len(df)} gönderildi ({throughput:.0f}/sn)")

    producer.flush()
    elapsed = time.time() - t0
    print(f"[producer] tamamlandı: {sent} mesaj, {elapsed:.1f}s, ort={sent/elapsed:.0f}/sn")


def run_consumer(args: argparse.Namespace) -> None:
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        group_id=args.group_id,
        auto_offset_reset=args.auto_offset_reset,
        enable_auto_commit=True,
        consumer_timeout_ms=args.timeout_sec * 1000,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")) if v is not None else None,
        key_deserializer=lambda v: int(v.decode("utf-8")) if v is not None else None,
    )

    out_file = open(args.out_file, "w", encoding="utf-8") if args.out_file else None
    print(f"[consumer] topic={args.topic}, bootstrap={args.bootstrap}, group_id={args.group_id}")

    count = 0
    try:
        for msg in consumer:
            event = msg.value or {}
            recv_ts = time.time()
            print(
                f"[msg {count+1}] key={msg.key} src={event.get('src')} dst={event.get('dst')} "
                f"time_step={event.get('time_step')} event_ts={event.get('event_ts')} recv_ts={recv_ts:.3f}"
            )
            if out_file is not None:
                out_file.write(json.dumps({
                    "key": msg.key,
                    **event,
                    "recv_ts": recv_ts,
                }) + "\n")
            count += 1
            if args.max_messages is not None and count >= args.max_messages:
                break
    finally:
        consumer.close()
        if out_file is not None:
            out_file.close()
        print(f"[consumer] kapatıldı, toplam {count} mesaj okundu")


def main() -> None:
    args = parse_args()
    if args.command == "produce":
        run_producer(args)
    elif args.command == "consume":
        run_consumer(args)
    else:
        raise ValueError(f"Bilinmeyen komut: {args.command}")


if __name__ == "__main__":
    main()
