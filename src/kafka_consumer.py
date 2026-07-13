"""Kafka consumer — Kafka topic'inden edge olayları okur ve JSON olarak yazdırır.

Kullanım:
  pip install kafka-python
  python src/kafka_consumer.py --bootstrap localhost:9092 --topic edges \
      --group-id edge-consumer --timeout-sec 30

Opsiyonel:
  --out-file events.jsonl ile alınan olayları JSONL dosyasına kaydedebilirsiniz.
"""
from __future__ import annotations

import argparse
import json
import time

from kafka import KafkaConsumer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap", default="localhost:9092",
                   help="Kafka bootstrap sunucusu")
    p.add_argument("--topic", default="edges",
                   help="Okunacak Kafka topic'i")
    p.add_argument("--group-id", default="elliptic-edge-consumer",
                   help="Kafka consumer group id")
    p.add_argument("--auto-offset-reset", choices=["earliest", "latest"],
                   default="earliest",
                   help="Consumer offset başlangıcı")
    p.add_argument("--timeout-sec", type=int, default=30,
                   help="Bu süre boyunca mesaj gelmezse consumer kapanır")
    p.add_argument("--max-messages", type=int, default=None,
                   help="Maksimum okunacak mesaj sayısı")
    p.add_argument("--out-file", default=None,
                   help="Alınan olayları JSONL olarak kaydet")
    return p.parse_args()


def main() -> None:
    args = parse_args()

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
            event = msg.value
            recv_ts = time.time()
            print(
                f"[msg {count+1}] key={msg.key} src={event.get('src')} dst={event.get('dst')} "
                f"time_step={event.get('time_step')} event_ts={event.get('event_ts'):.3f} "
                f"recv_ts={recv_ts:.3f}"
            )
            if out_file is not None:
                out_file.write(json.dumps({
                    "key": msg.key,
                    **(event or {}),
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


if __name__ == "__main__":
    main()
