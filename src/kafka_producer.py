"""Kafka producer/replayer — stream_edges.csv'yi kronolojik sırayla topic'e basar.

Her olay JSON: {"src": int, "dst": int, "time_step": int, "event_ts": float}
event_ts = gönderim anının epoch saniyesi -> consumer tarafında uçtan uca
gecikme (p50/p99) buradan ölçülür. --events-per-sec ile throughput kontrol
edilir (proposal'daki kontrollü throughput deneyleri).

Kullanım:
  pip install kafka-python
  python src/kafka_producer.py --stream processed/stream_edges.csv \
      --bootstrap localhost:9092 --topic edges --events-per-sec 500
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from kafka import KafkaProducer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stream", required=True, help="preprocess çıktısı stream_edges.csv")
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="edges")
    p.add_argument("--events-per-sec", type=float, default=500.0)
    p.add_argument("--from-ts", type=int, default=None,
                   help="yalnızca bu time step ve sonrasını gönder (vars: val_end+1 için 35)")
    args = p.parse_args()

    df = pd.read_csv(args.stream)
    if args.from_ts is not None:
        df = df[df.time_step >= args.from_ts]
    df = df.sort_values("time_step").reset_index(drop=True)

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode(),
        key_serializer=lambda k: str(k).encode(),
        linger_ms=5,
    )
    interval = 1.0 / args.events_per_sec
    t0, sent = time.time(), 0
    print(f"[producer] {len(df)} olay, hedef {args.events_per_sec}/sn, topic={args.topic}")
    for r in df.itertuples(index=False):
        producer.send(args.topic, key=int(r.src),
                      value={"src": int(r.src), "dst": int(r.dst),
                             "time_step": int(r.time_step), "event_ts": time.time()})
        sent += 1
        # basit hız kontrolü: hedef zamanın gerisindeysek bekleme yok
        target = t0 + sent * interval
        lag = target - time.time()
        if lag > 0:
            time.sleep(lag)
        if sent % 10000 == 0:
            print(f"  {sent}/{len(df)} gönderildi ({sent/(time.time()-t0):.0f}/sn)")
    producer.flush()
    print(f"[tamam] {sent} olay {time.time()-t0:.1f} sn'de gönderildi "
          f"(gerçekleşen {sent/(time.time()-t0):.0f}/sn)")


if __name__ == "__main__":
    main()
