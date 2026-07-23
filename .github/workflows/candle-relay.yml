#!/usr/bin/env python3
"""Hyperliquid candle relay.

Fetches fully CLOSED 15m/1h/4h candles + current funding for BTC/ETH/SOL/HYPE
from Hyperliquid's public API and writes compact JSON files under data/.
Consumed by the TR-F5-Crypto-LS-1H-03 routine via raw.githubusercontent.com.

No API keys. Public market data only.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

API = "https://api.hyperliquid.xyz/info"
ASSETS = ["BTC", "ETH", "SOL", "HYPE"]
TIMEFRAMES = {"15m": 15 * 60, "1h": 3600, "4h": 4 * 3600}
KEEP = 260          # candles kept per asset per timeframe (strategy needs >= 250)
OUT_DIR = "data"
SCHEMA = 1


def post(payload, retries=3, wait=5):
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(API, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001 - relay must degrade gracefully
            last = exc
            if attempt < retries - 1:
                time.sleep(wait)
    raise RuntimeError(f"Hyperliquid request failed after {retries} tries: {last}")


def fetch_candles(coin, interval, secs, now_ms):
    start = now_ms - (KEEP + 20) * secs * 1000
    raw = post({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval,
                "startTime": start, "endTime": now_ms},
    })
    rows = []
    for c in raw if isinstance(raw, list) else []:
        try:
            row = {
                "t": int(c["t"]),          # open time (ms, UTC)
                "T": int(c["T"]),          # close time (ms, UTC)
                "o": float(c["o"]),
                "h": float(c["h"]),
                "l": float(c["l"]),
                "c": float(c["c"]),
                "v": float(c["v"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if row["T"] <= now_ms:             # fully closed candles only
            rows.append(row)
    rows.sort(key=lambda r: r["t"])
    seen, deduped = set(), []
    for r in rows:
        if r["t"] in seen:
            continue
        seen.add(r["t"])
        deduped.append(r)
    return deduped[-KEEP:]


def fetch_funding():
    """Current hourly funding rate per asset (positive = longs pay shorts)."""
    try:
        meta, ctxs = post({"type": "metaAndAssetCtxs"})
        names = [u.get("name") for u in meta.get("universe", [])]
        out = {}
        for name, ctx in zip(names, ctxs):
            if name in ASSETS:
                try:
                    out[name] = float(ctx.get("funding"))
                except (TypeError, ValueError):
                    pass
        return out
    except Exception:  # noqa: BLE001
        return {}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    now_ms = int(time.time() * 1000)
    manifest = {
        "schema": SCHEMA,
        "source": "hyperliquid",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assets": ASSETS,
        "timeframes": list(TIMEFRAMES),
        "files": {},
        "funding": fetch_funding(),
        "errors": [],
    }
    for asset in ASSETS:
        for tf, secs in TIMEFRAMES.items():
            key = f"{asset}_{tf}"
            try:
                rows = fetch_candles(asset, tf, secs, now_ms)
            except Exception as exc:  # noqa: BLE001
                manifest["errors"].append(f"{key}: {exc}")
                print(f"FAIL {key}: {exc}", file=sys.stderr)
                continue
            if len(rows) < 200:
                manifest["errors"].append(f"{key}: only {len(rows)} candles")
            filename = f"{key}.json"
            with open(os.path.join(OUT_DIR, filename), "w") as fh:
                json.dump({"asset": asset, "interval": tf, "candles": rows},
                          fh, separators=(",", ":"))
            manifest["files"][key] = filename
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"OK files={len(manifest['files'])} errors={len(manifest['errors'])}")


if __name__ == "__main__":
    main()
