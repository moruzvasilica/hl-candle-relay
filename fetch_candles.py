#!/usr/bin/env python3
"""
Hyperliquid candle relay for TR-F5-Crypto-LS-1H-03.

Fetches fully CLOSED 15m, 1h and 4h perpetual candles plus current
funding rates from Hyperliquid's public API.

Writes backward-compatible JSON files under data/:

    data/BTC_15m.json
    data/BTC_1h.json
    data/BTC_4h.json
    ...
    data/manifest.json

Compatibility guarantees:
- manifest schema remains 1;
- manifest["files"][key] remains a filename string;
- candle fields remain t, T, o, h, l, c and v;
- t = candle open time in epoch milliseconds;
- T = candle close time in epoch milliseconds.

Additional diagnostics are stored in manifest["fileMeta"] without
breaking the existing Claude/SIGNUM consumer.

No API keys are required. Public market data only.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

API_URL = "https://api.hyperliquid.xyz/info"

ASSETS = [
    "BTC",
    "ETH",
    "XRP",
    "BNB",
    "SOL",
    "DOGE",
    "ADA",
    "TRX",
    "LINK",
    "AVAX",
    "SUI",
    "HYPE",
    "LTC",
    "DOT",
    "BCH",
]

# Value is timeframe duration in seconds.
TIMEFRAMES = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
}

# The trading strategy needs at least 200 candles.
# 320 provides extra indicator warm-up without producing large files.
KEEP = 320

# Request a few extra candles because the newest API candle may still be open,
# and malformed or duplicate records may need to be discarded.
FETCH_BUFFER = 30

MIN_REQUIRED_CANDLES = {
    "15m": 150,
    "1h": 200,
    "4h": 200,
}

OUT_DIR = Path("data")
SCHEMA = 1

CONNECT_TIMEOUT_SECONDS = 8
READ_TIMEOUT_SECONDS = 15

# Retries performed by our own request wrapper.
REQUEST_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 2.0

# Small delay between candle requests to avoid hammering the public endpoint.
REQUEST_DELAY_SECONDS = 0.25

# Candle timestamp tolerance.
# Hyperliquid candle close is normally interval_end - 1 millisecond.
TIMESTAMP_TOLERANCE_MS = 2_000

USER_AGENT = "hl-candle-relay/2.0"


# ---------------------------------------------------------------------------
# TIME AND SERIALIZATION HELPERS
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    """Return a timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


def iso_utc(value: datetime | None = None) -> str:
    """Return a stable ISO 8601 UTC timestamp ending in Z."""
    value = value or utc_now()
    return value.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def epoch_ms_to_iso(value_ms: int | None) -> str | None:
    """Convert epoch milliseconds to an ISO 8601 UTC timestamp."""
    if value_ms is None:
        return None

    return datetime.fromtimestamp(
        value_ms / 1000,
        tz=timezone.utc,
    ).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_json_write(
    path: Path,
    payload: Any,
    *,
    compact: bool,
) -> None:
    """
    Write JSON atomically.

    Data is first written to a temporary file and then moved into place.
    This prevents Claude/raw.githubusercontent.com from seeing a partially
    written JSON document if the process is interrupted during a write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(f".{path.name}.tmp")

    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        if compact:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        else:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temp_path, path)


# ---------------------------------------------------------------------------
# HTTP CLIENT
# ---------------------------------------------------------------------------

def create_session() -> requests.Session:
    """
    Create a reusable HTTP session.

    Connection pooling reduces overhead. Adapter-level retries are restricted
    to connection failures; application-level retries and logging are handled
    explicitly in post_json().
    """
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
    )

    adapter_retry = Retry(
        total=1,
        connect=1,
        read=0,
        status=0,
        redirect=0,
        backoff_factor=0,
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=adapter_retry,
        pool_connections=4,
        pool_maxsize=4,
    )

    session.mount("https://", adapter)
    return session


SESSION = create_session()


def post_json(
    payload: dict[str, Any],
    *,
    attempts: int = REQUEST_ATTEMPTS,
) -> Any:
    """
    POST JSON to Hyperliquid with bounded retries.

    Raises RuntimeError after all attempts fail. It never silently returns
    fabricated or empty data.
    """
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = SESSION.post(
                API_URL,
                json=payload,
                timeout=(
                    CONNECT_TIMEOUT_SECONDS,
                    READ_TIMEOUT_SECONDS,
                ),
            )

            response.raise_for_status()

            try:
                result = response.json()
            except ValueError as exc:
                preview = response.text[:200].replace("\n", " ")
                raise RuntimeError(
                    f"Invalid JSON response: {preview!r}"
                ) from exc

            return result

        except (
            requests.RequestException,
            RuntimeError,
        ) as exc:
            last_error = exc

            if attempt >= attempts:
                break

            # Exponential backoff plus small jitter.
            delay = (
                RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                + random.uniform(0.0, 0.5)
            )

            print(
                f"WARN request attempt {attempt}/{attempts} failed: "
                f"{type(exc).__name__}: {exc}; retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Hyperliquid request failed after {attempts} attempts: "
        f"{type(last_error).__name__ if last_error else 'UnknownError'}: "
        f"{last_error}"
    )


# ---------------------------------------------------------------------------
# CANDLE VALIDATION
# ---------------------------------------------------------------------------

def finite_number(value: Any) -> float:
    """Convert a value to a finite float or raise ValueError."""
    result = float(value)

    if not math.isfinite(result):
        raise ValueError("non-finite numeric value")

    return result


def parse_candle(
    raw: Any,
    *,
    expected_asset: str,
    expected_interval: str,
    interval_seconds: int,
    now_ms: int,
) -> tuple[dict[str, int | float] | None, str | None]:
    """
    Validate and normalize one Hyperliquid candle.

    Returns:
        (normalized_candle, None) on success
        (None, rejection_reason) when invalid or still forming
    """
    if not isinstance(raw, dict):
        return None, "not_an_object"

    try:
        open_ms = int(raw["t"])
        close_ms = int(raw["T"])

        open_price = finite_number(raw["o"])
        high_price = finite_number(raw["h"])
        low_price = finite_number(raw["l"])
        close_price = finite_number(raw["c"])
        volume = finite_number(raw["v"])

    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "invalid_or_missing_field"

    if open_ms <= 0 or close_ms <= 0:
        return None, "invalid_timestamp"

    if close_ms < open_ms:
        return None, "close_before_open"

    # Never include a forming or future candle.
    if close_ms > now_ms:
        return None, "forming_or_future"

    expected_duration_ms = interval_seconds * 1000
    observed_duration_ms = close_ms - open_ms + 1

    if abs(observed_duration_ms - expected_duration_ms) > TIMESTAMP_TOLERANCE_MS:
        return None, "unexpected_duration"

    if (
        open_price <= 0
        or high_price <= 0
        or low_price <= 0
        or close_price <= 0
    ):
        return None, "non_positive_price"

    if volume < 0:
        return None, "negative_volume"

    # Basic OHLC consistency.
    if high_price < max(open_price, close_price, low_price):
        return None, "invalid_high"

    if low_price > min(open_price, close_price, high_price):
        return None, "invalid_low"

    # Validate API identity fields when present.
    returned_asset = raw.get("s")
    if returned_asset is not None and str(returned_asset) != expected_asset:
        return None, "asset_mismatch"

    returned_interval = raw.get("i")
    if (
        returned_interval is not None
        and str(returned_interval) != expected_interval
    ):
        return None, "interval_mismatch"

    return {
        "t": open_ms,
        "T": close_ms,
        "o": open_price,
        "h": high_price,
        "l": low_price,
        "c": close_price,
        "v": volume,
    }, None


def validate_sequence(
    rows: list[dict[str, int | float]],
    *,
    interval_seconds: int,
) -> list[str]:
    """
    Inspect the final ordered sequence for gaps and ordering problems.

    Gaps are reported in metadata but do not automatically discard the file.
    Claude can then decide whether the affected asset remains usable.
    """
    warnings: list[str] = []

    if not rows:
        return warnings

    interval_ms = interval_seconds * 1000

    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]

        previous_open = int(previous["t"])
        current_open = int(current["t"])

        delta_ms = current_open - previous_open

        if delta_ms <= 0:
            warnings.append(
                f"non_chronological_at_index_{index}"
            )
            continue

        if delta_ms != interval_ms:
            missing_intervals = max(
                0,
                round(delta_ms / interval_ms) - 1,
            )

            warnings.append(
                "gap:"
                f"previous_open={previous_open},"
                f"current_open={current_open},"
                f"missing_intervals={missing_intervals}"
            )

    return warnings


# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_candles(
    asset: str,
    interval: str,
    interval_seconds: int,
    now_ms: int,
) -> tuple[
    list[dict[str, int | float]],
    dict[str, Any],
]:
    """
    Fetch and validate fully closed candles for one asset/timeframe.
    """
    requested_count = KEEP + FETCH_BUFFER
    start_ms = now_ms - requested_count * interval_seconds * 1000

    response = post_json(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": asset,
                "interval": interval,
                "startTime": start_ms,
                "endTime": now_ms,
            },
        }
    )

    if not isinstance(response, list):
        raise RuntimeError(
            f"Unexpected candleSnapshot response type: "
            f"{type(response).__name__}"
        )

    accepted: list[dict[str, int | float]] = []
    rejection_counts: dict[str, int] = {}

    for item in response:
        row, rejection_reason = parse_candle(
            item,
            expected_asset=asset,
            expected_interval=interval,
            interval_seconds=interval_seconds,
            now_ms=now_ms,
        )

        if row is None:
            reason = rejection_reason or "unknown_rejection"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        accepted.append(row)

    # Sort by open timestamp.
    accepted.sort(key=lambda candle: int(candle["t"]))

    # De-duplicate using open timestamp. When duplicates exist, keep the
    # latest occurrence returned by the API.
    by_open_time: dict[int, dict[str, int | float]] = {}

    for row in accepted:
        by_open_time[int(row["t"])] = row

    deduplicated = [
        by_open_time[open_time]
        for open_time in sorted(by_open_time)
    ]

    final_rows = deduplicated[-KEEP:]
    sequence_warnings = validate_sequence(
        final_rows,
        interval_seconds=interval_seconds,
    )

    diagnostics = {
        "apiRows": len(response),
        "acceptedRowsBeforeDeduplication": len(accepted),
        "duplicateRowsRemoved": len(accepted) - len(deduplicated),
        "rejections": rejection_counts,
        "sequenceWarnings": sequence_warnings,
    }

    return final_rows, diagnostics


def fetch_funding() -> tuple[dict[str, float], str | None]:
    """
    Fetch current hourly funding rates.

    Positive funding means longs pay shorts.
    Missing or malformed funding is omitted rather than fabricated.
    """
    try:
        response = post_json({"type": "metaAndAssetCtxs"})

        if (
            not isinstance(response, list)
            or len(response) != 2
            or not isinstance(response[0], dict)
            or not isinstance(response[1], list)
        ):
            raise RuntimeError(
                "Unexpected metaAndAssetCtxs response structure"
            )

        meta = response[0]
        contexts = response[1]
        universe = meta.get("universe", [])

        if not isinstance(universe, list):
            raise RuntimeError(
                "metaAndAssetCtxs universe is not a list"
            )

        funding: dict[str, float] = {}

        for market, context in zip(universe, contexts):
            if not isinstance(market, dict) or not isinstance(context, dict):
                continue

            asset = market.get("name")

            if asset not in ASSETS:
                continue

            raw_funding = context.get("funding")

            try:
                funding_value = finite_number(raw_funding)
            except (TypeError, ValueError, OverflowError):
                continue

            funding[str(asset)] = funding_value

        return funding, None

    except Exception as exc:  # noqa: BLE001
        return {}, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# MANIFEST METADATA
# ---------------------------------------------------------------------------

def build_file_metadata(
    *,
    filename: str,
    interval: str,
    interval_seconds: int,
    rows: list[dict[str, int | float]],
    diagnostics: dict[str, Any],
    reference_now_ms: int,
) -> dict[str, Any]:
    """Build diagnostics for one successfully fetched candle file."""
    first = rows[0] if rows else None
    latest = rows[-1] if rows else None

    latest_close_ms = int(latest["T"]) if latest else None

    close_age_seconds = (
        max(0, (reference_now_ms - latest_close_ms) // 1000)
        if latest_close_ms is not None
        else None
    )

    minimum_required = MIN_REQUIRED_CANDLES[interval]

    return {
        "filename": filename,
        "interval": interval,
        "intervalSeconds": interval_seconds,
        "count": len(rows),
        "minimumRequired": minimum_required,
        "minimumCountMet": len(rows) >= minimum_required,
        "firstOpenMs": int(first["t"]) if first else None,
        "firstOpenUtc": (
            epoch_ms_to_iso(int(first["t"])) if first else None
        ),
        "latestOpenMs": int(latest["t"]) if latest else None,
        "latestOpenUtc": (
            epoch_ms_to_iso(int(latest["t"])) if latest else None
        ),
        "latestCloseMs": latest_close_ms,
        "latestCloseUtc": epoch_ms_to_iso(latest_close_ms),
        "latestCloseAgeSecondsAtFetch": close_age_seconds,
        "latestCloseAgeMinutesAtFetch": (
            round(close_age_seconds / 60, 2)
            if close_age_seconds is not None
            else None
        ),
        "apiRows": diagnostics["apiRows"],
        "acceptedRowsBeforeDeduplication": diagnostics[
            "acceptedRowsBeforeDeduplication"
        ],
        "duplicateRowsRemoved": diagnostics["duplicateRowsRemoved"],
        "rejections": diagnostics["rejections"],
        "sequenceWarnings": diagnostics["sequenceWarnings"],
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_started = utc_now()
    run_started_ms = int(run_started.timestamp() * 1000)

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "source": "hyperliquid",

        # Backward-compatible field. It is updated again immediately before
        # the manifest is written, so it represents completion rather than
        # merely the beginning of the run.
        "generatedAtUtc": iso_utc(run_started),

        "runStartedAtUtc": iso_utc(run_started),
        "runCompletedAtUtc": None,
        "durationSeconds": None,

        "assets": ASSETS,
        "timeframes": list(TIMEFRAMES.keys()),

        # Backward compatibility:
        # value remains a filename string, not an object.
        "files": {},

        # New detailed metadata for freshness and diagnostics.
        "fileMeta": {},

        "funding": {},
        "fundingFetchedAtUtc": None,

        "errors": [],
        "warnings": [],
        "health": "starting",
        "successfulFileCount": 0,
        "expectedFileCount": len(ASSETS) * len(TIMEFRAMES),
    }

    print(
        f"START {iso_utc(run_started)} "
        f"assets={len(ASSETS)} "
        f"timeframes={len(TIMEFRAMES)} "
        f"expected_files={manifest['expectedFileCount']}"
    )

    # Funding failure does not prevent candles from being published.
    funding, funding_error = fetch_funding()
    manifest["funding"] = funding

    if funding:
        manifest["fundingFetchedAtUtc"] = iso_utc()
    else:
        manifest["warnings"].append(
            funding_error or "Funding response was empty"
        )
        print(
            f"WARN funding unavailable: "
            f"{funding_error or 'empty response'}",
            file=sys.stderr,
        )

    for asset in ASSETS:
        for interval, interval_seconds in TIMEFRAMES.items():
            key = f"{asset}_{interval}"
            filename = f"{key}.json"
            destination = OUT_DIR / filename

            # Small API politeness delay.
            time.sleep(REQUEST_DELAY_SECONDS)

            try:
                fetch_reference_ms = int(time.time() * 1000)

                rows, diagnostics = fetch_candles(
                    asset=asset,
                    interval=interval,
                    interval_seconds=interval_seconds,
                    now_ms=fetch_reference_ms,
                )

                if not rows:
                    raise RuntimeError("No valid fully closed candles returned")

                file_metadata = build_file_metadata(
                    filename=filename,
                    interval=interval,
                    interval_seconds=interval_seconds,
                    rows=rows,
                    diagnostics=diagnostics,
                    reference_now_ms=fetch_reference_ms,
                )

                if len(rows) < MIN_REQUIRED_CANDLES[interval]:
                    warning = (
                        f"{key}: only {len(rows)} valid candles; "
                        f"minimum required is "
                        f"{MIN_REQUIRED_CANDLES[interval]}"
                    )
                    manifest["warnings"].append(warning)
                    print(f"WARN {warning}", file=sys.stderr)

                if diagnostics["sequenceWarnings"]:
                    warning = (
                        f"{key}: sequence warnings="
                        f"{len(diagnostics['sequenceWarnings'])}"
                    )
                    manifest["warnings"].append(warning)
                    print(f"WARN {warning}", file=sys.stderr)

                candle_document = {
                    "asset": asset,
                    "interval": interval,
                    "candles": rows,
                }

                atomic_json_write(
                    destination,
                    candle_document,
                    compact=True,
                )

                # Keep this exact string format for compatibility with the
                # existing TR-F5 strategy.
                manifest["files"][key] = filename
                manifest["fileMeta"][key] = file_metadata

                print(
                    f"OK {key} "
                    f"count={len(rows)} "
                    f"latestOpen={file_metadata['latestOpenUtc']} "
                    f"latestClose={file_metadata['latestCloseUtc']} "
                    f"ageMin="
                    f"{file_metadata['latestCloseAgeMinutesAtFetch']}"
                )

            except Exception as exc:  # noqa: BLE001
                error_message = (
                    f"{key}: {type(exc).__name__}: {exc}"
                )

                manifest["errors"].append(error_message)

                # Important:
                # Do not overwrite a previously valid candle file with empty
                # or partial data. The failed key is excluded from the current
                # manifest, so a correct consumer must not treat the old file
                # as current.
                manifest["fileMeta"][key] = {
                    "filename": filename,
                    "interval": interval,
                    "intervalSeconds": interval_seconds,
                    "status": "fetch_failed",
                    "error": error_message,
                    "previousFileStillExists": destination.exists(),
                }

                print(
                    f"FAIL {error_message}",
                    file=sys.stderr,
                )

    run_completed = utc_now()
    duration_seconds = round(
        (run_completed - run_started).total_seconds(),
        3,
    )

    successful_count = len(manifest["files"])
    expected_count = int(manifest["expectedFileCount"])

    manifest["successfulFileCount"] = successful_count
    manifest["runCompletedAtUtc"] = iso_utc(run_completed)
    manifest["durationSeconds"] = duration_seconds

    # generatedAtUtc now represents the completion of the produced snapshot.
    manifest["generatedAtUtc"] = iso_utc(run_completed)

    critical_keys = {
        "BTC_1h",
        "BTC_4h",
    }

    missing_critical = sorted(
        key for key in critical_keys
        if key not in manifest["files"]
    )

    minimum_count_failures = sorted(
        key
        for key, metadata in manifest["fileMeta"].items()
        if isinstance(metadata, dict)
        and metadata.get("status") != "fetch_failed"
        and metadata.get("minimumCountMet") is False
    )

    manifest["missingCriticalFiles"] = missing_critical
    manifest["minimumCountFailures"] = minimum_count_failures

    if missing_critical:
        manifest["health"] = "critical"
    elif successful_count < expected_count or minimum_count_failures:
        manifest["health"] = "degraded"
    else:
        manifest["health"] = "ok"

    # Manifest is written last. Therefore, when Claude sees a new manifest,
    # all successfully listed candle files have already been written.
    atomic_json_write(
        OUT_DIR / "manifest.json",
        manifest,
        compact=False,
    )

    print(
        f"DONE health={manifest['health']} "
        f"files={successful_count}/{expected_count} "
        f"errors={len(manifest['errors'])} "
        f"warnings={len(manifest['warnings'])} "
        f"duration={duration_seconds}s "
        f"generatedAtUtc={manifest['generatedAtUtc']}"
    )

    # Return success after publishing a truthful partial manifest.
    #
    # This is intentional: many GitHub workflows perform the commit in a
    # later step. Returning non-zero here could prevent the updated manifest
    # and diagnostics from being committed, leaving raw.githubusercontent.com
    # stuck on an older snapshot.
    #
    # The trading routine must inspect manifest["health"],
    # manifest["files"], candle timestamps and minimum counts.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("CANCELLED by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # final unexpected safety net
        print(
            f"FATAL {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
