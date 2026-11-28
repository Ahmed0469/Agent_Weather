"""
History Backfill Script

Fetches 365 days of historical weather and air quality data from Open-Meteo
and OpenAQ APIs, engineers features, and saves them to the Hopsworks
Feature Store.

References logic from feature_pipeline.py:
  - pm25_to_aqi() for AQI calculation
  - save_to_hopsworks() pattern for Hopsworks integration
  - _request_with_retry() for resilient HTTP requests

Usage:
    python src/feature_pipeline/history_backfill.py

Environment variables:
    WEATHER_LAT, WEATHER_LON       - Location coordinates
    OPENAQ_API_KEY                 - OpenAQ v3 API key
    HOPSWORKS_API_KEY              - Hopsworks API key
    HOPSWORKS_PROJECT              - Hopsworks project name
    HOPSWORKS_HOST                 - (optional) Hopsworks host
    BACKFILL_DAYS                  - Number of days to backfill (default: 365)
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# from feature_pipeline import (
#     _request_with_retry,
#     pm25_to_aqi,
#     save_to_hopsworks,
# )
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from feature_pipeline import (
    _request_with_retry,
    pm25_to_aqi,
    save_to_hopsworks,
    fetch_openmeteo_weather,
    fetch_openmeteo_aqi,
    KARACHI_TZ,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("history_backfill")

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
OM_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OM_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENAQ_V3_URL = "https://api.openaq.org/v3"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAT = float(os.getenv("WEATHER_LAT", "24.8607"))
LON = float(os.getenv("WEATHER_LON", "67.0011"))
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", None)
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "365"))

# The archive API lags roughly ARCHIVE_LAG_DAYS behind today. Fill that tail
# from the forecast endpoint (past_days=7) so the backfill reaches the present.
ARCHIVE_LAG_DAYS = int(os.getenv("ARCHIVE_LAG_DAYS", "5"))
FORECAST_PAST_DAYS = int(os.getenv("FORECAST_PAST_DAYS", "7"))


def _parse_as_kachi(value: str) -> datetime:
    """Parse a date/datetime string as Karachi time.

    Naive values are treated as midnight Karachi time; aware values are
    converted to Karachi time.
    """
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KARACHI_TZ)
    return dt.astimezone(KARACHI_TZ)


# ---------------------------------------------------------------------------
# Fetch Functions (365-day historical)
# ---------------------------------------------------------------------------
# def fetch_openmeteo_weather_historical(
#     lat: float,
#     lon: float,
#     start_date: str,
#     end_date: str,
# ) -> Dict[str, Any]:
#     """
#     Fetch historical weather data from Open-Meteo Archive API.

#     Uses start_date/end_date to get hourly historical data.
#     Supports multi-year history.

#     Returns the raw JSON response dict.
#     """
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "start_date": start_date,
#         "end_date": end_date,
#         "hourly": [
#             "temperature_2m",
#             "relative_humidity_2m",
#             "precipitation",
#             "rain",
#             "wind_speed_10m",
#             "wind_direction_10m",
#         ],
#         "timezone": "UTC",
#     }
#     try:
#         resp = _request_with_retry("GET", OM_ARCHIVE_URL, params=params, timeout=120)
#         resp.raise_for_status()
#         data = resp.json()
#         hourly = data.get("hourly", {})
#         times = hourly.get("time", [])
#         logger.info(
#             f"Open-Meteo Archive Weather fetched: {len(times)} hourly records "
#             f"({times[0]} to {times[-1] if times else 'N/A'})"
#         )
#         return data
#     except Exception as exc:
#         logger.error(f"Open-Meteo Archive Weather fetch failed: {exc}")
#         return {}


# def fetch_openmeteo_aq_historical_from_range(
#     lat: float,
#     lon: float,
#     start_date: str,
#     end_date: str,
# ) -> Dict[str, Any]:
#     """
#     Fetch historical air quality data from Open-Meteo Air Quality API.

#     Uses start_date/end_date to get hourly historical UV index.
#     Supports multi-year history.

#     Returns the raw JSON response dict.
#     """
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "hourly": ["uv_index"],
#         "start_date": start_date,
#         "end_date": end_date,
#         "timezone": "UTC",
#     }
#     try:
#         resp = _request_with_retry("GET", OM_AQ_URL, params=params, timeout=120)
#         resp.raise_for_status()
#         data = resp.json()
#         hourly = data.get("hourly", {})
#         times = hourly.get("time", [])
#         logger.info(
#             f"Open-Meteo Archive Air Quality fetched: {len(times)} hourly records "
#             f"({times[0]} to {times[-1] if times else 'N/A'})"
#         )
#         return data
#     except Exception as exc:
#         logger.error(f"Open-Meteo Archive Air Quality fetch failed: {exc}")
#         return {}

def fetch_openaq_historical(    
    api_key: Optional[str] = None,
    start_date: str = None,
    end_date: str = None,
    chunk_days: int = 30,
    threshold: int = 650,
) -> Dict[str, Any]:
    """
    Fetch historical PM2.5 data from OpenAQ API v3.

    Splits the date range into monthly chunks (default 30 days).
    Uses location ID 4814327 as primary for every chunk. If a chunk returns
    fewer than `threshold` records, fetches location ID 4837117 for that same
    chunk and keeps whichever location returned more records.

    Returns a dict with keys: location_id, location_name, sensor_ids,
    latest (empty for backfill), hourly (dict of param -> list of records).
    """
    PRIMARY_LOC_ID = 4814327
    FALLBACK_LOC_ID = 4837117
    LOCATION_IDS = [PRIMARY_LOC_ID, FALLBACK_LOC_ID]

    if api_key is None:
        logger.warning(
            "OpenAQ API key not provided — skipping OpenAQ historical fetch. "
            "Set OPENAQ_API_KEY env var or pass api_key explicitly."
        )
        return {
            "location_id": PRIMARY_LOC_ID,
            "location_name": None,
            "sensor_ids": {},
            "latest": {},
            "hourly": {},
        }

    headers = {"X-API-Key": api_key}
    base_oaq = OPENAQ_V3_URL

    # ------------------------------------------------------------------
    # Fetch metadata & sensor IDs for each location
    # ------------------------------------------------------------------
    location_sensor_ids: Dict[int, int] = {}
    for loc_id in LOCATION_IDS:
        try:
            loc_res = _request_with_retry(
                "GET", f"{base_oaq}/locations/{loc_id}", headers=headers, timeout=30
            ).json()
        except Exception as exc:
            logger.error(f"OpenAQ location fetch failed for {loc_id}: {exc}")
            continue

        results = loc_res.get("results", [])
        loc_info = results[0] if isinstance(results, list) and results else (results or {})

        sensors = loc_info.get("sensors", [])
        for s in sensors:
            param_name = s.get("parameter", {}).get("name", "").lower()
            if param_name == "pm25":
                location_sensor_ids[loc_id] = s["id"]
                break

        if loc_id not in location_sensor_ids:
            logger.warning(f"No pm25 sensor found for OpenAQ location {loc_id}")

    if not location_sensor_ids:
        logger.error("No valid OpenAQ locations with PM2.5 sensors found")
        return {
            "location_id": PRIMARY_LOC_ID,
            "location_name": None,
            "sensor_ids": {},
            "latest": {},
            "hourly": {},
        }

    # Helper to fetch a single chunk for a specific location
    def _fetch_chunk_for_location(
        loc_id: int, from_str: str, to_str: str
    ) -> List[Dict[str, Any]]:
        sensor_id = location_sensor_ids.get(loc_id)
        if not sensor_id:
            return []
        try:
            h_res = _request_with_retry(
                "GET",
                f"{base_oaq}/sensors/{sensor_id}/hours",
                params={
                    "datetime_from": from_str,
                    "datetime_to": to_str,
                    "limit": 1000,
                },
                headers=headers,
                timeout=60,
            )
            records = h_res.json().get("results", [])
            parsed = []
            for r in records:
                dt_str = r.get("period", {}).get("datetimeTo", {}).get("utc")
                val = r.get("value")
                if dt_str and val is not None:
                    parsed.append({"time": dt_str, "pm25": float(val)})
            logger.info(f"  Location {loc_id} returned {len(parsed)} valid records for chunk")
            return parsed
        except Exception as exc:
            logger.warning(
                f"OpenAQ monthly chunk fetch failed for {loc_id} ({from_str} to {to_str}): {exc}"
            )
            return []

    # ------------------------------------------------------------------
    # Parse date range
    # ------------------------------------------------------------------
    if start_date is None:
        dt_from = datetime.now(KARACHI_TZ) - timedelta(days=BACKFILL_DAYS)
    else:
        dt_from = _parse_as_kachi(start_date)

    if end_date is None:
        dt_to = datetime.now(KARACHI_TZ)
    else:
        dt_to = _parse_as_kachi(end_date)

    # ------------------------------------------------------------------
    # Build monthly date chunks (30 days per request)
    # ------------------------------------------------------------------
    combined_records: List[Dict[str, Any]] = []
    usage_counts = {PRIMARY_LOC_ID: 0, FALLBACK_LOC_ID: 0}

    chunk_start = dt_from
    while chunk_start < dt_to:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), dt_to)
        # OpenAQ v3 requires UTC RFC3339 strings; convert from Karachi time
        chunk_from_str = chunk_start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        chunk_to_str = chunk_end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(
            f"Fetching OpenAQ PM2.5 monthly chunk: {chunk_from_str} to {chunk_to_str}"
        )

        # 1. Always fetch primary location first
        primary_records = _fetch_chunk_for_location(
            PRIMARY_LOC_ID, chunk_from_str, chunk_to_str
        )

        # 2. Only fetch fallback if primary is below threshold
        if len(primary_records) < threshold and FALLBACK_LOC_ID in location_sensor_ids:
            logger.info(
                f"  Primary location {PRIMARY_LOC_ID} returned {len(primary_records)} "
                f"(< {threshold}); checking fallback {FALLBACK_LOC_ID}..."
            )
            fallback_records = _fetch_chunk_for_location(
                FALLBACK_LOC_ID, chunk_from_str, chunk_to_str
            )

            # Compare both locations for this chunk and pick the one with more records
            if len(fallback_records) > len(primary_records):
                logger.info(
                    f"  Using fallback {FALLBACK_LOC_ID} ({len(fallback_records)} vs {len(primary_records)})"
                )
                combined_records.extend(fallback_records)
                usage_counts[FALLBACK_LOC_ID] += 1
            else:
                logger.info(
                    f"  Keeping primary {PRIMARY_LOC_ID} ({len(primary_records)} vs {len(fallback_records)})"
                )
                combined_records.extend(primary_records)
                usage_counts[PRIMARY_LOC_ID] += 1
        else:
            combined_records.extend(primary_records)
            usage_counts[PRIMARY_LOC_ID] += 1

        chunk_start = chunk_end

    # ------------------------------------------------------------------
    # Deduplicate by time across all chosen chunks & sort chronologically
    # ------------------------------------------------------------------
    seen_times = set()
    final_records = []
    for rec in combined_records:
        t = rec["time"]
        if t not in seen_times:
            seen_times.add(t)
            final_records.append(rec)

    final_records.sort(key=lambda r: r["time"])

    # Report whichever location contributed more chunks as the primary metadata ID
    dominant_loc_id = max(usage_counts, key=usage_counts.get)
    if dominant_loc_id not in location_sensor_ids:
        dominant_loc_id = next(iter(location_sensor_ids))

    logger.info(
        f"OpenAQ historical fetched: {len(final_records)} unique hourly PM2.5 records "
        f"({final_records[0]['time'] if final_records else 'N/A'} to "
        f"{final_records[-1]['time'] if final_records else 'N/A'}), "
        f"chunk usage: {usage_counts}"
    )

    return {
        "location_id": dominant_loc_id,
        "location_name": None,
        "sensor_ids": {"pm25": location_sensor_ids[dominant_loc_id]},
        "latest": {},
        "hourly": {"pm25": final_records},
    }


def fetch_all_sources_historical(
    lat: float,
    lon: float,
    openaq_api_key: Optional[str] = None,
    start_date: str = None,
    end_date: str = None,
) -> Dict[str, Any]:
    """
    Fetch historical data from all three sources and return a combined dict.

    Returns
    -------
    dict with keys: "om_weather", "om_aq", "openaq"
    """
    om_weather = fetch_openmeteo_weather(lat, lon, fetch_history_data=True, start_date=start_date, end_date=end_date)
    om_aq = fetch_openmeteo_aqi(lat, lon, fetch_history_data=True, start_date=start_date, end_date=end_date)
    openaq = fetch_openaq_historical(
        api_key=openaq_api_key, start_date=start_date, end_date=end_date
    )

    return {
        "om_weather": om_weather,
        "om_aq": om_aq,
        "openaq": openaq,
    }


# def fetch_openmeteo_weather_forecast_tail(
#     lat: float,
#     lon: float,
#     past_days: int = FORECAST_PAST_DAYS,
# ) -> Dict[str, Any]:
#     """
#     Fetch the recent past (tail) of weather data from the Open-Meteo Forecast API.

#     Uses ``past_days`` (max 7) to pull the most recent hourly history that the
#     archive endpoint has not yet caught up to. ``forecast_hours`` is omitted so
#     the response covers the past window; future forecast hours are dropped by
#     the caller.
#     """
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "hourly": [
#             "temperature_2m",
#             "relative_humidity_2m",
#             "precipitation",
#             "precipitation_probability",
#             "rain",
#             "wind_speed_10m",
#             "wind_direction_10m",
#         ],
#         "past_days": past_days,
#         "timezone": "UTC",
#     }
#     try:
#         resp = _request_with_retry("GET", OM_WEATHER_URL, params=params, timeout=60)
#         resp.raise_for_status()
#         data = resp.json()
#         hourly = data.get("hourly", {})
#         times = hourly.get("time", [])
#         logger.info(
#             f"Open-Meteo Forecast tail (past_days={past_days}) fetched: "
#             f"{len(times)} hourly records "
#             f"({times[0]} to {times[-1] if times else 'N/A'})"
#         )
#         return data
#     except Exception as exc:
#         logger.error(f"Open-Meteo Forecast tail fetch failed: {exc}")
#         return {}


# def fetch_openmeteo_aq_forecast_tail(
#     lat: float,
#     lon: float,
#     past_days: int = FORECAST_PAST_DAYS,
# ) -> Dict[str, Any]:
#     """
#     Fetch the recent past (tail) of air quality (UV index) from the
#     Open-Meteo Air Quality Forecast API using ``past_days``.
#     """
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "hourly": ["uv_index"],
#         "past_days": past_days,
#         "timezone": "UTC",
#     }
#     try:
#         resp = _request_with_retry("GET", OM_AQ_URL, params=params, timeout=60)
#         resp.raise_for_status()
#         data = resp.json()
#         hourly = data.get("hourly", {})
#         times = hourly.get("time", [])
#         logger.info(
#             f"Open-Meteo AQ Forecast tail (past_days={past_days}) fetched: "
#             f"{len(times)} hourly records "
#             f"({times[0]} to {times[-1] if times else 'N/A'})"
#         )
#         return data
#     except Exception as exc:
#         logger.error(f"Open-Meteo AQ Forecast tail fetch failed: {exc}")
#         return {}


# ---------------------------------------------------------------------------
# Feature Building (historical)
# ---------------------------------------------------------------------------
def build_features_historical(raw: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a feature DataFrame from 90 days of historical API responses.

    Combines Open-Meteo Weather, Open-Meteo Air Quality, and OpenAQ data
    into a unified feature set with one row per hour.

    Metrics:
      - Temperature (°C)
      - Precipitation (mm) and rain probability (%)
      - Humidity (%)
      - UV Index (from Open-Meteo AQ API)
      - Wind speed (m/s) and direction (degrees)
      - PM2.5 (from OpenAQ API)
      - AQI (calculated from PM2.5 using US EPA breakpoints)

    Also adds temporal, cyclical, rolling, lag/difference, and
    domain-specific interaction features.
    """
    w_hourly = raw.get("om_weather", {}).get("hourly", {})
    aq_hourly = raw.get("om_aq", {}).get("hourly", {})
    openaq_data = raw.get("openaq", {})

    times = w_hourly.get("time", [])
    if not times:
        logger.warning("No hourly weather data found — returning empty DataFrame.")
        return pd.DataFrame()

    # Build base DataFrame from Open-Meteo weather hourly data
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(times),
        "temp_c": w_hourly.get("temperature_2m", [np.nan] * len(times)),
        "humidity_pct": w_hourly.get("relative_humidity_2m", [np.nan] * len(times)),
        "precip_mm": w_hourly.get("precipitation", [np.nan] * len(times)),
        "rain_prob_pct": w_hourly.get("rain", [np.nan] * len(times)),
        # "rain_mm": w_hourly.get("rain", [np.nan] * len(times)),
        "wind_speed_ms": w_hourly.get("wind_speed_10m", [np.nan] * len(times)),
        "wind_dir_deg": w_hourly.get("wind_direction_10m", [np.nan] * len(times)),
    })

    # Floor timestamps to top of hour
    df["timestamp"] = df["timestamp"].dt.floor("h")

    # Merge UV index from Open-Meteo Air Quality
    aq_times = aq_hourly.get("time", [])
    aq_uv = aq_hourly.get("uv_index", [])
    if aq_times and aq_uv:
        aq_df = pd.DataFrame({
            "timestamp": pd.to_datetime(aq_times),
            "uv_index": aq_uv,
        })
        aq_df["timestamp"] = aq_df["timestamp"].dt.floor("h")
        df = df.merge(aq_df, on="timestamp", how="left")
    else:
        df["uv_index"] = np.nan

    # Merge PM2.5 from OpenAQ
    openaq_hourly = openaq_data.get("hourly", {})
    openaq_pm25_records = openaq_hourly.get("pm25", [])
    if openaq_pm25_records:
        openaq_df = pd.DataFrame(openaq_pm25_records)
        if not openaq_df.empty and "time" in openaq_df.columns and "pm25" in openaq_df.columns:
            openaq_df["timestamp"] = (
                pd.to_datetime(openaq_df["time"], utc=True)
                .dt.tz_convert(KARACHI_TZ)
                .dt.tz_localize(None)
                .dt.floor("h")
            )
            # Join as pm2_5 so downstream (lag/roll/AQI) uses a real column,
            # not the all-NaN placeholder the guard below would otherwise add.
            openaq_df = openaq_df.set_index("timestamp")[["pm25"]].rename(columns={"pm25": "pm2_5"})
            df = df.set_index("timestamp")
            df = df.join(openaq_df, how="left")
            df = df.reset_index()
        else:
            df["pm2_5"] = np.nan
    else:
        df["pm2_5"] = np.nan

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Ensure pm2_5 column exists (may be all NaN if OpenAQ data misaligned)
    if "pm2_5" not in df.columns:
        df["pm2_5"] = np.nan

    logger.info(
        f"Base historical DataFrame: {len(df)} rows "
        f"({df['timestamp'].min()} to {df['timestamp'].max()})"
    )
    return df


# ---------------------------------------------------------------------------
# Feature Engineering (historical)
# ---------------------------------------------------------------------------
def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the historical DataFrame.

    Adds temporal, cyclical, lag/difference, rolling, and
    domain-specific interaction features.
    """
    # ------------------------------------------------------------------
    # Temporal / cyclical features
    # ------------------------------------------------------------------
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Wind direction cyclical
    wind_dir_rad = np.deg2rad(df["wind_dir_deg"].fillna(0.0))
    df["wind_dir_sin"] = np.sin(wind_dir_rad)
    df["wind_dir_cos"] = np.cos(wind_dir_rad)

    # Source column -> short feature name used in feature_columns (and by
    # feature_pipeline.py), so these survive the final column filter.
    _col_alias = {
        "temp_c": "temp",
        "pm2_5": "pm2_5",
        "humidity_pct": "humidity",
        "wind_speed_ms": "wind_speed",
        "uv_index": "uv",
    }

    # ------------------------------------------------------------------
    # Lag / difference features (1h, 3h, 24h)
    # ------------------------------------------------------------------
    for col, periods in [
        ("temp_c", [1, 3]),
        ("pm2_5", [1, 3]),
        ("humidity_pct", [1, 3]),
        ("wind_speed_ms", [1, 3]),
        ("uv_index", [1]),
    ]:
        short = _col_alias[col]
        for p in periods:
            df[f"{short}_diff_{p}h"] = df[col].diff(periods=p)

    # ------------------------------------------------------------------
    # Rolling window features (6h, 24h)
    # ------------------------------------------------------------------
    for col, windows in [
        ("temp_c", [6, 24]),
        ("pm2_5", [6, 24]),
        ("humidity_pct", [6, 24]),
        ("wind_speed_ms", [6, 24]),
    ]:
        short = _col_alias[col]
        for w in windows:
            df[f"{short}_roll_mean_{w}h"] = df[col].rolling(window=w, min_periods=1).mean()

    df["precip_roll_sum_6h"] = df["precip_mm"].rolling(window=6, min_periods=1).sum()
    df["precip_roll_sum_24h"] = df["precip_mm"].rolling(window=24, min_periods=1).sum()

    df["pm2_5_roll_std_6h"] = df["pm2_5"].rolling(window=6, min_periods=1).std()
    df["temp_roll_std_6h"] = df["temp_c"].rolling(window=6, min_periods=1).std()

    # ------------------------------------------------------------------
    # Domain-specific interaction features
    # ------------------------------------------------------------------
    df["pollution_dispersion_idx"] = np.where(
        df["pm2_5"].notna() & df["wind_speed_ms"].notna(),
        df["pm2_5"] / (df["wind_speed_ms"] + 0.5),
        np.nan,
    )

    df["temp_humidity_index"] = np.where(
        df["temp_c"].notna() & df["humidity_pct"].notna(),
        df["temp_c"] * (df["humidity_pct"] / 100.0),
        np.nan,
    )

    # ------------------------------------------------------------------
    # AQI calculation from PM2.5
    # ------------------------------------------------------------------
    aqi_results = df["pm2_5"].apply(pm25_to_aqi)
    df["aqi"] = [r["aqi"] for r in aqi_results]

    # ------------------------------------------------------------------
    # Final column ordering
    # ------------------------------------------------------------------
    feature_columns = [
        "timestamp",
        "temp_c",
        "humidity_pct",
        "precip_mm",
        "rain_prob_pct",
        "uv_index",
        "wind_speed_ms",
        "wind_dir_deg",
        "pm2_5",
        "aqi",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "wind_dir_sin",
        "wind_dir_cos",
        "temp_diff_1h",
        "temp_diff_3h",
        "pm2_5_diff_1h",
        "pm2_5_diff_3h",
        "humidity_diff_1h",
        "humidity_diff_3h",
        "wind_speed_diff_1h",
        "wind_speed_diff_3h",
        "uv_diff_1h",
        "temp_roll_mean_6h",
        "temp_roll_mean_24h",
        "pm2_5_roll_mean_6h",
        "pm2_5_roll_mean_24h",
        "humidity_roll_mean_6h",
        "humidity_roll_mean_24h",
        "wind_speed_roll_mean_6h",
        "wind_speed_roll_mean_24h",
        "precip_roll_sum_6h",
        "precip_roll_sum_24h",
        "pm2_5_roll_std_6h",
        "temp_roll_std_6h",
        "pollution_dispersion_idx",
        "temp_humidity_index",
    ]

    # Keep only columns that exist
    existing_columns = [c for c in feature_columns if c in df.columns]
    df = df[existing_columns]

    # Final safety: fill any remaining boundary nulls in diff/lag/roll features
    # with 0 (no change / no variance at the start of the series).
    _fill_zero_cols = [
        c for c in df.columns
        if c not in ("timestamp",) and df[c].isna().any()
    ]
    if _fill_zero_cols:
        df[_fill_zero_cols] = df[_fill_zero_cols].fillna(0.0)

    logger.info(
        f"Final historical feature DataFrame: {len(df)} rows, {len(df.columns)} columns"
    )
    return df


# ---------------------------------------------------------------------------
# Missing-value imputation (historical backfill)
# ---------------------------------------------------------------------------
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute nulls using domain-appropriate strategies before feature engineering.

    Strategies (per feature):
      * temp_c        -> time-based linear interpolation
      * humidity_pct  -> linear interpolation, clipped to 0–100
      * precip_mm     -> zero-fill (missing rain = no rain)
      * rain_prob_pct -> linear interpolation
      * uv_index      -> interpolate daytime gaps; edge NaNs (likely night) -> 0
      * wind_speed_ms -> linear interpolation
      * wind_dir_deg  -> sin/cos transform -> interpolate -> convert back
                         (avoids averaging 350° and 10° into 180°)
      * pm2_5         -> linear interpolation limited to 6-hour gaps;
                         longer gaps stay NaN

    """
    df = df.copy()
    df = df.set_index("timestamp")

    # ------------------------------------------------------------------
    # temp_c: time-based linear interpolation
    # ------------------------------------------------------------------
    if "temp_c" in df.columns:
        df["temp_c"] = df["temp_c"].interpolate(method="time").bfill().ffill()

    # ------------------------------------------------------------------
    # humidity_pct: linear interpolation, clipped to 0–100
    # ------------------------------------------------------------------
    if "humidity_pct" in df.columns:
        df["humidity_pct"] = (
            df["humidity_pct"].interpolate(method="time").clip(0.0, 100.0).bfill().ffill()
        )

    # ------------------------------------------------------------------
    # precip_mm: zero-fill (missing rain reading means no rain occurred)
    # ------------------------------------------------------------------
    if "precip_mm" in df.columns:
        df["precip_mm"] = df["precip_mm"].fillna(0.0)

    # ------------------------------------------------------------------
    # rain_prob_pct: linear interpolation
    # ------------------------------------------------------------------
    if "rain_prob_pct" in df.columns:
        df["rain_prob_pct"] = df["rain_prob_pct"].interpolate(method="time").bfill().ffill()

    # ------------------------------------------------------------------
    # uv_index: interpolate daytime gaps; edge NaNs (likely night) -> 0
    # ------------------------------------------------------------------
    if "uv_index" in df.columns:
        df["uv_index"] = df["uv_index"].interpolate(method="time").fillna(0.0)

    # ------------------------------------------------------------------
    # wind_speed_ms: linear interpolation
    # ------------------------------------------------------------------
    if "wind_speed_ms" in df.columns:
        df["wind_speed_ms"] = df["wind_speed_ms"].interpolate(method="time").bfill().ffill()

    # ------------------------------------------------------------------
    # wind_dir_deg: sin/cos transform -> interpolate -> convert back
    # Avoids averaging 350° and 10° into 180° (opposite direction).
    # ------------------------------------------------------------------
    if "wind_dir_deg" in df.columns:
        wind_rad = np.deg2rad(df["wind_dir_deg"])
        wind_sin = np.sin(wind_rad)
        wind_cos = np.cos(wind_rad)
        wind_sin = wind_sin.interpolate(method="time").bfill().ffill()
        wind_cos = wind_cos.interpolate(method="time").bfill().ffill()
        df["wind_dir_deg"] = (np.rad2deg(np.arctan2(wind_sin, wind_cos)) % 360.0)

    # ------------------------------------------------------------------
    # pm2_5: Hour-of-Day Historical Median (grouped by hour and day_of_week
    # over a 30-day rolling window, causal/past-only)
    # For each missing value at timestamp T:
    #   - find rows where hour == T.hour and day_of_week == T.dayofweek
    #   - restrict to timestamps in [T - 30 days, T)
    #   - take the median of pm2_5 from those rows
    # ------------------------------------------------------------------
    if "pm2_5" in df.columns:
        # hour / day_of_week are not yet added (add_historical_features runs later),
        # so compute them from the DatetimeIndex for this imputation pass.
        df["_hour"] = df.index.hour
        df["_dow"] = df.index.dayofweek

        ref = df[df["pm2_5"].notna()][["pm2_5", "_hour", "_dow"]].copy()
        grouped = ref.groupby(["_hour", "_dow"])

        window = pd.Timedelta(days=30)
        missing_mask = df["pm2_5"].isna()

        for idx in df[missing_mask].index:
            hour = int(df.loc[idx, "_hour"])
            dow = int(df.loc[idx, "_dow"])
            window_start = idx - window

            try:
                group = grouped.get_group((hour, dow))
            except KeyError:
                continue

            hist = group[
                (group.index >= window_start) & (group.index < idx)
            ]["pm2_5"]

            if len(hist) > 0:
                df.loc[idx, "pm2_5"] = hist.median()

        # Final fallback: backfill any remaining edge nulls (e.g., first row
        # where no prior 30-day history exists).
        df["pm2_5"] = df["pm2_5"].bfill()

        df = df.drop(columns=["_hour", "_dow"])

    df = df.reset_index()
    return df


def main():
    """Entry point for running the history backfill script."""
    lat = float(os.getenv("WEATHER_LAT", LAT))
    lon = float(os.getenv("WEATHER_LON", LON))
    days = int(os.getenv("BACKFILL_DAYS", "365"))

    openaq_api_key = os.getenv("OPENAQ_API_KEY", None)
    if openaq_api_key:
        openaq_api_key = openaq_api_key.strip().strip('"').strip("'")

    # Compute date range in Karachi time so the backfill covers the right
    # local calendar dates.
    end_date = datetime.now(KARACHI_TZ).strftime("%Y-%m-%d")
    start_date = (datetime.now(KARACHI_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")

    logger.info(
        f"Starting history backfill for lat={lat}, lon={lon}, "
        f"start_date={start_date}, end_date={end_date}, days={days}"
    )

    # ------------------------------------------------------------------
    # Archive (lags ARCHIVE_LAG_DAYS behind today) + forecast tail fill
    # ------------------------------------------------------------------
    raw_archive = fetch_all_sources_historical(
        lat, lon, openaq_api_key=openaq_api_key,
        start_date=start_date, end_date=end_date,
    )

    raw_forecast_tail = {
        "om_weather": fetch_openmeteo_weather(lat, lon,past_days=FORECAST_PAST_DAYS,fetch_days=True),
        "om_aq": fetch_openmeteo_aqi(lat, lon,past_days=FORECAST_PAST_DAYS,fetch_days=True),
        "openaq": {},
    }

    # Save raw response for debugging
    raw_path = "raw_historical_response.json"
    with open(raw_path, "w") as f:
        json.dump(
            {"archive": raw_archive, "forecast_tail": raw_forecast_tail},
            f, indent=2, default=str,
        )
    logger.info(f"Raw historical API response saved to {raw_path}")

    # Build feature frames from both, then combine so archive wins wherever both
    # exist and the forecast tail fills the archive's recent gap.
    archive_df = build_features_historical(raw_archive)
    forecast_df = build_features_historical(raw_forecast_tail)

    # Drop future forecast hours so the backfill only covers up to now.
    # timestamp is tz-naive Karachi time, so keep now_khi tz-naive for the comparison.
    if not forecast_df.empty:
        now_khi = pd.Timestamp.now(tz=KARACHI_TZ).floor("h").tz_localize(None)
        forecast_df = forecast_df[forecast_df["timestamp"] <= now_khi]

    if not archive_df.empty and not forecast_df.empty:
        archive_i = archive_df.set_index("timestamp")
        forecast_i = forecast_df.set_index("timestamp")
        df = archive_i.combine_first(forecast_i).reset_index()
        logger.info(
            f"Combined archive ({len(archive_i)} rows) + forecast tail "
            f"({len(forecast_i)} rows) -> {len(df)} rows "
            "(archive wins where both exist)"
        )
    elif not archive_df.empty:
        df = archive_df
        logger.info("Using archive data only (forecast tail empty).")
    elif not forecast_df.empty:
        df = forecast_df
        logger.info("Using forecast tail only (archive empty).")
    else:
        df = pd.DataFrame()

    if df.empty:
        logger.error("No data fetched — aborting backfill.")
        return

    # Impute missing values before feature engineering
    df = impute_missing_values(df)

    # Add engineered features
    df = add_historical_features(df)

    # Save to CSV
    features_path = "weather_historical_features.csv"
    df.to_csv(features_path, index=False)
    logger.info(f"Historical features saved to {features_path} ({len(df)} rows)")

    # Print summary
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== Historical Feature Summary ===")
    print(df.head(3))
    print("\n=== Date Range ===")
    print(f"  From: {df['timestamp'].min()}")
    print(f"  To:   {df['timestamp'].max()}")
    print(f"  Rows: {len(df)}")

    print("\n=== Key Metrics ===")
    key_cols = [
        "aqi", "temp_c", "precip_mm", "rain_prob_pct",
        "humidity_pct", "uv_index", "wind_speed_ms", "wind_dir_deg",
    ]
    for col in key_cols:
        if col in df.columns:
            print(f"  {col:20s}: mean={df[col].mean():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")

    # ------------------------------------------------------------------
    # Cast dtypes to match Hopsworks feature group schema
    # Pandas infers bigint from whole-number API values, but Hopsworks
    # expects double for humidity/wind_dir and bigint (int64) for
    # hour/day_of_week/month.
    # ------------------------------------------------------------------
    for col in ("humidity_pct", "wind_dir_deg"):
        if col in df.columns:
            df[col] = df[col].astype("float64")
    for col in ("hour", "day_of_week", "month"):
        if col in df.columns:
            df[col] = df[col].astype("int64")

    # Save to Hopsworks Feature Store
    save_to_hopsworks(df)

    print("\n=== Backfill Complete ===")
    return df


if __name__ == "__main__":
    main()