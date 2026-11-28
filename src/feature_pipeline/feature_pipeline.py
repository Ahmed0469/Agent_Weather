"""
Feature Pipeline Script

Fetches weather and air quality metrics from:
  1. Open-Meteo Weather Forecast API  -> Temperature, Precipitation (mm & probability),
     Humidity, Wind (speed & direction)
  2. Open-Meteo Air Quality API       -> UV Index only
  3. OpenAQ API (location 4837117)    -> PM2.5 (latest + hourly history)

Metrics collected (as per project brief):
  - Air Quality Index (AQI)
  - Temperature (°C)
  - Precipitation (mm and rain probability)
  - Humidity (%)
  - UV Index
  - Wind (speed and direction)

Author: Agent-Weather
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Karachi time zone (Asia/Karachi, UTC+5, no DST) — used for all timestamps
# in this pipeline and in history_backfill.py (imports KARACHI_TZ from here).
KARACHI_TZ = ZoneInfo("Asia/Karachi")

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feature_pipeline")


# ---------------------------------------------------------------------------
# HTTP retry helper
# ---------------------------------------------------------------------------
def _request_with_retry(method: str, url: str, max_retries: int = 3, backoff_factor: float = 2.0, **kwargs):
    """Make an HTTP request with retry logic for transient failures.

    Retries on timeout and connection errors with exponential backoff.
    """
    timeout = kwargs.pop("timeout", 30)
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt < max_retries - 1:
                wait = backoff_factor ** attempt
                logger.warning(
                    f"Request attempt {attempt + 1}/{max_retries} failed: {exc}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"Request failed after {max_retries} attempts: {exc}")
                raise
        except requests.exceptions.RequestException as exc:
            logger.error(f"Request failed: {exc}")
            raise

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
OM_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OM_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENAQ_V3_URL = "https://api.openaq.org/v3"

# ---------------------------------------------------------------------------
# Fetch Functions
# ---------------------------------------------------------------------------
def fetch_openmeteo_weather(
    lat: float,
    lon: float,
    past_hours: int = 24,
    forecast_hours: int = 1,
    past_days : int = 7,
    fetch_days : bool = False,
    fetch_history_data  : bool = False,
    start_date : str = None,
    end_date : str = None,
) -> Dict[str, Any]:
    """
    Fetch current and hourly weather data from Open-Meteo Forecast API.

    Returns the raw JSON response dict.
    """
    if fetch_history_data:
        params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "precipitation_probability",
                    "rain",
                    "wind_speed_10m",
                    "wind_direction_10m",
                ],
                "timezone": "Asia/Karachi",
            }
    elif fetch_days:
        params = {
                "latitude": lat,
                "longitude": lon,                
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "rain",                    
                    "wind_speed_10m",
                    "wind_direction_10m",
                ],
                "past_days": past_days,
                "timezone": "Asia/Karachi",
            }
    else:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "wind_speed_10m",
                "wind_direction_10m",
            ],
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "wind_speed_10m",
                "wind_direction_10m",
            ],
            "past_hours": past_hours,
            "forecast_hours": forecast_hours,
            "timezone": "Asia/Karachi",
        }
    if fetch_history_data:
        try:
            resp = _request_with_retry("GET", OM_ARCHIVE_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            logger.info(
                f"Open-Meteo Archive Weather fetched: {len(times)} hourly records "
                f"({times[0]} to {times[-1] if times else 'N/A'})"
            )
            return data
        except Exception as exc:
            logger.error(f"Open-Meteo Archive Weather fetch failed: {exc}")
            return {}
    else:
        try:
            resp = _request_with_retry("GET", OM_WEATHER_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            logger.info("Open-Meteo Weather fetched successfully")
            return data
        except Exception as exc:
            logger.error(f"Open-Meteo Weather fetch failed: {exc}")
            return {}

def fetch_openmeteo_aqi(
    lat: float,
    lon: float,
    past_hours: int = 24,
    forecast_hours: int = 1,
    past_days : int = 7,
    fetch_days : bool = False,
    fetch_history_data  : bool = False,
    start_date : str = None,
    end_date : str = None,
) -> Dict[str, Any]:
    """
    Fetch current and hourly air quality data from Open-Meteo Air Quality API.

    Returns the raw JSON response dict.
    """
    if fetch_history_data:
        params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ["uv_index"],
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "Asia/Karachi",
            }
    elif fetch_days:
        params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ["uv_index"],
                "past_days": past_days,
                "timezone": "Asia/Karachi",
            }
    else:        
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "uv_index",
            ],
            "hourly": [
                "uv_index",
            ],
            "past_hours": past_hours,
            "forecast_hours": forecast_hours,
            "timezone": "Asia/Karachi",
        }
    try:
        resp = _request_with_retry("GET", OM_AQ_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        logger.info("Open-Meteo Air Quality fetched successfully")
        return data
    except Exception as exc:
        logger.error(f"Open-Meteo Air Quality fetch failed: {exc}")
        return {}

def fetch_openaq(
    location_id: int = 4814327,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch latest and hourly air quality data from OpenAQ API v3
    for a specific location ID (default 4837117), pm25 only.

    Uses the known PM2.5 sensor ID (13406434) directly, avoiding
    an extra API call to discover sensor metadata.

    Requires an API key for v3. If no key is provided, returns an empty dict.

    Parameters
    ----------
    location_id : int
        OpenAQ location ID (default 4837117).
    api_key : str, optional
        OpenAQ API v3 key.

    Returns
    -------
    dict
        Payload with keys: location_id, location_name, sensor_ids,
        latest (dict of param -> value), hourly (dict of param -> list of records).
    """
    # PM25_SENSOR_ID = 13406434  # Known PM2.5 sensor ID for location 4837117

    if api_key is None:
        logger.warning(
            "OpenAQ API key not provided — skipping OpenAQ fetch. "
            "Set OPENAQ_API_KEY env var or pass api_key explicitly."
        )
        return {}

    headers = {"X-API-Key": api_key}
    base_oaq = OPENAQ_V3_URL

    # ------------------------------------------------------------------
    # Time window for hourly history (past 26 hours), anchored to Karachi
    # now. OpenAQ v3 requires UTC RFC3339 strings, so convert at the API
    # boundary.
    # ------------------------------------------------------------------
    now_khi = datetime.now(KARACHI_TZ)
    dt_from = (now_khi - timedelta(hours=26)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dt_to = now_khi.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    openaq_payload = {
        "location_id": location_id,
        "location_name": None,
        "sensor_ids": {"pm25": None},
        "latest": {},
        "hourly": {},
    }

    # ------------------------------------------------------------------
    # Fetch latest measurements for the location
    # ------------------------------------------------------------------
    try:
        loc_res = _request_with_retry(
            "GET", f"{base_oaq}/locations/{location_id}", headers=headers, timeout=30
        ).json()
    except Exception as exc:
        logger.error(f"OpenAQ location fetch failed: {exc}")
        return {}

    results = loc_res.get("results", [])
    loc_info = results if results else {}
    if isinstance(loc_info, list):
        loc_info = loc_info[0] if loc_info else {}

    sensors = loc_info.get("sensors", [])

    sensor_ids = {}  # e.g., {'pm25': 12345}
    for s in sensors:
        param_name = s.get("parameter", {}).get("name", "").lower()
        if param_name == "pm25" and param_name not in sensor_ids:
            sensor_ids[param_name] = s["id"]

    if not sensor_ids:
        logger.warning("No pm25 sensor found for OpenAQ location %s", location_id)
        return {}
    
    try:
        latest_res = _request_with_retry(
            "GET", f"{base_oaq}/locations/{location_id}/latest", headers=headers, timeout=30
        ).json()
    except Exception as exc:
        logger.error(f"OpenAQ latest fetch failed: {exc}")
        latest_res = {}

    id_to_param = {v: k for k, v in sensor_ids.items()}
    for item in latest_res.get("results", []):
        p_name = id_to_param.get(item.get("sensorsId"))
        if p_name:
            openaq_payload["latest"][p_name] = item.get("value")
            openaq_payload["sensor_ids"][p_name] = item.get("sensorsId")


    # ------------------------------------------------------------------
    # Fetch past 24h hourly history for PM2.5 sensor
    # ------------------------------------------------------------------
    try:
        h_res = _request_with_retry(
            "GET",
            f"{base_oaq}/sensors/{openaq_payload['sensor_ids']["pm25"]}/hours",
            params={
                "datetime_from": dt_from,
                "datetime_to": dt_to,
                "limit": 48,
            },
            headers=headers,
            timeout=30,
        ).json()
    except Exception as exc:
        logger.error(f"OpenAQ hourly fetch failed for pm25: {exc}")
        return openaq_payload

    records = []
    for r in h_res.get("results", []):
        dt_str = r.get("period", {}).get("datetimeTo", {}).get("utc")
        val = r.get("value")
        if dt_str and val is not None:
            records.append({"time": dt_str, "pm25": float(val)})
    openaq_payload["hourly"]["pm25"] = records

    logger.info("OpenAQ data fetched successfully for location %s", location_id)
    return openaq_payload


def fetch_all_sources(
    lat: float,
    lon: float,
    openaq_api_key: Optional[str] = None,
    past_hours: int = 24,
    forecast_hours: int = 1,
) -> Dict[str, Any]:
    """
    Fetch data from all three sources and return a combined dict.

    Returns
    -------
    dict with keys: "om_weather", "om_aq", "openaq"
    """
    om_weather = fetch_openmeteo_weather(lat, lon, past_hours, forecast_hours)
    om_aq = fetch_openmeteo_aqi(lat, lon, past_hours, forecast_hours)
    # openaq = fetch_openaq(location_id=4837117, api_key=openaq_api_key)
    openaq = fetch_openaq(location_id=4814327, api_key=openaq_api_key)

    return {
        "om_weather": om_weather,
        "om_aq": om_aq,
        "openaq": openaq,
    }


# ---------------------------------------------------------------------------
# Feature Building
# ---------------------------------------------------------------------------
def _safe_float(val: Any, default: float = np.nan) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def pm25_to_aqi(pm25: float) -> dict:
    """
    Convert PM2.5 concentration to US EPA AQI value and category.

    Mirrors the JavaScript pm25ToAQI logic:
      - Truncate to 1 decimal place per EPA rules
      - Linear interpolation within breakpoint bands
      - Returns {aqi, category}

    Parameters
    ----------
    pm25 : float
        PM2.5 concentration in µg/m³.

    Returns
    -------
    dict
        {'aqi': int, 'category': str}
    """
    if pm25 is None or np.isnan(pm25):
        return {"aqi": np.nan, "category": "Unknown"}

    # Truncate to 1 decimal place per EPA rules
    c = np.floor(pm25 * 10) / 10

    breakpoints = [
        {"cLow": 0.0, "cHigh": 9.0, "iLow": 0, "iHigh": 50, "category": "Good"},
        {"cLow": 9.1, "cHigh": 35.4, "iLow": 51, "iHigh": 100, "category": "Moderate"},
        {"cLow": 35.5, "cHigh": 55.4, "iLow": 101, "iHigh": 150, "category": "Unhealthy for Sensitive Groups"},
        {"cLow": 55.5, "cHigh": 125.4, "iLow": 151, "iHigh": 200, "category": "Unhealthy"},
        {"cLow": 125.5, "cHigh": 225.4, "iLow": 201, "iHigh": 300, "category": "Very Unhealthy"},
        {"cLow": 225.5, "cHigh": 500.4, "iLow": 301, "iHigh": 500, "category": "Hazardous"},
    ]

    bp = next((b for b in breakpoints if b["cLow"] <= c <= b["cHigh"]), None)
    if bp is None:
        return {"aqi": 500, "category": "Hazardous"}  # Above 500.4

    aqi = int(round(((bp["iHigh"] - bp["iLow"]) / (bp["cHigh"] - bp["cLow"])) * (c - bp["cLow"]) + bp["iLow"]))

    return {"aqi": aqi, "category": bp["category"]}


def impute_feature_values(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Impute nulls in the hourly history DataFrame using the same domain
    strategies as history_backfill.impute_missing_values, so the single
    current row built from `hist` has no missing metric values.

    Strategies (per feature):
      * temp_c        -> time-based linear interpolation, then bfill/ffill
      * humidity_pct  -> linear interpolation, clipped 0–100, then bfill/ffill
      * precip_mm     -> zero-fill (missing rain = no rain)
      * rain_prob_pct -> linear interpolation, then bfill/ffill
      * uv_index      -> interpolate; edge NaNs (likely night) -> 0
      * wind_speed_ms -> linear interpolation, then bfill/ffill
      * wind_dir_deg  -> sin/cos transform -> interpolate -> arctan2 -> %360
                         (avoids averaging 350° and 10° into 180°)
      * pm2_5         -> linear interpolation limited to 6-hour gaps,
                         then bfill/ffill for edge gaps
    """
    df = hist.copy()
    df.index = pd.to_datetime(df.index)

    if "temperature_2m" in df.columns:
        df["temperature_2m"] = (
            df["temperature_2m"].interpolate(method="time").bfill().ffill()
        )

    if "relative_humidity_2m" in df.columns:
        df["relative_humidity_2m"] = (
            df["relative_humidity_2m"]
            .interpolate(method="time")
            .clip(0.0, 100.0)
            .bfill()
            .ffill()
        )

    if "precipitation" in df.columns:
        # Missing rain reading means no rain occurred.
        df["precipitation"] = df["precipitation"].fillna(0.0)

    if "rain" in df.columns:
        df["rain"] = df["rain"].interpolate(method="time").bfill().ffill()

    if "uv_index" in df.columns:
        # Interpolate daytime gaps; edge NaNs (likely night) -> 0.
        df["uv_index"] = df["uv_index"].interpolate(method="time").fillna(0.0)

    if "wind_speed_10m" in df.columns:
        df["wind_speed_10m"] = (
            df["wind_speed_10m"].interpolate(method="time").bfill().ffill()
        )

    if "wind_direction_10m" in df.columns:
        # Sin/cos transform so 350° and 10° average near 0°, not 180°.
        wind_rad = np.deg2rad(df["wind_direction_10m"])
        wind_sin = np.sin(wind_rad)
        wind_cos = np.cos(wind_rad)
        wind_sin = wind_sin.interpolate(method="time").bfill().ffill()
        wind_cos = wind_cos.interpolate(method="time").bfill().ffill()
        df["wind_direction_10m"] = (
            np.rad2deg(np.arctan2(wind_sin, wind_cos)) % 360.0
        )

    if "pm2_5" in df.columns:
        # Linear interpolation for gaps < 6 hours; longer/edge gaps filled.
        df["pm2_5"] = (
            df["pm2_5"].interpolate(method="time", limit=6).bfill().ffill()
        )

    return df


def build_features(raw: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from the raw API responses.

    Combines Open-Meteo Weather, Open-Meteo Air Quality, and OpenAQ data
    into a unified feature set with the metrics from the project brief:

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
    w_curr = raw.get("om_weather", {}).get("current", {})
    aq_curr = raw.get("om_aq", {}).get("current", {})
    openaq_data = raw.get("openaq", {})

    timestamp_str = w_curr.get("time")
    timestamp = (
        pd.to_datetime(timestamp_str).floor("h")
        if timestamp_str
        else pd.Timestamp.now(tz=KARACHI_TZ).floor("h").tz_localize(None)
    )

    temp_c = _safe_float(w_curr.get("temperature_2m"))
    humidity_pct = _safe_float(w_curr.get("relative_humidity_2m"))
    precip_mm = _safe_float(w_curr.get("precipitation"))
    # rain_prob_pct = _safe_float(w_curr.get("precipitation_probability"))
    rain_mm = _safe_float(w_curr.get("rain"))
    wind_speed_ms = _safe_float(w_curr.get("wind_speed_10m"))
    wind_dir_deg = _safe_float(w_curr.get("wind_direction_10m"))

    om_aq_uv_index = _safe_float(aq_curr.get("uv_index"))

    openaq_pm2_5 = np.nan

    try:
        openaq_latest = openaq_data.get("latest", {})
        openaq_hourly = openaq_data.get("hourly", {})

        # Extract latest pm25 value from OpenAQ
        openaq_pm2_5 = _safe_float(openaq_latest.get("pm25"))
    except Exception as exc:
        logger.warning(f"OpenAQ parsing warning: {exc}")

    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month
    is_weekend = 1 if day_of_week >= 5 else 0

    # wind_dir_rad is recomputed below from the imputed base_row["wind_dir_deg"]
    # (after missing hourly values are filled in).
    wind_dir_rad = 0.0

    # precip_val = rain_mm if not np.isnan(rain_mm) else precip_mm

    base_row = {
        "timestamp": timestamp,
        "temp_c": temp_c,
        "humidity_pct": humidity_pct,
        "precip_mm": precip_mm,
        "rain_prob_pct": rain_mm,
        "uv_index": om_aq_uv_index,
        "wind_speed_ms": wind_speed_ms,
        "wind_dir_deg": wind_dir_deg,
        "pm2_5": openaq_pm2_5,
    }

    w_hourly = raw.get("om_weather", {}).get("hourly", {})
    aq_hourly = raw.get("om_aq", {}).get("hourly", {})

    times = w_hourly.get("time", [])
    if not times:
        return pd.DataFrame([base_row])

    hist = pd.DataFrame({
        "timestamp": pd.to_datetime(times),
        "temperature_2m": w_hourly.get("temperature_2m", [np.nan] * len(times)),
        "relative_humidity_2m": w_hourly.get("relative_humidity_2m", [np.nan] * len(times)),
        "precipitation": w_hourly.get("precipitation", [np.nan] * len(times)),
        "rain": w_hourly.get("rain", [np.nan] * len(times)),
        "wind_speed_10m": w_hourly.get("wind_speed_10m", [np.nan] * len(times)),
        "wind_direction_10m": w_hourly.get("wind_direction_10m", [np.nan] * len(times)),
    }).set_index("timestamp")
    
    # Merge OpenAQ pm2_5 hourly history into hist
    openaq_hourly = openaq_data.get("hourly", {})
    openaq_pm25_records = openaq_hourly.get("pm25", [])
    if openaq_pm25_records:
        openaq_df = pd.DataFrame(openaq_pm25_records)
        if not openaq_df.empty and "time" in openaq_df.columns and "pm25" in openaq_df.columns:
            openaq_df["timestamp"] = pd.to_datetime(openaq_df["time"]).dt.tz_localize(None)

            openaq_df = openaq_df.set_index("timestamp")[["pm25"]]
            # Reindex OpenAQ pm25 onto OpenMeteo's hourly timestamps
            # (OpenAQ may start 1 hour earlier/later; reindex + nearest handles misalignment)
            openaq_pm25_reindexed = openaq_df.reindex(hist.index, method="nearest", tolerance=pd.Timedelta("1h"))
            if "pm2_5" not in hist.columns:
                hist["pm2_5"] = openaq_pm25_reindexed["pm25"]
            else:
                hist["pm2_5"] = hist["pm2_5"].fillna(openaq_pm25_reindexed["pm25"])
    if "pm2_5" not in hist.columns:
        hist["pm2_5"] = np.nan

    # Add uv_index from OpenMeteo AQ hourly
    aq_hourly = raw.get("om_aq", {}).get("hourly", {})
    aq_times = aq_hourly.get("time", [])
    if aq_times and "uv_index" in aq_hourly:
        aq_uv_series = pd.Series(aq_hourly["uv_index"], index=pd.to_datetime(aq_times))
        hist = hist.join(aq_uv_series.rename("uv_index"), how="left")
    if "uv_index" not in hist.columns:
        hist["uv_index"] = np.nan

    # ------------------------------------------------------------------
    # Impute missing hourly values using the same domain strategies as the
    # history backfill script, so the current row / derived features have
    # no gaps.
    # ------------------------------------------------------------------
    hist = impute_feature_values(hist)

    # Fill any current-row NaNs from the imputed hourly history (latest value)
    if not hist.empty:
        _last = hist.iloc[-1]
        for _col, _hist_col in [
            ("temp_c", "temperature_2m"),
            ("humidity_pct", "relative_humidity_2m"),
            ("precip_mm", "precipitation"),
            ("rain_prob_pct", "rain"),
            ("uv_index", "uv_index"),
            ("wind_speed_ms", "wind_speed_10m"),
            ("wind_dir_deg", "wind_direction_10m"),
            ("pm2_5", "pm2_5"),
        ]:
            if _hist_col in _last.index and pd.isna(base_row[_col]):
                base_row[_col] = float(_last[_hist_col])

    def get_diff(series: pd.Series, periods: int) -> float:
        if len(series) < periods + 1:
            return np.nan
        return float(series.iloc[-1] - series.iloc[-1 - periods])

    def get_roll_mean(series: pd.Series, window: int) -> float:
        if len(series) < window:
            return np.nan
        return float(series.tail(window).mean())

    def get_roll_std(series: pd.Series, window: int) -> float:
        if len(series) < window:
            return np.nan
        return float(series.tail(window).std())

    # Recompute from imputed base_row wind_dir_deg (filled from hourly history).
    wind_dir_rad = (
        np.deg2rad(base_row["wind_dir_deg"])
        if not np.isnan(base_row["wind_dir_deg"])
        else 0.0
    )

    engineered_row = dict(base_row)

    engineered_row.update({
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
        "wind_dir_sin": np.sin(wind_dir_rad),
        "wind_dir_cos": np.cos(wind_dir_rad),
    })

    engineered_row.update({
        "temp_diff_1h": get_diff(hist["temperature_2m"], 1),
        "temp_diff_3h": get_diff(hist["temperature_2m"], 3),
        "pm2_5_diff_1h": get_diff(hist["pm2_5"], 1),
        "pm2_5_diff_3h": get_diff(hist["pm2_5"], 3),
        "humidity_diff_1h": get_diff(hist["relative_humidity_2m"], 1),
        "humidity_diff_3h": get_diff(hist["relative_humidity_2m"], 3),
        "wind_speed_diff_1h": get_diff(hist["wind_speed_10m"], 1),
        "wind_speed_diff_3h": get_diff(hist["wind_speed_10m"], 3),
        "uv_diff_1h": get_diff(hist["uv_index"], 1) if "uv_index" in hist.columns else np.nan,
    })

    engineered_row.update({
        "temp_roll_mean_6h": get_roll_mean(hist["temperature_2m"], 6),
        "temp_roll_mean_24h": get_roll_mean(hist["temperature_2m"], 24),
        "pm2_5_roll_mean_6h": get_roll_mean(hist["pm2_5"], 6),
        "pm2_5_roll_mean_24h": get_roll_mean(hist["pm2_5"], 24),
        "humidity_roll_mean_6h": get_roll_mean(hist["relative_humidity_2m"], 6),
        "humidity_roll_mean_24h": get_roll_mean(hist["relative_humidity_2m"], 24),
        "wind_speed_roll_mean_6h": get_roll_mean(hist["wind_speed_10m"], 6),
        "wind_speed_roll_mean_24h": get_roll_mean(hist["wind_speed_10m"], 24),
        "precip_roll_sum_6h": float(hist["precipitation"].tail(6).sum()) if len(hist) >= 6 else np.nan,
        "precip_roll_sum_24h": float(hist["precipitation"].tail(24).sum()) if len(hist) >= 24 else np.nan,
        "pm2_5_roll_std_6h": get_roll_std(hist["pm2_5"], 6),
        "temp_roll_std_6h": get_roll_std(hist["temperature_2m"], 6),
    })

    # Use the imputed pm2_5 (OpenAQ latest, filled from hourly history if missing).
    pm25_val = base_row["pm2_5"]

    aqi_result = pm25_to_aqi(pm25_val)

    engineered_row.update({
        "pollution_dispersion_idx": pm25_val / (wind_speed_ms + 0.5) if not np.isnan(pm25_val) else np.nan,
        "temp_humidity_index": temp_c * (humidity_pct / 100.0) if not np.isnan(temp_c) and not np.isnan(humidity_pct) else np.nan,
        "aqi": aqi_result["aqi"],
        # "aqi_category": aqi_result["category"],
    })

    return pd.DataFrame([engineered_row])


# ---------------------------------------------------------------------------
# Hopsworks Feature Store integration
# ---------------------------------------------------------------------------
def save_to_hopsworks(
    features_df: pd.DataFrame,
    feature_group_name: str = "weather_air_quality_features",
    feature_group_version: int = 1,
    batch_size: int = 10000,
) -> bool:
    """
    Save the feature DataFrame to the Hopsworks Feature Store.

    Requires hopsworks library and Hopsworks credentials via environment variables:
      - HOPSWORKS_API_KEY: Hopsworks API key
      - HOPSWORKS_PROJECT: Hopsworks project name
      - HOPSWORKS_HOST: (optional) Hopsworks host, defaults to cloud

    Inserts data in batches to avoid timeouts with large DataFrames.

    Parameters
    ----------
    features_df : pd.DataFrame
        The engineered features DataFrame (single row or many rows).
    feature_group_name : str
        Name of the feature group to write to.
    feature_group_version : int
        Version of the feature group.
    batch_size : int
        Number of rows per insert batch.

    Returns
    -------
    bool
        True if save succeeded, False otherwise.
    """
    try:
        import hopsworks
    except ImportError:
        logger.warning(
            "hopsworks not installed — skipping Hopsworks save. "
            "Install with: uv add hopsworks[python]"
        )
        return False

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")
    host = os.getenv("HOPSWORKS_HOST")

    # Clean values: strip whitespace and remove surrounding quotes (in case secrets were copy-pasted with quotes)
    if api_key:
        api_key = api_key.strip().strip('"').strip("'")
    if project_name:
        project_name = project_name.strip().strip('"').strip("'")
    if host:
        host = host.strip().strip('"').strip("'")

    if not api_key or not project_name:
        logger.warning(
            "HOPSWORKS_API_KEY or HOPSWORKS_PROJECT not set — "
            "skipping Hopsworks save."
        )
        return False

    if not host:
        logger.warning(
            "HOPSWORKS_HOST not set — using default Hopsworks cloud host."
        )

    try:
        # project = hopsworks.login(
        #     host=host,
        #     project=project_name,
        #     api_key=api_key,
        # )
        project = hopsworks.login(
            host=host,
            project=project_name,
            api_key_value=api_key
        )
        fs = project.get_feature_store()

        # Get or create the feature group (single method in the new API)
        fg = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=feature_group_version,
            primary_key=["timestamp"],
            description="Weather and air quality features from Open-Meteo and OpenAQ",
            online_enabled=True,
            time_travel_format="HUDI",
        )

        # Insert in batches to avoid timeouts with large DataFrames
        total_rows = len(features_df)
        num_batches = (total_rows + batch_size - 1) // batch_size

        logger.info(
            f"Inserting {total_rows} rows into Hopsworks in {num_batches} batch(es)"
        )

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = features_df.iloc[start_idx:end_idx]

            fg.insert(batch_df)
            logger.info(
                f"  Batch {i + 1}/{num_batches}: inserted {len(batch_df)} rows "
                f"({start_idx + 1}-{end_idx})"
            )

        logger.info(
            f"Features saved to Hopsworks feature group '{feature_group_name}' "
            f"v{feature_group_version} ({total_rows} total rows)"
        )
        return True

    except Exception as exc:
        logger.error(f"Hopsworks save failed: {exc}")
        return False


def _clean_secret(val: Optional[str]) -> Optional[str]:
    """Strip whitespace and surrounding quotes from an env-secret value."""
    if val is None:
        return None
    return val.strip().strip('"').strip("'")


def load_last_day_from_hopsworks(
    feature_group_name: str = "weather_air_quality_features",
    feature_group_version: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Read the Hopsworks feature group and return the feature row for the same
    hour on the previous day in Karachi time.

    Used as a fallback when the live weather APIs fail: the pipeline copies
    last day's current-hour data so it still emits a feature row.

    Returns None if Hopsworks is unavailable, has no credentials, or the
    feature group holds no matching row.
    """
    try:
        import hopsworks
    except ImportError:
        logger.warning("hopsworks not installed — cannot use Hopsworks fallback.")
        return None

    api_key = _clean_secret(os.getenv("HOPSWORKS_API_KEY"))
    project_name = _clean_secret(os.getenv("HOPSWORKS_PROJECT"))
    host = _clean_secret(os.getenv("HOPSWORKS_HOST"))

    if not api_key or not project_name:
        logger.warning(
            "HOPSWORKS_API_KEY or HOPSWORKS_PROJECT not set — "
            "cannot use Hopsworks fallback."
        )
        return None

    try:
        project = hopsworks.login(
            host=host,
            project=project_name,
            api_key_value=api_key,
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(
            name=feature_group_name,
            version=feature_group_version,
        )
        df = fg.read()
    except Exception as exc:
        logger.error(f"Hopsworks fallback read failed: {exc}")
        return None

    if df is None or df.empty or "timestamp" not in df.columns:
        logger.warning("Hopsworks fallback feature group is empty — no row to copy.")
        return None

    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(KARACHI_TZ).dt.tz_localize(None)
    df["timestamp"] = ts

    now_khi = pd.Timestamp.now(tz=KARACHI_TZ).floor("h").tz_localize(None)
    target = now_khi - pd.Timedelta(days=1)

    match = df[df["timestamp"] == target]
    if match.empty:
        # Nearest previous row (asof) if the exact hour is missing.
        prev = df.sort_values("timestamp")
        prev = prev[prev["timestamp"] <= target]
        if prev.empty:
            logger.warning("Hopsworks fallback: no row at/before last day's hour.")
            return None
        match = prev.tail(1)

    row = match.iloc[0].to_dict()
    logger.info(
        f"Hopsworks fallback: copying last-day data from {row['timestamp']}"
    )
    return row


def _synthesize_raw_from_fallback(
    fb: Dict[str, Any],
    now_khi,
) -> Dict[str, Any]:
    """
    Build a minimal raw payload from a Hopsworks fallback row so that
    build_features can emit a complete single-row feature DataFrame stamped
    with the current Karachi time but carrying last day's values.
    """
    now_str = now_khi.strftime("%Y-%m-%dT%H:%M")

    return {
        "om_weather": {
            "current": {
                "time": now_str,
                "temperature_2m": fb.get("temp_c"),
                "relative_humidity_2m": fb.get("humidity_pct"),
                "precipitation": fb.get("precip_mm"),
                "rain": fb.get("rain_prob_pct"),
                "wind_speed_10m": fb.get("wind_speed_ms"),
                "wind_direction_10m": fb.get("wind_dir_deg"),
            },
            "hourly": {
                "time": [now_str],
                "temperature_2m": [fb.get("temp_c")],
                "relative_humidity_2m": [fb.get("humidity_pct")],
                "precipitation": [fb.get("precip_mm")],
                "rain": [fb.get("rain_prob_pct")],
                "wind_speed_10m": [fb.get("wind_speed_ms")],
                "wind_direction_10m": [fb.get("wind_dir_deg")],
            },
        },
        "om_aq": {
            "current": {"uv_index": fb.get("uv_index")},
            "hourly": {"time": [now_str], "uv_index": [fb.get("uv_index")]},
        },
        "openaq": {
            "latest": {"pm25": fb.get("pm2_5")},
            "hourly": {"pm25": [{"time": now_str, "pm25": fb.get("pm2_5")}]},
        },
    }


def main():
    """Entry point for running the pipeline as a script."""
    lat = float(os.getenv("WEATHER_LAT", 24.8607))
    lon = float(os.getenv("WEATHER_LON", 67.0011))

    openaq_api_key = os.getenv("OPENAQ_API_KEY", None)
    if openaq_api_key:
        openaq_api_key = openaq_api_key.strip().strip('"').strip("'")

    logger.info(f"Starting feature pipeline for lat={lat}, lon={lon}")

    raw_data = fetch_all_sources(lat, lon, openaq_api_key=openaq_api_key)

    # ------------------------------------------------------------------
    # Fallback: if the live weather API returned no data, copy last day's
    # same-hour row from the Hopsworks feature store so the pipeline still
    # emits a feature row (stamped with the current Karachi time).
    # ------------------------------------------------------------------
    if not raw_data.get("om_weather"):
        logger.warning(
            "Open-Meteo weather API returned no data — falling back to "
            "Hopsworks feature store (last day, same hour)."
        )
        now_khi = pd.Timestamp.now(tz=KARACHI_TZ).floor("h").tz_localize(None)
        fb_row = load_last_day_from_hopsworks()
        if fb_row is None:
            logger.error(
                "Hopsworks fallback unavailable — proceeding with empty API data."
            )
        else:
            raw_data = _synthesize_raw_from_fallback(fb_row, now_khi)

    raw_path = "raw_api_response.json"
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2, default=str)
    logger.info(f"Raw API response saved to {raw_path}")
    # raw_data = None
    # with open("raw_api_response.json",'r') as file:
    #     raw_data = json.load(file)

    features_df = build_features(raw_data)

    features_path = "weather_features.csv"
    features_df.to_csv(features_path, index=False)
    logger.info(f"Features saved to {features_path}")

    # Save to Hopsworks Feature Store
    save_to_hopsworks(features_df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== Feature Summary ===")
    print(features_df.T)

    print("\n=== Key Metrics (Project Brief) ===")
    key_cols = [
        "aqi", "temp_c", "precip_mm", "rain_prob_pct",
        "humidity_pct", "uv_index", "wind_speed_ms", "wind_dir_deg",
    ]
    for col in key_cols:
        if col in features_df.columns:
            val = features_df.iloc[0][col]
            print(f"  {col:20s}: {val}")

    return features_df


if __name__ == "__main__":
    main()