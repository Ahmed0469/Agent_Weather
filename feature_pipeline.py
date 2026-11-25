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
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta

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
# API Endpoints
# ---------------------------------------------------------------------------
OM_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OM_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENAQ_V3_URL = "https://api.openaq.org/v3"

# ---------------------------------------------------------------------------
# Fetch Functions
# ---------------------------------------------------------------------------
def fetch_openmeteo_weather(
    lat: float,
    lon: float,
    past_hours: int = 24,
    forecast_hours: int = 1,
) -> Dict[str, Any]:
    """
    Fetch current and hourly weather data from Open-Meteo Forecast API.

    Returns the raw JSON response dict.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "precipitation_probability",
            "rain",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "precipitation_probability",
            "rain",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "wind_speed_unit": "ms",
        "past_hours": past_hours,
        "forecast_hours": forecast_hours,
        "timezone": "UTC",
    }
    try:
        resp = requests.get(OM_WEATHER_URL, params=params, timeout=15)
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
) -> Dict[str, Any]:
    """
    Fetch current and hourly air quality data from Open-Meteo Air Quality API.

    Returns the raw JSON response dict.
    """
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
        "timezone": "UTC",
    }
    try:
        resp = requests.get(OM_AQ_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        logger.info("Open-Meteo Air Quality fetched successfully")
        return data
    except Exception as exc:
        logger.error(f"Open-Meteo Air Quality fetch failed: {exc}")
        return {}

def fetch_openaq(
    location_id: int = 4837117,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch latest and hourly air quality data from OpenAQ API v3
    for a specific location ID (default 4837117), pm25 only.

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
    if api_key is None:
        logger.warning(
            "OpenAQ API key not provided — skipping OpenAQ fetch. "
            "Set OPENAQ_API_KEY env var or pass api_key explicitly."
        )
        return {}

    headers = {"X-API-Key": api_key}
    base_oaq = OPENAQ_V3_URL

    # ------------------------------------------------------------------
    # Fetch metadata & sensor IDs for the location
    # ------------------------------------------------------------------
    try:
        loc_res = requests.get(
            f"{base_oaq}/locations/{location_id}",
            headers=headers,
            timeout=15,
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

    # ------------------------------------------------------------------
    # Time window for hourly history (past 26 hours)
    # ------------------------------------------------------------------
    now_utc = datetime.now(timezone.utc)
    dt_from = (now_utc - timedelta(hours=26)).strftime("%Y-%m-%dT%H:%M:%SZ")
    dt_to = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    openaq_payload = {
        "location_id": location_id,
        "location_name": loc_info.get("name") if isinstance(loc_info, dict) else None,
        "sensor_ids": sensor_ids,
        "latest": {},
        "hourly": {},
    }

    # ------------------------------------------------------------------
    # Fetch latest measurements for the location
    # ------------------------------------------------------------------
    try:
        latest_res = requests.get(
            f"{base_oaq}/locations/{location_id}/latest",
            headers=headers,
            timeout=15,
        ).json()
    except Exception as exc:
        logger.error(f"OpenAQ latest fetch failed: {exc}")
        latest_res = {}

    id_to_param = {v: k for k, v in sensor_ids.items()}
    for item in latest_res.get("results", []):
        p_name = id_to_param.get(item.get("sensorsId"))
        if p_name:
            openaq_payload["latest"][p_name] = item.get("value")

    # ------------------------------------------------------------------
    # Fetch past 24h hourly history per sensor
    # ------------------------------------------------------------------
    for param_name, s_id in sensor_ids.items():
        if param_name == "pm25":
            try:
                h_res = requests.get(
                    f"{base_oaq}/sensors/{s_id}/hours",
                    params={
                        "datetime_from": dt_from,
                        "datetime_to": dt_to,
                        "limit": 48,
                    },
                    headers=headers,
                    timeout=15,
                ).json()
            except Exception as exc:
                logger.error(f"OpenAQ hourly fetch failed for {param_name}: {exc}")
                continue

            records = []
            for r in h_res.get("results", []):
                dt_str = r.get("period", {}).get("datetimeTo", {}).get("utc")
                val = r.get("value")
                if dt_str and val is not None:
                    records.append({"time": dt_str, param_name: float(val)})
            openaq_payload["hourly"][param_name] = records

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
    openaq = fetch_openaq(location_id=4837117, api_key=openaq_api_key)

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

    Also adds temporal, cyclical, rolling, lag/difference, and
    domain-specific interaction features.
    """
    w_curr = raw.get("om_weather", {}).get("current", {})
    aq_curr = raw.get("om_aq", {}).get("current", {})
    openaq_data = raw.get("openaq", {})

    timestamp_str = w_curr.get("time")
    timestamp_utc = pd.to_datetime(timestamp_str) if timestamp_str else pd.Timestamp.utcnow()

    temp_c = _safe_float(w_curr.get("temperature_2m"))
    humidity_pct = _safe_float(w_curr.get("relative_humidity_2m"))
    precip_mm = _safe_float(w_curr.get("precipitation"))
    rain_prob_pct = _safe_float(w_curr.get("precipitation_probability"))
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

    hour = timestamp_utc.hour
    day_of_week = timestamp_utc.dayofweek
    month = timestamp_utc.month
    is_weekend = 1 if day_of_week >= 5 else 0

    wind_dir_rad = np.deg2rad(wind_dir_deg) if not np.isnan(wind_dir_deg) else 0.0

    precip_val = rain_mm if not np.isnan(rain_mm) else precip_mm

    base_row = {
        "timestamp_utc": timestamp_utc,
        "temp_c": temp_c,
        "humidity_pct": humidity_pct,
        "precip_mm": precip_val,
        "rain_prob_pct": rain_prob_pct,
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
        "precipitation_probability": w_hourly.get("precipitation_probability", [np.nan] * len(times)),
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

    pm25_val = openaq_pm2_5

    engineered_row.update({
        "pollution_dispersion_idx": pm25_val / (wind_speed_ms + 0.5) if not np.isnan(pm25_val) else np.nan,
        "temp_humidity_index": temp_c * (humidity_pct / 100.0) if not np.isnan(temp_c) and not np.isnan(humidity_pct) else np.nan,
    })

    return pd.DataFrame([engineered_row])


# ---------------------------------------------------------------------------
# Hopsworks Feature Store integration
# ---------------------------------------------------------------------------
def save_to_hopsworks(
    features_df: pd.DataFrame,
    feature_group_name: str = "weather_air_quality_features",
    feature_group_version: int = 1,
) -> bool:
    """
    Save the feature DataFrame to the Hopsworks Feature Store.

    Requires hsfs library and Hopsworks credentials via environment variables:
      - HOPSWORKS_API_KEY: Hopsworks API key
      - HOPSWORKS_PROJECT: Hopsworks project name
      - HOPSWORKS_HOST: (optional) Hopsworks host, defaults to cloud

    Parameters
    ----------
    features_df : pd.DataFrame
        The engineered features DataFrame (single row).
    feature_group_name : str
        Name of the feature group to write to.
    feature_group_version : int
        Version of the feature group.

    Returns
    -------
    bool
        True if save succeeded, False otherwise.
    """
    try:
        import hsfs
    except ImportError:
        logger.warning(
            "hsfs not installed — skipping Hopsworks save. "
            "Install with: uv add hopsworks[python]"
        )
        return False

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = os.getenv("HOPSWORKS_PROJECT")
    host = os.getenv("HOPSWORKS_HOST")

    if not api_key or not project:
        logger.warning(
            "HOPSWORKS_API_KEY or HOPSWORKS_PROJECT not set — "
            "skipping Hopsworks save."
        )
        return False

    try:
        conn = hsfs.connection(
            host=host,
            project=project,
            api_key=api_key,
        )
        fs = conn.get_feature_store()

        # Get or create the feature group
        try:
            fg = fs.get_feature_group(
                name=feature_group_name,
                version=feature_group_version,
            )
        except Exception:
            fg = fs.create_feature_group(
                name=feature_group_name,
                version=feature_group_version,
                primary_key=["timestamp_utc"],
                description="Weather and air quality features from Open-Meteo and OpenAQ",
                online_enabled=True,
            )

        fg.insert(features_df, write_options={"wait_for_job": False})
        logger.info(
            f"Features saved to Hopsworks feature group '{feature_group_name}' "
            f"v{feature_group_version}"
        )
        return True

    except Exception as exc:
        logger.error(f"Hopsworks save failed: {exc}")
        return False


def main():
    """Entry point for running the pipeline as a script."""
    lat = float(os.getenv("WEATHER_LAT", None))
    lon = float(os.getenv("WEATHER_LON", None))

    openaq_api_key = os.getenv("OPENAQ_API_KEY", None)

    logger.info(f"Starting feature pipeline for lat={lat}, lon={lon}")

    raw_data = fetch_all_sources(lat, lon, openaq_api_key=openaq_api_key)

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