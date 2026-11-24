from datetime import datetime, timedelta, timezone
import json
import numpy as np
import pandas as pd
import requests

def compute_us_aqi(pm25: float, pm10: float = np.nan) -> float:
    """Calculates US EPA AQI from PM2.5 and PM10 concentrations (µg/m³)."""

    def calc_sub_index(conc, breakpoints):
        if pd.isna(conc) or conc < 0:
            return np.nan
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= conc <= c_high:
                return round(
                    ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)
                    + i_low
                )
        return 500.0 if conc > breakpoints[-1] else np.nan

    # US EPA PM2.5 & PM10 breakpoints
    pm25_bp = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 500.4, 301, 500),
    ]
    pm10_bp = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ]

    aqi_25 = calc_sub_index(
        round(pm25, 1) if pd.notna(pm25) else np.nan, pm25_bp
    )
    aqi_10 = calc_sub_index(round(pm10) if pd.notna(pm10) else np.nan, pm10_bp)

    if pd.isna(aqi_25) and pd.isna(aqi_10):
        return np.nan
    return float(np.nanmax([aqi_25, aqi_10]))

def fetch_all_sources(
    lat: float,
    lon: float,
    openaq_api_key: str,
    location_id: int = 4837117,
) -> dict:
    # 1. Open-Meteo Weather API (Current + 24h Hourly for weather diffs & rolling)
    om_weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
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
        "wind_speed_unit": "ms",
        "past_hours": 24,
        "forecast_hours": 1,
        "timezone": "UTC",
    }
    weather_res = requests.get(
        om_weather_url, params=weather_params, timeout=15
    ).json()

    # 2. Open-Meteo Air Quality API (SIRF CURRENT VALUES - No historical AQ)
    om_aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["us_aqi", "uv_index", "pm2_5", "pm10"],
        "timezone": "UTC",
    }
    aq_res = requests.get(om_aq_url, params=aq_params, timeout=15).json()

    # 3. OpenAQ API v3 for Specific Location ID (4837117)
    headers = {"X-API-Key": openaq_api_key}
    base_oaq = "https://api.openaq.org/v3"

    # Fetch metadata & sensor IDs for location 4837117
    loc_res = requests.get(
        f"{base_oaq}/locations/{location_id}",
        headers=headers,
        timeout=15,
    ).json()

    results = loc_res.get("results", [])
    loc_info = results if results else {}
    sensors = loc_info.get("sensors", [])

    sensor_ids = {}  # e.g., {'pm25': 12345, 'pm10': 12346, 'pm1': 12347}
    for s in sensors:
        param_name = s.get("parameter", {}).get("name", "").lower()
        if param_name in ("pm25", "pm10", "pm1") and param_name not in sensor_ids:
            sensor_ids[param_name] = s["id"]

    now_utc = datetime.now(timezone.utc)
    dt_from = (now_utc - timedelta(hours=26)).strftime("%Y-%m-%dT%H:%M:%SZ")
    dt_to = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    openaq_payload = {
        "location_id": location_id,
        "location_name": loc_info.get("name"),
        "sensor_ids": sensor_ids,
        "latest": {},
        "hourly": {},
    }

    # Fetch latest measurements for location 4837117
    latest_res = requests.get(
        f"{base_oaq}/locations/{location_id}/latest",
        headers=headers,
        timeout=15,
    ).json()

    id_to_param = {v: k for k, v in sensor_ids.items()}
    for item in latest_res.get("results", []):
        p_name = id_to_param.get(item.get("sensorsId"))
        if p_name:
            openaq_payload["latest"][p_name] = item.get("value")

    # Fetch past 24h hourly history per sensor directly from OpenAQ
    for param_name, s_id in sensor_ids.items():
        if param_name in ("pm25", "pm10"):
            h_res = requests.get(
                f"{base_oaq}/sensors/{s_id}/hours",
                params={"datetime_from": dt_from, "datetime_to": dt_to, "limit": 48},
                headers=headers,
                timeout=15,
            ).json()
            records = []
            for r in h_res.get("results", []):
                dt_str = r.get("period", {}).get("datetimeTo", {}).get("utc")
                val = r.get("value")
                if dt_str and val is not None:
                    records.append({"time": dt_str, param_name: float(val)})
            openaq_payload["hourly"][param_name] = records

    return {"om_weather": weather_res, "om_aq": aq_res, "openaq": openaq_payload}

def build_features(raw: dict) -> pd.DataFrame:
    w_curr = raw["om_weather"].get("current", {})
    aq_curr = raw["om_aq"].get("current", {})
    oaq_latest = raw["openaq"].get("latest", {})

    oaq_pm25 = oaq_latest.get("pm25", np.nan)
    oaq_pm10 = oaq_latest.get("pm10", np.nan)
    oaq_pm1 = oaq_latest.get("pm1", np.nan)

    row = {
        "timestamp_utc": pd.to_datetime(w_curr.get("time")),
        # OpenAQ Primary + Open-Meteo Current Only
        "openaq_us_aqi": compute_us_aqi(oaq_pm25, oaq_pm10),
        "om_current_us_aqi": aq_curr.get("us_aqi", np.nan),
        "openaq_pm25": float(oaq_pm25) if pd.notna(oaq_pm25) else np.nan,
        "openaq_pm10": float(oaq_pm10) if pd.notna(oaq_pm10) else np.nan,
        "openaq_pm1": float(oaq_pm1) if pd.notna(oaq_pm1) else np.nan,
        "om_current_pm25": aq_curr.get("pm2_5", np.nan),
        "om_current_pm10": aq_curr.get("pm10", np.nan),
        "om_aq_uv_index": aq_curr.get("uv_index", np.nan),
        # Open-Meteo Weather
        "om_temp_c": w_curr.get("temperature_2m", np.nan),
        "om_humidity_pct": w_curr.get("relative_humidity_2m", np.nan),
        "om_precip_mm": w_curr.get("precipitation", np.nan),
        "om_rain_prob_pct": w_curr.get("rain", np.nan),
        "om_wind_speed_ms": w_curr.get("wind_speed_10m", np.nan),
        "om_wind_dir_deg": w_curr.get("wind_direction_10m", np.nan),
    }
    return pd.DataFrame([row])
