import requests
import numpy as np
import pandas as pd
import json

def fetch_all_sources(lat: float, lon: float, aqicn_token: str) -> dict:
    # 1. Open-Meteo Weather Forecast API (Current + Hourly for rolling features)
    om_weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            # "precipitation_probability",
            'rain',
            "wind_speed_10m",
            "wind_direction_10m",
            # "uv_index",
        ],
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            # "precipitation_probability",
            'rain',
            "wind_speed_10m",
            "wind_direction_10m",
            # "uv_index",
        ],
        "wind_speed_unit": "ms",  # Match AQICN m/s standard
        "past_hours": 24,
        "forecast_hours": 1,
        "timezone": "UTC",
    }
    weather_res = requests.get(om_weather_url, params=weather_params, timeout=10).json()

    # 2. Open-Meteo Air Quality API
    om_aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        # "current": ["us_aqi", "european_aqi", "uv_index", "pm2_5", "pm10"],
        "current": ["uv_index", "pm2_5", "pm10"],
        "hourly": ["uv_index", "pm2_5", "pm10"],
        "past_hours": 24,
        "forecast_hours": 1,
        "timezone": "UTC",
    }
    aq_res = requests.get(om_aq_url, params=aq_params, timeout=10).json()

    # 3. AQICN Geo-localized Feed API
    # aqicn_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"
    aqicn_url = "https://api.waqi.info/feed/@-545332/"
    aqicn_res = requests.get(aqicn_url, params={"token": aqicn_token}, timeout=10).json()

    return {"om_weather": weather_res, "om_aq": aq_res, "aqicn": aqicn_res}

def build_features(raw: dict) -> pd.DataFrame:
    w_curr = raw["om_weather"].get("current", {})
    aq_curr = raw["om_aq"].get("current", {})
    aqicn_data = raw["aqicn"].get("data", {})
    iaqi = aqicn_data.get("iaqi", {})

    def get_iaqi(key):
        val = iaqi.get(key, {}).get("v")
        return float(val) if val is not None else np.nan    

    # --- A. Separate Features for Overlapping / Similar Metrics ---
    row = {
        "timestamp_utc": pd.to_datetime(w_curr.get("time")),
        # AQI (Separate per source)
        # "om_us_aqi": aq_curr.get("us_aqi", np.nan),
        # "om_eu_aqi": aq_curr.get("european_aqi", np.nan),
        "aqicn_us_aqi": float(aqicn_data["aqi"]) if str(aqicn_data.get("aqi", "")).isdigit() else np.nan,

        # Temperature °C (Separate per source)
        "om_temp_c": w_curr.get("temperature_2m", np.nan),
        "aqicn_temp_c": get_iaqi("t"),

        # Humidity % (Separate per source)
        "om_humidity_pct": w_curr.get("relative_humidity_2m", np.nan),
        "aqicn_humidity_pct": get_iaqi("h"),

        # Precipitation mm (Separate per source)
        "om_precip_mm": w_curr.get("precipitation", np.nan),
        # "aqicn_precip_mm": get_iaqi("r"),

        # Wind Speed m/s (Separate per source)
        "om_wind_speed_ms": w_curr.get("wind_speed_10m", np.nan),
        # "aqicn_wind_speed_ms": get_iaqi("w"),

        # Wind Direction deg (Separate per source)
        "om_wind_dir_deg": w_curr.get("wind_direction_10m", np.nan),
        # "aqicn_wind_dir_deg": get_iaqi("wd"),

        # UV Index (Separate across Open-Meteo Weather, Open-Meteo AQ, and AQICN Daily Max)
        # "om_weather_uv_index": w_curr.get("uv_index", np.nan),
        "om_aq_uv_index": aq_curr.get("uv_index", np.nan),
        # "aqicn_daily_uv_max": aqicn_uv_max,

        # --- B. Single-Source Unique Metrics ---
        "om_rain_prob_pct": w_curr.get("rain", np.nan),

        # "om_pm1": aq_curr.get("",np.nan),
        "aqicn_pm1": get_iaqi("pm1"),

        "om_pm10": aq_curr.get("pm10",np.nan),
        "aqicn_pm10": get_iaqi("pm10"),

        "om_pm25": aq_curr.get("pm2_5",np.nan),
        "aqicn_pm25": get_iaqi("pm25"),

    }

    df = pd.DataFrame([row])
    return df

def build_engineered_features(
    raw: dict, base_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Creates engineered features for Hopsworks Feature Store and ML training.
    Uses raw API payloads for historical hourly lags/rolling stats and base_df for current fused targets.
    """
    if base_df is None:
        base_df = build_features(raw)

    curr = base_df.iloc[0]

    # --- 1. Primary Key & Event Time ---
    ts_floored = pd.to_datetime(curr["timestamp_utc"]).floor("h")

    # --- 2. Fused & Ground-Truth Target Columns ---
    # Fallback to Open-Meteo if AQICN sensor drops a reading
    aqi_val = (
        curr["aqicn_us_aqi"]
        if pd.notna(curr["aqicn_us_aqi"])
        else raw["om_aq"].get("current", {}).get("us_aqi", np.nan)
    )
    pm25_val = (
        curr["aqicn_pm25"] if pd.notna(curr["aqicn_pm25"]) else curr["om_pm25"]
    )
    pm10_val = (
        curr["aqicn_pm10"] if pd.notna(curr["aqicn_pm10"]) else curr["om_pm10"]
    )
    precip_val = (
        curr["om_precip_mm"] if pd.notna(curr["om_precip_mm"]) else 0.0
    )

    # --- 3. Prepare Historical Hourly Series for Lags & Rolling Stats ---
    w_hourly = pd.DataFrame(raw["om_weather"].get("hourly", {}))
    aq_hourly = pd.DataFrame(raw["om_aq"].get("hourly", {}))

    w_hourly["time"] = pd.to_datetime(w_hourly["time"])
    aq_hourly["time"] = pd.to_datetime(aq_hourly["time"])

    hist = pd.merge(w_hourly, aq_hourly, on="time", how="inner").sort_values(
        "time"
    )
    # Keep only hours up to the current floored timestamp
    hist = hist[hist["time"] <= ts_floored].reset_index(drop=True)

    # Align the final row's values with our current fused values so diffs/rolling reflect t
    if not hist.empty and hist.iloc[-1]["time"] == ts_floored:
        hist.loc[hist.index[-1], "temperature_2m"] = curr["om_temp_c"]
        hist.loc[hist.index[-1], "relative_humidity_2m"] = curr[
            "om_humidity_pct"
        ]
        hist.loc[hist.index[-1], "wind_speed_10m"] = curr["om_wind_speed_ms"]
        hist.loc[hist.index[-1], "wind_direction_10m"] = curr[
            "om_wind_dir_deg"
        ]
        hist.loc[hist.index[-1], "precipitation"] = precip_val
        hist.loc[hist.index[-1], "us_aqi"] = aqi_val
        hist.loc[hist.index[-1], "pm2_5"] = pm25_val
        hist.loc[hist.index[-1], "pm10"] = pm10_val
        hist.loc[hist.index[-1], "uv_index"] = curr["om_aq_uv_index"]

    def get_diff(series: pd.Series, lag: int) -> float:
        if len(series) > lag:
            return float(series.iloc[-1] - series.iloc[-(lag + 1)])
        return np.nan

    def get_roll_mean(series: pd.Series, window: int) -> float:
        return (
            float(series.tail(window).mean()) if not series.empty else np.nan
        )

    def get_roll_std(series: pd.Series, window: int) -> float:
        return (
            float(series.tail(window).std()) if len(series) >= 2 else np.nan
        )

    # --- 4. Time & Cyclical Features ---
    hour = ts_floored.hour
    day_of_week = ts_floored.dayofweek
    month = ts_floored.month
    is_weekend = 1 if day_of_week >= 5 else 0

    wind_dir_rad = np.radians(curr["om_wind_dir_deg"])

    # --- 5. Assemble All Engineered Features ---
    engineered_row = {
        # Primary Key & Event Time
        "timestamp_utc": ts_floored,
        # Fused & Ground-Truth Targets
        "aqi": aqi_val,
        "temp_c": curr["om_temp_c"],
        "humidity_pct": curr["om_humidity_pct"],
        "precip_mm": precip_val,
        "rain_prob_pct": curr["om_rain_prob_pct"],
        "uv_index": curr["om_aq_uv_index"],
        "wind_speed_ms": curr["om_wind_speed_ms"],
        "wind_dir_deg": curr["om_wind_dir_deg"],
        "pm25": pm25_val,
        "pm10": pm10_val,
        "pm1": curr["aqicn_pm1"],
        # Time & Cyclical Features
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
        # Rate-of-Change / Lag Difference Features (Short-Term Momentum)
        "temp_diff_1h": get_diff(hist["temperature_2m"], 1),
        "temp_diff_3h": get_diff(hist["temperature_2m"], 3),
        "aqi_diff_1h": get_diff(hist["us_aqi"], 1),
        "aqi_diff_3h": get_diff(hist["us_aqi"], 3),
        "pm25_diff_1h": get_diff(hist["pm2_5"], 1),
        "pm25_diff_3h": get_diff(hist["pm2_5"], 3),
        "pm10_diff_1h": get_diff(hist["pm10"], 1),
        "humidity_diff_1h": get_diff(hist["relative_humidity_2m"], 1),
        "humidity_diff_3h": get_diff(hist["relative_humidity_2m"], 3),
        "wind_speed_diff_1h": get_diff(hist["wind_speed_10m"], 1),
        "wind_speed_diff_3h": get_diff(hist["wind_speed_10m"], 3),
        "uv_diff_1h": get_diff(hist["uv_index"], 1),
        # Rolling Window Statistics (Baseline Trends & Volatility)
        "temp_roll_mean_6h": get_roll_mean(hist["temperature_2m"], 6),
        "temp_roll_mean_24h": get_roll_mean(hist["temperature_2m"], 24),
        "aqi_roll_mean_6h": get_roll_mean(hist["us_aqi"], 6),
        "aqi_roll_mean_24h": get_roll_mean(hist["us_aqi"], 24),
        "pm25_roll_mean_6h": get_roll_mean(hist["pm2_5"], 6),
        "pm25_roll_mean_24h": get_roll_mean(hist["pm2_5"], 24),
        "pm10_roll_mean_6h": get_roll_mean(hist["pm10"], 6),
        "pm10_roll_mean_24h": get_roll_mean(hist["pm10"], 24),
        "humidity_roll_mean_6h": get_roll_mean(hist["relative_humidity_2m"], 6),
        "humidity_roll_mean_24h": get_roll_mean(
            hist["relative_humidity_2m"], 24
        ),
        "wind_speed_roll_mean_6h": get_roll_mean(hist["wind_speed_10m"], 6),
        "wind_speed_roll_mean_24h": get_roll_mean(hist["wind_speed_10m"], 24),
        "precip_roll_sum_6h": float(hist["precipitation"].tail(6).sum()),
        "precip_roll_sum_24h": float(hist["precipitation"].tail(24).sum()),
        "pm25_roll_std_6h": get_roll_std(hist["pm2_5"], 6),
        "aqi_roll_std_6h": get_roll_std(hist["us_aqi"], 6),
        "temp_roll_std_6h": get_roll_std(hist["temperature_2m"], 6),
        # Domain-Specific Interaction Features (Dispersion & Aerosol Ratios)
        "pm25_to_pm10_ratio": pm25_val / (pm10_val + 1e-5),
        "pollution_dispersion_idx": pm25_val
        / (curr["om_wind_speed_ms"] + 0.5),
        "temp_humidity_index": curr["om_temp_c"]
        * (curr["om_humidity_pct"] / 100.0),
    }

    return pd.DataFrame([engineered_row])

raw_data = fetch_all_sources(lat=24.8607, lon=67.0011, aqicn_token="aa707710756712ae9d5fc69bfc9f97a429c4e3e8")
with open('json_file.json','w')as file:
    json.dump(raw_data,file,indent=True)
features_df = build_engineered_features(raw_data)
features_df.to_csv('Weather_Fetches.csv',index=False)
print(features_df.T)
