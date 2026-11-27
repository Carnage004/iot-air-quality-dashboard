"""
Maharashtra IoT Air Quality Dashboard (Streamlit)
Run:  streamlit run app.py

Data sources
  1. Simulated demo (14 days, 15-min readings, Maharashtra cities)
  2. Live CPCB feed via data.gov.in (station snapshot)
  3. Upload CSV: three formats are detected automatically
       a) IoT time series      : timestamp, node, pm25, co2, temp, humidity
       b) CPCB snapshot        : state, city, station, pollutant_id, pollutant_avg, latitude, longitude
       c) Daily city AQI history: city, date, aqi and/or pm25 (+ pm10, no2, so2, co, o3 ...)
"""
import calendar
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="IoT Based Air Quality Dashboard", page_icon="🌫️", layout="wide")

# ------------------------------------------------------------------ constants
# name: (region, lat, lon, pollution factor, temp offset °C, humidity offset %)
# Pollution factors are illustrative (urban/industrial scale), NOT measurements.
CITIES = {
    # Konkan / Mumbai Metropolitan Region
    "Mumbai": ("Konkan", 19.076, 72.878, 1.30, 3, 12),
    "Thane": ("Konkan", 19.218, 72.978, 1.25, 3, 10),
    "Navi Mumbai": ("Konkan", 19.033, 73.030, 1.20, 3, 10),
    "Kalyan-Dombivli": ("Konkan", 19.235, 73.130, 1.20, 3, 8),
    "Vasai-Virar": ("Konkan", 19.391, 72.839, 1.15, 3, 10),
    "Ratnagiri": ("Konkan", 16.990, 73.312, 0.55, 3, 12),
    # Western Maharashtra
    "Pune": ("Western Maharashtra", 18.520, 73.857, 1.15, 0, 0),
    "Pimpri-Chinchwad": ("Western Maharashtra", 18.629, 73.800, 1.20, 0, 0),
    "Satara": ("Western Maharashtra", 17.680, 74.018, 0.65, -1, 4),
    "Kolhapur": ("Western Maharashtra", 16.705, 74.243, 0.85, -1, 6),
    "Sangli": ("Western Maharashtra", 16.854, 74.581, 0.85, 0, 2),
    "Solapur": ("Western Maharashtra", 17.659, 75.906, 1.00, 2, -6),
    # North Maharashtra
    "Nashik": ("North Maharashtra", 19.998, 73.790, 1.00, 0, 0),
    "Ahilyanagar": ("North Maharashtra", 19.095, 74.750, 0.90, 1, -4),
    "Dhule": ("North Maharashtra", 20.902, 74.774, 0.95, 1, -4),
    "Jalgaon": ("North Maharashtra", 21.007, 75.563, 1.05, 2, -5),
    # Marathwada
    "Chhatrapati Sambhajinagar": ("Marathwada", 19.876, 75.343, 1.10, 1, -3),
    "Jalna": ("Marathwada", 19.841, 75.886, 0.95, 1, -4),
    "Beed": ("Marathwada", 18.989, 75.756, 0.85, 1, -5),
    "Latur": ("Marathwada", 18.408, 76.560, 0.90, 1, -4),
    "Nanded": ("Marathwada", 19.138, 77.321, 0.95, 2, -3),
    # Vidarbha
    "Nagpur": ("Vidarbha", 21.146, 79.088, 1.35, 2, -4),
    "Amravati": ("Vidarbha", 20.933, 77.753, 1.00, 2, -4),
    "Akola": ("Vidarbha", 20.709, 77.002, 1.00, 2, -5),
    "Chandrapur": ("Vidarbha", 19.962, 79.296, 1.50, 3, -2),
}
DEFAULT_PICK = ["Mumbai", "Pune", "Nagpur", "Nashik", "Chhatrapati Sambhajinagar"]

# Coordinates for common Indian cities, used to draw a map when an uploaded file has no lat/lon columns
CITY_COORDS = {
    "Delhi": (28.614, 77.209), "Mumbai": (19.076, 72.878), "Bengaluru": (12.972, 77.594),
    "Bangalore": (12.972, 77.594), "Kolkata": (22.573, 88.364), "Chennai": (13.083, 80.270),
    "Hyderabad": (17.385, 78.487), "Lucknow": (26.847, 80.947), "Jaipur": (26.912, 75.787),
    "Pune": (18.520, 73.857), "Ahmedabad": (23.023, 72.572), "Patna": (25.594, 85.138),
    "Bhopal": (23.259, 77.413), "Chandigarh": (30.733, 76.779), "Nagpur": (21.146, 79.088),
    "Kanpur": (26.449, 80.332), "Indore": (22.719, 75.857), "Surat": (21.170, 72.831),
    "Gurugram": (28.459, 77.027), "Noida": (28.535, 77.391), "Ghaziabad": (28.667, 77.449),
    "Thane": (19.218, 72.978), "Nashik": (19.998, 73.790), "Kochi": (9.931, 76.267),
    "Thiruvananthapuram": (8.524, 76.936), "Guwahati": (26.144, 91.736),
    "Bhubaneswar": (20.296, 85.824), "Ranchi": (23.344, 85.310), "Amritsar": (31.634, 74.873),
    "Varanasi": (25.317, 82.973), "Agra": (27.177, 78.008), "Coimbatore": (11.017, 76.956),
    "Vadodara": (22.307, 73.181), "Raipur": (21.251, 81.630), "Dehradun": (30.316, 78.032),
    "Visakhapatnam": (17.687, 83.218), "Jodhpur": (26.239, 73.024), "Faridabad": (28.409, 77.313),
}

# PM2.5 categories (CPCB, 24h average, µg/m³)
BINS = [0, 30, 60, 90, 120, 250, 1e4]
CATS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
COLS = ["#2e9e4f", "#9ccc3c", "#f2d12e", "#f28c28", "#d9402b", "#7a1f1f"]
CAT_COLOR = dict(zip(CATS, COLS))
# National AQI categories (CPCB): 0-50 Good ... 401+ Severe
AQI_BINS = [0, 50, 100, 200, 300, 400, 1e4]
PM10_BINS = [0, 50, 100, 250, 350, 430, 1e4]
SENSORS = ["pm25", "co2", "temp", "humidity"]
SENSOR_LABEL = {"pm25": "PM2.5 (µg/m³)", "co2": "CO₂ (ppm)", "temp": "Temperature (°C)", "humidity": "Humidity (%)"}
PALETTE = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24


def category(v):
    return CATS[int(np.digitize(v, BINS[1:-1]))]


# ------------------------------------------------------------------ data
@st.cache_data
def simulate(seed: int = 42, days: int = 14):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-11-09", periods=days * 96, freq="15min")
    h = np.asarray(idx.hour + idx.minute / 60, dtype=float)
    frames = []
    for name, (region, lat, lon, k, t_off, h_off) in CITIES.items():
        rush = 28 * np.exp(-((h - 9) ** 2) / 4) + 35 * np.exp(-((h - 19.5) ** 2) / 5)
        inv = 18 * np.exp(-((h - 3) ** 2) / 12)
        amp = 4.5 if region == "Konkan" else 7
        temp = 24 + t_off + amp * np.sin((h - 9) / 24 * 2 * np.pi) + rng.normal(0, .4, len(idx))
        hum = np.clip(62 + h_off - 1.6 * (temp - 24 - t_off) + rng.normal(0, 2, len(idx)), 20, 99)
        pm25 = k * (45 + rush + inv + 0.35 * (hum - 60)) + rng.normal(0, 4, len(idx))
        for s in rng.choice(len(idx) - 12, 4, replace=False):          # pollution events
            pm25[s:s + rng.integers(2, 10)] += rng.uniform(60, 130)
        co2 = 420 + 1.6 * rush * k + rng.normal(0, 12, len(idx))
        df = pd.DataFrame({"timestamp": idx, "node": name, "pm25": pm25,
                           "co2": co2, "temp": temp, "humidity": hum})
        n = len(df)
        df.loc[rng.choice(n, int(.02 * n), replace=False), "pm25"] = np.nan   # dropouts
        df.loc[rng.choice(n, 12, replace=False), "pm25"] = rng.uniform(900, 1500, 12)  # spikes
        df.loc[rng.choice(n, 6, replace=False), "temp"] = -127                # sensor error code
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return pd.concat([raw, raw.sample(300, random_state=1)])                  # duplicate MQTT msgs


@st.cache_data
def clean(raw: pd.DataFrame):
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    faults = {"Duplicate messages": int(raw.duplicated(["timestamp", "node"]).sum()),
              "Missing values": int(raw[SENSORS].isna().sum().sum()),
              "PM2.5 out of range (>500)": int((raw.pm25 > 500).sum()),
              "Temp error code (< -40)": int((raw.temp < -40).sum())}
    df = raw.drop_duplicates(["timestamp", "node"]).sort_values(["node", "timestamp"]).copy()
    df.loc[df.pm25 > 500, "pm25"] = np.nan
    df.loc[df.temp < -40, "temp"] = np.nan
    df[SENSORS] = df.groupby("node")[SENSORS].transform(
        lambda s: s.interpolate(limit=6).bfill().ffill())
    df["category"] = pd.cut(df.pm25, BINS, labels=CATS)
    df["hour"] = df.timestamp.dt.hour
    df["date"] = df.timestamp.dt.date
    return df.reset_index(drop=True), faults


def build_meta(df):
    rows = []
    for n, g in df.groupby("node"):
        if n in CITIES:
            r, la, lo = CITIES[n][:3]
        else:
            r = g["region"].iloc[0] if "region" in g else "Other"
            la = g["lat"].iloc[0] if "lat" in g else np.nan
            lo = g["lon"].iloc[0] if "lon" in g else np.nan
        rows.append((n, r, la, lo))
    return pd.DataFrame(rows, columns=["node", "region", "lat", "lon"]).set_index("node")


@st.cache_data
def detect_anomalies(df: pd.DataFrame, contamination: float):
    out = df.copy()
    out["pm_diff"] = out.groupby("node").pm25.diff().fillna(0)
    out["anomaly"] = False
    for n, g in out.groupby("node"):
        iso = IsolationForest(contamination=contamination, random_state=0)
        out.loc[g.index, "anomaly"] = iso.fit_predict(g[SENSORS + ["pm_diff"]]) == -1
    return out


@st.cache_data
def forecast(df: pd.DataFrame, node: str, test_days: int):
    hr = df[df.node == node].set_index("timestamp")[SENSORS].resample("1h").mean()
    hr["hs"] = np.sin(2 * np.pi * hr.index.hour / 24)
    hr["hc"] = np.cos(2 * np.pi * hr.index.hour / 24)
    for l in (1, 2, 3, 24):
        hr[f"lag{l}"] = hr.pm25.shift(l)
    hr["target"] = hr.pm25.shift(-1)
    hr = hr.dropna()
    X, y = hr.drop(columns="target"), hr.target
    cut = hr.index.max() - pd.Timedelta(days=test_days)
    Xtr, ytr, Xte, yte = X[X.index <= cut], y[y.index <= cut], X[X.index > cut], y[y.index > cut]
    rf = RandomForestRegressor(200, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    pred = pd.Series(rf.predict(Xte), index=Xte.index)
    base = Xte.pm25
    m = {"RF MAE": mean_absolute_error(yte, pred), "RF R²": r2_score(yte, pred),
         "Baseline MAE": mean_absolute_error(yte, base), "Baseline R²": r2_score(yte, base)}
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    return yte, pred, base, m, imp


def delta_1h(g):
    last = g.iloc[-1]
    ref = g[g.timestamp <= last.timestamp - pd.Timedelta(hours=1)]
    return last.pm25 - (ref.iloc[-1].pm25 if len(ref) else last.pm25)


# ------------------------------------------------------------------ CSV format detection
ALIASES = {"pm2.5": "pm25", "pm2_5": "pm25", "pm_25": "pm25", "pm 2.5": "pm25",
           "date_time": "timestamp", "datetime": "timestamp", "city_name": "city"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case / trim headers, map common aliases, and adapt near-miss files to a supported format."""
    df = df.copy()
    df.columns = [ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
    cols = set(df.columns)
    sensors = {"co2", "temp", "humidity"} <= cols
    has_val = "aqi" in cols or "pm25" in cols
    if "node" not in cols and "city" in cols and "timestamp" in cols and sensors:
        df = df.rename(columns={"city": "node"})                 # sensor file that calls its nodes "city"
    elif "timestamp" not in cols and "date" in cols and "node" in cols and sensors:
        df = df.rename(columns={"date": "timestamp"})
    elif "timestamp" in cols and "node" not in cols and "city" not in cols and "pm25" in cols and sensors:
        df["node"] = "Sensor 1"                                  # single-sensor IoT file without a node column
    elif "node" not in cols and has_val and ("timestamp" in cols or "date" in cols):
        if "timestamp" in cols and "date" not in cols:           # daily/hourly history that uses "timestamp"
            df = df.rename(columns={"timestamp": "date"})
        if "city" not in cols:
            df["city"] = "Uploaded series"                       # single-location file without a city column
    return df


def is_historical(cols: set) -> bool:
    return {"city", "date"} <= cols and ("aqi" in cols or "pm25" in cols) and not {"timestamp", "node"} <= cols


# ------------------------------------------------------------------ live feed (data.gov.in)
RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"   # "Real time Air Quality Index from various location"
API_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36",
           "Accept": "application/json,text/plain,*/*"}
RENAME = {"min_value": "pollutant_min", "max_value": "pollutant_max", "avg_value": "pollutant_avg"}


@st.cache_data(ttl=900, show_spinner="Fetching live data from data.gov.in ...")
def fetch_cpcb(api_key: str, state: str):
    """Download the current CPCB station feed. Cached for 15 minutes. Raises RuntimeError with a key-safe message."""
    rows, offset, page = [], 0, 500
    try:
        while True:
            params = {"api-key": api_key, "format": "json", "limit": page, "offset": offset}
            if state != "All India":
                params["filters[state]"] = state
            for attempt in range(3):                       # a couple of retries for slow/flaky responses
                try:
                    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=(15, 90))
                    break
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))
            r.raise_for_status()
            j = r.json()
            recs = j.get("records", [])
            rows += recs
            total = int(j.get("total") or 0)
            offset += len(recs)
            if not recs or (total and offset >= total) or offset >= 20000:
                break
    except requests.HTTPError as e:      # never put the URL in the message: it contains the API key
        code = e.response.status_code if e.response is not None else "?"
        hint = " (check that your API key is valid)" if code in (401, 403) else ""
        raise RuntimeError(f"data.gov.in returned HTTP {code}{hint}.") from None
    except requests.RequestException as e:   # class name only: safe, never includes the URL/key
        raise RuntimeError(f"Could not reach data.gov.in ({type(e).__name__}). "
                           "Check your internet connection or try a different network.") from None
    except (ValueError, TypeError):
        raise RuntimeError("data.gov.in returned an unreadable response.") from None
    if not rows:
        raise RuntimeError("data.gov.in returned no records for this selection.")
    return pd.DataFrame(rows).rename(columns=RENAME)


# ------------------------------------------------------------------ snapshot mode (data.gov.in / CPCB live feed)
UNIT = {"PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³", "SO2": "µg/m³", "O3": "µg/m³", "NH3": "µg/m³", "CO": "mg/m³"}


def snap_category(pol, v):
    if pol == "PM2.5":
        return CATS[int(np.digitize(v, BINS[1:-1]))]
    if pol == "PM10":
        return CATS[int(np.digitize(v, PM10_BINS[1:-1]))]
    return None


def snapshot_view(raw: pd.DataFrame, source_label: str = "uploaded CSV"):
    """Dashboard for a single-timestamp, long-format feed (one row per station x pollutant)."""
    d = raw.copy()
    d["pollutant_id"] = d.pollutant_id.astype(str).str.upper().replace({"OZONE": "O3"})
    for c in ("pollutant_min", "pollutant_max", "pollutant_avg", "latitude", "longitude"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    stamp = d["last_update"].iloc[0] if "last_update" in d and len(d) else "n/a"
    st.title("India Air Quality: Station Snapshot")
    st.caption(f"Real CPCB station data · single snapshot, last update **{stamp}** · "
               "no time series, so trends, forecast and anomaly detection are not available for this file")

    states = sorted(d.state.dropna().unique())
    state = st.sidebar.selectbox("State", states, index=states.index("Maharashtra") if "Maharashtra" in states else 0)
    d = d[d.state == state]
    pols = sorted(d.pollutant_id.unique())
    pol = st.sidebar.selectbox("Pollutant", pols, index=pols.index("PM2.5") if "PM2.5" in pols else 0)
    st.sidebar.caption(f"Data source: {source_label} (snapshot)")

    s = d[(d.pollutant_id == pol) & d.pollutant_avg.notna()]
    if s.empty:
        st.warning(f"No {pol} readings for {state} in this file.")
        st.stop()
    unit = UNIT.get(pol, "")
    city = (s.groupby("city").agg(avg=("pollutant_avg", "mean"), stations=("station", "nunique"),
                                  lat=("latitude", "mean"), lon=("longitude", "mean")).reset_index())
    city["avg"] = city.avg.round(1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stations reporting", f"{s.station.nunique()}")
    k2.metric("Cities", f"{city.city.nunique()}")
    k3.metric(f"Mean {pol}", f"{s.pollutant_avg.mean():.1f} {unit}")
    worst = city.loc[city.avg.idxmax()]
    k4.metric("Highest city", worst.city, f"{worst.avg:.0f} {unit}", delta_color="off")

    tm, tr, tp, tc, td = st.tabs(["🗺️ Map", "📊 City ranking", "🧪 All pollutants", "📍 Station detail", "🗂️ Data"])

    with tm:
        s2 = s.copy()
        s2["cat"] = s2.pollutant_avg.apply(lambda v: snap_category(pol, v))
        s2["size"] = s2.pollutant_avg.clip(lower=1)
        span = max(s2.latitude.max() - s2.latitude.min(), s2.longitude.max() - s2.longitude.min())
        kw = dict(lat="latitude", lon="longitude", size="size", hover_name="station",
                  hover_data={"city": True, "pollutant_avg": True, "pollutant_min": True, "pollutant_max": True,
                              "latitude": False, "longitude": False, "size": False},
                  zoom=5.2 if span < 8 else 3.6, size_max=22, map_style="open-street-map", height=620,
                  center={"lat": float(s2.latitude.mean()), "lon": float(s2.longitude.mean())})
        if pol in ("PM2.5", "PM10"):
            fig = px.scatter_map(s2, color="cat", color_discrete_map=CAT_COLOR, category_orders={"cat": CATS}, **kw)
        else:
            fig = px.scatter_map(s2, color="pollutant_avg", color_continuous_scale="YlOrRd", **kw)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, title_text=""))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Each dot is one monitoring station ({pol}, {unit}). "
                   + ("Colours follow CPCB categories." if pol in ("PM2.5", "PM10") else "Colour shows the value."))

    with tr:
        rk = city.sort_values("avg")
        rk["category"] = rk.avg.apply(lambda v: snap_category(pol, v))
        if pol in ("PM2.5", "PM10"):
            bar = px.bar(rk, x="avg", y="city", color="category", orientation="h", color_discrete_map=CAT_COLOR,
                         category_orders={"category": CATS}, hover_data=["stations"], labels={"avg": f"{pol} ({unit})", "city": ""})
        else:
            bar = px.bar(rk, x="avg", y="city", orientation="h", hover_data=["stations"],
                         labels={"avg": f"{pol} ({unit})", "city": ""})
        bar.update_layout(height=max(420, 22 * len(rk) + 120), title=f"{pol} by city (average across stations)")
        st.plotly_chart(bar, width="stretch")

    with tp:
        allp = d[d.pollutant_avg.notna()].pivot_table(index="city", columns="pollutant_id", values="pollutant_avg", aggfunc="mean")
        scaled = (allp - allp.min()) / (allp.max() - allp.min()).replace(0, np.nan)
        hm = go.Figure(go.Heatmap(z=scaled.values, x=allp.columns, y=allp.index, colorscale="YlOrRd",
                                  text=allp.round(0).astype("Int64").astype(str).replace("<NA>", "").values,
                                  texttemplate="%{text}", showscale=False, hoverongaps=False))
        hm.update_layout(height=max(420, 22 * len(allp) + 120), title="All pollutants by city (numbers = value, colour = relative within each pollutant)")
        st.plotly_chart(hm, width="stretch")
        st.caption("Units: CO in mg/m³, all others in µg/m³. Colours are scaled per pollutant, so compare cities, not columns.")

    with tc:
        cities = sorted(s.city.unique())
        cc = st.selectbox("City", cities, index=cities.index(worst.city))
        sc = s[s.city == cc].sort_values("pollutant_avg")
        fig = go.Figure(go.Bar(x=sc.pollutant_avg, y=sc.station, orientation="h",
                               error_x=dict(type="data", symmetric=False,
                                            array=(sc.pollutant_max - sc.pollutant_avg).fillna(0),
                                            arrayminus=(sc.pollutant_avg - sc.pollutant_min).fillna(0)),
                               marker_color="#c44e52"))
        fig.update_layout(height=max(300, 60 * len(sc) + 120), title=f"{cc}: {pol} per station (bars = average, whiskers = min to max)",
                          xaxis_title=f"{pol} ({unit})")
        st.plotly_chart(fig, width="stretch")

    with td:
        cols = [c for c in ["state", "city", "station", "last_update", "pollutant_id", "pollutant_min",
                            "pollutant_max", "pollutant_avg", "latitude", "longitude"] if c in d.columns]
        st.dataframe(d[cols], width="stretch", hide_index=True)
        st.download_button("⬇️ Download this state's data (CSV)", d[cols].to_csv(index=False).encode(),
                           f"{state.lower().replace(' ', '_')}_snapshot.csv", "text/csv")
    st.stop()


# ------------------------------------------------------------------ daily city history mode (city, date, aqi / pm25 / ...)
HIST_POLS = ["aqi", "pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]
HIST_LABEL = {"aqi": "AQI", "pm25": "PM2.5 (µg/m³)", "pm10": "PM10 (µg/m³)", "no2": "NO₂ (µg/m³)",
              "so2": "SO₂ (µg/m³)", "co": "CO (mg/m³)", "o3": "O₃ (µg/m³)", "nh3": "NH₃ (µg/m³)"}
NAAQS_24H = {"pm25": 60, "pm10": 100, "no2": 80, "so2": 80}          # India NAAQS 24-hour limits
HIST_RULE = {"Raw (hourly)": None, "Daily": "D", "Weekly": "W", "Monthly": "MS", "Yearly": "YS"}
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
          6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"}
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-monsoon"]


def metric_bins(m):
    return {"aqi": AQI_BINS, "pm25": BINS, "pm10": PM10_BINS}.get(m)


def cat_for(v, bins):
    return CATS[int(np.digitize(v, bins[1:-1]))]


@st.cache_data(show_spinner=False)
def prep_historical(raw: pd.DataFrame):
    d = raw.copy()
    n0 = len(d)
    d = d.rename(columns={"latitude": "lat", "longitude": "lon"})
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if d["date"].isna().all() and {"year", "month", "day"} <= set(d.columns):     # rebuild from parts
        parts = d[["year", "month", "day"] + (["hour"] if "hour" in d.columns else [])].apply(pd.to_numeric, errors="coerce")
        d["date"] = pd.to_datetime(parts, errors="coerce")
    d["city"] = d["city"].astype(str).str.strip()
    pols = [c for c in HIST_POLS if c in d.columns]
    for c in pols + [c for c in ("lat", "lon") if c in d.columns]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    q = {"Rows in file": n0, "Unreadable dates": int(d.date.isna().sum())}
    d = d.dropna(subset=["date"])
    q["Duplicate city-date rows"] = int(d.duplicated(["city", "date"]).sum())
    d = d.drop_duplicates(["city", "date"]).sort_values(["city", "date"]).reset_index(drop=True)
    q["Missing values"] = int(d[pols].isna().sum().sum())
    q["Negative values"] = int((d[pols] < 0).sum().sum())
    d[pols] = d[pols].where(d[pols] >= 0)
    d[pols] = d.groupby("city")[pols].transform(lambda s: s.interpolate(limit=7).bfill().ffill())
    d["year"] = d.date.dt.year
    d["month"] = d.date.dt.month
    d["season"] = d.month.map(SEASON)
    return d, pols, q


def city_coords(d: pd.DataFrame):
    out = {}
    if {"lat", "lon"} <= set(d.columns):
        g = d.groupby("city")[["lat", "lon"]].mean().dropna()
        out = {c: (r.lat, r.lon) for c, r in g.iterrows()}
    for c in d.city.unique():
        if c not in out and c in CITY_COORDS:
            out[c] = CITY_COORDS[c]
    return out


def historical_view(raw: pd.DataFrame, source_label: str = "uploaded CSV"):
    """Dashboard for a daily, multi-city history (one row per city per day)."""
    d, pols, q = prep_historical(raw)
    if d.empty:
        st.title("India Air Quality: City History")
        st.error("No usable rows found: the `date` column could not be read as dates.")
        st.stop()

    # ---- sidebar controls
    all_cities = sorted(d.city.unique())
    main = pols[0]
    default = all_cities if len(all_cities) <= 10 else list(d.groupby("city")[main].mean().nlargest(8).index)
    sel = st.sidebar.multiselect("Cities", all_cities, default=default)
    dmin, dmax = d.date.dt.date.min(), d.date.dt.date.max()
    rng_dates = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    d0, d1 = rng_dates if isinstance(rng_dates, tuple) and len(rng_dates) == 2 else (dmin, dmax)
    metric = st.sidebar.selectbox("Metric", pols, format_func=HIST_LABEL.get)
    subdaily = bool((d.date != d.date.dt.normalize()).any())      # hourly (or finer) readings?
    res_opts = list(HIST_RULE) if subdaily else ["Daily", "Weekly", "Monthly", "Yearly"]
    res = st.sidebar.selectbox("Time resolution", res_opts, index=res_opts.index("Monthly"))
    show_bands = st.sidebar.checkbox("Show CPCB category bands", True)
    st.sidebar.caption(f"Data source: {source_label} (daily city history)")

    st.title("India Air Quality: City History")
    if not sel:
        st.warning("Select at least one city in the sidebar.")
        st.stop()
    df = d[d.city.isin(sel) & (d.date.dt.date >= d0) & (d.date.dt.date <= d1)].copy()
    if df.empty:
        st.warning("No data for this selection. Widen the date range.")
        st.stop()

    label, bins = HIST_LABEL[metric], metric_bins(metric)
    ncolor = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(all_cities)}
    if bins:
        df["category"] = pd.cut(df[metric], bins, labels=CATS, include_lowest=True)
    st.caption(f"{'Hourly' if subdaily else 'Daily'} readings · **{df.city.nunique()}** location(s) · {df.date.min():%d %b %Y} to {df.date.max():%d %b %Y} · "
               f"{len(df):,} rows" + (" · categories follow the CPCB national AQI scale" if metric == "aqi" else ""))

    # ---- KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"Mean {label}", f"{df[metric].mean():.1f}")
    pk = df.loc[df[metric].idxmax()]
    k2.metric("Peak", f"{pk[metric]:.0f}", f"{pk.city}, {pk.date:%d %b %Y}" if not subdaily else f"{pk.city}, {pk.date:%d %b %Y %H:%M}", delta_color="off")
    if bins:
        k3.metric("Days 'Poor' or worse", f"{(df[metric] > bins[3]).mean() * 100:.2f}%")
    else:
        k3.metric("Median", f"{df[metric].median():.1f}")
    cm = df.groupby("city")[metric].mean()
    k4.metric("Most polluted city", cm.idxmax(), f"{cm.max():.0f}", delta_color="off")
    ym = df.groupby("year")[metric].mean()
    if len(ym) >= 2:
        k5.metric(f"Change {ym.index[0]} to {ym.index[-1]}", f"{ym.iloc[-1]:.1f}",
                  f"{ym.iloc[-1] - ym.iloc[0]:+.1f} vs {ym.index[0]}", delta_color="inverse")
    else:
        k5.metric("Cleanest city", cm.idxmin(), f"{cm.min():.0f}", delta_color="off")

    t1, t2, t3, t4, t5, t6 = st.tabs(["📈 Trends", "🏙️ City comparison", "🗓️ Seasonality",
                                      "🧪 Pollutants", "🗺️ Map", "🗂️ Data"])

    # ---- trends
    with t1:
        rule = None if (res == "Daily" and not subdaily) else HIST_RULE[res]
        series = {}
        for c in sel:
            g = df[df.city == c].set_index("date")[metric]
            series[c] = g if rule is None else g.resample(rule).mean()
        fig = go.Figure()
        if show_bands and bins:
            top = max(float(s.max()) for s in series.values()) * 1.1
            for lo, hi, col in zip(bins[:-1], bins[1:], COLS):
                if lo < top:
                    fig.add_hrect(y0=lo, y1=min(hi, top), fillcolor=col, opacity=.10, line_width=0)
        for c, s in series.items():
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=c, mode="lines", line=dict(width=1.4, color=ncolor[c])))
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                          legend=dict(orientation="h", y=1.1), yaxis_title=label)
        st.plotly_chart(fig, width="stretch")
        if len(sel) > 6:
            st.caption("Many cities selected: click legend entries to hide lines, or pick fewer cities.")

        yr = df.groupby(["year", "city"])[metric].mean().reset_index()
        yb = px.bar(yr, x="year", y=metric, color="city", barmode="group", color_discrete_map=ncolor,
                    title=f"Yearly mean {label}", labels={metric: label, "year": ""})
        yb.update_layout(height=380, xaxis=dict(dtick=1))
        st.plotly_chart(yb, width="stretch")

    # ---- city comparison
    with t2:
        stats = df.groupby("city")[metric].agg(mean="mean", median="median",
                                               p95=lambda s: s.quantile(.95), max="max").round(1)
        if bins:
            stats["% days Poor+"] = df.groupby("city")[metric].apply(lambda s: (s > bins[3]).mean() * 100).round(1)
        stats = stats.sort_values("mean", ascending=False)
        c1, c2 = st.columns(2)
        rk = stats.reset_index().sort_values("mean")
        if bins:
            rk["category"] = rk["mean"].apply(lambda v: cat_for(v, bins))
            bar = px.bar(rk, x="mean", y="city", color="category", orientation="h", color_discrete_map=CAT_COLOR,
                         category_orders={"category": CATS}, title=f"City ranking by mean {label}",
                         labels={"mean": label, "city": ""})
        else:
            bar = px.bar(rk, x="mean", y="city", orientation="h", title=f"City ranking by mean {label}",
                         labels={"mean": label, "city": ""})
        bar.update_layout(height=max(380, 30 * len(rk) + 120), legend_title_text="")
        c1.plotly_chart(bar, width="stretch")
        box = px.box(df, x="city", y=metric, color="city", color_discrete_map=ncolor, points=False,
                     title=f"{label} distribution by city")
        box.update_layout(height=max(380, 30 * len(rk) + 120), showlegend=False, xaxis_title="", yaxis_title=label)
        c2.plotly_chart(box, width="stretch")

        if bins:
            share = df.groupby("city")["category"].value_counts(normalize=True).mul(100).rename("pct").reset_index()
            sb = px.bar(share, x="city", y="pct", color="category", color_discrete_map=CAT_COLOR,
                        category_orders={"category": CATS}, title="% of days in each CPCB category")
            sb.update_layout(height=380, yaxis_title="%", xaxis_title="", legend_title_text="")
            st.plotly_chart(sb, width="stretch")

        piv = df.pivot_table(index="city", columns="year", values=metric, aggfunc="mean")
        piv.columns = piv.columns.astype(str)
        hm = px.imshow(piv, text_auto=".0f", aspect="auto", color_continuous_scale="YlOrRd",
                       labels=dict(x="Year", y="", color=label), title=f"City × year: mean {label}")
        hm.update_layout(height=max(320, 34 * len(piv) + 120))
        st.plotly_chart(hm, width="stretch")
        st.dataframe(stats.rename(columns={"mean": "Mean", "median": "Median", "p95": "95th pct", "max": "Max"}),
                     width="stretch")

    # ---- seasonality
    with t3:
        mp = df.groupby(["city", "month"])[metric].mean().reset_index()
        ml = px.line(mp, x="month", y=metric, color="city", markers=True, color_discrete_map=ncolor,
                     title=f"Average {label} by month of year", labels={metric: label, "month": ""})
        ml.update_layout(height=400, xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
                                                ticktext=[calendar.month_abbr[m] for m in range(1, 13)]))
        st.plotly_chart(ml, width="stretch")

        se = df.groupby(["city", "season"])[metric].mean().reset_index()
        sb = px.bar(se, x="city", y=metric, color="season", barmode="group",
                    category_orders={"season": SEASON_ORDER},
                    color_discrete_map={"Winter": "#4c78a8", "Summer": "#f58518", "Monsoon": "#54a24b",
                                        "Post-monsoon": "#b279a2"},
                    title=f"Mean {label} by season", labels={metric: label, "city": ""})
        sb.update_layout(height=380, legend_title_text="")
        st.plotly_chart(sb, width="stretch")

        if subdaily:
            hp_ = df.assign(hour=df.date.dt.hour).groupby(["city", "hour"])[metric].mean().reset_index()
            hl = px.line(hp_, x="hour", y=metric, color="city", markers=True, color_discrete_map=ncolor,
                         title=f"Average {label} by hour of day", labels={metric: label, "hour": "Hour of day"})
            hl.update_layout(height=380)
            st.plotly_chart(hl, width="stretch")

        hc = st.selectbox("Month × year heatmap for", sel)
        p2 = df[df.city == hc].pivot_table(index="year", columns="month", values=metric, aggfunc="mean")
        p2.columns = [calendar.month_abbr[m] for m in p2.columns]
        p2.index = p2.index.astype(str)
        h2 = px.imshow(p2, text_auto=".0f", aspect="auto", color_continuous_scale="YlOrRd",
                       labels=dict(x="Month", y="Year", color=label), title=f"{hc}: mean {label} by month and year")
        h2.update_layout(height=420)
        st.plotly_chart(h2, width="stretch")
        st.caption("Indian cities typically peak in winter (Nov to Feb) and are cleanest in the monsoon (Jun to Sep).")

    # ---- pollutants
    with t4:
        if len(pols) < 2:
            st.info("This file only has one pollutant column, so there is nothing to compare.")
        else:
            c1, c2 = st.columns(2)
            cmx = px.imshow(df[pols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Pollutant correlation matrix")
            cmx.update_layout(height=430)
            c1.plotly_chart(cmx, width="stretch")
            samp = df.sample(min(3000, len(df)), random_state=0)
            others = [p for p in pols if p != metric]
            xp = c2.selectbox("Compare against", others, format_func=HIST_LABEL.get)
            sc = px.scatter(samp, x=xp, y=metric, color="city", color_discrete_map=ncolor, opacity=.5,
                            title=f"{HIST_LABEL[xp]} vs {label}", labels={xp: HIST_LABEL[xp], metric: label})
            sc.update_layout(height=380)
            c2.plotly_chart(sc, width="stretch")

            allp = df.groupby("city")[pols].mean()
            scaled = (allp - allp.min()) / (allp.max() - allp.min()).replace(0, np.nan)
            hp = go.Figure(go.Heatmap(z=scaled.values, x=[p.upper() for p in allp.columns], y=allp.index,
                                      colorscale="YlOrRd", text=allp.round(1).astype(str).values,
                                      texttemplate="%{text}", showscale=False))
            hp.update_layout(height=max(360, 34 * len(allp) + 120),
                             title="Mean pollutant levels by city (numbers = value, colour = relative within each pollutant)")
            st.plotly_chart(hp, width="stretch")

            ex = [(p, lim) for p, lim in NAAQS_24H.items() if p in pols]
            if ex:
                rows = [{"city": c, "pollutant": p.upper(), "pct": (g[p] > lim).mean() * 100}
                        for c, g in df.groupby("city") for p, lim in ex]
                eb = px.bar(pd.DataFrame(rows), x="city", y="pct", color="pollutant", barmode="group",
                            title="% of days above the India NAAQS 24-hour limit (" +
                                  ", ".join(f"{p.upper()} {lim}" for p, lim in ex) + " µg/m³)",
                            labels={"pct": "% of days", "city": ""})
                eb.update_layout(height=380, legend_title_text="")
                st.plotly_chart(eb, width="stretch")

    # ---- map
    with t5:
        coords = city_coords(d)
        mp = df.groupby("city")[metric].mean().rename("value").reset_index()
        mp["lat"] = mp.city.map(lambda c: coords.get(c, (np.nan, np.nan))[0])
        mp["lon"] = mp.city.map(lambda c: coords.get(c, (np.nan, np.nan))[1])
        miss = mp[mp.lat.isna()].city.tolist()
        mp = mp.dropna(subset=["lat", "lon"])
        if mp.empty:
            st.info("No coordinates known for these cities. Add `lat` and `lon` columns to your CSV to see a map.")
        else:
            mp["value"] = mp.value.round(1)
            kw = dict(lat="lat", lon="lon", size="value", hover_name="city", hover_data={"value": True, "lat": False, "lon": False},
                      zoom=3.6, size_max=34, map_style="open-street-map", height=620,
                      center={"lat": float(mp.lat.mean()), "lon": float(mp.lon.mean())})
            if bins:
                mp["category"] = mp.value.apply(lambda v: cat_for(v, bins))
                fig = px.scatter_map(mp, color="category", color_discrete_map=CAT_COLOR, category_orders={"category": CATS}, **kw)
            else:
                fig = px.scatter_map(mp, color="value", color_continuous_scale="YlOrRd", **kw)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, title_text=""))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Each dot is a city, sized and coloured by its mean {label} for the selected dates."
                       + (f" Not shown (no coordinates): {', '.join(miss)}." if miss else ""))

    # ---- data
    with t6:
        st.subheader("Data quality")
        qa, qb = st.columns([1, 2])
        qa.dataframe(pd.DataFrame({"Check": list(q), "Count": list(q.values())}), width="stretch", hide_index=True)
        qb.markdown("**Cleaning steps:** parse dates → drop duplicate city-day rows → treat negative readings as missing → "
                    "interpolate short gaps (up to 7 days) within each city → forward/back-fill the rest.")
        show = df.drop(columns=["year", "month"])
        st.dataframe(show.tail(1000), width="stretch", hide_index=True)
        st.download_button("⬇️ Download filtered data (CSV)", show.to_csv(index=False).encode(),
                           "filtered_city_history.csv", "text/csv")
    st.stop()


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌫️ Controls")
MODES = ["Simulated demo", "Live: CPCB via data.gov.in", "Upload CSV"]
mode = st.sidebar.radio("Data source", MODES)

if mode == MODES[1]:
    try:
        key = st.secrets.get("DATA_GOV_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = st.sidebar.text_input("data.gov.in API key", type="password",
                                    help="Free: register at data.gov.in, then copy the key from My Account. "
                                         "On a deployed app, store it in Streamlit Secrets as DATA_GOV_API_KEY.")
    fetch_state = st.sidebar.selectbox("State to fetch", ["Maharashtra", "All India"])
    if st.sidebar.button("🔄 Refresh now"):
        fetch_cpcb.clear()
    if not key:
        st.title("IoT Based Air Quality Dashboard")
        st.info("Enter your data.gov.in API key in the sidebar to load live CPCB readings.")
        st.stop()
    try:
        live = fetch_cpcb(key.strip(), fetch_state)
    except RuntimeError as e:
        st.title("IoT Based Air Quality Dashboard")
        st.error(str(e))
        st.stop()
    snapshot_view(live, "live CPCB feed (data.gov.in), cached 15 min")   # stops afterwards

elif mode == MODES[2]:
    up = st.sidebar.file_uploader("Upload a CSV", type="csv",
                                  help="Accepted formats: (1) IoT time series: timestamp, node, pm25, co2, temp, humidity "
                                       "(optional: region, lat, lon); (2) daily city history: city, date, aqi and/or pm25 "
                                       "(+ pm10, no2, so2, co, o3); (3) CPCB snapshot from data.gov.in.")
    if up is None:
        st.title("IoT Based Air Quality Dashboard")
        st.info("Upload a CSV in the sidebar, or switch the data source to *Simulated demo* or *Live*.")
        st.stop()
    try:
        raw = pd.read_csv(up)
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
        st.stop()
    raw = normalize_columns(raw)
    cols_up = set(raw.columns)
    if {"pollutant_id", "pollutant_avg", "city", "state"} <= cols_up:
        snapshot_view(raw)          # data.gov.in / CPCB snapshot format (stops afterwards)
    if is_historical(cols_up):
        historical_view(raw)        # daily city AQI history (stops afterwards)
    missing = {"timestamp", "node", *SENSORS} - cols_up
    if missing:
        st.title("IoT Based Air Quality Dashboard")
        st.sidebar.error("This file is not in a supported format. Found columns: "
                         f"{', '.join(map(str, raw.columns[:12]))}{' ...' if raw.shape[1] > 12 else ''}. "
                         "Supported: (1) time series: timestamp, node, pm25, co2, temp, humidity; "
                         "(2) daily city history: city, date, aqi and/or pm25; "
                         "(3) CPCB snapshot: state, city, station, pollutant_id, pollutant_avg, latitude, longitude.")
        st.error(f"Unsupported CSV format. For a time-series file the missing columns are: {', '.join(sorted(missing))}.")
        st.stop()
    source = "Uploaded CSV"
else:
    raw, source = simulate(), f"Simulated: {len(CITIES)} Maharashtra cities, 14 days"

df0, faults = clean(raw)
contam = st.sidebar.slider("Anomaly sensitivity (contamination)", 0.005, 0.05, 0.01, 0.005)
df_all = detect_anomalies(df0, contam)
meta = build_meta(df_all)
df_all["region"] = df_all.node.map(meta.region)

regions_avail = sorted(df_all.region.unique())
sel_regions = st.sidebar.multiselect("Regions", regions_avail, default=regions_avail)
pool = sorted(df_all[df_all.region.isin(sel_regions)].node.unique())
defaults = [c for c in DEFAULT_PICK if c in pool] or pool[:5]
nodes = st.sidebar.multiselect("Cities / sensor nodes", pool, default=defaults)

dmin, dmax = df_all.timestamp.dt.date.min(), df_all.timestamp.dt.date.max()
rng_dates = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
d0, d1 = rng_dates if isinstance(rng_dates, tuple) and len(rng_dates) == 2 else (dmin, dmax)
RES = {"15 min (raw)": None, "1 hour": "1h", "6 hours": "6h", "1 day": "1D"}
res = st.sidebar.selectbox("Time resolution", list(RES), index=1)
show_bands = st.sidebar.checkbox("Show CPCB category bands", True)
st.sidebar.caption(f"Data source: {source}")

if not sel_regions:
    st.warning("Select at least one region in the sidebar.")
    st.stop()

df_state = df_all[(df_all.region.isin(sel_regions)) &
                  (df_all.timestamp.dt.date >= d0) & (df_all.timestamp.dt.date <= d1)]
df = df_state[df_state.node.isin(nodes)]
if not nodes or df.empty:
    st.warning("Select at least one city in the sidebar.")
    st.stop()

ncolor = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(sorted(df_all.node.unique()))}

# ------------------------------------------------------------------ header + KPIs
st.title("IoT Based Air Quality Dashboard")
st.caption("IoT sensor analytics across Maharashtra: PM2.5, CO₂, temperature, humidity · "
           "CPCB (India) PM2.5 categories" + (" · ⚠️ simulated demo data" if mode == MODES[0] else ""))

if len(nodes) <= 5:
    cols = st.columns(len(nodes))
    for c, n in zip(cols, nodes):
        g = df[df.node == n]
        last = g.iloc[-1]
        cat = category(last.pm25)
        c.metric(f"{n}: latest PM2.5", f"{last.pm25:.0f} µg/m³", f"{delta_1h(g):+.0f} vs 1h ago",
                 delta_color="inverse")
        c.markdown(f"<span style='background:{CAT_COLOR[cat]};color:{'#000' if cat in ('Moderate', 'Satisfactory') else '#fff'};"
                   f"padding:2px 10px;border-radius:10px;font-size:13px'>{cat}</span>", unsafe_allow_html=True)
else:
    latest = (df.sort_values("timestamp").groupby("node").last()[["region", "pm25", "co2", "temp", "humidity"]]
              .assign(category=lambda x: x.pm25.apply(category)).round(1))
    with st.expander(f"Latest readings for {len(nodes)} selected cities", expanded=False):
        st.dataframe(latest, width="stretch")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Mean PM2.5 (selection)", f"{df.pm25.mean():.1f} µg/m³")
k2.metric("Peak PM2.5", f"{df.pm25.max():.0f} µg/m³")
k3.metric("Time in 'Poor' or worse", f"{(df.pm25 > 90).mean() * 100:.1f}%")
k4.metric("Anomalies flagged", f"{int(df.anomaly.sum())}")

# ------------------------------------------------------------------ tabs
t0, t1, t2, t3, t4, t5, t6 = st.tabs(["🗺️ Maharashtra map", "📈 Overview", "🕒 Patterns", "🔗 Relationships",
                                      "🔮 Forecast", "🧹 Data quality", "🗂️ Data"])

with t0:
    basis = st.radio("Show", ["Average over selected dates", "Latest reading"], horizontal=True)
    if basis.startswith("Average"):
        agg = df_state.groupby("node").pm25.mean()
    else:
        agg = df_state.sort_values("timestamp").groupby("node").pm25.last()
    mp = agg.rename("pm25").reset_index()
    mp = mp.join(meta, on="node")
    mp["category"] = mp.pm25.apply(category)
    mp["pm25"] = mp.pm25.round(1)

    plot_mp = mp.dropna(subset=["lat", "lon"])
    if plot_mp.empty:
        st.info("No coordinates available for the map. Add `lat` and `lon` columns to your CSV.")
    else:
        fig = px.scatter_map(plot_mp, lat="lat", lon="lon", color="category", size="pm25",
                             hover_name="node", hover_data={"pm25": True, "region": True, "lat": False, "lon": False},
                             color_discrete_map=CAT_COLOR, category_orders={"category": CATS},
                             zoom=5.2, center={"lat": 19.4, "lon": 76.4}, size_max=26,
                             map_style="open-street-map", height=620)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, width="stretch")

    w, c_ = mp.loc[mp.pm25.idxmax()], mp.loc[mp.pm25.idxmin()]
    a1, a2, a3 = st.columns(3)
    a1.metric("Most polluted", f"{w.node}", f"{w.pm25:.0f} µg/m³", delta_color="off")
    a2.metric("Cleanest", f"{c_.node}", f"{c_.pm25:.0f} µg/m³", delta_color="off")
    a3.metric("Cities at 'Poor' or worse", f"{int((mp.pm25 > 90).sum())} of {len(mp)}")

    r1, r2 = st.columns(2)
    rk = mp.sort_values("pm25")
    bar = px.bar(rk, x="pm25", y="node", color="category", orientation="h", color_discrete_map=CAT_COLOR,
                 category_orders={"category": CATS}, title="City ranking by PM2.5", labels={"pm25": "µg/m³", "node": ""})
    bar.add_vline(x=60, line_dash="dash", line_color="grey")
    bar.update_layout(height=max(380, 22 * len(rk) + 120), showlegend=False)
    r1.plotly_chart(bar, width="stretch")
    reg = df_state.groupby("region").agg(mean_pm25=("pm25", "mean"), peak=("pm25", "max"),
                                         poor_pct=("pm25", lambda s: (s > 90).mean() * 100)).round(1)
    rb = px.box(df_state, x="region", y="pm25", color="region", points=False, title="PM2.5 by region")
    rb.update_layout(height=380, showlegend=False, xaxis_title="")
    r2.plotly_chart(rb, width="stretch")
    r2.dataframe(reg.rename(columns={"mean_pm25": "Mean PM2.5", "peak": "Peak", "poor_pct": "% time Poor+"}),
                 width="stretch")

with t1:
    sensor = st.radio("Sensor", SENSORS, horizontal=True, format_func=SENSOR_LABEL.get)
    if len(nodes) > 8:
        st.caption("Many cities selected: the chart may get crowded. Use the legend to hide lines, or pick fewer cities.")
    rule = RES[res]
    fig = go.Figure()
    if sensor == "pm25" and show_bands:
        top = min(float(df.pm25.max()) * 1.1, 400)
        for lo, hi, col in zip(BINS[:-1], BINS[1:], COLS):
            if lo < top:
                fig.add_hrect(y0=lo, y1=min(hi, top), fillcolor=col, opacity=.10, line_width=0)
    for n in nodes:
        g = df[df.node == n].set_index("timestamp")
        s = g[sensor] if rule is None else g[sensor].resample(rule).mean()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=n, mode="lines",
                                 line=dict(width=1.3, color=ncolor[n])))
        if len(nodes) <= 4:
            an = g[g.anomaly]
            if len(an):
                fig.add_trace(go.Scatter(x=an.index, y=an[sensor], mode="markers", name=f"{n} anomaly",
                                         marker=dict(color="red", size=6, symbol="x")))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.1), yaxis_title=SENSOR_LABEL[sensor])
    st.plotly_chart(fig, width="stretch")
    if len(nodes) > 4:
        st.caption("Anomaly markers are shown when 4 or fewer cities are selected.")

    c1, c2 = st.columns(2)
    box = px.box(df, x="node", y="pm25", color="node", color_discrete_map=ncolor,
                 title="PM2.5 distribution by city", points=False)
    box.add_hline(y=60, line_dash="dash", line_color="grey", annotation_text="NAAQS 24h limit (60)")
    box.update_layout(showlegend=False, height=380, margin=dict(t=50), xaxis_title="")
    c1.plotly_chart(box, width="stretch")
    share = (df.groupby("node").category.value_counts(normalize=True).mul(100).rename("pct").reset_index())
    bar = px.bar(share, x="node", y="pct", color="category", color_discrete_map=CAT_COLOR,
                 category_orders={"category": CATS}, title="% of time in each CPCB category")
    bar.update_layout(height=380, margin=dict(t=50), yaxis_title="%", xaxis_title="")
    c2.plotly_chart(bar, width="stretch")

with t2:
    prof = df.groupby(["node", "hour"]).pm25.mean().reset_index()
    f = px.line(prof, x="hour", y="pm25", color="node", markers=True, color_discrete_map=ncolor,
                title="Average PM2.5 by hour of day: rush-hour peaks")
    f.update_layout(height=380, yaxis_title="µg/m³")
    st.plotly_chart(f, width="stretch")

    ch = df.groupby(["node", "hour"]).pm25.mean().unstack()
    cmap = px.imshow(ch, aspect="auto", color_continuous_scale="YlOrRd",
                     labels=dict(x="Hour of day", y="", color="µg/m³"), title="City × hour PM2.5 heatmap")
    cmap.update_layout(height=max(300, 26 * len(ch) + 120))
    st.plotly_chart(cmap, width="stretch")

    hn = st.selectbox("Day-by-day heatmap for", nodes)
    piv = df[df.node == hn].pivot_table(index="hour", columns="date", values="pm25")
    hm = px.imshow(piv, aspect="auto", origin="lower", color_continuous_scale="YlOrRd",
                   labels=dict(x="Date", y="Hour", color="µg/m³"), title=f"{hn}: PM2.5 by hour × day")
    hm.update_layout(height=420)
    st.plotly_chart(hm, width="stretch")

with t3:
    c1, c2 = st.columns(2)
    cm = df[SENSORS].corr()
    chm = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Sensor correlation matrix")
    chm.update_layout(height=420)
    c1.plotly_chart(chm, width="stretch")
    samp = df.sample(min(3000, len(df)), random_state=0)
    sc = px.scatter(samp, x="humidity", y="pm25", color="node", color_discrete_map=ncolor,
                    opacity=.5, title="Humidity vs PM2.5")
    sc.update_layout(height=420)
    c2.plotly_chart(sc, width="stretch")
    st.info("Values near +1 or -1 mean two sensors move together (or oppositely). Temperature and humidity "
            "are strongly negatively correlated; PM2.5 rises with CO₂ because both follow traffic.")

with t4:
    c1, c2 = st.columns(2)
    all_nodes = sorted(df_all.node.unique())
    fn = c1.selectbox("Forecast city", all_nodes, index=all_nodes.index("Pune") if "Pune" in all_nodes else 0)
    td = c2.slider("Hold-out test days", 1, 5, 3)
    yte, pred, base, m, imp = forecast(df_all, fn, td)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Random Forest MAE", f"{m['RF MAE']:.2f}")
    m2.metric("Random Forest R²", f"{m['RF R²']:.3f}")
    m3.metric("Baseline MAE", f"{m['Baseline MAE']:.2f}")
    m4.metric("Baseline R²", f"{m['Baseline R²']:.3f}")
    ff = go.Figure()
    ff.add_trace(go.Scatter(x=yte.index, y=yte, name="Actual", line=dict(color="black")))
    ff.add_trace(go.Scatter(x=pred.index, y=pred, name="Random Forest", line=dict(color="#d62728")))
    ff.add_trace(go.Scatter(x=base.index, y=base, name="Baseline (last hour)", line=dict(color="grey", dash="dash")))
    ff.update_layout(height=400, title=f"{fn}: next-hour PM2.5 forecast on hold-out period",
                     hovermode="x unified", yaxis_title="µg/m³")
    st.plotly_chart(ff, width="stretch")
    fi = px.bar(imp, orientation="h", title="Feature importance", labels={"value": "Importance", "index": ""})
    fi.update_layout(showlegend=False, height=340)
    st.plotly_chart(fi, width="stretch")
    st.caption("Baseline = predicting that next hour equals the current hour. Simulated data is smoother "
               "than real sensor data, so expect lower accuracy on a real deployment.")

with t5:
    fq = pd.DataFrame({"Fault type": list(faults), "Count": list(faults.values())})
    st.plotly_chart(px.bar(fq, x="Count", y="Fault type", orientation="h", text="Count",
                           title="Faults found in raw data and how they were handled")
                    .update_traces(marker_color="#c44e52").update_layout(height=340), width="stretch")
    st.markdown("""
**Cleaning steps:** remove duplicate MQTT messages → mark impossible values (PM2.5 > 500, temperature < -40 °C) as missing →
interpolate short gaps (up to 6 readings) → forward/back-fill the rest.
""")

with t6:
    show = df.drop(columns=["hour", "date"])
    st.dataframe(show.tail(1000), width="stretch", hide_index=True)
    st.download_button("⬇️ Download filtered data (CSV)", show.to_csv(index=False).encode(),
                       "filtered_air_quality.csv", "text/csv")