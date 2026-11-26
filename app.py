"""
IoT Air Quality Dashboard (Streamlit)
Run:  streamlit run app.py
NOTE: data is SIMULATED for demonstration. Upload a CSV to analyse real sensor data.
"""
import numpy as np
import pandas as pd
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

BINS = [0, 30, 60, 90, 120, 250, 1e4]
CATS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
COLS = ["#2e9e4f", "#9ccc3c", "#f2d12e", "#f28c28", "#d9402b", "#7a1f1f"]
CAT_COLOR = dict(zip(CATS, COLS))
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


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌫️ Controls")
up = st.sidebar.file_uploader("Upload your own sensor CSV (optional)", type="csv",
                              help="Columns: timestamp, node, pm25, co2, temp, humidity. "
                                   "Optional: region, lat, lon (for the map).")
if up is not None:
    try:
        raw = pd.read_csv(up)
        missing = {"timestamp", "node", *SENSORS} - set(raw.columns)
        if missing:
            st.sidebar.error(f"Missing columns: {', '.join(sorted(missing))}")
            st.stop()
        source = "Uploaded CSV"
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
        st.stop()
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
           "CPCB (India) PM2.5 categories" + (" · ⚠️ simulated demo data" if up is None else ""))

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