"""
IoT Urban Air Quality - interactive Streamlit dashboard
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="IoT Air Quality Dashboard", page_icon="🌫️", layout="wide")

# ------------------------------------------------------------------ constants
NODES = {"Akurdi": 1.00, "Hinjewadi": 1.15, "Swargate": 1.35}
BINS = [0, 30, 60, 90, 120, 250, 1e4]
CATS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
COLS = ["#2e9e4f", "#9ccc3c", "#f2d12e", "#f28c28", "#d9402b", "#7a1f1f"]
CAT_COLOR = dict(zip(CATS, COLS))
NODE_COLOR = {"Akurdi": "#1f77b4", "Hinjewadi": "#ff7f0e", "Swargate": "#2ca02c"}
SENSORS = ["pm25", "co2", "temp", "humidity"]


def category(v):
    return CATS[int(np.digitize(v, BINS[1:-1]))]


# ------------------------------------------------------------------ data
@st.cache_data
def simulate(seed: int = 42, days: int = 14):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-11-09", periods=days * 288, freq="5min")
    frames = []
    for name, k in NODES.items():
        h = np.asarray(idx.hour + idx.minute / 60, dtype=float)
        rush = 28 * np.exp(-((h - 9) ** 2) / 4) + 35 * np.exp(-((h - 19.5) ** 2) / 5)
        inv = 18 * np.exp(-((h - 3) ** 2) / 12)
        temp = 24 + 7 * np.sin((h - 9) / 24 * 2 * np.pi) + rng.normal(0, .4, len(idx))
        hum = np.clip(68 - 1.6 * (temp - 24) + rng.normal(0, 2, len(idx)), 25, 99)
        pm25 = k * (45 + rush + inv + 0.35 * (hum - 60)) + rng.normal(0, 4, len(idx))
        for s in rng.choice(len(idx) - 40, 5, replace=False):
            pm25[s:s + rng.integers(8, 36)] += rng.uniform(60, 130)
        co2 = 420 + 1.6 * rush * k + rng.normal(0, 12, len(idx))
        df = pd.DataFrame({"timestamp": idx, "node": name, "pm25": pm25,
                           "co2": co2, "temp": temp, "humidity": hum})
        n = len(df)
        df.loc[rng.choice(n, int(.02 * n), replace=False), "pm25"] = np.nan
        df.loc[rng.choice(n, 25, replace=False), "pm25"] = rng.uniform(900, 1500, 25)
        df.loc[rng.choice(n, 12, replace=False), "temp"] = -127
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return pd.concat([raw, raw.sample(150, random_state=1)])


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


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌫️ Controls")
up = st.sidebar.file_uploader("Upload your own sensor CSV (optional)", type="csv",
                              help="Columns: timestamp, node, pm25, co2, temp, humidity")
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
    raw, source = simulate(), "Simulated (3 Pune nodes, 14 days)"

df0, faults = clean(raw)
contam = st.sidebar.slider("Anomaly sensitivity (contamination)", 0.005, 0.05, 0.01, 0.005)
df_all = detect_anomalies(df0, contam)

all_nodes = sorted(df_all.node.unique())
nodes = st.sidebar.multiselect("Sensor nodes", all_nodes, default=all_nodes)
dmin, dmax = df_all.timestamp.dt.date.min(), df_all.timestamp.dt.date.max()
rng_dates = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(rng_dates, tuple) and len(rng_dates) == 2:
    d0, d1 = rng_dates
else:
    d0, d1 = dmin, dmax
res = st.sidebar.selectbox("Time resolution", ["5 min (raw)", "15 min", "1 hour"], index=1)
show_bands = st.sidebar.checkbox("Show CPCB category bands", True)
st.sidebar.caption(f"Data source: {source}")

if not nodes:
    st.warning("Select at least one sensor node in the sidebar.")
    st.stop()

df = df_all[(df_all.node.isin(nodes)) & (df_all.timestamp.dt.date >= d0) & (df_all.timestamp.dt.date <= d1)]
if df.empty:
    st.warning("No data for this selection.")
    st.stop()

# ------------------------------------------------------------------ header + KPIs
st.title("IoT Urban Air Quality Dashboard")
st.caption("Multi-node sensor analytics: PM2.5, CO₂, temperature, humidity · CPCB (India) PM2.5 categories")

cols = st.columns(len(nodes))
for c, n in zip(cols, nodes):
    g = df[df.node == n]
    last = g.iloc[-1]
    cat = category(last.pm25)
    prev = g.iloc[-13].pm25 if len(g) > 13 else last.pm25
    c.metric(f"{n}: latest PM2.5", f"{last.pm25:.0f} µg/m³", f"{last.pm25 - prev:+.0f} vs 1h ago", delta_color="inverse")
    c.markdown(f"<span style='background:{CAT_COLOR[cat]};color:{'#000' if cat in ('Moderate','Satisfactory') else '#fff'};"
               f"padding:2px 10px;border-radius:10px;font-size:13px'>{cat}</span>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Mean PM2.5", f"{df.pm25.mean():.1f} µg/m³")
k2.metric("Peak PM2.5", f"{df.pm25.max():.0f} µg/m³")
k3.metric("Time in 'Poor' or worse", f"{(df.pm25 > 90).mean() * 100:.1f}%")
k4.metric("Anomalies flagged", f"{int(df.anomaly.sum())}")

# ------------------------------------------------------------------ tabs
t1, t2, t3, t4, t5, t6 = st.tabs(["📈 Overview", "🕒 Patterns", "🔗 Relationships",
                                  "🔮 Forecast", "🧹 Data quality", "🗂️ Data"])

with t1:
    sensor = st.radio("Sensor", SENSORS, horizontal=True,
                      format_func=lambda s: {"pm25": "PM2.5 (µg/m³)", "co2": "CO₂ (ppm)",
                                             "temp": "Temperature (°C)", "humidity": "Humidity (%)"}[s])
    rule = {"5 min (raw)": None, "15 min": "15min", "1 hour": "1h"}[res]
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
                                 line=dict(width=1.3, color=NODE_COLOR.get(n))))
        an = g[g.anomaly]
        if len(an):
            fig.add_trace(go.Scatter(x=an.index, y=an[sensor], mode="markers", name=f"{n} anomaly",
                                     marker=dict(color="red", size=6, symbol="x")))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    box = px.box(df, x="node", y="pm25", color="node", color_discrete_map=NODE_COLOR,
                 title="PM2.5 distribution by location", points=False)
    box.add_hline(y=60, line_dash="dash", line_color="grey", annotation_text="NAAQS 24h limit (60)")
    box.update_layout(showlegend=False, height=360, margin=dict(t=50))
    c1.plotly_chart(box, width="stretch")
    share = (df.groupby("node").category.value_counts(normalize=True).mul(100)
               .rename("pct").reset_index())
    bar = px.bar(share, x="node", y="pct", color="category", color_discrete_map=CAT_COLOR,
                 category_orders={"category": CATS}, title="% of time in each CPCB category")
    bar.update_layout(height=360, margin=dict(t=50), yaxis_title="%")
    c2.plotly_chart(bar, width="stretch")

with t2:
    prof = df.groupby(["node", "hour"]).pm25.mean().reset_index()
    f = px.line(prof, x="hour", y="pm25", color="node", markers=True, color_discrete_map=NODE_COLOR,
                title="Average PM2.5 by hour of day: rush-hour peaks")
    f.update_layout(height=380, yaxis_title="µg/m³")
    st.plotly_chart(f, width="stretch")

    hn = st.selectbox("Heatmap node", nodes)
    piv = df[df.node == hn].pivot_table(index="hour", columns="date", values="pm25")
    hm = px.imshow(piv, aspect="auto", origin="lower", color_continuous_scale="YlOrRd",
                   labels=dict(x="Date", y="Hour", color="µg/m³"), title=f"{hn}: PM2.5 by hour × day")
    hm.update_layout(height=420)
    st.plotly_chart(hm, width="stretch")
    worst = prof.loc[prof.groupby("node").pm25.idxmax()][["node", "hour", "pm25"]]
    st.dataframe(worst.rename(columns={"node": "Node", "hour": "Worst hour", "pm25": "Avg PM2.5 then"})
                 .round(1), hide_index=True)

with t3:
    c1, c2 = st.columns(2)
    cm = df[SENSORS].corr()
    ch = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                   title="Sensor correlation matrix")
    ch.update_layout(height=420)
    c1.plotly_chart(ch, width="stretch")
    samp = df.sample(min(3000, len(df)), random_state=0)
    sc = px.scatter(samp, x="humidity", y="pm25", color="node", color_discrete_map=NODE_COLOR,
                    opacity=.5, title="Humidity vs PM2.5", trendline=None)
    sc.update_layout(height=420)
    c2.plotly_chart(sc, width="stretch")
    st.info("Reading the matrix: values near +1 or -1 mean two sensors move together (or oppositely). "
            "Temperature and humidity are strongly negatively correlated; PM2.5 rises with CO₂ because "
            "both follow traffic.")

with t4:
    c1, c2 = st.columns(2)
    fn = c1.selectbox("Forecast node", all_nodes)
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
    ff.update_layout(height=400, title="Next-hour PM2.5 forecast on hold-out period",
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
    st.dataframe(df.drop(columns=["hour", "date"]).tail(1000), width="stretch", hide_index=True)
    st.download_button("⬇️ Download filtered data (CSV)",
                       df.drop(columns=["hour", "date"]).to_csv(index=False).encode(),
                       "filtered_air_quality.csv", "text/csv")
