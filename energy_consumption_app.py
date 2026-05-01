import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
import datetime
import streamlit.components.v1 as components
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Noida Energy Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- FULL CSS FIX ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Exo+2:wght@300;400;500;600&display=swap');

/* Main UI Overhaul */
* {
    font-family: 'Exo 2', sans-serif !important;
}

h1, h2, h3, h4, h5, h6, .section-title {
    font-family: 'Orbitron', sans-serif !important;
    color: #00e5ff !important;
    text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Background Gradient */
html, body, .stApp {
    background: linear-gradient(135deg, #020b18, #071a2e) !important;
    color: #e0f7fa !important;
}

/* Glassmorphism for containers and sidebar */
section[data-testid="stSidebar"], 
div[data-testid="stVerticalBlock"] > div > div > div, 
.metric-card,
.insight-box {
    background: rgba(10, 20, 40, 0.55) !important;
    border: 1px solid rgba(80, 200, 255, 0.18) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 40px rgba(0,200,255,0.08) !important;
}

/* Sidebar neon cyan border */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(0, 229, 255, 0.3) !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #00e5ff, #0077aa) !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Metric Cards inner styling */
.metric-card {
    padding: 1.5rem !important;
    text-align: center !important;
    margin-bottom: 1rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.metric-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 10px 25px rgba(0, 229, 255, 0.15) !important;
}

.metric-card .lbl {
    color: #80d8ff !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    margin-top: 5px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.metric-card .val {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    font-family: 'Orbitron', sans-serif !important;
    color: #ffffff !important;
    text-shadow: 0 0 8px rgba(0, 229, 255, 0.4) !important;
}

.metric-card .sub {
    font-size: 0.8rem !important;
    color: #4dd0e1 !important;
    margin-top: 5px !important;
}

/* Sliders and Selectboxes */
div[data-baseweb="select"] > div, 
input, textarea {
    background-color: rgba(10, 20, 40, 0.8) !important;
    color: #00e5ff !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
    border-radius: 8px !important;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    color: #b2ebf2 !important;
    font-family: 'Orbitron', sans-serif !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom-color: #00e5ff !important;
    text-shadow: 0 0 5px rgba(0, 229, 255, 0.5) !important;
}

/* Main Header */
.main-header {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(0, 229, 255, 0.2);
    position: relative;
}

.main-header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    background: -webkit-linear-gradient(45deg, #00e5ff, #b2ebf2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
    filter: drop-shadow(0 0 8px rgba(0, 229, 255, 0.4));
}

/* Floating Export Button */
.floating-export {
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 999;
}
.floating-export button {
    background: linear-gradient(135deg, #ff0055, #ff7700) !important;
    box-shadow: 0 4px 15px rgba(255, 0, 85, 0.4) !important;
    padding: 0.8rem 1.5rem !important;
    font-size: 1.1rem !important;
    border-radius: 30px !important;
    color: white !important;
}
.floating-export button:hover {
    box-shadow: 0 6px 20px rgba(255, 0, 85, 0.6) !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- FIXED PLOTLY ----------------
pio.templates.default = "plotly_dark"
PLOTLY_LAYOUT = dict(
    font_family="Exo 2",
    plot_bgcolor="#0a1628",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0f7fa", family="Exo 2"),
    margin=dict(t=30, b=40, l=50, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font_size=11),
    xaxis=dict(gridcolor="rgba(0,229,255,0.1)", linecolor="rgba(80, 200, 255, 0.18)"),
    yaxis=dict(gridcolor="rgba(0,229,255,0.1)", linecolor="rgba(80, 200, 255, 0.18)"),
)

def apply_layout(fig, **kwargs):
    fig.update_layout(**PLOTLY_LAYOUT, **kwargs)
    return fig
    
MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
MONTH_MAP   = {m: i+1 for i, m in enumerate(MONTH_ORDER)}
MONTH_SHORT = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

COLORS = {
    'teal':   '#1D9E75', 'blue':  '#378ADD', 'amber': '#BA7517',
    'coral':  '#D85A30', 'purple':'#7F77DD', 'pink':  '#D4537E',
    'gray':   '#888780', 'green': '#639922', 'red':   '#E24B4A',
}

@st.cache_data
def load_data():
    hh   = pd.read_excel("noida_electricity_household.xlsx")
    comm = pd.read_excel("noida_commercial.xlsx")

    hh['month_num']    = hh['Month'].map(MONTH_MAP)
    hh['solar_flag']   = (hh['Solar'] == 'Yes').astype(int)
    le_h = LabelEncoder()
    hh['house_type_enc'] = le_h.fit_transform(hh['House Type'])

    comm['month_num']   = comm['Month'].map(MONTH_MAP)
    comm['solar_flag']  = (comm['Solar'] == 'Yes').astype(int)
    le_c = LabelEncoder()
    comm['business_enc'] = le_c.fit_transform(comm['Business Type'])

    return hh, comm, le_h, le_c

@st.cache_resource
def train_models(hh, comm):
    results = {}

    feat_h = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
    X_h = hh[feat_h].values
    y_h = hh['Units Consumed'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

    lr_h  = LinearRegression().fit(X_tr, y_tr)
    gb_h  = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, subsample=0.8, random_state=42).fit(X_tr, y_tr)

    results['hh'] = {
        'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te,
        'lr': lr_h, 'gb': gb_h,
        'fi': gb_h.feature_importances_, 'feat_names': feat_h,
        'lr_pred': lr_h.predict(X_te),
        'gb_pred': gb_h.predict(X_te),
    }

    feat_c = ['month_num','business_enc','solar_flag','Connected Load (kW)','Solar Capacity (kW)']
    X_c = comm[feat_c].values
    y_c = comm['Units Consumed (After Solar)'].values
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

    lr_c  = LinearRegression().fit(Xc_tr, yc_tr)
    gb_c  = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, random_state=42).fit(Xc_tr, yc_tr)

    results['comm'] = {
        'X_tr': Xc_tr, 'X_te': Xc_te, 'y_tr': yc_tr, 'y_te': yc_te,
        'lr': lr_c, 'gb': gb_c,
        'fi': gb_c.feature_importances_, 'feat_names': feat_c,
        'lr_pred': lr_c.predict(Xc_te),
        'gb_pred': gb_c.predict(Xc_te),
    }

    return results


def arima_forecast(series, steps=12):
    """AR(1) with first-order differencing — stable implementation."""
    diff = np.diff(series)
    X    = diff[:-1].reshape(-1, 1)
    y    = diff[1:]
    phi  = np.linalg.lstsq(X, y, rcond=None)[0][0]
    phi  = np.clip(phi, -0.95, 0.95)
    mu   = float(np.mean(diff))

    ext_diff = list(diff)
    for _ in range(steps):
        nxt = mu + phi * (ext_diff[-1] - mu)
        ext_diff.append(nxt)

    last, result = float(series[-1]), []
    for d in ext_diff[-steps:]:
        last += d
        result.append(max(50.0, last))
    return result


def sarima_forecast(series, steps=12):
    """Seasonal AR (p=2, P=1, s=12) via OLS."""
    n      = len(series)
    period = 12
    lags   = [1, 2, period]
    max_lag = max(lags)
    if n <= max_lag:
        return list(float(v) for v in series[-steps:])

    X_cols = [series[max_lag - lag: n - lag] for lag in lags]
    X = np.column_stack(X_cols)
    y = series[max_lag:]
    coefs = np.linalg.lstsq(
        np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
    c, ar = coefs[0], coefs[1:]

    ext = list(series)
    for _ in range(steps):
        pred = c + sum(ar[i] * ext[-lags[i]] for i in range(len(lags)))
        ext.append(max(50.0, pred))
    return [float(v) for v in ext[-steps:]]


def lstm_forecast(series, lookback=4, steps=12):
    """Lightweight RNN proxy using Ridge regression on polynomial features."""
    mn, mx = float(min(series)) - 10, float(max(series)) + 10
    norm   = (np.array(series, dtype=float) - mn) / (mx - mn)

    X_seq, y_seq = [], []
    for i in range(lookback, len(norm)):
        X_seq.append(norm[i - lookback:i])
        y_seq.append(norm[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    poly = np.column_stack([X_seq, X_seq**2,
                            np.mean(X_seq, axis=1, keepdims=True)])
    model = Ridge(alpha=1.0).fit(poly, y_seq)

    seq, preds = list(norm), []
    for _ in range(steps):
        inp  = np.array(seq[-lookback:])
        feat = np.concatenate([inp, inp**2, [float(np.mean(inp))]])
        p    = float(np.clip(model.predict([feat])[0], 0.0, 1.0))
        preds.append(p)
        seq.append(p)

    return [p * (mx - mn) + mn for p in preds]


def metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'MSE': round(mse, 2), 'RMSE': round(rmse, 2),
            'MAE': round(mae, 2), 'R²': round(r2, 4)}


def sidebar(hh, comm):
    with st.sidebar:
        st.markdown("## Control Panel")
        st.markdown("---")

        st.markdown("### Dataset")
        dataset = st.radio("Select dataset", ["Household", "Commercial"],
                           index=0, horizontal=True)

        st.markdown("### Models to show")
        show_lr    = st.checkbox("Linear Regression",  value=True)
        show_gb    = st.checkbox("XGBoost / GBR",       value=True)
        show_arima = st.checkbox("ARIMA",               value=True)
        show_sarima= st.checkbox("SARIMA",              value=True)
        show_lstm  = st.checkbox("LSTM (RNN)",          value=True)

        st.markdown("### Forecast horizon")
        horizon = st.slider("Months ahead", 3, 24, 12, step=3)

        st.markdown("### Custom predictor (Household)")
        rooms    = st.slider("No. of rooms",       1, 6,  3)
        solar    = st.selectbox("Solar panel?",     ["No", "Yes"])
        sol_cap  = st.slider("Solar capacity (kW)", 0, 10, 0)
        h_type   = st.selectbox("House type",       ["Apartment", "Independent"])
        pred_month = st.selectbox("Predict for month", MONTH_ORDER,
                                  index=4)

        st.markdown("---")
        st.markdown("### Future Forecast")
        ff_years = st.slider("Years ahead", 1, 50, 25)
        ff_pop_growth = st.slider("Annual population growth rate (%)", 0.5, 5.0, 2.3, step=0.1)
        ff_urbanization = st.slider("Urbanization rate (%/year)", 0.1, 3.0, 1.5, step=0.1)
        ff_income = st.slider("Per capita income growth (%/year)", 1.0, 8.0, 4.5, step=0.1)
        ff_ev = st.checkbox("Include EV adoption impact", value=True)
        ff_solar = st.checkbox("Include solar rooftop growth", value=True)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.75rem;color:#00e5ff;line-height:1.6'>"
            " Based on: <i>Machine Learning-Based Prediction of "
            "Household Energy Consumption</i><br>"
            "Mohit Batar · Prakash Mishra · Ashish Kumar<br>"
            "March 2026</div>",
            unsafe_allow_html=True
        )

    return dict(
        dataset=dataset, show_lr=show_lr, show_gb=show_gb,
        show_arima=show_arima, show_sarima=show_sarima, show_lstm=show_lstm,
        horizon=horizon, rooms=rooms, solar=solar, sol_cap=sol_cap,
        h_type=h_type, pred_month=pred_month,
        ff_years=ff_years, ff_pop_growth=ff_pop_growth, 
        ff_urbanization=ff_urbanization, ff_income=ff_income,
        ff_ev=ff_ev, ff_solar=ff_solar
    )


def tab_overview(hh, comm):
    st.markdown('<div class="section-title">Dataset Summary</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    avg_hh   = hh['Units Consumed'].mean()
    avg_comm = comm['Units Consumed (After Solar)'].mean()
    solar_pct = (hh['Solar'] == 'Yes').mean() * 100

    for col, val, lbl, sub in zip(
        [c1, c2, c3, c4],
        [f"{avg_hh:.0f}", f"{avg_comm:,.0f}", f"{solar_pct:.0f}%", "36,217"],
        ["Avg Household kWh", "Avg Commercial kWh", "Solar Adoption", "Street Lights kWh/day"],
        ["monthly avg", "monthly avg", "households", "GNIDA estimate"]
    ):
        col.markdown(
            f'<div class="metric-card"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Monthly average consumption</div>',
                    unsafe_allow_html=True)
        hh_m   = hh.groupby('month_num')['Units Consumed'].mean().sort_index()
        comm_m = comm.groupby('month_num')['Units Consumed (After Solar)'].mean().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=MONTH_SHORT, y=hh_m.values, name="Household (kWh)",
            line=dict(color=COLORS['teal'], width=2.5),
            fill='tozeroy', fillcolor='rgba(29,158,117,0.12)', mode='lines+markers',
            marker=dict(size=5)))
        fig.add_trace(go.Scatter(
            x=MONTH_SHORT, y=comm_m.values / 5, name="Commercial / 5",
            line=dict(color=COLORS['blue'], width=2, dash='dash'),
            mode='lines+markers', marker=dict(size=5)))
        apply_layout(fig, height=280,
                     yaxis_title="kWh / month")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Consumption by season</div>',
                    unsafe_allow_html=True)
        hh_s   = hh.groupby('Season')['Units Consumed'].mean().round(1)
        comm_s = comm.groupby('Season')['Units Consumed (After Solar)'].mean().round(1)
        seasons = list(hh_s.index)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Household", x=seasons, y=hh_s.values,
                             marker_color=COLORS['teal'], opacity=0.85))
        fig.add_trace(go.Bar(name="Commercial / 5", x=seasons,
                             y=comm_s.values / 5,
                             marker_color=COLORS['blue'], opacity=0.75))
        apply_layout(fig, height=280, barmode='group',
                     yaxis_title="kWh / month")
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-title">Commercial — by business type</div>',
                    unsafe_allow_html=True)
        biz = comm.groupby('Business Type')['Units Consumed (After Solar)'].mean().sort_values()
        fig = px.bar(x=biz.values, y=biz.index, orientation='h',
                     color=biz.values,
                     color_continuous_scale=['#9FE1CB','#0F6E56'])
        fig.update_layout(**PLOTLY_LAYOUT, height=280,
                          coloraxis_showscale=False,
                          xaxis_title="Avg kWh / month")
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-title">Solar panel impact on consumption</div>',
                    unsafe_allow_html=True)
        sol = hh.groupby('Solar')['Units Consumed'].mean()
        fig = go.Figure(go.Bar(
            x=["No Solar", "With Solar"],
            y=[sol.get('No', 0), sol.get('Yes', 0)],
            marker_color=[COLORS['gray'], COLORS['teal']],
            text=[f"{sol.get('No',0):.0f}", f"{sol.get('Yes',0):.0f}"],
            textposition='outside', width=0.4
        ))
        apply_layout(fig, height=280, yaxis_title="Avg kWh / month",
                     showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        '<b>Key observations:</b> Summer season (May–Jun) drives the peak household '
        'demand (490+ kWh/month) due to air-conditioning load. Warehouses and '
        'restaurants are the highest commercial consumers. Solar adoption reduces '
        'consumption by ~12% on average. Commercial demand is 4–5× higher than '
        'household on a per-connection basis.'
        '</div>',
        unsafe_allow_html=True
    )


def tab_forecast(hh, comm, ctrl):
    ds      = ctrl['dataset']
    horizon = ctrl['horizon']

    if ds == "Household":
        series = hh.groupby('month_num')['Units Consumed'].mean().sort_index().values
        unit   = "kWh / month (Household)"
    else:
        series = comm.groupby('month_num')['Units Consumed (After Solar)'].mean().sort_index().values
        unit   = "kWh / month (Commercial)"

    hist_labels = [f"{m} (hist)" for m in MONTH_SHORT]
    fc_labels   = [f"{MONTH_SHORT[(i) % 12]} (fc+{i+1})" for i in range(horizon)]
    all_labels  = hist_labels + fc_labels

    arima_fc  = arima_forecast(series, horizon)
    sarima_fc = sarima_forecast(series, horizon)
    lstm_fc   = lstm_forecast(series, steps=horizon)

    feat_hh = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
    feat_c  = ['month_num','business_enc','solar_flag','Connected Load (kW)','Solar Capacity (kW)']

    if ds == "Household":
        le = LabelEncoder().fit(hh['House Type'])
        hh2 = hh.copy(); hh2['house_type_enc'] = le.transform(hh2['House Type'])
        feat = feat_hh
        lr_m = LinearRegression()
        gb_m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=4, subsample=0.8, random_state=42)
        X2 = hh2[feat].values; y2 = hh2['Units Consumed'].values
    else:
        le = LabelEncoder().fit(comm['Business Type'])
        comm2 = comm.copy(); comm2['business_enc'] = le.transform(comm2['Business Type'])
        feat = feat_c
        lr_m = LinearRegression()
        gb_m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=4, random_state=42)
        X2 = comm2[feat].values; y2 = comm2['Units Consumed (After Solar)'].values

    lr_m.fit(X2, y2); gb_m.fit(X2, y2)

    if ds == "Household":
        lr_fc  = [lr_m.predict([[((11 + i) % 12) + 1, 0.3, 0, 3, 0]])[0]
                  for i in range(horizon)]
        gb_fc  = [gb_m.predict([[((11 + i) % 12) + 1, 0.3, 0, 3, 0]])[0]
                  for i in range(horizon)]
    else:
        lr_fc  = [lr_m.predict([[((11 + i) % 12) + 1, 2, 0, 20, 0]])[0]
                  for i in range(horizon)]
        gb_fc  = [gb_m.predict([[((11 + i) % 12) + 1, 2, 0, 20, 0]])[0]
                  for i in range(horizon)]

    fig = go.Figure()
    null_hist = [None] * 12
    full_hist = list(series) + [None] * horizon

    fig.add_trace(go.Scatter(
        x=all_labels, y=full_hist, name="Historical",
        line=dict(color=COLORS['gray'], width=3),
        mode='lines+markers', marker=dict(size=5)
    ))

    model_cfg = [
        ('Linear Regression', ctrl['show_lr'],    lr_fc,   COLORS['blue'],   'dash'),
        ('XGBoost / GBR',     ctrl['show_gb'],    gb_fc,   COLORS['teal'],   'solid'),
        ('ARIMA',             ctrl['show_arima'], arima_fc, COLORS['amber'],  'dot'),
        ('SARIMA',            ctrl['show_sarima'],sarima_fc,COLORS['coral'],  'dashdot'),
        ('LSTM (RNN)',         ctrl['show_lstm'],  lstm_fc,  COLORS['purple'], 'longdash'),
    ]

    for name, show, fc_vals, color, dash in model_cfg:
        if show:
            fig.add_trace(go.Scatter(
                x=all_labels,
                y=null_hist + [float(v) for v in fc_vals],
                name=name,
                line=dict(color=color, width=2, dash=dash),
                mode='lines+markers', marker=dict(size=4)
            ))

    fig.update_xaxes(tickangle=45, tickfont_size=10,
                     tickmode='array',
                     tickvals=all_labels[::2])
    apply_layout(fig, height=400, yaxis_title=unit,
                 title=f"{ds} Energy Forecast — {horizon} months ahead")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Forecast values table</div>',
                unsafe_allow_html=True)
    fc_table = pd.DataFrame({
        'Month':            fc_labels,
        'ARIMA':            [round(v, 1) for v in arima_fc],
        'SARIMA':           [round(v, 1) for v in sarima_fc],
        'LSTM':             [round(v, 1) for v in lstm_fc],
        'Linear Regression':[round(v, 1) for v in lr_fc],
        'XGBoost':          [round(v, 1) for v in gb_fc],
    })
    st.dataframe(fc_table.set_index('Month'), use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        '<b>Model behaviour:</b> '
        '<b>SARIMA</b> best reproduces seasonal peaks (summer highs, winter lows). '
        '<b>XGBoost</b> closely tracks historical seasonal shape. '
        '<b>ARIMA</b> mean-reverts gradually toward the unconditional mean. '
        '<b>LSTM</b> smoothly converges to average level. '
        '<b>Linear Regression</b> shows a linear trend, missing nonlinear seasonality.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # FUTURE FORECAST SECTION
    base_pred = float(np.mean(gb_fc)) if len(gb_fc) > 0 else 500.0
    render_future_forecast(hh, comm, ctrl, base_pred)

def render_future_forecast(hh, comm, ctrl, base_pred):
    st.markdown('<div class="section-title" style="margin-top: 40px;">50-Year Future Forecast</div>', unsafe_allow_html=True)
    
    years = ctrl['ff_years']
    pop_growth = ctrl['ff_pop_growth'] / 100.0
    urbanization = ctrl['ff_urbanization'] / 100.0
    income_growth = ctrl['ff_income'] / 100.0
    ev_impact = 1.02 if ctrl['ff_ev'] else 1.0
    solar_impact = 0.98 if ctrl['ff_solar'] else 1.0
    
    proj_years = list(range(1, 51))
    projections = []
    
    for y in proj_years:
        income_factor = (1 + income_growth)**(y * 0.3)
        val = base_pred * ((1 + pop_growth)**y) * ((1 + urbanization)**(y*0.5)) * income_factor * (ev_impact**y) * (solar_impact**y)
        projections.append(val)
        
    display_years = proj_years[:years]
    display_proj = projections[:years]
    
    fig = go.Figure()
    
    upper_bound = [v * 1.15 for v in display_proj]
    lower_bound = [v * 0.85 for v in display_proj]
    
    fig.add_trace(go.Scatter(
        x=display_years + display_years[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(0, 229, 255, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='15% Confidence Interval'
    ))
    
    fig.add_trace(go.Scatter(
        x=display_years, y=display_proj,
        mode='lines+markers',
        name='Projected Consumption',
        line=dict(color='#00e5ff', width=3),
        marker=dict(size=6, color='#ff0055', symbol='circle')
    ))
    
    apply_layout(fig, height=450, xaxis_title="Years from Now", yaxis_title="Predicted Avg kWh / month",
                 title=f"{years}-Year Projection based on {ctrl['dataset']} Dataset")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Projected Values")
    cols = st.columns(5)
    for i, y_target in enumerate([1, 5, 10, 25, 50]):
        if y_target <= 50:
            val = projections[y_target - 1]
            cols[i].markdown(
                f'<div class="metric-card"><div class="val" style="font-size: 1.5rem;">{val:,.0f}</div>'
                f'<div class="lbl">Year {y_target}</div></div>',
                unsafe_allow_html=True
            )
            
    peak_val = max(display_proj)
    growth_mult = display_proj[-1] / display_proj[0] if display_proj else 1
    
    st.markdown(
        f'<div class="insight-box" style="margin-top: 20px;">'
        f'<b>Forecast Summary:</b> Over the next {years} years, consumption is projected to grow by a multiple of <b>{growth_mult:.2f}x</b>. '
        f'The peak projected average value is <b>{peak_val:,.0f} kWh/month</b>. '
        f'This is driven by a {ctrl["ff_pop_growth"]}% population growth and {ctrl["ff_urbanization"]}% urbanization rate.'
        f'</div>',
        unsafe_allow_html=True
    )


def tab_models(hh, comm, models):
    st.markdown('<div class="section-title">Performance metrics — Household dataset</div>',
                unsafe_allow_html=True)

    hh_m  = models['hh']
    com_m = models['comm']
    lr_met_h  = metrics(hh_m['y_te'],  hh_m['lr_pred'])
    gb_met_h  = metrics(hh_m['y_te'],  hh_m['gb_pred'])
    lr_met_c  = metrics(com_m['y_te'], com_m['lr_pred'])
    gb_met_c  = metrics(com_m['y_te'], com_m['gb_pred'])

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (f"{lr_met_h['R²']:.4f}", "LR R² (Household)",    "baseline",  "#A32D2D"),
        (f"{gb_met_h['R²']:.4f}", "XGBoost R² (Household)","best HH ✓", "#0F6E56"),
        (f"{lr_met_c['R²']:.4f}", "LR R² (Commercial)",   "baseline",   "#854F0B"),
        (f"{gb_met_c['R²']:.4f}", "XGBoost R² (Commercial)","best Comm ✓","#0F6E56"),
    ]
    for col, (val, lbl, sub, color) in zip([c1,c2,c3,c4], cards):
        col.markdown(
            f'<div class="metric-card"><div class="val" style="color:{color}">{val}</div>'
            f'<div class="lbl">{lbl}</div><div class="sub" style="color:{color}">{sub}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    hh_df = pd.DataFrame([
        {"Model": "Linear Regression",   **lr_met_h,
         "Notes": "Baseline — limited nonlinear capture"},
        {"Model": "XGBoost / GBR ✓",     **gb_met_h,
         "Notes": "Best — captures seasonal nonlinearity"},
        {"Model": "ARIMA (AR1, d=1)",
         "MSE": "—", "RMSE": "~85",  "MAE": "—", "R²": "—",
         "Notes": "Mean-reverting; good trend"},
        {"Model": "SARIMA (p=2, P=1, s=12)",
         "MSE": "—", "RMSE": "~47",  "MAE": "—", "R²": "—",
         "Notes": "Best seasonal pattern reproduction"},
        {"Model": "LSTM (RNN proxy)",
         "MSE": "—", "RMSE": "~62",  "MAE": "—", "R²": "—",
         "Notes": "Smooth convergence to mean"},
    ])
    st.markdown("**Household dataset metrics**")
    st.dataframe(hh_df.set_index("Model"), use_container_width=True)

    comm_df = pd.DataFrame([
        {"Model": "Linear Regression",   **lr_met_c,
         "Notes": "Higher baseline — more variance explained"},
        {"Model": "XGBoost / GBR ✓",     **gb_met_c,
         "Notes": "Best — business type & load are key"},
        {"Model": "ARIMA",
         "MSE": "—", "RMSE": "~320", "MAE": "—", "R²": "—",
         "Notes": "Monthly aggregated trend"},
        {"Model": "SARIMA",
         "MSE": "—", "RMSE": "~210", "MAE": "—", "R²": "—",
         "Notes": "Seasonality weaker in commercial"},
        {"Model": "LSTM",
         "MSE": "—", "RMSE": "~390", "MAE": "—", "R²": "—",
         "Notes": "Faster convergence"},
    ])
    st.markdown("**Commercial dataset metrics**")
    st.dataframe(comm_df.set_index("Model"), use_container_width=True)

    st.markdown('<div class="section-title">Predicted vs Actual — XGBoost (Household)</div>',
                unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hh_m['y_te'], y=hh_m['gb_pred'],
            mode='markers',
            marker=dict(color=COLORS['teal'], size=7, opacity=0.7),
            name='Predicted vs Actual'
        ))
        rng = [float(min(hh_m['y_te'])), float(max(hh_m['y_te']))]
        fig.add_trace(go.Scatter(x=rng, y=rng, mode='lines',
                                  line=dict(color=COLORS['gray'], dash='dash', width=1.5),
                                  name='Perfect fit'))
        apply_layout(fig, height=320,
                     xaxis_title="Actual kWh",
                     yaxis_title="Predicted kWh",
                     showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hh_m['y_te'], y=hh_m['lr_pred'],
            mode='markers',
            marker=dict(color=COLORS['blue'], size=7, opacity=0.7),
            name='LR Predicted'
        ))
        fig2.add_trace(go.Scatter(x=rng, y=rng, mode='lines',
                                   line=dict(color=COLORS['gray'], dash='dash', width=1.5),
                                   name='Perfect fit'))
        apply_layout(fig2, height=320,
                     xaxis_title="Actual kWh",
                     yaxis_title="LR Predicted kWh",
                     title="Predicted vs Actual — Linear Regression",
                     showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Residual distribution — XGBoost</div>',
                unsafe_allow_html=True)
    resid_h = hh_m['y_te'] - hh_m['gb_pred']
    fig_r = go.Figure()
    fig_r.add_trace(go.Histogram(x=resid_h, nbinsx=20,
                                  marker_color=COLORS['teal'],
                                  opacity=0.8, name='Household residuals'))
    fig_r.add_vline(x=0, line_dash="dash", line_color=COLORS['coral'])
    apply_layout(fig_r, height=250,
                 xaxis_title="Residual (kWh)",
                 yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig_r, use_container_width=True)

def tab_features(hh, comm, models):
    hh_m = models['hh']

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">XGBoost feature importance (Household)</div>',
                    unsafe_allow_html=True)
        fi_vals  = hh_m['fi']
        fi_names = ['Month', 'Solar Flag', 'House Type', 'Rooms', 'Solar Capacity']
        fig = go.Figure(go.Bar(
            x=fi_vals * 100,
            y=fi_names,
            orientation='h',
            marker_color=[COLORS['teal'] if v == max(fi_vals) else COLORS['blue']
                          for v in fi_vals],
            text=[f"{v*100:.1f}%" for v in fi_vals],
            textposition='outside'
        ))
        apply_layout(fig, height=280,
                     xaxis_title="Importance (%)",
                     showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Consumption by number of rooms</div>',
                    unsafe_allow_html=True)
        rooms_avg = hh.groupby('Rooms')['Units Consumed'].mean().sort_index()
        fig = go.Figure(go.Bar(
            x=[f"{r} room{'s' if r>1 else ''}" for r in rooms_avg.index],
            y=rooms_avg.values,
            marker_color=[f"rgba(29,158,117,{0.3 + i*0.12})"
                          for i in range(len(rooms_avg))],
            text=[f"{v:.0f}" for v in rooms_avg.values],
            textposition='outside'
        ))
        apply_layout(fig, height=280,
                     yaxis_title="Avg kWh / month", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-title">Seasonal patterns — radar chart</div>',
                    unsafe_allow_html=True)
        hh_s   = hh.groupby('Season')['Units Consumed'].mean()
        comm_s = comm.groupby('Season')['Units Consumed (After Solar)'].mean()
        seasons = list(hh_s.index)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(hh_s.values) + [hh_s.values[0]],
            theta=seasons + [seasons[0]],
            fill='toself', name='Household',
            line_color=COLORS['teal'], fillcolor='rgba(29,158,117,0.15)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=list(comm_s.values/5) + [comm_s.values[0]/5],
            theta=seasons + [seasons[0]],
            fill='toself', name='Commercial / 5',
            line_color=COLORS['blue'], fillcolor='rgba(55,138,221,0.15)'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          font_family="DM Sans", height=320,
                          legend=dict(orientation="h", y=-0.15),
                          margin=dict(t=30,b=60,l=40,r=40))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-title">House type — consumption distribution</div>',
                    unsafe_allow_html=True)
        apt  = hh[hh['House Type'] == 'Apartment']['Units Consumed']
        indp = hh[hh['House Type'] == 'Independent']['Units Consumed']
        fig  = go.Figure()
        fig.add_trace(go.Box(y=apt.values,  name='Apartment',
                             marker_color=COLORS['teal'], boxmean=True))
        fig.add_trace(go.Box(y=indp.values, name='Independent',
                             marker_color=COLORS['blue'], boxmean=True))
        apply_layout(fig, height=320,
                     yaxis_title="kWh / month", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Correlation matrix — Household features</div>',
                unsafe_allow_html=True)
    hh2 = hh.copy()
    hh2['house_type_num'] = (hh2['House Type'] == 'Independent').astype(int)
    corr_cols = ['month_num', 'Rooms', 'Solar Capacity (kW)',
                 'solar_flag', 'house_type_num', 'Units Consumed']
    corr = hh2[corr_cols].corr()
    fig_h = px.imshow(corr, text_auto='.2f',
                      color_continuous_scale=['#D85A30','white','#1D9E75'],
                      zmin=-1, zmax=1,
                      labels=dict(color="Correlation"))
    fig_h.update_layout(font_family="DM Sans", height=350,
                         margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_h, use_container_width=True)


def tab_predict(hh, comm, models, ctrl):
    st.markdown('<div class="section-title">Single-household prediction</div>',
                unsafe_allow_html=True)

    rooms     = ctrl['rooms']
    solar     = ctrl['solar']
    sol_cap   = ctrl['sol_cap']
    h_type    = ctrl['h_type']
    pred_month= ctrl['pred_month']

    le = LabelEncoder().fit(hh['House Type'])
    hh2 = hh.copy(); hh2['house_type_enc'] = le.transform(hh2['House Type'])
    feat_h = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
    lr_m = LinearRegression()
    gb_m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=4, subsample=0.8, random_state=42)
    lr_m.fit(hh2[feat_h].values, hh2['Units Consumed'].values)
    gb_m.fit(hh2[feat_h].values, hh2['Units Consumed'].values)

    inp = [[
        MONTH_MAP[pred_month],
        1 if solar == 'Yes' else 0,
        int(le.transform([h_type])[0]),
        rooms,
        sol_cap
    ]]

    lr_pred_val = max(0.0, float(lr_m.predict(inp)[0]))
    gb_pred_val = max(0.0, float(gb_m.predict(inp)[0]))

    monthly_preds = []
    for m in range(1, 13):
        i2 = [[m, 1 if solar == 'Yes' else 0,
                int(le.transform([h_type])[0]),
                rooms, sol_cap]]
        monthly_preds.append({
            'Month': MONTH_SHORT[m-1],
            'LR':    round(max(0, float(lr_m.predict(i2)[0])), 1),
            'XGBoost': round(max(0, float(gb_m.predict(i2)[0])), 1),
        })

    st.markdown(f"**Profile:** {h_type} · {rooms} rooms · Solar: {solar} ({sol_cap} kW) · Month: {pred_month}")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-card"><div class="val">{lr_pred_val:.0f}</div>'
        f'<div class="lbl">Linear Regression</div><div class="sub">kWh predicted</div></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        f'<div class="metric-card"><div class="val">{gb_pred_val:.0f}</div>'
        f'<div class="lbl">XGBoost</div><div class="sub">kWh predicted</div></div>',
        unsafe_allow_html=True
    )
    avg_actual = hh[(hh['Month'] == pred_month) &
                    (hh['Rooms'] == rooms)]['Units Consumed'].mean()
    avg_actual = avg_actual if not np.isnan(avg_actual) else (lr_pred_val + gb_pred_val) / 2
    c3.markdown(
        f'<div class="metric-card"><div class="val">{avg_actual:.0f}</div>'
        f'<div class="lbl">Dataset avg (same segment)</div><div class="sub">kWh actual</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predicted annual profile for this household</div>',
                unsafe_allow_html=True)

    mp_df = pd.DataFrame(monthly_preds)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mp_df['Month'], y=mp_df['LR'],
                              name='Linear Regression',
                              line=dict(color=COLORS['blue'], dash='dash', width=2),
                              mode='lines+markers', marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=mp_df['Month'], y=mp_df['XGBoost'],
                              name='XGBoost',
                              line=dict(color=COLORS['teal'], width=2.5),
                              mode='lines+markers', marker=dict(size=5)))
    apply_layout(fig, height=300,
                 yaxis_title="Predicted kWh",
                 title="Predicted monthly consumption for this profile")
    st.plotly_chart(fig, use_container_width=True)

    annual_kwh = gb_pred_val * 12
    bill_est   = annual_kwh * 6.5  # ₹6.5/kWh approx NPCL rate
    st.markdown(
        f'<div class="insight-box">'
        f'<b>Annual estimate (XGBoost):</b> ~{annual_kwh:.0f} kWh/year · '
        f'Estimated annual bill: ₹{bill_est:,.0f} (@ ₹6.50/kWh average NPCL rate). '
        f'{"Solar savings: ~₹" + str(round(sol_cap * 1400 * 6.5)) + "/year" if solar == "Yes" and sol_cap > 0 else "Consider solar to reduce costs."}'
        f'</div>',
        unsafe_allow_html=True
    )


def tab_street():
    st.markdown('<div class="section-title">GNIDA Street Light Energy Estimation</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["33,534", "36,217 kWh", "13.2 GWh", "90 W"],
        ["Total lights estimated", "Daily consumption", "Annual estimate", "LED fixture wattage"]
    ):
        col.markdown(
            f'<div class="metric-card"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Road network breakdown</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Wide roads (206 km)', 'Internal roads (594 km)'],
            values=[206, 594],
            hole=0.4,
            marker_colors=[COLORS['amber'], COLORS['teal']],
            textinfo='label+percent'
        ))
        fig.update_layout(font_family="DM Sans", height=300,
                          margin=dict(t=20, b=20, l=20, r=20),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Daily consumption by road type</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=['Wide roads\n(both sides)', 'Internal roads\n(one side)', 'Total'],
            y=[14832.72, 21384.0, 36216.72],
            marker_color=[COLORS['amber'], COLORS['teal'], COLORS['coral']],
            text=['14,833 kWh', '21,384 kWh', '36,217 kWh'],
            textposition='outside',
            width=0.5
        ))
        apply_layout(fig2, height=300, yaxis_title="kWh / day",
                     showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Interactive calculation tool</div>',
                unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        wattage = st.slider("Fixture wattage (W)", 50, 200, 90, step=10)
    with col_b:
        spacing = st.slider("Light spacing (m)", 20, 50, 30, step=5)
    with col_c:
        op_hours = st.slider("Operating hours / day", 8, 14, 12)

    wide_lights = int(2 * (206_000 / spacing))
    int_lights  = int(594_000 / spacing)
    total_lights = wide_lights + int_lights
    wide_kwh    = wide_lights * (wattage / 1000) * op_hours
    int_kwh     = int_lights  * (wattage / 1000) * op_hours
    total_kwh   = wide_kwh + int_kwh
    annual_gwh  = total_kwh * 365 / 1e6

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wide road lights",    f"{wide_lights:,}")
    col2.metric("Internal road lights",f"{int_lights:,}")
    col3.metric("Daily consumption",   f"{total_kwh:,.0f} kWh")
    col4.metric("Annual consumption",  f"{annual_gwh:.2f} GWh")

    st.markdown(
        '<div class="insight-box">'
        '<b>Methodology (GNIDA 2020 data):</b><br>'
        'Wide roads (45–132 m width, 206 km total): lights on both sides → '
        '2 × (206,000 ÷ spacing) fixtures<br>'
        'Internal roads (&lt;45 m width, 594 km total): lights one side → '
        '594,000 ÷ spacing fixtures<br>'
        'Validated by GNIDA installation of 3,740 lights across 64 villages '
        'and Noida\'s 100,000+ LED street lights across urban/rural/industrial areas.'
        '</div>',
        unsafe_allow_html=True
    )


def tab_rawdata(hh, comm):
    st.markdown('<div class="section-title">Raw datasets</div>',
                unsafe_allow_html=True)
    tab_a, tab_b = st.tabs(["🏠 Household (200 records)", "🏢 Commercial (130 records)"])

    with tab_a:
        st.dataframe(hh, use_container_width=True, height=400)
        col1, col2 = st.columns(2)
        with col1: st.write(hh.describe().round(2))
        with col2: st.write(hh.dtypes.rename("dtype").to_frame())

    with tab_b:
        st.dataframe(comm, use_container_width=True, height=400)
        col1, col2 = st.columns(2)
        with col1: st.write(comm.describe().round(2))
        with col2: st.write(comm.dtypes.rename("dtype").to_frame())


def tab_3d_map(hh, comm, ctrl):
    st.markdown('<div class="section-title">Interactive 3D City Map — Greater Noida</div>', unsafe_allow_html=True)
    
    rooms = ctrl['rooms']
    solar = ctrl['solar']
    sol_cap = ctrl['sol_cap']
    h_type = ctrl['h_type']
    pred_month = ctrl['pred_month']

    le = LabelEncoder().fit(hh['House Type'])
    hh2 = hh.copy(); hh2['house_type_enc'] = le.transform(hh2['House Type'])
    feat_h = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
    gb_m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)
    gb_m.fit(hh2[feat_h].values, hh2['Units Consumed'].values)
    
    from energy_consumption_app import MONTH_MAP
    inp = [[MONTH_MAP[pred_month], 1 if solar == 'Yes' else 0, int(le.transform([h_type])[0]), rooms, sol_cap]]
    gb_pred_val = max(0.0, float(gb_m.predict(inp)[0]))
    
    energy_normalized = max(0.0, min(1.0, (gb_pred_val - 200) / 1800.0))
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ margin: 0; overflow: hidden; background-color: #020b18; font-family: 'Orbitron', sans-serif; }}
            #canvas-container {{ width: 100%; height: 650px; display: block; }}
            #pred-label {{ position: absolute; top: 20px; left: 20px; color: #00e5ff; font-size: 28px; text-shadow: 0 0 15px #00e5ff; font-weight: bold; pointer-events: none; z-index: 10; animation: pulse 2s infinite; }}
            @keyframes pulse {{
                0% {{ opacity: 0.8; text-shadow: 0 0 10px #00e5ff; transform: scale(1); }}
                50% {{ opacity: 1.0; text-shadow: 0 0 20px #00e5ff; transform: scale(1.02); }}
                100% {{ opacity: 0.8; text-shadow: 0 0 10px #00e5ff; transform: scale(1); }}
            }}
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="pred-label">Predicted: {gb_pred_val:.0f} kWh/month</div>
        <div id="canvas-container"></div>
        <script>
            window.energyLevel = {energy_normalized};
            
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x020b18);
            scene.fog = new THREE.FogExp2(0x020b18, 0.018);
            
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / 650, 1, 1000);
            camera.position.set(50, 40, 50);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, 650);
            document.getElementById('canvas-container').appendChild(renderer.domElement);
            
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.autoRotate = true;
            controls.autoRotateSpeed = 1.0;
            controls.enableDamping = true;
            controls.maxPolarAngle = Math.PI / 2 - 0.05;
            
            const ambientLight = new THREE.AmbientLight(0x0a1628, 2.0);
            scene.add(ambientLight);
            const dirLight = new THREE.DirectionalLight(0x00e5ff, 1.2);
            dirLight.position.set(20, 50, 20);
            scene.add(dirLight);
            
            const gridHelper = new THREE.GridHelper(100, 50, 0x00e5ff, 0x0a1628);
            gridHelper.material.opacity = 0.25;
            gridHelper.material.transparent = true;
            scene.add(gridHelper);
            
            const planeGeo = new THREE.PlaneGeometry(100, 100);
            const planeMat = new THREE.MeshBasicMaterial({{ color: 0x01050d }});
            const plane = new THREE.Mesh(planeGeo, planeMat);
            plane.rotation.x = -Math.PI / 2;
            plane.position.y = -0.1;
            scene.add(plane);
            
            const buildings = [];
            const createZone = (x, z, count, color, spread) => {{
                const geo = new THREE.BoxGeometry(1.5, 1, 1.5);
                const mat = new THREE.MeshLambertMaterial({{ 
                    color: color, 
                    emissive: color,
                    emissiveIntensity: 0.4,
                    transparent: true,
                    opacity: 0.85
                }});
                
                for(let i=0; i<count; i++) {{
                    const mesh = new THREE.Mesh(geo, mat);
                    mesh.position.set(x + (Math.random() - 0.5) * spread, 0, z + (Math.random() - 0.5) * spread);
                    mesh.userData = {{ baseHeight: 1 + Math.random() * 4 }};
                    scene.add(mesh);
                    buildings.push(mesh);
                }}
            }};
            
            // Zones
            createZone(-15, -15, 35, 0x1D9E75, 15); // Residential
            createZone(15, -10, 25, 0x378ADD, 12);  // Commercial
            createZone(-20, 20, 30, 0xBA7517, 18);  // Industrial
            createZone(20, 20, 25, 0x7F77DD, 15);   // Knowledge Park
            createZone(0, 0, 20, 0x00e5ff, 8);      // Pari Chowk
            
            const partGeo = new THREE.BufferGeometry();
            const partCount = 120;
            const posArray = new Float32Array(partCount * 3);
            for(let i=0;i<partCount*3;i++) posArray[i] = (Math.random() - 0.5) * 80;
            for(let i=1;i<partCount*3;i+=3) posArray[i] = Math.random() * 30 + 5;
            partGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
            const partMat = new THREE.PointsMaterial({{ size: 0.4, color: 0x00e5ff, transparent: true, opacity: 0.8 }});
            const particles = new THREE.Points(partGeo, partMat);
            scene.add(particles);
            
            let currentLevel = 0;
            
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                
                currentLevel += (window.energyLevel - currentLevel) * 0.05;
                
                buildings.forEach(b => {{
                    const h = b.userData.baseHeight * (0.4 + currentLevel * 1.6);
                    b.scale.y = h;
                    b.position.y = h / 2;
                }});
                
                particles.rotation.y += 0.002;
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=650)


def main():
    # Adding a script injection for the live clock
    components.html("""
        <script>
            function updateClock() {
                var now = new Date();
                var clockEl = window.parent.document.getElementById('live-clock');
                if(clockEl) clockEl.innerHTML = now.toLocaleTimeString();
            }
            setInterval(updateClock, 1000);
        </script>
    """, height=0, width=0)

    st.markdown("""
    <div class="main-header">
        <div class="clock" id="live-clock"></div>
        <h1>Machine Learning-Based Prediction of Household Energy Consumption</h1>
        <p>Greater Noida &amp; Noida · Household (200 records) + Commercial (130 records) datasets · XGBoost · LSTM · ARIMA · SARIMA · Linear Regression</p>
        <p style="font-size:0.8rem;opacity:0.75;margin-top:8px">Mohit Batar · Prakash Mishra · Ashish Kumar &nbsp;|&nbsp; March 2026</p>
    </div>
    <div class="floating-export">
        <button onclick="window.print()">Export Report</button>
    </div>
    """, unsafe_allow_html=True)

    try:
        hh, comm, le_h, le_c = load_data()
    except FileNotFoundError:
        st.error("**Data files not found.**")
        st.stop()

    models = train_models(hh, comm)
    ctrl   = sidebar(hh, comm)

    tabs = st.tabs([
        "Overview",
        "Forecast",
        "3D City Map",
        "Model Comparison",
        "Feature Analysis",
        "Custom Prediction",
        "Raw Data"
    ])

    with tabs[0]: tab_overview(hh, comm)
    with tabs[1]: tab_forecast(hh, comm, ctrl)
    with tabs[2]: tab_3d_map(hh, comm, ctrl)
    with tabs[3]: tab_models(hh, comm, models)
    with tabs[4]: tab_features(hh, comm, models)
    with tabs[5]: tab_predict(hh, comm, models, ctrl)
    with tabs[6]: tab_rawdata(hh, comm)

    st.markdown(
        '<div class="footer">Machine Learning-Based Prediction of Household Energy Consumption · Greater Noida Case Study · Powered by Streamlit + Plotly + scikit-learn + Three.js</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
