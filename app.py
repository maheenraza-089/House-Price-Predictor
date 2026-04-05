import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HomeValue AI · House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,600;0,700;1,400;1,600&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #080604 !important;
    color: #EDE8DC;
}
.main, .block-container {
    background: #080604 !important;
    padding: 0 !important;
    max-width: 100% !important;
}
header { display: none !important; }
footer { display: none !important; }

.hero {
    background: #080604;
    padding: 52px 64px 44px;
    border-bottom: 1px solid #1E1810;
    position: relative;
    overflow: hidden;
}
.hero-glow-1 {
    position: absolute; top: -100px; right: -60px;
    width: 480px; height: 480px;
    background: radial-gradient(circle, rgba(201,168,76,0.10) 0%, transparent 65%);
    pointer-events: none;
}
.hero-glow-2 {
    position: absolute; bottom: -100px; left: 100px;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(139,69,19,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 10px; letter-spacing: 0.35em; text-transform: uppercase;
    color: #C9A84C; margin-bottom: 20px;
}
.hero-tag::before { content: ''; width: 24px; height: 1px; background: #C9A84C; display: inline-block; }
.hero-h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 76px; font-weight: 700; line-height: 0.92;
    color: #EDE8DC; letter-spacing: -0.02em; margin: 0 0 8px;
}
.hero-h1 em { font-style: italic; color: #C9A84C; }
.hero-p {
    font-size: 14px; font-weight: 300; color: #6A6050;
    margin-top: 18px; letter-spacing: 0.03em; line-height: 1.6;
}
.kpi-row {
    display: flex; gap: 48px; margin-top: 36px;
    padding-top: 28px; border-top: 1px solid #1E1810;
}
.kpi-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 34px; font-weight: 600; color: #C9A84C; line-height: 1;
}
.kpi-lbl { font-size: 9px; letter-spacing: 0.25em; text-transform: uppercase; color: #4A4038; margin-top: 5px; }

.body-wrap { padding: 44px 64px 0; }

.s-eyebrow { font-size: 9px; letter-spacing: 0.35em; text-transform: uppercase; color: #C9A84C; margin-bottom: 4px; }
.s-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px; font-weight: 600; color: #EDE8DC;
    margin-bottom: 22px; letter-spacing: -0.01em;
}

.panel {
    background: #0F0C06; border: 1px solid #1E1810;
    border-radius: 2px; padding: 32px; position: relative;
}
.panel::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #C9A84C 0%, #8B4513 60%, transparent 100%);
    border-radius: 2px 2px 0 0;
}

.result-outer {
    background: #0F0C06; border: 1px solid #2A2010;
    border-radius: 2px; padding: 36px 28px;
    text-align: center; position: relative; overflow: hidden;
}
.result-outer::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #C9A84C, transparent);
}
.rl { font-size: 9px; letter-spacing: 0.3em; text-transform: uppercase; color: #4A4038; margin-bottom: 10px; }
.rp {
    font-family: 'Cormorant Garamond', serif;
    font-size: 60px; font-weight: 700; color: #C9A84C;
    line-height: 1; letter-spacing: -0.03em;
}
.rrange { font-size: 12px; color: #5A5040; margin-top: 10px; font-style: italic; }
.conf-wrap { margin-top: 22px; background: #1A1208; height: 3px; border-radius: 1px; overflow: hidden; }
.conf-bar { height: 100%; background: linear-gradient(90deg, #8B4513, #C9A84C, #E8D5A3); border-radius: 1px; }
.conf-lbl { font-size: 9px; letter-spacing: 0.2em; color: #4A4038; margin-top: 5px; }

.mstrip { display: flex; gap: 12px; margin-top: 16px; }
.mcard {
    flex: 1; background: #0F0C06; border: 1px solid #1E1810;
    border-radius: 2px; padding: 16px; text-align: center;
}
.mv { font-family: 'Cormorant Garamond', serif; font-size: 24px; font-weight: 600; color: #C9A84C; }
.ml { font-size: 9px; letter-spacing: 0.2em; text-transform: uppercase; color: #4A4038; margin-top: 3px; }

.fi-row { display: flex; align-items: center; gap: 10px; padding: 9px 0; border-bottom: 1px solid #12100A; }
.fi-name { font-size: 11px; color: #7A7060; width: 120px; flex-shrink: 0; }
.fi-bg { flex: 1; background: #1A1208; height: 3px; border-radius: 1px; overflow: hidden; }
.fi-fill { height: 100%; background: linear-gradient(90deg, #8B4513, #C9A84C); border-radius: 1px; }
.fi-pct { font-size: 10px; color: #C9A84C; width: 32px; text-align: right; }

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1E1810, #C9A84C 40%, #1E1810, transparent);
    margin: 52px 0;
}

.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1E1810 !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #4A4038 !important;
    font-size: 11px !important; letter-spacing: 0.15em !important;
    text-transform: uppercase !important; font-family: 'DM Sans', sans-serif !important;
    padding: 10px 22px !important; border: none !important;
}
.stTabs [aria-selected="true"] { color: #C9A84C !important; border-bottom: 2px solid #C9A84C !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 24px 0 0 !important; background: transparent !important; }

.stButton > button {
    background: linear-gradient(135deg, #C9A84C, #A07828) !important;
    color: #080604 !important; border: none !important; border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 11px !important;
    font-weight: 500 !important; letter-spacing: 0.2em !important;
    text-transform: uppercase !important; padding: 16px 40px !important;
    width: 100% !important; box-shadow: 0 6px 32px rgba(201,168,76,0.22) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 10px 42px rgba(201,168,76,0.38) !important;
    transform: translateY(-2px) !important;
}

.stSlider > div > div > div > div { background: #C9A84C !important; }
.stSlider > div > div > div { background: #2A2010 !important; }

label {
    color: #6A6050 !important; font-size: 10px !important;
    font-weight: 400 !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
.stSelectbox > div > div { background: #080604 !important; border-color: #1E1810 !important; color: #EDE8DC !important; }
.stRadio > div { gap: 16px !important; }
.stRadio label { font-size: 13px !important; text-transform: none !important; letter-spacing: 0 !important; color: #8A8070 !important; }

.footer {
    text-align: center; padding: 28px; border-top: 1px solid #12100A;
    color: #2A2018; font-size: 10px; letter-spacing: 0.15em; margin-top: 48px;
}
</style>
""", unsafe_allow_html=True)


# ─── LOAD & TRAIN ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("data.csv")
    df = df.dropna(subset=['price'])
    df = df[(df['price'] > 50_000) & (df['price'] < 5_000_000)]

    feats = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
             'waterfront','view','condition','sqft_above','sqft_basement',
             'yr_built','yr_renovated']
    df = df.dropna(subset=feats)

    X = df[feats]
    y = np.log1p(df['price'])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc  = sc.transform(X_te)

    mdl = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05,
        max_depth=5, subsample=0.8, random_state=42
    )
    mdl.fit(X_tr_sc, y_tr)

    preds_log = mdl.predict(X_te_sc)
    preds_act = np.expm1(preds_log)
    y_te_act  = np.expm1(y_te)

    r2  = r2_score(y_te_act, preds_act)
    mae = mean_absolute_error(y_te_act, preds_act)

    return mdl, sc, feats, r2, mae, df

model, scaler, features, r2, mae, df = load_and_train()

FEAT_LABELS = {
    'sqft_living':'Living Area','sqft_above':'Above Ground','yr_built':'Year Built',
    'bathrooms':'Bathrooms','bedrooms':'Bedrooms','sqft_lot':'Lot Size',
    'view':'View Quality','condition':'Condition','floors':'Floors',
    'waterfront':'Waterfront','sqft_basement':'Basement','yr_renovated':'Renovated'
}
PLOT_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#6A6050', family='DM Sans', size=11),
)


# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-glow-1"></div>
  <div class="hero-glow-2"></div>
  <div class="hero-tag">HomeValue AI &nbsp;·&nbsp; Machine Learning Valuation</div>
  <div class="hero-h1">Estimate Your<br>Home's <em>True Worth</em></div>
  <div class="hero-p">
    Gradient Boosting model trained on {len(df):,} verified Washington real estate transactions.<br>
    Adjust property details below to instantly receive a data-driven valuation.
  </div>
  <div class="kpi-row">
    <div><div class="kpi-num">{r2*100:.1f}%</div><div class="kpi-lbl">R² Accuracy</div></div>
    <div><div class="kpi-num">{len(df):,}</div><div class="kpi-lbl">Transactions</div></div>
    <div><div class="kpi-num">${mae/1000:.0f}K</div><div class="kpi-lbl">Avg Error (MAE)</div></div>
    <div><div class="kpi-num">${df['price'].median()/1000:.0f}K</div><div class="kpi-lbl">Median Price</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── BODY ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="body-wrap">', unsafe_allow_html=True)

col_l, col_r = st.columns([1.05, 0.95], gap="large")

with col_l:
    st.markdown('<div class="s-eyebrow">Configure Property</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-title">Property Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        bedrooms    = st.slider("Bedrooms", 1, 9, 3)
        sqft_living = st.slider("Living Area (sqft)", 370, 10000, 1900, step=50)
        floors      = st.select_slider("Floors", options=[1.0,1.5,2.0,2.5,3.0], value=1.0)
        condition   = st.select_slider("Condition (1–5)", options=[1,2,3,4,5], value=3)
        yr_built    = st.slider("Year Built", 1900, 2014, 1985)
    with b:
        bathrooms    = st.slider("Bathrooms", 1.0, 6.0, 2.0, step=0.25)
        sqft_lot     = st.slider("Lot Size (sqft)", 638, 50000, 7500, step=100)
        view         = st.select_slider("View Quality (0–4)", options=[0,1,2,3,4], value=0)
        waterfront   = st.radio("Waterfront?", [0,1], format_func=lambda x: "Yes ✦" if x else "No", horizontal=True)
        yr_renovated = st.slider("Year Renovated (0 = Never)", 0, 2014, 0)

    sqft_above    = int(sqft_living * 0.75)
    sqft_basement = sqft_living - sqft_above
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("✦  Estimate Property Value")

with col_r:
    st.markdown('<div class="s-eyebrow">Valuation Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-title">Price Estimate</div>', unsafe_allow_html=True)

    inp    = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot,
                        floors, waterfront, view, condition,
                        sqft_above, sqft_basement, yr_built, yr_renovated]])
    inp_sc = scaler.transform(inp)
    pred   = np.expm1(model.predict(inp_sc)[0])
    low, high = pred * 0.88, pred * 1.12
    conf   = min(r2 * 100, 94)
    psf    = pred / sqft_living if sqft_living else 0
    vs_med = (pred - df['price'].median()) / df['price'].median() * 100

    st.markdown(f"""
    <div class="result-outer">
      <div class="rl">Estimated Market Value</div>
      <div class="rp">${pred:,.0f}</div>
      <div class="rrange">Range &nbsp;·&nbsp; ${low:,.0f} — ${high:,.0f}</div>
      <div class="conf-wrap"><div class="conf-bar" style="width:{conf:.0f}%;"></div></div>
      <div class="conf-lbl">{conf:.0f}% confidence score</div>
    </div>
    <div class="mstrip">
      <div class="mcard"><div class="mv">${psf:,.0f}</div><div class="ml">Per Sqft</div></div>
      <div class="mcard"><div class="mv">{vs_med:+.1f}%</div><div class="ml">vs Median</div></div>
      <div class="mcard"><div class="mv">${pred/1000:.0f}K</div><div class="ml">Thousands</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="s-eyebrow">Key Value Drivers</div>', unsafe_allow_html=True)

    imp_sorted = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)[:6]
    max_imp    = imp_sorted[0][1]
    rows = "".join([
        f'<div class="fi-row"><div class="fi-name">{FEAT_LABELS.get(f,f)}</div>'
        f'<div class="fi-bg"><div class="fi-fill" style="width:{i/max_imp*100:.0f}%;"></div></div>'
        f'<div class="fi-pct">{i*100:.1f}%</div></div>'
        for f, i in imp_sorted
    ])
    st.markdown(f'<div class="panel" style="padding:20px;">{rows}</div>', unsafe_allow_html=True)


# ─── DIVIDER + CHARTS ──────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="s-eyebrow">Market Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="s-title">Data Insights</div>', unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["  Price Distribution  ", "  Size vs Price  ", "  City Averages  "])

with t1:
    cap = df[df['price'] < 2_500_000]['price']
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=cap, nbinsx=55,
        marker=dict(color='#C9A84C', opacity=0.75, line=dict(color='#8B4513', width=0.4)),
    ))
    fig.add_vline(x=pred, line_color='#EDE8DC', line_width=1.5,
                  annotation_text=f'  Your estimate: ${pred:,.0f}',
                  annotation_font=dict(color='#EDE8DC', size=10))
    fig.update_layout(**PLOT_BASE,
        xaxis=dict(title='Price (USD)', gridcolor='#12100A', tickformat='$,.0f', tickfont=dict(size=10)),
        yaxis=dict(title='Count', gridcolor='#12100A', tickfont=dict(size=10)),
        margin=dict(l=0,r=0,t=16,b=0), showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

with t2:
    samp = df[df['price'] < 2_000_000].sample(min(700, len(df)), random_state=1)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=samp['sqft_living'], y=samp['price'], mode='markers',
        marker=dict(color=samp['condition'],
            colorscale=[[0,'#12100A'],[0.5,'#8B4513'],[1,'#C9A84C']],
            size=4.5, opacity=0.65, showscale=True,
            colorbar=dict(title='Cond.', tickfont=dict(color='#6A6050', size=9), titlefont=dict(color='#6A6050', size=9))),
        name='Properties'))
    fig2.add_trace(go.Scatter(
        x=[sqft_living], y=[pred], mode='markers',
        marker=dict(color='#EDE8DC', size=13, symbol='star', line=dict(color='#C9A84C', width=2)),
        name='Your Property'))
    fig2.update_layout(**PLOT_BASE,
        xaxis=dict(title='Living Area (sqft)', gridcolor='#12100A', tickfont=dict(size=10)),
        yaxis=dict(title='Price (USD)', gridcolor='#12100A', tickformat='$,.0f', tickfont=dict(size=10)),
        margin=dict(l=0,r=0,t=16,b=0), height=300,
        legend=dict(font=dict(color='#6A6050', size=10)))
    st.plotly_chart(fig2, use_container_width=True)

with t3:
    city_avg = df.groupby('city')['price'].mean().sort_values().tail(14)
    fig3 = go.Figure(go.Bar(
        x=city_avg.values, y=city_avg.index, orientation='h',
        marker=dict(color=city_avg.values,
            colorscale=[[0,'#1A1208'],[0.5,'#8B4513'],[1,'#C9A84C']], line=dict(width=0))
    ))
    fig3.update_layout(**PLOT_BASE,
        xaxis=dict(title='Average Price (USD)', gridcolor='#12100A', tickformat='$,.0f', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#12100A', tickfont=dict(size=10)),
        margin=dict(l=0,r=0,t=16,b=0), height=370, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  HomeValue AI &nbsp;·&nbsp; Python · Streamlit · Scikit-Learn &nbsp;·&nbsp;
  Gradient Boosting Regressor &nbsp;·&nbsp; Washington Real Estate Dataset
</div>
""", unsafe_allow_html=True)
