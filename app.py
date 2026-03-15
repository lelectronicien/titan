import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.optimize as sco
from datetime import datetime, timedelta
import warnings

# Bibliothèques Avancées pour Markov & ML
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

# ==========================================
# 0. CONFIGURATION TITAN QUANTUM
# ==========================================
st.set_page_config(page_title="TITAN QUANTUM V18", layout="wide", page_icon="🧬")
warnings.filterwarnings("ignore")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    .stApp { background-color: #000000; color: #c0c0c0; font-family: 'Roboto Mono', monospace; }
    h1, h2, h3, h4 { color: #d4af37 !important; font-family: 'Roboto Mono'; text-transform: uppercase; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #d4af37; font-weight: bold; }
    .dataframe { background-color: #111; border: 1px solid #333; }
    button[data-baseweb="tab"] { background-color: #111; color: #666; border-bottom: 2px solid #333; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: #222; color: #d4af37; border-bottom: 2px solid #d4af37; }
    .sig-box { padding: 10px; border: 1px solid #444; text-align: center; margin-bottom: 10px; background: #050505; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. BASE DE DONNÉES & DATA FETCHING
# ==========================================
ASSET_DB = {
    "INDICES MONDIAUX": ['^GSPC', '^NDX', '^DJI', '^RUT', '^VIX', '^FCHI', '^GDAXI', '^N225', '^HSI', '^BVSP'],
    "ETFs (TRACKERS)": ['SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'USO', 'TLT', 'IEF', 'HYG', 'LQD', 'EEM', 'VGK'],
    "FOREX MAJOR": ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X'],
    "COMMODITIES": ['GC=F', 'CL=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZW=F', 'ZS=F', 'KC=F', 'SB=F'],
    "CRYPTO": ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD'],
    "US TECH": ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'CRM', 'INTC', 'QCOM'],
    "US FINANCE": ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'V', 'MA', 'AXP', 'C'],
    "EU STOCKS": ['MC.PA', 'OR.PA', 'TTE.PA', 'SAP.DE', 'SIE.DE', 'ASML.AS', 'AIR.PA', 'SAN.PA']
}
ALL_LIST = sorted([item for sublist in ASSET_DB.values() for item in sublist])

@st.cache_data(ttl=600)
def get_data_history(ticker, period="10y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            # Select the ticker level if it exists to get a flat DataFrame
            if ticker in df.columns.get_level_values(1):
                df = df.xs(ticker, axis=1, level=1)
            else:
                df.columns = df.columns.get_level_values(0)
        
        # Final safety: if there are duplicate columns after flattening, pick the first one
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except: return None

# ==========================================
# 2. MOTEUR QUANTITATIF STANDARD (MRAT/FLUX)
# ==========================================
def strategy_engine_single(df, short_l=21, long_l=200):
    # Robustly get Close as a Series
    if 'Close' not in df.columns: return df
    close = df['Close']
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    
    df['SMA_S'] = close.rolling(short_l).mean()
    df['SMA_L'] = close.rolling(long_l).mean()
    df['MRAT'] = df['SMA_S'] / df['SMA_L']
    
    df['Buy_Flux'] = df['Volume'].where(df['Close'] >= df['Open'], 0).rolling(14).mean()
    df['Sell_Flux'] = df['Volume'].where(df['Close'] < df['Open'], 0).rolling(14).mean()
    
    df['Signal'] = np.where(df['MRAT'] > 1.0, 1, -1)
    df['Pos'] = df['Signal'].shift(1).fillna(0)
    df['Returns'] = df['Close'].pct_change().fillna(0)
    
    df['Ret_LongOnly'] = np.where(df['Pos'] == 1, df['Returns'], 0)
    df['Ret_Trading'] = df['Pos'] * df['Returns']
    df['Ret_Bench'] = df['Returns']
    return df.dropna()

def run_portfolio_simulation(tickers, capital=100000):
    if not tickers: return None, None
    eq_lo, eq_tr, eq_bh = pd.Series(0.0), pd.Series(0.0), pd.Series(0.0)
    count = 0
    for t in tickers:
        df = get_data_history(t)
        if df is not None and len(df) > 250:
            res = strategy_engine_single(df)
            if count == 0:
                eq_lo, eq_tr, eq_bh = res['Ret_LongOnly'], res['Ret_Trading'], res['Ret_Bench']
            else:
                eq_lo = eq_lo.add(res['Ret_LongOnly'], fill_value=0)
                eq_tr = eq_tr.add(res['Ret_Trading'], fill_value=0)
                eq_bh = eq_bh.add(res['Ret_Bench'], fill_value=0)
            count += 1
    if count == 0: return None, None
    df_final = pd.DataFrame({'Ret_LongOnly': eq_lo/count, 'Ret_Trading': eq_tr/count, 'Ret_Bench': eq_bh/count}).dropna()
    df_final['Eq_LongOnly'] = capital * (1 + df_final['Ret_LongOnly']).cumprod()
    df_final['Eq_Trading'] = capital * (1 + df_final['Ret_Trading']).cumprod()
    df_final['Eq_Bench'] = capital * (1 + df_final['Ret_Bench']).cumprod()
    return df_final, None

# ==========================================
# 3. NOUVEAUX MOTEURS AVANCÉS (MARKOV & ML)
# ==========================================

def run_markov_regime_model(df, capital=100000):
    """
    Modèle de Régime Markovien à 2 états (Volatilité Faible vs Haute).
    Stratégie : Long si régime 'Calme' (Proba Vol < 0.5), Cash si 'Panique'.
    """
    try:
        data = df['Returns'].iloc[1:] * 100 # Scaling pour statsmodels
        # Modèle Markov (2 régimes, switching variance)
        model = MarkovRegression(data, k_regimes=2, trend='c', switching_variance=True)
        res = model.fit(disp=False)
        
        # Proba d'être dans le régime 0 (souvent le régime calme/bullish)
        # On vérifie quel régime a la variance la plus faible
        if res.params['sigma2[0]'] < res.params['sigma2[1]']:
            low_vol_regime = 0
        else:
            low_vol_regime = 1
            
        df['Regime_Prob'] = res.smoothed_marginal_probabilities[low_vol_regime]
        
        # Signal : 1 si proba régime calme > 50%, sinon 0 (Cash)
        df['Signal_Markov'] = np.where(df['Regime_Prob'] > 0.5, 1, 0)
        df['Pos_Markov'] = df['Signal_Markov'].shift(1).fillna(0)
        
        df['Ret_Markov'] = df['Pos_Markov'] * df['Returns']
        df['Eq_Markov'] = capital * (1 + df['Ret_Markov']).cumprod()
        
        return df, True
    except:
        return df, False

def run_ml_strategy(df, capital=100000):
    """
    Modèle Random Forest (Machine Learning).
    Features : RSI, Volatilité, Returns Lag, SMA Dist
    """
    try:
        df = df.copy()
        # Features Engineering
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().clip(lower=0).rolling(14).mean() / df['Close'].pct_change().clip(upper=0).abs().rolling(14).mean()))
        df['Vol_20'] = df['Close'].pct_change().rolling(20).std()
        df['Ret_Lag1'] = df['Close'].pct_change().shift(1)
        df['SMA_Dist'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
        
        df = df.dropna()
        
        # Target : 1 si le prix monte demain, 0 sinon
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        features = ['RSI', 'Vol_20', 'Ret_Lag1', 'SMA_Dist']
        X = df[features]
        y = df['Target']
        
        # Train/Test Split (On entraîne sur les 70% premiers, on teste sur les 30% récents)
        split = int(len(df) * 0.7)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions sur TOUT le dataset pour la courbe
        df['Pred'] = model.predict(X)
        
        # Stratégie : Achat si Pred=1, Cash si Pred=0
        # On décale la prediction de 1 jour (signal aujourd'hui pour demain)
        df['Pos_ML'] = df['Pred'].shift(1).fillna(0)
        
        df['Ret_ML'] = df['Pos_ML'] * df['Returns']
        df['Eq_ML'] = capital * (1 + df['Ret_ML']).cumprod()
        
        # On ne garde que la partie "Test" pour l'équité pour être honnête
        df_test = df.iloc[split:].copy()
        df_test['Eq_ML_Test'] = capital * (1 + df_test['Ret_ML']).cumprod()
        
        return df_test, True
    except:
        return df, False

def monte_carlo_sim(returns, start_val, sims=1000, days=252):
    mu, sigma = returns.mean(), returns.std()
    paths = np.zeros((days, sims))
    paths[0] = start_val
    shocks = np.random.normal(mu, sigma, (days, sims))
    for t in range(1, days): paths[t] = paths[t-1] * (1 + shocks[t])
    return paths

# ==========================================
# 4. INTERFACE UTILISATEUR
# ==========================================
st.sidebar.title("TITAN QUANTUM")
st.sidebar.caption("Système V18 - Markov & ML")

st.sidebar.header("1. ESPIONNAGE")
cat = st.sidebar.selectbox("Catégorie", list(ASSET_DB.keys()))
single_ticker = st.sidebar.selectbox("Actif", ASSET_DB[cat])

st.sidebar.header("2. PORTEFEUILLE")
port_assets = st.sidebar.multiselect("Constitution du Panier", ALL_LIST, default=['SPY', 'QQQ', 'GLD', 'NVDA'])
capital = st.sidebar.number_input("Capital ($)", value=100000, step=10000)

df_single = get_data_history(single_ticker, period="10y") # 10 ans pour Markov/ML
if df_single is not None:
    df_single = strategy_engine_single(df_single)
    last = df_single.iloc[-1]

st.title(f"POSTE DE COMMANDE : {single_ticker}")

# NOUVEL ONGLET 6
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 CHART & FLUX", 
    "🎲 OPTIONS GEX", 
    "💰 PORTEFEUILLE", 
    "🧪 QUANT LAB", 
    "🧠 MACRO",
    "🤖 AI & REGIMES"
])

# TAB 1: CHART
with tab1:
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, column_widths=[0.85, 0.15], row_heights=[0.5, 0.25, 0.25], specs=[[{},{"rowspan":3}],[{},None],[{},None]], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df_single.index, open=df_single['Open'], high=df_single['High'], low=df_single['Low'], close=df_single['Close'], name='Prix'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_single.index, y=df_single['SMA_S'], line=dict(color='orange'), name='SMA 21'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_single.index, y=df_single['SMA_L'], line=dict(color='blue'), name='SMA 200'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_single.index, y=df_single['Buy_Flux'], fill='tozeroy', line=dict(color='#00ff00', width=0), name='Flux Achat'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_single.index, y=df_single['Sell_Flux'], fill='tozeroy', line=dict(color='#ff0000', width=0), name='Flux Vente'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_single.index, y=df_single['MRAT'], line=dict(color='#d4af37'), name='MRAT'), row=3, col=1)
    fig.add_hline(y=1, line_dash='dash', line_color='white', row=3, col=1)
    vp = df_single.groupby(pd.cut(df_single['Close'], bins=50))['Volume'].sum()
    fig.add_trace(go.Bar(x=vp.values, y=[i.mid for i in vp.index], orientation='h', marker_color='#333', name='VPVR'), row=1, col=2)
    fig.update_layout(template="plotly_dark", height=800, title="Analyse Technique", dragmode='pan')
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# TAB 2: GEX (FIXED)
with tab2:
    st.subheader(f"Gamma Exposure : {single_ticker}")
    PROXY_MAP = {'^GSPC': 'SPY', '^NDX': 'QQQ', '^RUT': 'IWM', '^DJI': 'DIA'}
    target_ticker = PROXY_MAP.get(single_ticker, single_ticker)
    if target_ticker != single_ticker: st.info(f"Proxy utilisé : {target_ticker}")
    try:
        tk = yf.Ticker(target_ticker)
        dates = tk.options
        if dates:
            dte = st.selectbox("Expiration", dates)
            if st.button("Charger GEX"):
                opt = tk.option_chain(dte)
                calls, puts = opt.calls, opt.puts
                fig_gex = go.Figure()
                fig_gex.add_trace(go.Bar(x=calls['strike'], y=calls['openInterest'], marker_color='green', name='Calls'))
                fig_gex.add_trace(go.Bar(x=puts['strike'], y=puts['openInterest'], marker_color='red', name='Puts'))
                fig_gex.update_layout(template="plotly_dark", barmode='overlay', title=f"OI {target_ticker} - {dte}")
                st.plotly_chart(fig_gex, use_container_width=True)
        else: st.warning("Pas d'options.")
    except Exception as e: st.error(f"Erreur Options: {e}")

# TAB 3: PORTFOLIO
with tab3:
    st.subheader("Performance Panier")
    if len(port_assets) > 0:
        with st.spinner("Calcul..."):
            df_port, _ = run_portfolio_simulation(port_assets, capital)
        if df_port is not None:
            c1, c2 = st.columns(2)
            c1.metric("Long Only", f"{df_port['Eq_LongOnly'].iloc[-1]:,.0f} $")
            c2.metric("Trading", f"{df_port['Eq_Trading'].iloc[-1]:,.0f} $")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=df_port.index, y=df_port['Eq_LongOnly'], line=dict(color='#00ff00'), name='Long Only'))
            fig_eq.add_trace(go.Scatter(x=df_port.index, y=df_port['Eq_Trading'], line=dict(color='#d4af37'), name='Trading'))
            fig_eq.add_trace(go.Scatter(x=df_port.index, y=df_port['Eq_Bench'], line=dict(color='gray', dash='dot'), name='Benchmark'))
            fig_eq.update_layout(template="plotly_dark", height=500, title="Equity Curve")
            st.plotly_chart(fig_eq, use_container_width=True)
            
            st.subheader("Monte Carlo Portfolio (Trading Strat)")
            if st.button("Simuler Risque Portfolio"):
                paths = monte_carlo_sim(df_port['Ret_Trading'], df_port['Eq_Trading'].iloc[-1])
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(y=paths.mean(axis=1), line=dict(color='white'), name='Mean'))
                for i in range(100): fig_mc.add_trace(go.Scatter(y=paths[:,i], line=dict(color='orange', width=1), opacity=0.1, showlegend=False))
                fig_mc.update_layout(template="plotly_dark")
                st.plotly_chart(fig_mc, use_container_width=True)

# TAB 4: QUANT
with tab4:
    st.subheader("Laboratoire Quant")
    if len(port_assets) > 1:
        if st.button("Lancer Markowitz"):
            p_data = yf.download(port_assets, period="1y", progress=False)['Close'].pct_change().dropna()
            corr = p_data.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            fig_corr.update_layout(template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)

# TAB 5: MACRO
with tab5:
    st.subheader("Rapport Macro")
    try:
        m = yf.download(['^VIX', '^TNX', 'DX-Y.NYB', 'CL=F'], period='5d', progress=False)['Close'].iloc[-1]
        st.markdown(f"""
        **RÉGIME ACTUEL :** {'RISK-OFF' if m['^VIX'] > 20 else 'NEUTRE/BULLISH'}
        *   **VIX:** {m['^VIX']:.2f} | **TNX:** {m['^TNX']:.2f}% | **USD:** {m['DX-Y.NYB']:.2f}
        """)
    except: st.error("No Data.")

# ==========================================
# TAB 6: AI & REGIMES (NOUVEAU)
# ==========================================
with tab6:
    st.markdown("## 🤖 INTELLIGENCE ARTIFICIELLE & RÉGIMES")
    st.caption(f"Analyse Avancée sur : {single_ticker} (Comparaison Benchmark S&P 500)")
    
    # Récupération Benchmark
    bench_ticker = "^GSPC"
    df_bench = get_data_history(bench_ticker, period="10y")
    
    col_ai1, col_ai2 = st.columns(2)
    
    with col_ai1:
        st.subheader("1. Modèle de Régime de Markov (Switching)")
        st.info("Détecte mathématiquement les régimes de 'Calme' vs 'Panique'. Si Proba Calme > 50% => Achat.")
        
        with st.spinner("Entraînement Modèle Markov..."):
            df_markov, success_markov = run_markov_regime_model(df_single, capital)
            
        if success_markov:
            # Graphique Proba
            fig_prob = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig_prob.add_trace(go.Scatter(x=df_markov.index, y=df_markov['Close'], name='Prix', line=dict(color='gray')), row=1, col=1)
            # Colorier les zones de risque
            crisis = df_markov[df_markov['Regime_Prob'] < 0.5]
            fig_prob.add_trace(go.Scatter(x=crisis.index, y=crisis['Close'], mode='markers', marker=dict(color='red', size=3), name='Régime Panique'), row=1, col=1)
            
            fig_prob.add_trace(go.Scatter(x=df_markov.index, y=df_markov['Regime_Prob'], name='Proba Régime Calme', line=dict(color='#00ff00')), row=2, col=1)
            fig_prob.add_hline(y=0.5, line_color='white', line_dash='dash', row=2, col=1)
            fig_prob.update_layout(template="plotly_dark", height=500, title="Détection de Régime (Markov)")
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Monte Carlo sur Markov
            st.write("**Monte Carlo sur Stratégie Markov (1000 Sims)**")
            paths_markov = monte_carlo_sim(df_markov['Ret_Markov'], df_markov['Eq_Markov'].iloc[-1])
            fig_mc_mk = go.Figure()
            fig_mc_mk.add_trace(go.Scatter(y=paths_markov.mean(axis=1), line=dict(color='cyan', width=2), name='Moyenne Markov'))
            for i in range(50): fig_mc_mk.add_trace(go.Scatter(y=paths_markov[:,i], line=dict(color='cyan', width=1), opacity=0.05, showlegend=False))
            fig_mc_mk.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_mc_mk, use_container_width=True)
            
    with col_ai2:
        st.subheader("2. Modèle Machine Learning (Random Forest)")
        st.info("Apprend des indicateurs (RSI, Vol, SMA) pour prédire la direction future (Train sur 70% data).")
        
        with st.spinner("Entraînement Random Forest..."):
            df_ml, success_ml = run_ml_strategy(df_single, capital)
            
        if success_ml:
            # Comparaison Benchmark
            # On aligne le benchmark sur la date de début du test ML
            start_date_ml = df_ml.index[0]
            if df_bench is not None:
                df_bench_cut = df_bench[df_bench.index >= start_date_ml]
                # Rebase 100k
                df_bench_cut['Eq_Bench'] = capital * (1 + df_bench_cut['Close'].pct_change().fillna(0)).cumprod()
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=df_ml.index, y=df_ml['Eq_ML_Test'], name='Stratégie AI (ML)', line=dict(color='#d4af37', width=2)))
            if df_bench is not None:
                fig_comp.add_trace(go.Scatter(x=df_bench_cut.index, y=df_bench_cut['Eq_Bench'], name='S&P 500 (Benchmark)', line=dict(color='gray', dash='dot')))
            
            fig_comp.update_layout(template="plotly_dark", height=500, title="Performance AI vs Market (Out-of-Sample)")
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Métriques
            final_ml = df_ml['Eq_ML_Test'].iloc[-1]
            ret_ml = ((final_ml/capital)-1)*100
            st.metric("Résultat IA (ML)", f"{final_ml:,.0f} $", f"{ret_ml:+.2f}%")