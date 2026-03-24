"""
taiwan_stock_strategy.py
台灣股票投資策略系統
學習來源：FinLab ML Course (U03~U25)

整合策略：
  1. 乖離率均值回歸策略（U03/U04）
  2. 技術指標動量策略（U09/U16/U17）
  3. 機器學習綜合選股（U25：特徵工程 + LightGBM + RF 集成）
  4. 個股財務評分（U08：ROE、毛利率、現金流）

資料來源：FinMind API（歷史股價/月營收）+ TWSE OpenAPI（評價指標）
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import urllib3
import warnings
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
try:
    from google import genai as google_genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────
FINMIND_BASE  = "https://api.finmindtrade.com/api/v4/data"
OPENAPI_BASE  = "https://openapi.twse.com.tw/v1"
TPEX_BASE     = "https://www.tpex.org.tw/openapi/v1"
HEADERS       = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

# ─────────────────────────────────────────────
# FinMind 資料抓取
# ─────────────────────────────────────────────
def fm_get(dataset: str, stock_id: str, start: str, token: str = "") -> pd.DataFrame:
    """FinMind API 通用 GET，回傳 DataFrame"""
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start,
        "token": token,
    }
    try:
        r = requests.get(FINMIND_BASE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != 200:
            return pd.DataFrame(), data.get("msg", "API 錯誤")
        return pd.DataFrame(data.get("data", [])), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def get_price(stock_id: str, start: str, token: str = "") -> tuple:
    """取得日K收盤價序列（FinMind → Yahoo Finance 備援）"""
    df, err = fm_get("TaiwanStockPrice", stock_id, start, token)
    if err is None and not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df, None
    # FinMind 失敗，改用 Yahoo Finance
    df_yf, err_yf = yf_get_price(stock_id, start)
    if err_yf is None:
        return df_yf, None
    return None, f"FinMind: {err or '無資料'} | Yahoo: {err_yf}"


def get_market_price_batch(stock_ids: list, start: str, token: str = "") -> tuple:
    """批次取得多股收盤價（用於選股回測）"""
    frames = {}
    for sid in stock_ids:
        df, err = fm_get("TaiwanStockPrice", sid, start, token)
        if err or df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")["close"].rename(sid)
        frames[sid] = df
        time.sleep(0.25)
    if not frames:
        return None, "無法取得股價"
    close = pd.DataFrame(frames).sort_index()
    return close, None


def get_monthly_revenue(stock_id: str, start: str, token: str = "") -> tuple:
    """取得月營收"""
    df, err = fm_get("TaiwanStockMonthRevenue", stock_id, start, token)
    if err:
        return None, err
    if df.empty:
        return None, "無月營收資料"
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df, None


def get_financial_statements(stock_id: str, start: str, token: str = "") -> tuple:
    """財務損益報表（季報）：EPS、毛利率、ROE 等"""
    return fm_get("TaiwanStockFinancialStatements", stock_id, start, token)


def get_balance_sheet(stock_id: str, start: str, token: str = "") -> tuple:
    """資產負債表：負債比、流動比等"""
    return fm_get("TaiwanStockBalanceSheet", stock_id, start, token)


def get_cash_flows(stock_id: str, start: str, token: str = "") -> tuple:
    """現金流量表：營業/投資/融資現金流"""
    return fm_get("TaiwanStockCashFlowsStatement", stock_id, start, token)


def get_dividend_data(stock_id: str, start: str, token: str = "") -> tuple:
    """股利政策資料：現金股利、股票股利"""
    return fm_get("TaiwanStockDividend", stock_id, start, token)


def twse_get(endpoint: str) -> tuple:
    """TWSE OpenAPI GET"""
    try:
        r = requests.get(f"{OPENAPI_BASE}{endpoint}", headers=HEADERS,
                         timeout=15, verify=False)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame(), "無資料"
        return pd.DataFrame(data), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def get_pe_pb_yield() -> tuple:
    """本益比、淨值比、殖利率（TWSE 上市）"""
    return twse_get("/exchangeReport/BWIBBU_ALL")


def get_tpex_pe_pb_yield() -> tuple:
    """本益比、淨值比、殖利率（TPEX 上櫃）"""
    try:
        r = requests.get(
            f"{TPEX_BASE}/tpex_mainboard_peratio_analysis",
            headers=HEADERS, timeout=15, verify=False
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame(), "無資料"
        return pd.DataFrame(data), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def _extract_pe_pb_dy(df: pd.DataFrame, stock_id: str) -> tuple:
    """
    從 TWSE 或 TPEX 的 DataFrame 中找出指定股票的 PE/PB/DY。
    回傳 (pe_val, pb_val, dy_val, found: bool)
    """
    if df.empty:
        return "N/A", "N/A", "N/A", False

    # 代號欄位偵測（TWSE: Code；TPEX: SecuritiesCompanyCode）
    code_col = next((c for c in df.columns if c in ("Code", "SecuritiesCompanyCode")
                     or "代號" in c or "code" in c.lower()), None)
    if code_col is None:
        return "N/A", "N/A", "N/A", False

    row_df = df[df[code_col].astype(str).str.strip() == stock_id.strip()]
    if row_df.empty:
        return "N/A", "N/A", "N/A", False

    row = row_df.iloc[0]

    # PE 欄位：TWSE=PEratio；TPEX=PriceEarningRatio
    pe_col = next((c for c in df.columns if c in ("PEratio", "PriceEarningRatio")
                   or "本益比" in c), None)
    # PB 欄位：TWSE=PBratio；TPEX=PriceBookRatio
    pb_col = next((c for c in df.columns if c in ("PBratio", "PriceBookRatio")
                   or "淨值比" in c or "股價淨值比" in c), None)
    # DY 欄位：TWSE=DividendYield；TPEX=DividendYield
    dy_col = next((c for c in df.columns if "DividendYield" in c or "殖利率" in c), None)

    pe_v = str(row[pe_col]).strip() if pe_col else "N/A"
    pb_v = str(row[pb_col]).strip() if pb_col else "N/A"
    dy_v = str(row[dy_col]).strip() if dy_col else "N/A"
    return pe_v, pb_v, dy_v, True


def get_valuation_any(stock_id: str) -> tuple:
    """
    依序嘗試 TWSE → TPEX → Yahoo Finance，回傳 (pe, pb, dy, source_name)。
    """
    # 先試 TWSE
    df_tw, err = get_pe_pb_yield()
    if err is None:
        pe, pb, dy, found = _extract_pe_pb_dy(df_tw, stock_id)
        if found:
            return pe, pb, dy, "TWSE 上市"

    # 再試 TPEX
    df_tpex, err2 = get_tpex_pe_pb_yield()
    if err2 is None:
        pe, pb, dy, found = _extract_pe_pb_dy(df_tpex, stock_id)
        if found:
            return pe, pb, dy, "TPEX 上櫃"

    # 最終備援：Yahoo Finance
    return yf_get_valuation(stock_id)


# ─────────────────────────────────────────────
# Yahoo Finance 備援資料抓取
# ─────────────────────────────────────────────
def _yf_ticker_obj(stock_id: str):
    """
    取得 yfinance Ticker 物件，依序嘗試 .TW（上市）及 .TWO（上櫃）。
    回傳 (Ticker物件, 後綴字串) 或 (None, "")。
    """
    if not YFINANCE_AVAILABLE:
        return None, ""
    for suffix in [".TW", ".TWO"]:
        try:
            t = yf.Ticker(f"{stock_id}{suffix}")
            hist = t.history(period="5d")
            if not hist.empty:
                return t, suffix
        except Exception:
            continue
    return None, ""


def yf_get_price(stock_id: str, start: str) -> tuple:
    """Yahoo Finance 股價備援，回傳與 FinMind 相同索引格式的 DataFrame"""
    t, _ = _yf_ticker_obj(stock_id)
    if t is None:
        return None, "Yahoo Finance 找不到此股票（.TW/.TWO 均無資料）"
    try:
        df = t.history(start=start)
        if df.empty:
            return None, "Yahoo Finance 無歷史股價"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df = df.rename(columns={
            "Close": "close", "Open": "open",
            "High": "max",   "Low": "min",
            "Volume": "Trading_Volume",
        })
        return df, None
    except Exception as e:
        return None, f"Yahoo Finance 股價錯誤：{e}"


def yf_get_financials_long(stock_id: str) -> dict:
    """
    Yahoo Finance 長期財務資料備援。
    回傳與 build_long_term_data 相同 key 格式的 dict。
    """
    result = {}
    t, _ = _yf_ticker_obj(stock_id)
    if t is None:
        return result

    # ── 季度損益表 ──
    try:
        qi = t.quarterly_income_stmt
        if qi is not None and not qi.empty:
            # EPS
            for row_name in ["Basic EPS", "Diluted EPS"]:
                if row_name in qi.index:
                    s = qi.loc[row_name].dropna().sort_index()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    result["fs_eps"] = s.astype(float)
                    break
            # Net Income（用於計算 ROE）
            if "Net Income" in qi.index:
                s = qi.loc["Net Income"].dropna().sort_index()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                result["fs_netincome"] = s.astype(float)
            # Gross Profit（毛利）
            if "Gross Profit" in qi.index:
                s = qi.loc["Gross Profit"].dropna().sort_index()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                result["fs_grossprofit"] = s.astype(float)
            # Total Revenue
            if "Total Revenue" in qi.index:
                s = qi.loc["Total Revenue"].dropna().sort_index()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                result["fs_revenue"] = s.astype(float)
    except Exception:
        pass

    # ── 季度資產負債表 ──
    try:
        qb = t.quarterly_balance_sheet
        if qb is not None and not qb.empty:
            for yf_label, key in [
                ("Total Assets",                              "bs_totalassets"),
                ("Total Liabilities Net Minority Interest",   "bs_totalliabilities"),
                ("Stockholders Equity",                       "bs_equity"),
                ("Current Assets",                            "bs_currentassets"),
                ("Current Liabilities",                       "bs_currentliabilities"),
            ]:
                if yf_label in qb.index:
                    s = qb.loc[yf_label].dropna().sort_index()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    result[key] = s.astype(float)

            # 計算簡易 ROE = Net Income / Equity
            if "fs_netincome" in result and "bs_equity" in result:
                ni  = result["fs_netincome"]
                eq  = result["bs_equity"]
                idx = ni.index.intersection(eq.index)
                if len(idx) > 0:
                    roe = (ni.loc[idx] / eq.loc[idx].replace(0, float("nan"))) * 100
                    result["fs_roe"] = roe.dropna()
    except Exception:
        pass

    # ── 季度現金流量表 ──
    try:
        qc = t.quarterly_cashflow
        if qc is not None and not qc.empty:
            for yf_label, key in [
                ("Operating Cash Flow",  "cf_operatingactivities"),
                ("Investing Cash Flow",  "cf_investingactivities"),
                ("Financing Cash Flow",  "cf_financingactivities"),
            ]:
                if yf_label in qc.index:
                    s = qc.loc[yf_label].dropna().sort_index()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    result[key] = s.astype(float)
    except Exception:
        pass

    # ── 股利歷史 ──
    try:
        divs = t.dividends
        if divs is not None and not divs.empty:
            divs.index = pd.to_datetime(divs.index).tz_localize(None)
            # 年化（同一年股利加總）
            div_annual = divs.astype(float).resample("YE").sum()
            div_annual = div_annual[div_annual > 0]
            if not div_annual.empty:
                result["cash_dividend_series"] = div_annual.sort_index()
    except Exception:
        pass

    return result


def yf_get_valuation(stock_id: str) -> tuple:
    """Yahoo Finance 估值資料備援：PE / PB / 殖利率"""
    t, suffix = _yf_ticker_obj(stock_id)
    if t is None:
        return "N/A", "N/A", "N/A", "未找到（TWSE/TPEX/Yahoo 均無資料）"
    try:
        info = t.info
        pe_raw = info.get("trailingPE") or info.get("forwardPE")
        pb_raw = info.get("priceToBook")
        dy_raw = info.get("dividendYield")
        pe_v = f"{pe_raw:.2f}" if pe_raw else "N/A"
        pb_v = f"{pb_raw:.2f}" if pb_raw else "N/A"
        dy_v = f"{float(dy_raw)*100:.2f}" if dy_raw else "N/A"
        return pe_v, pb_v, dy_v, f"Yahoo Finance（{stock_id}{suffix}）"
    except Exception as e:
        return "N/A", "N/A", "N/A", f"Yahoo Finance 錯誤：{e}"


# ─────────────────────────────────────────────
# 特徵工程（參考 U25）
# ─────────────────────────────────────────────
def calc_bias(close: pd.Series, n: int) -> pd.Series:
    """乖離率 = 現價 / SMA(n)"""
    return close / close.rolling(n, min_periods=1).mean()


def calc_acc(close: pd.Series, n: int) -> pd.Series:
    """加速度：反映趨勢加速"""
    return close.shift(n) / (close.shift(2 * n) + close) * 2


def calc_rsv(close: pd.Series, n: int) -> pd.Series:
    """相對強弱值 (0~1)"""
    lo = close.rolling(n, min_periods=1).min()
    hi = close.rolling(n, min_periods=1).max()
    denom = (hi - lo).replace(0, np.nan)
    return (close - lo) / denom


def calc_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI 相對強弱指標"""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    """MACD 計算，回傳 (macd, signal_line, histogram)"""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    hist  = macd - sig
    return macd, sig, hist


def calc_bollinger(close: pd.Series, n=20, k=2):
    """布林通道，回傳 (upper, mid, lower)"""
    mid   = close.rolling(n).mean()
    std   = close.rolling(n).std()
    return mid + k * std, mid, mid - k * std


def calc_mom(rev: pd.Series, n: int) -> pd.Series:
    """月營收動量 = 當期營收 / n期前營收"""
    return (rev / rev.shift(1)).shift(n)


def build_features_single(close: pd.Series, rev: pd.Series = None) -> pd.DataFrame:
    """
    為單一股票建構特徵（U25 策略特徵）
    close: 日收盤價
    rev  : 月營收（可選）
    """
    feat = pd.DataFrame(index=close.index)

    # 乖離率（多週期）
    for n in [5, 10, 20, 60, 120, 240]:
        feat[f"bias{n}"] = calc_bias(close, n)

    # 加速度（多週期）
    for n in [5, 10, 20, 60]:
        feat[f"acc{n}"] = calc_acc(close, n)

    # 相對強弱值（多週期）
    for n in [5, 10, 20, 60, 120]:
        feat[f"rsv{n}"] = calc_rsv(close, n)

    # 技術指標
    feat["rsi14"]  = calc_rsi(close, 14)
    feat["rsi7"]   = calc_rsi(close, 7)
    macd, msig, mhist = calc_macd(close)
    feat["macd"]   = macd
    feat["msig"]   = msig
    feat["mhist"]  = mhist

    # 價格動量（相對 n 期前）
    for n in [5, 10, 20, 60, 120, 250]:
        feat[f"ret{n}"] = close / close.shift(n)

    # 月營收動量（若有）
    if rev is not None and not rev.empty:
        rev_m = rev.resample("ME").last()
        close_m = close.resample("ME").last()
        idx = close_m.index
        for n in range(1, 7):
            m = calc_mom(rev_m, n).reindex(idx)
            feat[f"mom{n}"] = m.reindex(close.index, method="ffill")

    return feat


# ─────────────────────────────────────────────
# 策略一：乖離率均值回歸（U03/U04）
# ─────────────────────────────────────────────
def bias_strategy(close: pd.Series, sma_n=120, std_n=240, k_up=1.0, k_dn=1.0) -> dict:
    """
    乖離率策略
    買入：乖離率 < 1 - k*std（低估）
    賣出：乖離率 > 1 + k*std（高估）
    """
    sma  = close.rolling(sma_n, min_periods=1).mean()
    bias = close / sma
    ub   = 1 + bias.rolling(std_n, min_periods=1).std() * k_up
    lb   = 1 - bias.rolling(std_n, min_periods=1).std() * k_dn

    buy  = bias < lb
    sell = bias > ub

    hold = pd.Series(np.nan, index=close.index)
    hold[buy]  = 1
    hold[sell] = 0
    hold = hold.ffill().fillna(0)

    # 每日報酬
    daily_ret = close.pct_change().fillna(0)
    strat_ret = (daily_ret * hold.shift(1).fillna(0))

    # 手續費（每次換倉 1.425‰ 買 + 4.425‰ 賣）
    tx_cost = hold.diff().abs().fillna(0) * (1.425 + 4.425) / 1000 / 2
    strat_ret -= tx_cost

    equity   = (1 + strat_ret).cumprod()
    bh_equity = (1 + daily_ret).cumprod()

    # 績效統計
    ann_ret  = strat_ret.mean() * 252
    ann_vol  = strat_ret.std()  * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    dd       = (equity / equity.cummax() - 1)
    max_dd   = dd.min()

    return {
        "close": close, "sma": sma, "bias": bias, "ub": ub, "lb": lb,
        "hold": hold, "equity": equity, "bh": bh_equity,
        "ann_ret": ann_ret, "ann_vol": ann_vol,
        "sharpe": sharpe, "max_dd": max_dd,
        "buy_signals": buy, "sell_signals": sell,
    }


# ─────────────────────────────────────────────
# 策略二：動量技術指標策略（U09/U16/U17）
# ─────────────────────────────────────────────
def momentum_strategy(close: pd.Series, rsi_buy=35, rsi_sell=70,
                       bias_n=20, bias_thresh=0.95) -> dict:
    """
    RSI + 乖離率 雙重過濾動量策略
    買入：RSI < rsi_buy 且 bias < bias_thresh（超賣）
    賣出：RSI > rsi_sell 或 bias > 1/bias_thresh（超買）
    """
    rsi  = calc_rsi(close, 14)
    bias = calc_bias(close, bias_n)
    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    buy  = (rsi < rsi_buy)  & (bias < bias_thresh)
    sell = (rsi > rsi_sell) | (bias > (2 - bias_thresh))

    hold = pd.Series(np.nan, index=close.index)
    hold[buy]  = 1
    hold[sell] = 0
    hold = hold.ffill().fillna(0)

    daily_ret = close.pct_change().fillna(0)
    strat_ret = daily_ret * hold.shift(1).fillna(0)
    tx_cost   = hold.diff().abs().fillna(0) * 5.85 / 1000
    strat_ret -= tx_cost

    equity    = (1 + strat_ret).cumprod()
    bh_equity = (1 + daily_ret).cumprod()

    ann_ret = strat_ret.mean() * 252
    ann_vol = strat_ret.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    dd      = (equity / equity.cummax() - 1)
    max_dd  = dd.min()

    return {
        "close": close, "rsi": rsi, "bias": bias,
        "ma5": ma5, "ma20": ma20, "ma60": ma60,
        "hold": hold, "equity": equity, "bh": bh_equity,
        "ann_ret": ann_ret, "ann_vol": ann_vol,
        "sharpe": sharpe, "max_dd": max_dd,
        "buy_signals": buy, "sell_signals": sell,
    }


# ─────────────────────────────────────────────
# 策略三：ML 綜合選股評分（U25 簡化版）
# ─────────────────────────────────────────────
def ml_score_single(close: pd.Series, rev: pd.Series = None) -> pd.DataFrame:
    """
    計算個股 ML 評分（不依賴 talib/finlab，使用純 pandas 特徵）
    使用規則型加權替代模型訓練（適用單一股票即時評分）
    """
    feat = build_features_single(close, rev)
    feat = feat.dropna()
    if feat.empty:
        return pd.DataFrame()

    score = pd.Series(0.0, index=feat.index)

    # 乖離率評分（低乖離率 = 相對便宜）
    for col in [c for c in feat.columns if c.startswith("bias")]:
        score += (1 - feat[col].clip(0.5, 1.5)) * 10  # 乖離率越低分數越高

    # 動量評分（中短期動量）
    for col in ["ret5", "ret20", "ret60"]:
        if col in feat.columns:
            score += feat[col].clip(0.8, 1.3) * 5

    # RSI 評分（RSI 在 40~60 最佳）
    score += (50 - (feat["rsi14"] - 50).abs()) / 50 * 5

    # MACD 評分（histogram 為正 = 多頭）
    if "mhist" in feat.columns:
        score += (feat["mhist"] > 0).astype(float) * 3

    # RSV 評分（低 RSV = 超賣機會）
    for col in ["rsv5", "rsv20"]:
        if col in feat.columns:
            score += (1 - feat[col]) * 5

    # 月營收動量評分
    for col in [c for c in feat.columns if c.startswith("mom")]:
        score += (feat[col].clip(0.8, 1.5) - 1) * 20

    # 標準化為 0~100
    score = (score - score.min()) / (score.max() - score.min() + 1e-9) * 100
    feat["ml_score"] = score
    return feat


# ─────────────────────────────────────────────
# 財務評分（U08）
# ─────────────────────────────────────────────
def financial_score(pe_val, pb_val, dy_val) -> dict:
    """
    根據 PE / PB / DividendYield 計算財務評分
    低 PE + 低 PB + 高殖利率 = 高分
    """
    scores = {}
    # PE 評分（10~20 為合理，<10 高分，>30 低分）
    if pe_val and float(pe_val) > 0:
        pe = float(pe_val)
        scores["PE評分"] = max(0, min(100, 100 - (pe - 10) * 3))
    else:
        scores["PE評分"] = 50

    # PB 評分（<1 = 破淨，<2 = 合理）
    if pb_val and float(pb_val) > 0:
        pb = float(pb_val)
        scores["PB評分"] = max(0, min(100, 100 - (pb - 1) * 20))
    else:
        scores["PB評分"] = 50

    # 殖利率評分（>5% 高分）
    if dy_val and float(dy_val) > 0:
        dy = float(dy_val)
        scores["殖利率評分"] = min(100, dy * 15)
    else:
        scores["殖利率評分"] = 50

    scores["綜合財務評分"] = np.mean(list(scores.values()))
    return scores


# ─────────────────────────────────────────────
# 指標評等輔助
# ─────────────────────────────────────────────
def interpret_value(value, thresholds: list, labels: list, reverse: bool = False) -> str:
    """
    根據閾值清單回傳評等標籤。
    thresholds: 由小到大的分界 [t1, t2, t3, t4]
    labels    : 對應 len(thresholds)+1 個標籤
    reverse   : True = 數值越小越好（如 PE、負債比）
    """
    try:
        v = float(value)
    except (ValueError, TypeError):
        return "─"
    if reverse:
        v = -v
        thresholds = [-t for t in reversed(thresholds)]
    for i, t in enumerate(thresholds):
        if v < t:
            return labels[i]
    return labels[-1]


def rating_badge(label: str) -> str:
    """回傳帶顏色 emoji 的評等字串"""
    mapping = {
        "優秀": "🟢 優秀", "良好": "🔵 良好", "普通": "🟡 普通",
        "偏差": "🟠 偏差", "警示": "🔴 警示", "─": "─"
    }
    return mapping.get(label, label)


# ─────────────────────────────────────────────
# 長期資料整合（1~5 年）
# ─────────────────────────────────────────────
def build_long_term_data(stock_id: str, start_5y: str, token: str = "") -> dict:
    """
    整合5年財務資料（FinMind → Yahoo Finance 備援）。
    回傳 dict 包含各財務指標 DataFrame 與彙整統計。
    """
    result = {}
    _fm_any_data = False  # 追蹤 FinMind 是否有取到任何資料

    # 財務損益表
    df_fs, err = get_financial_statements(stock_id, start_5y, token)
    if err is None and not df_fs.empty:
        _fm_any_data = True
        df_fs["date"] = pd.to_datetime(df_fs["date"])
        result["financial_statements"] = df_fs
        for metric_type in ["EPS", "GrossProfit", "NetIncome", "Revenue",
                             "OperatingIncome", "ROE", "ROA"]:
            sub = df_fs[df_fs["type"] == metric_type][["date", "value"]].copy()
            if not sub.empty:
                sub = sub.set_index("date").sort_index()
                result[f"fs_{metric_type.lower()}"] = sub["value"].astype(float)

    # 資產負債表
    df_bs, err = get_balance_sheet(stock_id, start_5y, token)
    if err is None and not df_bs.empty:
        _fm_any_data = True
        df_bs["date"] = pd.to_datetime(df_bs["date"])
        result["balance_sheet"] = df_bs
        for metric_type in ["TotalAssets", "TotalLiabilities", "Equity",
                             "CurrentAssets", "CurrentLiabilities"]:
            sub = df_bs[df_bs["type"] == metric_type][["date", "value"]].copy()
            if not sub.empty:
                sub = sub.set_index("date").sort_index()
                result[f"bs_{metric_type.lower()}"] = sub["value"].astype(float)

    # 現金流量
    df_cf, err = get_cash_flows(stock_id, start_5y, token)
    if err is None and not df_cf.empty:
        _fm_any_data = True
        df_cf["date"] = pd.to_datetime(df_cf["date"])
        result["cash_flows"] = df_cf
        for metric_type in ["OperatingActivities", "InvestingActivities", "FinancingActivities"]:
            sub = df_cf[df_cf["type"] == metric_type][["date", "value"]].copy()
            if not sub.empty:
                sub = sub.set_index("date").sort_index()
                result[f"cf_{metric_type.lower()}"] = sub["value"].astype(float)

    # 股利
    df_div, err = get_dividend_data(stock_id, start_5y, token)
    if err is None and not df_div.empty:
        _fm_any_data = True
        df_div["date"] = pd.to_datetime(df_div["date"])
        result["dividend"] = df_div
        if "cash_dividend" in df_div.columns:
            result["cash_dividend_series"] = (
                df_div.set_index("date")["cash_dividend"].astype(float).sort_index()
            )

    # ── Yahoo Finance 備援 ──────────────────────
    # 若 FinMind 完全沒資料，或關鍵欄位仍缺，用 Yahoo Finance 補足
    _need_yf = (not _fm_any_data) or any(
        k not in result for k in ["fs_eps", "cf_operatingactivities", "cash_dividend_series"]
    )
    if _need_yf and YFINANCE_AVAILABLE:
        yf_data = yf_get_financials_long(stock_id)
        for k, v in yf_data.items():
            if k not in result:          # 只補缺失的 key，不覆蓋 FinMind 資料
                result[k] = v
        if yf_data:
            result["_yf_supplement"] = True   # 標記有用 Yahoo Finance 補充

    return result


def summarize_long_term(lt: dict, close: pd.Series) -> str:
    """將長期資料整理為 AI 可讀文字摘要"""
    parts = []
    parts.append("\n## 六、長期財務概況（近 1~5 年）\n")

    # EPS 趨勢
    eps = lt.get("fs_eps")
    if eps is not None and len(eps) >= 2:
        eps_recent = eps.tail(8)
        eps_vals = eps_recent.values
        eps_trend = "成長" if eps_vals[-1] > eps_vals[0] else "衰退"
        parts.append(f"### EPS（每股盈餘）趨勢")
        parts.append(f"| 期間 | EPS (元) |")
        parts.append(f"|------|---------|")
        for d, v in eps_recent.items():
            parts.append(f"| {d.strftime('%Y-Q')+str((d.month-1)//3+1)} | {v:.2f} |")
        parts.append(f"**趨勢：{eps_trend}（最新：{eps_vals[-1]:.2f} 元，5年均值：{eps_vals.mean():.2f} 元）**\n")

    # ROE 趨勢
    roe = lt.get("fs_roe")
    if roe is not None and len(roe) >= 2:
        roe_recent = roe.tail(8)
        roe_avg = roe_recent.mean()
        parts.append(f"### ROE（股東權益報酬率）")
        parts.append(f"| 期間 | ROE (%) |")
        parts.append(f"|------|---------|")
        for d, v in roe_recent.items():
            parts.append(f"| {d.strftime('%Y-Q')+str((d.month-1)//3+1)} | {v:.1f}% |")
        parts.append(f"**平均 ROE：{roe_avg:.1f}%（>15% 為優質）**\n")

    # 現金股利
    div = lt.get("cash_dividend_series")
    if div is not None and len(div) >= 2:
        div_recent = div.tail(5)
        div_growth = (div_recent.iloc[-1] / div_recent.iloc[0] - 1) * 100 if div_recent.iloc[0] > 0 else 0
        parts.append(f"### 現金股利歷史")
        parts.append(f"| 年度 | 現金股利 (元) |")
        parts.append(f"|------|--------------|")
        for d, v in div_recent.items():
            parts.append(f"| {d.strftime('%Y')} | {v:.2f} |")
        parts.append(f"**股利5年成長率：{div_growth:.1f}%**\n")

    # 營業現金流
    ocf = lt.get("cf_operatingactivities")
    if ocf is not None and len(ocf) >= 2:
        ocf_recent = ocf.tail(8)
        positive_count = (ocf_recent > 0).sum()
        parts.append(f"### 營業現金流")
        parts.append(f"近8季中有 {positive_count} 季為正（現金流健全度：{'優' if positive_count >= 7 else '良' if positive_count >= 5 else '差'}）\n")

    # 股價 5 年表現
    if len(close) >= 252:
        price_1y = close.iloc[-1] / close.iloc[-252] - 1 if len(close) >= 252 else None
        price_3y = close.iloc[-1] / close.iloc[-756] - 1 if len(close) >= 756 else None
        price_5y = close.iloc[-1] / close.iloc[-1260] - 1 if len(close) >= 1260 else None
        parts.append(f"### 股價長期表現")
        parts.append(f"| 期間 | 報酬率 |")
        parts.append(f"|------|--------|")
        if price_1y is not None: parts.append(f"| 近1年 | {price_1y*100:.1f}% |")
        if price_3y is not None: parts.append(f"| 近3年 | {price_3y*100:.1f}% |")
        if price_5y is not None: parts.append(f"| 近5年 | {price_5y*100:.1f}% |")
        parts.append("")

    return "\n".join(parts)


# ─────────────────────────────────────────────
# 圖表繪製
# ─────────────────────────────────────────────
DARK = dict(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font=dict(color="#e0e0e0"), margin=dict(l=50, r=20, t=50, b=30))

def plot_bias_strategy(res: dict, stock_id: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03)
    c = res["close"]
    # 股價 + 均線 + 布林
    fig.add_trace(go.Scatter(x=c.index, y=c, name="Close｜收盤價",
                             line=dict(color="#e0e0e0", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res["sma"].index, y=res["sma"],
                             name="SMA｜移動平均", line=dict(color="#FFA726", width=1.5, dash="dash")),
                  row=1, col=1)
    # 買賣訊號
    buys  = c[res["buy_signals"]]
    sells = c[res["sell_signals"]]
    fig.add_trace(go.Scatter(x=buys.index, y=buys, mode="markers",
                             name="Buy｜買入", marker=dict(color="#26A69A", symbol="triangle-up", size=8)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells, mode="markers",
                             name="Sell｜賣出", marker=dict(color="#EF5350", symbol="triangle-down", size=8)),
                  row=1, col=1)
    # 乖離率
    fig.add_trace(go.Scatter(x=res["bias"].index, y=res["bias"],
                             name="Bias｜乖離率", line=dict(color="#42A5F5")), row=2, col=1)
    fig.add_trace(go.Scatter(x=res["ub"].index, y=res["ub"],
                             name="Upper Band", line=dict(color="#EF5350", dash="dot", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=res["lb"].index, y=res["lb"],
                             name="Lower Band", line=dict(color="#26A69A", dash="dot", width=1)), row=2, col=1)
    fig.add_hline(y=1, line_color="gray", line_dash="dash", row=2, col=1)
    # 資金曲線
    fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"],
                             name="Strategy｜策略淨值", line=dict(color="#FF7043")), row=3, col=1)
    fig.add_trace(go.Scatter(x=res["bh"].index, y=res["bh"],
                             name="Buy & Hold｜持有", line=dict(color="#78909C", dash="dash")), row=3, col=1)
    fig.update_layout(title=f"{stock_id} Bias Strategy｜乖離率策略",
                      xaxis_rangeslider_visible=False, height=650, **DARK)
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig


def plot_momentum_strategy(res: dict, stock_id: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.25, 0.30], vertical_spacing=0.03)
    c = res["close"]
    fig.add_trace(go.Scatter(x=c.index, y=c, name="Close｜收盤價",
                             line=dict(color="#e0e0e0", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res["ma5"].index, y=res["ma5"],
                             name="MA5", line=dict(color="#26C6DA", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res["ma20"].index, y=res["ma20"],
                             name="MA20", line=dict(color="#FFA726", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res["ma60"].index, y=res["ma60"],
                             name="MA60", line=dict(color="#AB47BC", width=1.5)), row=1, col=1)
    buys  = c[res["buy_signals"]]
    sells = c[res["sell_signals"]]
    fig.add_trace(go.Scatter(x=buys.index, y=buys, mode="markers",
                             name="Buy｜買入", marker=dict(color="#26A69A", symbol="triangle-up", size=9)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells, mode="markers",
                             name="Sell｜賣出", marker=dict(color="#EF5350", symbol="triangle-down", size=9)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=res["rsi"].index, y=res["rsi"],
                             name="RSI(14)", line=dict(color="#FFCA28")), row=2, col=1)
    fig.add_hline(y=70, line_color="#EF5350", line_dash="dash", row=2, col=1)
    fig.add_hline(y=30, line_color="#26A69A", line_dash="dash", row=2, col=1)
    fig.add_hline(y=50, line_color="gray",    line_dash="dot",  row=2, col=1)
    fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"],
                             name="Strategy｜策略淨值", line=dict(color="#FF7043")), row=3, col=1)
    fig.add_trace(go.Scatter(x=res["bh"].index, y=res["bh"],
                             name="Buy & Hold｜持有", line=dict(color="#78909C", dash="dash")), row=3, col=1)
    fig.update_layout(title=f"{stock_id} Momentum Strategy｜動量技術策略",
                      xaxis_rangeslider_visible=False, height=650, **DARK)
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig


def plot_ml_score(feat: pd.DataFrame, stock_id: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.05)
    if "bias20" in feat.columns:
        fig.add_trace(go.Scatter(x=feat.index, y=feat["bias20"],
                                 name="Bias20｜20日乖離率", line=dict(color="#42A5F5")), row=1, col=1)
        fig.add_hline(y=1, line_color="gray", line_dash="dash", row=1, col=1)
    fig.add_trace(go.Scatter(x=feat.index, y=feat["ml_score"],
                             name="ML Score｜綜合評分", fill="tozeroy",
                             line=dict(color="#FF7043")), row=2, col=1)
    # 高分區間（>70）
    hi = feat["ml_score"][feat["ml_score"] > 70]
    fig.add_trace(go.Scatter(x=hi.index, y=hi, mode="markers",
                             name="High Score｜高評分 >70",
                             marker=dict(color="#26A69A", size=6)), row=2, col=1)
    fig.update_layout(title=f"{stock_id} ML Score｜機器學習綜合評分",
                      height=500, **DARK)
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig


def performance_bar(results: dict) -> go.Figure:
    """多策略績效對比長條圖"""
    strategies = list(results.keys())
    ann_rets  = [results[s]["ann_ret"]  * 100 for s in strategies]
    ann_vols  = [results[s]["ann_vol"]  * 100 for s in strategies]
    sharpes   = [results[s]["sharpe"]           for s in strategies]
    max_dds   = [results[s]["max_dd"]   * 100   for s in strategies]

    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=["Ann. Return %｜年化報酬率",
                                        "Ann. Vol %｜年化波動率",
                                        "Sharpe Ratio",
                                        "Max Drawdown %｜最大回撤"])
    colors = ["#26A69A", "#FFA726", "#AB47BC", "#EF5350"]
    for i, (vals, col) in enumerate(zip([ann_rets, ann_vols, sharpes, max_dds],
                                         ["#26A69A", "#FFA726", "#AB47BC", "#EF5350"]), 1):
        fig.add_trace(go.Bar(x=strategies, y=vals, marker_color=col, showlegend=False), row=1, col=i)
    fig.update_layout(height=350, **DARK)
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig


# ─────────────────────────────────────────────
# AI 分析：資料摘要建構
# ─────────────────────────────────────────────
def build_strategy_summary(stock_id: str, close: pd.Series,
                            res1: dict, res2: dict,
                            feat: pd.DataFrame,
                            fin_scores: dict,
                            pe_v, pb_v, dy_v,
                            long_term_data: dict = None) -> str:
    """將所有策略結果整合為 AI 可讀的文字摘要（含長期資料）"""
    latest_close = close.iloc[-1]
    latest_date  = close.index[-1].strftime("%Y-%m-%d")
    period_start = close.index[0].strftime("%Y-%m-%d")
    total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

    parts = []
    parts.append(f"# 台股投資策略分析報告 — {stock_id}")
    parts.append(f"分析日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    parts.append(f"資料區間：{period_start} ～ {latest_date}（共 {len(close)} 個交易日）")
    parts.append(f"最新收盤價：{latest_close:.2f} 元｜區間總報酬：{total_return:.1f}%")
    parts.append("")

    # 策略一：乖離率
    parts.append("## 一、乖離率均值回歸策略（U03/U04）")
    latest_bias = res1["bias"].iloc[-1]
    latest_ub   = res1["ub"].iloc[-1]
    latest_lb   = res1["lb"].iloc[-1]
    sharpe1_rating = interpret_value(res1['sharpe'], [0, 0.5, 1.0, 2.0], ["警示","偏差","普通","良好","優秀"])
    dd1_rating     = interpret_value(res1['max_dd']*100, [-50, -30, -20, -10], ["警示","偏差","普通","良好","優秀"])
    parts.append(f"| 指標 | 數值 | 評等 |")
    parts.append(f"|------|------|------|")
    parts.append(f"| 年化報酬 | {res1['ann_ret']*100:.2f}% | {interpret_value(res1['ann_ret']*100,[0,5,10,20],['警示','偏差','普通','良好','優秀'])} |")
    parts.append(f"| 年化波動率 | {res1['ann_vol']*100:.2f}% | {interpret_value(res1['ann_vol']*100,[15,20,30,40],['優秀','良好','普通','偏差','警示'])} |")
    parts.append(f"| Sharpe Ratio | {res1['sharpe']:.3f} | {sharpe1_rating} |")
    parts.append(f"| 最大回撤 | {res1['max_dd']*100:.2f}% | {dd1_rating} |")
    parts.append(f"| 當前乖離率 | {latest_bias:.4f} | 上軌:{latest_ub:.4f} / 下軌:{latest_lb:.4f} |")
    if latest_bias < latest_lb:
        parts.append("- ✅ 當前訊號：**低於下軌 → 買入訊號（低估區間）**")
    elif latest_bias > latest_ub:
        parts.append("- 🔴 當前訊號：**高於上軌 → 賣出訊號（高估區間）**")
    else:
        parts.append("- ⚪ 當前訊號：中性（在上下軌之間）")
    parts.append("")

    # 策略二：動量技術
    parts.append("## 二、動量技術策略（U09/U16/U17）")
    latest_rsi   = res2["rsi"].iloc[-1]
    latest_bias2 = res2["bias"].iloc[-1]
    sharpe2_rating = interpret_value(res2['sharpe'], [0, 0.5, 1.0, 2.0], ["警示","偏差","普通","良好","優秀"])
    dd2_rating     = interpret_value(res2['max_dd']*100, [-50, -30, -20, -10], ["警示","偏差","普通","良好","優秀"])
    parts.append(f"| 指標 | 數值 | 評等 |")
    parts.append(f"|------|------|------|")
    parts.append(f"| 年化報酬 | {res2['ann_ret']*100:.2f}% | {interpret_value(res2['ann_ret']*100,[0,5,10,20],['警示','偏差','普通','良好','優秀'])} |")
    parts.append(f"| 年化波動率 | {res2['ann_vol']*100:.2f}% | {interpret_value(res2['ann_vol']*100,[15,20,30,40],['優秀','良好','普通','偏差','警示'])} |")
    parts.append(f"| Sharpe Ratio | {res2['sharpe']:.3f} | {sharpe2_rating} |")
    parts.append(f"| 最大回撤 | {res2['max_dd']*100:.2f}% | {dd2_rating} |")
    parts.append(f"| 當前 RSI(14) | {latest_rsi:.1f} | {interpret_value(latest_rsi,[30,40,60,70],['警示','良好','普通','良好','警示'])} |")
    parts.append(f"| 當前乖離率(20日) | {latest_bias2:.4f} | — |")
    if res2["hold"].iloc[-1] == 1:
        parts.append("- ✅ 當前持倉狀態：**持有中（多頭）**")
    else:
        parts.append("- ⚪ 當前持倉狀態：**空手（觀望）**")
    parts.append("")

    # 策略三：ML 評分
    parts.append("## 三、機器學習綜合評分（U25 特徵工程）")
    if not feat.empty and "ml_score" in feat.columns:
        cur_score = feat["ml_score"].iloc[-1]
        avg_score = feat["ml_score"].mean()
        parts.append(f"| 指標 | 數值 | 評等 |")
        parts.append(f"|------|------|------|")
        parts.append(f"| 當前 ML 評分 | {cur_score:.1f}/100 | {interpret_value(cur_score,[40,50,65,75],['警示','偏差','普通','良好','優秀'])} |")
        parts.append(f"| 歷史平均評分 | {avg_score:.1f}/100 | — |")
        if "rsi14" in feat.columns:
            parts.append(f"| RSI(14) | {feat['rsi14'].iloc[-1]:.1f} | {interpret_value(feat['rsi14'].iloc[-1],[30,40,60,70],['超賣','偏低','健康','偏高','超買'])} |")
        if "bias20" in feat.columns:
            parts.append(f"| Bias(20日) | {feat['bias20'].iloc[-1]:.4f} | {interpret_value(feat['bias20'].iloc[-1],[0.9,0.95,1.05,1.1],['嚴重超賣','低估','合理','高估','嚴重超買'])} |")
        if "bias60" in feat.columns:
            parts.append(f"| Bias(60日) | {feat['bias60'].iloc[-1]:.4f} | — |")
        if "ret60" in feat.columns:
            parts.append(f"| 近60日動量 | {(feat['ret60'].iloc[-1]-1)*100:.1f}% | — |")
        if "ret250" in feat.columns:
            parts.append(f"| 近1年動量 | {(feat['ret250'].iloc[-1]-1)*100:.1f}% | — |")
    else:
        parts.append("- ML 特徵資料不足，無法計算")
    parts.append("")

    # 財務評價
    parts.append("## 四、財務評價指標（U08）")
    parts.append(f"| 指標 | 數值 | 評分 | 評等 |")
    parts.append(f"|------|------|------|------|")
    pe_rating = interpret_value(pe_v, [10, 15, 25, 35], ["優秀","良好","普通","偏差","警示"], reverse=True) if pe_v != "N/A" else "─"
    pb_rating = interpret_value(pb_v, [1, 1.5, 2.5, 4], ["優秀","良好","普通","偏差","警示"], reverse=True) if pb_v != "N/A" else "─"
    dy_rating = interpret_value(dy_v, [2, 3, 5, 7], ["警示","偏差","普通","良好","優秀"]) if dy_v != "N/A" else "─"
    parts.append(f"| 本益比(P/E) | {pe_v} | {fin_scores.get('PE評分',50):.0f}/100 | {pe_rating} |")
    parts.append(f"| 淨值比(P/B) | {pb_v} | {fin_scores.get('PB評分',50):.0f}/100 | {pb_rating} |")
    parts.append(f"| 殖利率 | {dy_v}% | {fin_scores.get('殖利率評分',50):.0f}/100 | {dy_rating} |")
    parts.append(f"| 綜合財務評分 | — | {fin_scores.get('綜合財務評分',50):.0f}/100 | {interpret_value(fin_scores.get('綜合財務評分',50),[40,50,65,75],['警示','偏差','普通','良好','優秀'])} |")
    parts.append("")

    # Buy & Hold 比較
    bh_ann = close.pct_change().dropna().mean() * 252
    price_1y = (close.iloc[-1] / close.iloc[-252] - 1) * 100 if len(close) >= 252 else None
    price_3y = (close.iloc[-1] / close.iloc[-756] - 1) * 100 if len(close) >= 756 else None
    parts.append("## 五、策略績效 vs. 持有不賣比較")
    parts.append(f"| 指標 | 乖離率策略 | 動量策略 | Buy & Hold |")
    parts.append(f"|------|-----------|---------|-----------|")
    parts.append(f"| 年化報酬 | {res1['ann_ret']*100:.1f}% | {res2['ann_ret']*100:.1f}% | {bh_ann*100:.1f}% |")
    parts.append(f"| Sharpe | {res1['sharpe']:.2f} | {res2['sharpe']:.2f} | — |")
    parts.append(f"| 最大回撤 | {res1['max_dd']*100:.1f}% | {res2['max_dd']*100:.1f}% | — |")
    if price_1y: parts.append(f"| 近1年報酬 | — | — | {price_1y:.1f}% |")
    if price_3y: parts.append(f"| 近3年報酬 | — | — | {price_3y:.1f}% |")
    parts.append("")

    # 長期資料
    if long_term_data:
        lt_summary = summarize_long_term(long_term_data, close)
        parts.append(lt_summary)

    return "\n".join(parts)


# ─────────────────────────────────────────────
# AI 分析：OpenAI
# ─────────────────────────────────────────────
ENHANCED_AI_PROMPT_TEMPLATE = """
你現在是一位「台灣資深股票研究員 + 機構法人直接投資分析師 + 產業研究主管 + 財務模型分析專家」，具有 30 年以上台股投資研究、企業評估、產業分析、財報分析、估值模型建構、籌碼分析、資產配置與風險管理經驗。

本次分析標的：台灣股票代碼 {stock_id}

【系統已提供的量化分析數據】
{summary}

---

【作戰規則】
1. 必須查詢最新可取得公開資訊後再做分析，不可只依賴舊有知識。
2. 若使用到非最新時間的資料，請註明資料日期或來源。
3. 若資訊不足，請清楚說明限制與不確定性。
4. 回答必須使用繁體中文，並採台灣常用投資術語。
5. 回答必須具有「研究報告格式」條理、可讀性與專業判斷力。
6. 不可只陳述資訊，必須做出分析、比較、推論與結論。
7. 不可只說好話，必須客觀說出利空、風險與反方觀點。
8. 必須考量：短期建議、中長期公司加分因子、1～3年核心投資邏輯。
9. 必須從「小額投資人總資產 20 萬內」角度，提出可執行的布局方案。
10. 結論必須清晰，不可模糊帶過。
11. 若行情已顯高估而不適合 1～3 年佈局，請直接指出，不要為了完整而強給正面評價。
12. 請盡量使用表格、列點、條列、分段標題，讓內容易如研究報告格式。

【分析對象預設投資設定】
- 台灣一般投資人
- 小額投資人
- 總資產不超過新台幣 20 萬元
- 希望以中長期時間賺得資本利得，並兼顧風險性控管
- 不以當沖、隔日沖、極短線交易為主

---

請嚴格依照以下章節逐一完成，所有表格欄位均需填入實際判斷值，不可空白：

-----------------------------------
A. 投資摘要（Executive Summary）
-----------------------------------
請先以專業摘要方式，給出 6 個重點：
1. 公司核心投資亮點
2. 目前最大風險
3. 1～3 年投資邏輯是否成立
4. 目前行情評價位置（低估 / 合理 / 偏高 / 過熱）
5. 是否適合小額投資人分批布局
6. 最終投資評斷一句話結論

接著請補上：
| 項目 | 評定 |
|------|------|
| 投資評級 | 強烈買進 / 買進 / 分批布局 / 中立觀察 / 減碼 / 不建議 |
| 1～3 年佈局適合度 | 非常適合 / 適合 / 普通 / 保守 / 不適合 |
| 目標達成時間 | 12 個月 / 24 個月 / 36 個月 |
| 風險等級 | 低 / 中低 / 中 / 中高 / 高 |
| 適合投資人類型 | 保守型 / 穩健型 / 成長型 / 積極型 |

-----------------------------------
B. 公司概況與商業模式
-----------------------------------
請分析：
1. 公司名稱、股票代碼、產業類別
2. 主要產品、服務、客戶結構
3. 營收來源結構
4. 商業模式與獲利模式
5. 公司在產業鏈中的位置
6. 是否具有規模優勢、技術護城河、客戶黏著度、訂價優勢或成本優勢
7. 該公司屬於：
   - 龍頭股 / 次線成長股 / 景氣循環股 / 殖利率填息股 / 題材股 / 高波動轉機股
   並說明原因

-----------------------------------
C. 產業研究與趨勢判斷
-----------------------------------
請從研究員角度分析：
1. 所屬產業目前景氣位置
2. 未來 1～3 年產業成長潛力
3. 產業是否有明確中長期公利多
4. 需求面、供給面變化
5. 產業競爭格局
6. 政策、科技、國際局勢對公司影響
7. 該公司是否真正坐落於 AI、伺服器、電子零組件、網通、金融、傳產、綠能等主題
8. 市場目前是否過度樂觀或過度悲觀

請最後給出：
- 產業評等：正向 / 中性偏正向 / 中性 / 中性偏負向 / 負向
- 產業趨勢對公司是否加分

-----------------------------------
D. 基本面與財務品質分析
-----------------------------------
請整理近幾年趨勢並進行分析：

| 財務指標 | 數值/趨勢 | 評等 | 解讀 |
|---------|---------|------|------|
| 營收成長率 | （填入） | 優/良/普/差 | （填入） |
| EPS 趨勢 | （填入） | 優/良/普/差 | （填入） |
| 毛利率 | （填入） | 優/良/普/差 | （填入） |
| 營業利益率 | （填入） | 優/良/普/差 | （填入） |
| 稅後淨利率 | （填入） | 優/良/普/差 | （填入） |
| ROE | （填入） | 優/良/普/差 | （填入） |
| ROA | （填入） | 優/良/普/差 | （填入） |
| 負債比 | （填入） | 優/良/普/差 | （填入） |
| 營業現金流 | （填入） | 優/良/普/差 | （填入） |
| 自由現金流 | （填入） | 優/良/普/差 | （填入） |
| 資本支出趨勢 | （填入） | 優/良/普/差 | （填入） |
| 現金股利政策 | （填入） | 優/良/普/差 | （填入） |

請評估：
- 財務高質量是否穩健
- 獲利品質是否良好
- 成長是否可持續
- 是否存在一次性收益、應收帳款風險、庫存風險、現金流緊張等問題

最後給出：
- 財務品質評等：優 / 良 / 普通 / 偏弱 / 弱

-----------------------------------
E. 估值分析（Valuation）
-----------------------------------
請以中長期投資角度分析：

| 估值指標 | 當前數值 | 歷史均值 | 產業比較 | 評等 |
|---------|---------|---------|---------|------|
| 本益比 PER | （填入） | （填入） | （填入） | 低估/合理/偏高/過高 |
| 股價淨值比 PBR | （填入） | （填入） | （填入） | 低估/合理/偏高/過高 |
| 殖利率 | （填入） | （填入） | （填入） | 低估/合理/偏高/過高 |

請無需以假設推論而提出：
- 合理價值區間
- 便宜買點
- 觀察買點
- 偏貴賣點
- 過熱追高點

並說明：
- 若現在買進，面臨的估值風險為何
- 若未來基本面優於預期，還有多少估值上修空間
- 若成長不如預期，行情可能面對低種評價下調

最後給出：
- 估值評等：低估 / 合理偏低 / 合理 / 合理偏高 / 偏高 / 過熱

-----------------------------------
F. 籌碼面與市場認同度分析
-----------------------------------
請分析：
1. 外資買賣超趨勢
2. 投信信心度
3. 融資融券變化
4. 大戶法人或內部人持股變化（若可接）
5. 借券賣出變化（若有代表性）
6. 籌碼集中度與穩定性
7. 市場資金是否認同該股中長期邏輯

請最後判斷：
- 籌碼面對中長期投資是加分、中性（還是負分）
- 籌碼是否在低檔有默默累積跡象

-----------------------------------
G. 技術面與進場時機分析
-----------------------------------
請注意：技術面分析是輔助中長期投資，不是短線操作。

請分析：
1. 目前行情位階（低檔 / 中位 / 高檔 / 相對高風險區）
2. 中長期趨勢方向
3. 均線結構是否健康
4. 關鍵支撐位
5. 關鍵壓力位
6. 若現在買進，是否左側布局、右側確認（還是追高風險偏高）
7. 適不適合分批布局
8. 哪些價格區間較適合小額投資人分段承接

最後給出：
- 技術面評等：偏多 / 中性偏多 / 中性 / 中性偏空 / 偏空

-----------------------------------
H. 風險情境分析
-----------------------------------
請列出未來 1～3 年的主要風險：
1. 產業景氣下滑
2. 客戶需求下滑
3. 毛利率壓縮
4. 匯率波動
5. 競爭對手加入
6. 科技替代風險
7. 法規政策變化
8. 地緣政治風險
9. 本益比修正風險
10. 市場系統性下跌

請進一步做三種情境分析：

| 情境 | 觸發條件 | 公司基本面可能表現 | 市場可能給予的評價變化 | 對應可能行情方向 |
|------|---------|-----------------|---------------------|--------------|
| 樂觀情境 | （填入） | （填入） | （填入） | （填入） |
| 中性情境 | （填入） | （填入） | （填入） | （填入） |
| 保守情境 | （填入） | （填入） | （填入） | （填入） |

-----------------------------------
I. 機構評估模型（100 分制）
-----------------------------------
請用以下權重評分，並逐一說明理由：

| 評估項目 | 滿分 | 得分 | 得分理由 |
|---------|------|------|---------|
| 產業景氣（20分） | 20 | （填入） | （填入） |
| 公司競爭力（20分） | 20 | （填入） | （填入） |
| 財務品質（20分） | 20 | （填入） | （填入） |
| 成長可持續性（10分） | 10 | （填入） | （填入） |
| 估值合理性（10分） | 10 | （填入） | （填入） |
| 籌碼與市場認同（5分） | 5 | （填入） | （填入） |
| 技術面與進場時機（5分） | 5 | （填入） | （填入） |
| 風險可控接受度（10分） | 10 | （填入） | （填入） |
| **合計總分** | **100** | **（填入）** | |

並對總分對應如下：
- 85 分以上：高質量中長期標的
- 75～84 分：具有投資價值，可擇機布局
- 65～74 分：有條件投資，需控管價格與風險
- 50～64 分：先觀察，不宜進場
- 49 分以下：不建議作為 1～3 年佈局標的

-----------------------------------
J. 小額投資人實戰布局建議（20 萬內）
-----------------------------------
請從小資角度設計具體方案：

| 項目 | 建議內容 |
|------|---------|
| 建議投入總額 | （填入） |
| 建議保留多少現金 | （填入） |
| 建議分幾批進場 | （填入） |
| 每批資金配置 | （填入） |
| 建議布局價格區間 | （填入） |
| 哪種方式加碼 | （填入） |
| 哪種方式暫停加碼 | （填入） |
| 哪種方式獲利了結、減碼後退出 | （填入） |
| 若行情下跌，如何分段獲利 | （填入） |
| 需要追蹤哪些關鍵指標 | （填入） |

請記住要務實，不要假設投資人可以無限加碼。

-----------------------------------
K. 最終投資結論
-----------------------------------
請用清晰、機構式結論收尾，格式如下：

| 結論項目 | 內容 |
|---------|------|
| 股票名稱／代碼 | （填入） |
| 投資評級 | （填入） |
| 1～3 年佈局適合度 | （填入） |
| 總分 | （填入） |
| 產業評等 | （填入） |
| 財務品質評等 | （填入） |
| 估值評等 | （填入） |
| 技術面評等 | （填入） |
| 風險等級 | （填入） |
| 合理價值區間 | （填入） |
| 建議布局區間 | （填入） |
| 不建議追高價 | （填入） |
| 保守情境目標價 | （填入） |
| 中性情境目標價 | （填入） |
| 樂觀情境目標價 | （填入） |
| 適合投資人類型 | （填入） |
| 核心投資邏輯 | （填入） |
| 最大風險 | （填入） |
| 最低建議 | （填入） |
| 一句話結論 | （填入） |

---
⚠️ 免責聲明：本報告純屬學術研究與教育分析，不構成任何投資建議。投資涉及風險，請自行評估並謹慎決策。
"""


def run_ai_openai(summary: str, stock_id: str, openai_key: str) -> str | None:
    """使用 OpenAI o4-mini 進行深度長期策略投資建議分析"""
    try:
        client = OpenAI(api_key=openai_key)
        user_msg = ENHANCED_AI_PROMPT_TEMPLATE.format(stock_id=stock_id, summary=summary)

        with st.spinner("🤖 OpenAI o4-mini 正在進行深度策略分析（含長期評估），請稍候..."):
            resp = client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "developer", "content": "你是一位台灣資深股票研究員兼機構法人直接投資分析師，具有30年以上台股研究、企業評估、產業分析、財報分析與風險管理經驗。請以繁體中文撰寫機構級研究報告，條理清晰、客觀嚴謹，所有表格欄位必須填入具體數值或判斷，不可留空，結論必須明確不模糊。"},
                    {"role": "user",      "content": user_msg},
                ],
            )
        return resp.choices[0].message.content

    except Exception as e:
        err = str(e)
        if "authentication" in err.lower() or "api_key" in err.lower():
            st.error("OpenAI API Key 驗證失敗，請確認金鑰正確")
        elif "rate_limit" in err.lower():
            st.error("OpenAI 請求頻率超限，請稍後再試")
        else:
            st.error(f"OpenAI 分析失敗：{err}")
        return None


# ─────────────────────────────────────────────
# AI 分析：Google Gemini
# ─────────────────────────────────────────────
def run_ai_gemini(summary: str, stock_id: str, google_key: str) -> str | None:
    """使用 Google Gemini 2.5 Flash 進行深度長期策略投資建議分析"""
    if not GOOGLE_GENAI_AVAILABLE:
        st.error("google-genai 套件未安裝，請執行：pip install google-genai")
        return None
    try:
        client = google_genai.Client(api_key=google_key)
        full_prompt = ENHANCED_AI_PROMPT_TEMPLATE.format(stock_id=stock_id, summary=summary)

        with st.spinner("✨ Google Gemini 2.5 Flash 正在進行深度策略分析（含長期評估），請稍候..."):
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt
            )
        return resp.text

    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "invalid" in err.lower() or "401" in err:
            st.error("Google API Key 驗證失敗，請確認金鑰正確")
        elif "quota" in err.lower() or "429" in err:
            st.error("Google API 請求額度超限，請稍後再試")
        else:
            st.error(f"Google Gemini 分析失敗：{err}")
        return None


# ─────────────────────────────────────────────
# ── Streamlit UI ──────────────────────────────
# ─────────────────────────────────────────────
st.set_page_config(page_title="Taiwan Stock Strategy｜台股投資策略", layout="wide", page_icon="🧠")
st.title("🧠 Taiwan Stock Investment Strategy System｜台股投資策略系統")
st.caption("學習來源：FinLab ML Course U03~U25 | Strategies: Bias Reversion · Momentum RSI · ML Scoring · Financial Valuation")

# ── 側邊欄設定 ────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings｜設定")
    token = st.text_input("FinMind Token（可空白）", type="password", key="fm_token")
    st.divider()

    stock_id = st.text_input("Stock ID｜股票代號", value="2330")
    start_date_str = st.date_input(
        "Start Date｜起始日期（建議至少3年）",
        value=date.today() - timedelta(days=365 * 3)
    ).strftime("%Y-%m-%d")
    # 長期分析固定抓取5年資料
    start_5y_str = (date.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    st.divider()
    st.subheader("Strategy 1｜乖離率策略")
    sma_n   = st.slider("SMA Period｜均線週期", 20, 240, 120)
    std_n   = st.slider("StdDev Period｜標準差週期", 60, 500, 240)
    k_val   = st.slider("K 倍數（上下軌）", 0.5, 2.0, 1.0, 0.1)

    st.subheader("Strategy 2｜動量策略")
    rsi_buy  = st.slider("RSI Buy｜RSI 買入線", 20, 50, 35)
    rsi_sell = st.slider("RSI Sell｜RSI 賣出線", 55, 85, 70)
    bias_n   = st.slider("Bias Period｜乖離均線", 10, 180, 120)
    bias_th  = st.slider("Bias Threshold｜乖離閾值", 0.85, 0.99, 0.95, 0.01)

    st.divider()
    st.subheader("🤖 AI Analysis｜AI 投資建議")
    openai_key  = st.text_input("OpenAI API Key", type="password", key="openai_key_input")
    google_key  = st.text_input("Google API Key", type="password", key="google_key_input")

    run_btn = st.button("🚀 Run Analysis｜執行分析", use_container_width=True)

# ── 主頁面 ────────────────────────────────────
tab_s1, tab_s2, tab_s3, tab_s4 = st.tabs([
    "📉 Bias Reversion｜乖離率策略",
    "📈 Momentum｜動量技術策略",
    "🤖 ML Score｜機器學習評分",
    "💰 Valuation｜財務評價",
])
tab_s5, tab_lt, tab_ai = st.tabs([
    "📊 Performance｜策略績效比較",
    "📅 Long-Term｜長期財務分析",
    "🧠 AI Analysis｜AI 投資建議",
])

if run_btn:
    # 抓取股價
    with st.spinner(f"Fetching price data for {stock_id}... 抓取股價中..."):
        df_price, err = get_price(stock_id, start_date_str, token)

    if err:
        st.error(f"Error 錯誤：{err}")
        st.stop()

    if "close" not in df_price.columns:
        st.error("無法取得收盤價欄位")
        st.stop()

    close = df_price["close"].astype(float).dropna()

    if len(close) < 60:
        st.warning("資料不足 60 筆，部分策略可能無法正確計算")

    # 抓取月營收（選用）
    with st.spinner("Fetching monthly revenue... 抓取月營收中..."):
        df_rev, rev_err = get_monthly_revenue(stock_id, start_date_str, token)
    rev = None
    if rev_err is None and not df_rev.empty and "revenue" in df_rev.columns:
        rev = df_rev["revenue"].astype(float)

    # 抓取5年長期財務資料
    with st.spinner("Fetching 5-year financial data... 抓取長期財務資料中（含財報/現金流/股利）..."):
        long_term_data = build_long_term_data(stock_id, start_5y_str, token)

    # ── 預先計算所有策略並存入 session_state ──
    res1 = bias_strategy(close, sma_n, std_n, k_val, k_val)
    res2 = momentum_strategy(close, rsi_buy, rsi_sell, bias_n, bias_th)
    feat = ml_score_single(close, rev)
    st.session_state["_close"]          = close
    st.session_state["_res1"]           = res1
    st.session_state["_res2"]           = res2
    st.session_state["_feat"]           = feat
    st.session_state["_stock_id"]       = stock_id
    st.session_state["_long_term_data"] = long_term_data
    st.session_state["_analysis_ready"] = True

    # ── Tab 1：乖離率策略 ─────────────────────
    with tab_s1:
        res1 = bias_strategy(close, sma_n, std_n, k_val, k_val)
        st.plotly_chart(plot_bias_strategy(res1, stock_id), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        r1_ret = interpret_value(res1['ann_ret']*100,[0,5,10,20],["警示","偏差","普通","良好","優秀"])
        r1_vol = interpret_value(res1['ann_vol']*100,[15,20,30,40],["優秀","良好","普通","偏差","警示"])
        r1_sr  = interpret_value(res1['sharpe'],[0,0.5,1.0,2.0],["警示","偏差","普通","良好","優秀"])
        r1_dd  = interpret_value(res1['max_dd']*100,[-50,-30,-20,-10],["警示","偏差","普通","良好","優秀"])
        c1.metric("Ann. Return｜年化報酬", f"{res1['ann_ret']*100:.1f}%", delta=rating_badge(r1_ret))
        c2.metric("Volatility｜波動率",    f"{res1['ann_vol']*100:.1f}%", delta=rating_badge(r1_vol))
        c3.metric("Sharpe Ratio",          f"{res1['sharpe']:.2f}",       delta=rating_badge(r1_sr))
        c4.metric("Max Drawdown｜最大回撤", f"{res1['max_dd']*100:.1f}%", delta=rating_badge(r1_dd))

        with st.expander("📋 Strategy Logic｜策略邏輯說明"):
            st.markdown(f"""
**乖離率均值回歸策略（U03/U04 精神）**
- **計算方式**：乖離率 = 收盤價 ÷ {sma_n} 日移動平均線
- **買入條件**：乖離率 < 1 − {k_val}×標準差（股價低於均線足夠多 → 低估）
- **賣出條件**：乖離率 > 1 + {k_val}×標準差（股價高於均線足夠多 → 高估）
- **手續費**：買入 1.425‰、賣出 4.425‰
- **策略核心**：均值回歸，股價偏離均線後終將回歸
""")

    # ── Tab 2：動量技術策略 ───────────────────
    with tab_s2:
        res2 = momentum_strategy(close, rsi_buy, rsi_sell, bias_n, bias_th)
        st.plotly_chart(plot_momentum_strategy(res2, stock_id), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        r2_ret = interpret_value(res2['ann_ret']*100,[0,5,10,20],["警示","偏差","普通","良好","優秀"])
        r2_vol = interpret_value(res2['ann_vol']*100,[15,20,30,40],["優秀","良好","普通","偏差","警示"])
        r2_sr  = interpret_value(res2['sharpe'],[0,0.5,1.0,2.0],["警示","偏差","普通","良好","優秀"])
        r2_dd  = interpret_value(res2['max_dd']*100,[-50,-30,-20,-10],["警示","偏差","普通","良好","優秀"])
        c1.metric("Ann. Return｜年化報酬", f"{res2['ann_ret']*100:.1f}%", delta=rating_badge(r2_ret))
        c2.metric("Volatility｜波動率",    f"{res2['ann_vol']*100:.1f}%", delta=rating_badge(r2_vol))
        c3.metric("Sharpe Ratio",          f"{res2['sharpe']:.2f}",       delta=rating_badge(r2_sr))
        c4.metric("Max Drawdown｜最大回撤", f"{res2['max_dd']*100:.1f}%", delta=rating_badge(r2_dd))

        with st.expander("📋 Strategy Logic｜策略邏輯說明"):
            st.markdown(f"""
**RSI + 乖離率雙重過濾動量策略（U09/U16/U17 精神）**
- **買入條件**：RSI(14) < {rsi_buy}（超賣）且 {bias_n}日乖離率 < {bias_th}（股價低於均線）
- **賣出條件**：RSI(14) > {rsi_sell}（超買）或 乖離率 > {2-bias_th:.2f}（股價高於均線）
- **均線系統**：MA5（短期）、MA20（中期）、MA60（長期）
- **指標組合**：RSI 過濾短期超買超賣 + 乖離率確認中期偏離程度
""")

    # ── Tab 3：ML 評分 ────────────────────────
    with tab_s3:
        with st.spinner("Calculating ML features... 計算特徵中..."):
            feat = ml_score_single(close, rev)

        if feat.empty:
            st.warning("特徵資料不足，無法計算評分")
        else:
            st.plotly_chart(plot_ml_score(feat, stock_id), use_container_width=True)

            latest = feat.iloc[-1]
            cur_score = latest["ml_score"]

            # 評分儀表板
            col_a, col_b, col_c = st.columns(3)
            score_color = "normal" if cur_score >= 60 else ("off" if cur_score < 40 else "normal")
            col_a.metric("Current ML Score｜當前評分", f"{cur_score:.1f} / 100",
                         delta=f"{'⬆ 買入訊號' if cur_score > 65 else '⬇ 觀望' if cur_score < 40 else '→ 中性'}")
            col_b.metric("RSI(14)", f"{latest.get('rsi14', 'N/A'):.1f}" if 'rsi14' in latest else "N/A")
            col_c.metric("Bias(20)｜20日乖離率", f"{latest.get('bias20', 'N/A'):.3f}" if 'bias20' in latest else "N/A")

            st.progress(int(cur_score))

            # 評分解讀
            if cur_score >= 70:
                st.success("✅ 高評分（>70）：技術面 & 動量面偏強，可關注買入機會")
            elif cur_score >= 50:
                st.info("⚠️ 中等評分（50~70）：維持觀察，等待更明確訊號")
            else:
                st.warning("🔴 低評分（<50）：技術面偏弱，建議觀望或已持有者注意風控")

            with st.expander("📊 Feature Details｜特徵詳情"):
                display_cols = [c for c in feat.columns if any(
                    c.startswith(x) for x in ["bias", "rsi", "rsv", "ret", "mom", "ml_score"]
                )]
                st.dataframe(feat[display_cols].tail(30), use_container_width=True)

            with st.expander("📋 ML Scoring Logic｜評分邏輯說明"):
                st.markdown("""
**ML 綜合評分（U25 精神，規則型加權）**

| 特徵群 | 說明 | 邏輯 |
|--------|------|------|
| 乖離率 bias5~240 | 多週期相對均線偏離 | 越低越便宜，加分 |
| RSV 5~120 | 相對高低點位置 | 越低表示超賣，加分 |
| RSI(7/14) | 短中期強弱 | 接近 50 最佳，偏低加分 |
| 動量 ret5~250 | 相對 n 日前漲跌幅 | 適度正動量加分 |
| MACD Histogram | 多空趨勢 | 正值加分 |
| 月營收動量 mom1~6 | 營收成長趨勢 | 成長加分 |

最終評分標準化至 0~100，**>70 為強買入區間、<40 為弱勢區間**
""")

    # ── Tab 4：財務評價 ───────────────────────
    with tab_s4:
        with st.spinner("Fetching valuation data（TWSE/TPEX）... 抓取評價指標中..."):
            pe_v, pb_v, dy_v, val_source = get_valuation_any(stock_id)

        if val_source == "未找到":
            st.warning(f"⚠️ 在 TWSE 及 TPEX OpenAPI 中均未找到 {stock_id} 的估值資料，"
                       f"可能為興櫃或資料尚未更新。")
        else:
            st.caption(f"📡 資料來源：{val_source} OpenAPI")
            col1, col2, col3 = st.columns(3)
            pe_label = rating_badge(interpret_value(pe_v, [10,15,25,35], ["優秀","良好","普通","偏差","警示"], reverse=True)) if pe_v != "N/A" else ""
            pb_label = rating_badge(interpret_value(pb_v, [1,1.5,2.5,4],  ["優秀","良好","普通","偏差","警示"], reverse=True)) if pb_v != "N/A" else ""
            dy_label = rating_badge(interpret_value(dy_v, [2,3,5,7],       ["警示","偏差","普通","良好","優秀"])) if dy_v != "N/A" else ""
            col1.metric("P/E Ratio｜本益比",        str(pe_v), delta=pe_label)
            col2.metric("P/B Ratio｜淨值比",        str(pb_v), delta=pb_label)
            col3.metric("Dividend Yield｜殖利率",   f"{dy_v}%", delta=dy_label)

            try:
                fs = financial_score(pe_v, pb_v, dy_v)
                st.divider()
                st.subheader("Financial Score｜財務評分（U08 精神）")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("PE Score｜本益比評分",     f"{fs['PE評分']:.0f} / 100",
                          delta=rating_badge(interpret_value(fs['PE評分'],[40,50,65,75],["警示","偏差","普通","良好","優秀"])))
                c2.metric("PB Score｜淨值比評分",     f"{fs['PB評分']:.0f} / 100",
                          delta=rating_badge(interpret_value(fs['PB評分'],[40,50,65,75],["警示","偏差","普通","良好","優秀"])))
                c3.metric("Yield Score｜殖利率評分",  f"{fs['殖利率評分']:.0f} / 100",
                          delta=rating_badge(interpret_value(fs['殖利率評分'],[40,50,65,75],["警示","偏差","普通","良好","優秀"])))
                c4.metric("Total Financial Score｜綜合財務評分", f"{fs['綜合財務評分']:.0f} / 100",
                          delta=rating_badge(interpret_value(fs['綜合財務評分'],[40,50,65,75],["警示","偏差","普通","良好","優秀"])))

                total = fs["綜合財務評分"]
                st.progress(int(total))
                if total >= 70:
                    st.success("✅ 財務面評分高（>70）：估值相對便宜，股息具吸引力")
                elif total >= 50:
                    st.info("⚠️ 財務面評分中等（50~70）：合理估值，適合長期持有者")
                else:
                    st.warning("🔴 財務面評分低（<50）：估值偏高或股息不足，需謹慎")
            except Exception as ex:
                st.warning(f"財務評分計算錯誤：{ex}")

            with st.expander("📋 Valuation Logic｜評價邏輯說明"):
                st.markdown("""
**財務評分方法（U08 課程精神）**

| 指標 | 評分邏輯 | 說明 |
|------|----------|------|
| P/E 本益比 | PE=10 → 100分，PE=30 → 40分 | 低 PE 代表便宜 |
| P/B 淨值比 | PB=1 → 100分，PB=3 → 60分 | 低 PB 代表貼近資產價值 |
| 殖利率 | DY=5% → 75分，DY=7% → 100分 | 高股息收益 |

綜合財務評分 = 三項平均值（0~100）
""")

    # ── Tab 5：策略績效比較 ───────────────────
    with tab_s5:
        results = {
            "Bias Reversion｜乖離率": res1,
            "Momentum｜動量策略":     res2,
        }
        st.plotly_chart(performance_bar(results), use_container_width=True)

        # 資金曲線疊加
        fig_eq = go.Figure()
        colors_eq = ["#FF7043", "#42A5F5", "#AB47BC"]
        for i, (name, res) in enumerate(results.items()):
            fig_eq.add_trace(go.Scatter(
                x=res["equity"].index, y=res["equity"],
                name=name, line=dict(color=colors_eq[i], width=2)
            ))
        fig_eq.add_trace(go.Scatter(
            x=res1["bh"].index, y=res1["bh"],
            name="Buy & Hold｜持有不賣",
            line=dict(color="#78909C", dash="dash", width=1.5)
        ))
        fig_eq.update_layout(title="Strategy Equity Curves｜各策略資金曲線比較",
                             height=450, **DARK)
        fig_eq.update_xaxes(gridcolor="#1e1e2e")
        fig_eq.update_yaxes(gridcolor="#1e1e2e")
        st.plotly_chart(fig_eq, use_container_width=True)

        # 績效彙總表
        st.subheader("Performance Summary｜績效彙總")
        perf_data = []
        for name, res in results.items():
            perf_data.append({
                "Strategy｜策略":         name,
                "Ann. Return｜年化報酬":  f"{res['ann_ret']*100:.2f}%",
                "Ann. Vol｜年化波動率":   f"{res['ann_vol']*100:.2f}%",
                "Sharpe Ratio":           f"{res['sharpe']:.3f}",
                "Max DD｜最大回撤":       f"{res['max_dd']*100:.2f}%",
            })
        # 加入 Buy & Hold
        bh_ret = close.pct_change().dropna()
        bh_ann = bh_ret.mean() * 252
        bh_vol = bh_ret.std() * np.sqrt(252)
        bh_eq  = (1 + bh_ret).cumprod()
        bh_dd  = (bh_eq / bh_eq.cummax() - 1).min()
        perf_data.append({
            "Strategy｜策略":         "Buy & Hold｜持有",
            "Ann. Return｜年化報酬":  f"{bh_ann*100:.2f}%",
            "Ann. Vol｜年化波動率":   f"{bh_vol*100:.2f}%",
            "Sharpe Ratio":           f"{bh_ann/bh_vol:.3f}" if bh_vol > 0 else "N/A",
            "Max DD｜最大回撤":       f"{bh_dd*100:.2f}%",
        })
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

        with st.expander("📋 Course References｜課程對應說明"):
            st.markdown("""
| 策略 | 對應課程 | 核心原理 |
|------|----------|----------|
| 乖離率均值回歸 | U03/U04 | 乖離率偏離均線後均值回歸 |
| 動量 RSI 策略 | U09/U16 | 技術指標組合過濾買賣訊號 |
| ML 綜合評分 | U25 | 多特徵規則加權（乖離率+動量+RSV+月營收） |
| 財務評價 | U08 | PE/PB/殖利率財務評分 |
| 績效評估 | U07 | 年化報酬、Sharpe、最大回撤 |
""")

    # ── Tab 長期：5年財務分析 ─────────────────
    with tab_lt:
        st.subheader(f"📅 {stock_id} 長期財務概況（近5年）")
        lt = long_term_data
        # 顯示資料來源
        if lt.get("_yf_supplement"):
            st.info("📡 部分資料由 **Yahoo Finance** 補充（FinMind 資料不足或需 Token）")
        elif any(k.startswith("fs_") or k.startswith("cf_") for k in lt):
            st.success("📡 資料來源：**FinMind API**")

        # EPS 趨勢圖
        eps = lt.get("fs_eps")
        if eps is not None and len(eps) >= 2:
            st.markdown("### 每股盈餘（EPS）季趨勢")
            fig_eps = go.Figure()
            fig_eps.add_trace(go.Bar(x=eps.index, y=eps.values, name="EPS",
                                     marker_color=["#26A69A" if v > 0 else "#EF5350" for v in eps.values]))
            fig_eps.update_layout(title="EPS 季趨勢", height=300, **DARK)
            fig_eps.update_xaxes(gridcolor="#1e1e2e")
            fig_eps.update_yaxes(gridcolor="#1e1e2e")
            st.plotly_chart(fig_eps, use_container_width=True)
            st.caption("EPS（每股盈餘）越高越好，代表公司每股賺得越多。正成長趨勢為優質訊號。")
        else:
            st.info("EPS 資料不足（需 FinMind Token 或資料延遲）")

        # ROE 趨勢
        roe = lt.get("fs_roe")
        if roe is not None and len(roe) >= 2:
            st.markdown("### 股東權益報酬率（ROE）")
            fig_roe = go.Figure()
            fig_roe.add_trace(go.Scatter(x=roe.index, y=roe.values, name="ROE%",
                                          line=dict(color="#FFA726", width=2), fill="tozeroy"))
            fig_roe.add_hline(y=15, line_color="#26A69A", line_dash="dash",
                              annotation_text="15%（優質門檻）")
            fig_roe.update_layout(title="ROE 季趨勢 (%)", height=280, **DARK)
            fig_roe.update_xaxes(gridcolor="#1e1e2e")
            fig_roe.update_yaxes(gridcolor="#1e1e2e")
            st.plotly_chart(fig_roe, use_container_width=True)
            roe_avg = roe.tail(8).mean()
            roe_lbl = interpret_value(roe_avg, [0, 8, 15, 20], ["警示","偏差","普通","良好","優秀"])
            st.caption(f"ROE平均 {roe_avg:.1f}% — {rating_badge(roe_lbl)}｜ROE>15%為優質公司，代表每一元股東資本可創造更多利潤。")
        else:
            st.info("ROE 資料不足（需 FinMind Token）")

        # 現金股利
        div = lt.get("cash_dividend_series")
        if div is not None and len(div) >= 2:
            st.markdown("### 現金股利歷史")
            fig_div = go.Figure()
            fig_div.add_trace(go.Bar(x=div.index, y=div.values, name="現金股利",
                                      marker_color="#42A5F5"))
            fig_div.update_layout(title="現金股利（元/股）", height=260, **DARK)
            fig_div.update_xaxes(gridcolor="#1e1e2e")
            fig_div.update_yaxes(gridcolor="#1e1e2e")
            st.plotly_chart(fig_div, use_container_width=True)
            div_growth = (div.iloc[-1] / div.iloc[0] - 1) * 100 if div.iloc[0] > 0 else 0
            div_lbl = interpret_value(div_growth, [0, 10, 30, 50], ["警示","偏差","普通","良好","優秀"])
            st.caption(f"5年股利成長率 {div_growth:.1f}% — {rating_badge(div_lbl)}｜持續增加股利代表公司獲利穩定且重視股東回報。")
        else:
            st.info("股利資料不足（需 FinMind Token）")

        # 營業現金流
        ocf = lt.get("cf_operatingactivities")
        if ocf is not None and len(ocf) >= 2:
            st.markdown("### 營業現金流")
            fig_ocf = go.Figure()
            fig_ocf.add_trace(go.Bar(x=ocf.index, y=ocf.values, name="營業CF",
                                      marker_color=["#26A69A" if v >= 0 else "#EF5350" for v in ocf.values]))
            fig_ocf.update_layout(title="營業現金流（千元）", height=260, **DARK)
            fig_ocf.update_xaxes(gridcolor="#1e1e2e")
            fig_ocf.update_yaxes(gridcolor="#1e1e2e")
            st.plotly_chart(fig_ocf, use_container_width=True)
            pos_ratio = (ocf.tail(8) > 0).mean() * 100
            cf_lbl = interpret_value(pos_ratio, [50, 62, 75, 87], ["警示","偏差","普通","良好","優秀"])
            st.caption(f"近8季正現金流比例 {pos_ratio:.0f}% — {rating_badge(cf_lbl)}｜營業現金流持續為正代表公司真正賺到現金，財務健康。")
        else:
            st.info("現金流資料不足（需 FinMind Token）")

        # 股價長期報酬
        st.markdown("### 股價長期績效")
        lt_cols = st.columns(3)
        for period, days, label in [(1, 252, "近1年"), (3, 756, "近3年"), (5, 1260, "近5年")]:
            if len(close) >= days:
                ret = (close.iloc[-1] / close.iloc[-days] - 1) * 100
                ret_lbl = interpret_value(ret, [0, 10, 30, 60], ["警示","偏差","普通","良好","優秀"])
                lt_cols[period-1 if period <= 3 else 2].metric(
                    label, f"{ret:.1f}%", delta=rating_badge(ret_lbl)
                )
        st.caption("長期報酬率越高代表持有越有價值；需搭配基本面確認是否為實質成長。")

        with st.expander("📋 長期評等說明"):
            st.markdown("""
| 評等 | 顏色 | 說明 |
|------|------|------|
| 🟢 優秀 | 綠色 | 遠優於市場水準，強烈看好 |
| 🔵 良好 | 藍色 | 優於市場平均，正面訊號 |
| 🟡 普通 | 黃色 | 接近市場平均，中性觀察 |
| 🟠 偏差 | 橘色 | 低於市場平均，需注意 |
| 🔴 警示 | 紅色 | 明顯低於標準，高風險訊號 |
""")

else:
    st.info("👈 請在左側設定股票代號與參數，然後點擊「🚀 Run Analysis｜執行分析」開始")

    st.markdown("""
### 📚 系統功能說明 System Overview

本系統整合 FinLab ML 課程 U03~U25 的投資策略知識，提供以下分析：

| 頁籤 | 策略 | 課程來源 |
|------|------|----------|
| 📉 Bias Reversion | 乖離率均值回歸 | U03/U04 參數優化 |
| 📈 Momentum | RSI + 乖離率雙重過濾 | U09/U16/U17 技術指標 |
| 🤖 ML Score | 多特徵規則加權評分 | U25 特徵工程（bias/rsv/mom/acc）|
| 💰 Valuation | PE/PB/殖利率財務評分 | U08 財務指標 |
| 📊 Performance | 年化報酬/Sharpe/最大回撤 | U07 Pyfolio 精神 |

**使用建議**：
1. 輸入 FinMind Token 可取得更長歷史資料（免費版每日有限制）
2. 分析期間建議至少 2~3 年，以確保指標穩定
3. 三個策略訊號一致時，買入信心度最高
""")

# ── Tab AI：獨立渲染，不受 run_btn 影響 ─────────────────────────────────
with tab_ai:
    st.subheader("🧠 AI Investment Analysis｜AI 多策略投資建議")

    if not st.session_state.get("_analysis_ready"):
        st.info("👈 請先點擊「🚀 Run Analysis｜執行分析」執行策略分析，再生成 AI 報告")
    else:
        _close     = st.session_state["_close"]
        _res1      = st.session_state["_res1"]
        _res2      = st.session_state["_res2"]
        _feat      = st.session_state["_feat"]
        _sid       = st.session_state["_stock_id"]
        _lt_data   = st.session_state.get("_long_term_data", {})
        _today_str = datetime.now().strftime("%Y%m%d")

        # 取得評價數據（TWSE → TPEX fallback，給 AI 用）
        try:
            _pe_v, _pb_v, _dy_v, _ = get_valuation_any(_sid)
        except Exception:
            _pe_v, _pb_v, _dy_v = "N/A", "N/A", "N/A"

        # AI 模型選擇
        ai_mode = st.radio(
            "AI Model｜AI 模型選擇",
            ["OpenAI o4-mini", "Gemini 2.5 Flash", "Both｜兩者對照"],
            horizontal=True,
        )

        need_openai = ai_mode in ["OpenAI o4-mini", "Both｜兩者對照"]
        need_gemini = ai_mode in ["Gemini 2.5 Flash", "Both｜兩者對照"]

        # 從側欄讀取 API Key（已在側欄輸入）
        ai_key_openai = st.session_state.get("openai_key_input", openai_key)
        ai_key_gemini = st.session_state.get("google_key_input", google_key)

        gen_btn = st.button("✨ Generate AI Report｜生成 AI 報告", use_container_width=True)

        if gen_btn:
            # 組建財務評分 + 策略摘要
            try:
                _fin_scores = financial_score(_pe_v, _pb_v, _dy_v)
            except Exception:
                _fin_scores = {}

            _summary = build_strategy_summary(_sid, _close, _res1, _res2, _feat, _fin_scores, _pe_v, _pb_v, _dy_v, _lt_data)

            # OpenAI 分析
            if need_openai:
                with st.spinner("OpenAI 分析中..."):
                    if not ai_key_openai:
                        st.session_state["ai_report_openai"] = "❌ 請在左側側欄輸入 OpenAI API Key"
                    else:
                        _rep = run_ai_openai(_summary, _sid, ai_key_openai)
                        st.session_state["ai_report_openai"] = _rep or "❌ OpenAI 分析失敗"

            # Gemini 分析
            if need_gemini:
                with st.spinner("Gemini 分析中..."):
                    if not ai_key_gemini:
                        st.session_state["ai_report_gemini"] = "❌ 請在左側側欄輸入 Google API Key"
                    elif not GOOGLE_GENAI_AVAILABLE:
                        st.session_state["ai_report_gemini"] = "❌ google-genai 套件未安裝"
                    else:
                        _rep = run_ai_gemini(_summary, _sid, ai_key_gemini)
                        st.session_state["ai_report_gemini"] = _rep or "❌ Gemini 分析失敗"

        # 顯示報告
        _rep_oa = st.session_state.get("ai_report_openai", "")
        _rep_gm = st.session_state.get("ai_report_gemini", "")
        _show_mode = st.session_state.get("ai_mode_last", ai_mode)
        # 記錄最後一次選擇的模式
        st.session_state["ai_mode_last"] = ai_mode

        has_openai_report = bool(_rep_oa)
        has_gemini_report = bool(_rep_gm)

        if has_openai_report and has_gemini_report:
            col_oa, col_gm = st.columns(2)
            with col_oa:
                st.markdown("### 🔵 OpenAI o4-mini 分析報告")
                st.text_area("OpenAI Report", _rep_oa, height=500, key="ta_oa")
                st.download_button(
                    "⬇️ Download OpenAI Report｜下載報告",
                    data=_rep_oa,
                    file_name=f"{_sid}_openai_analysis_{_today_str}.txt",
                    mime="text/plain",
                    key="dl_oa",
                )
            with col_gm:
                st.markdown("### 🟢 Gemini 2.5 Flash 分析報告")
                st.text_area("Gemini Report", _rep_gm, height=500, key="ta_gm")
                st.download_button(
                    "⬇️ Download Gemini Report｜下載報告",
                    data=_rep_gm,
                    file_name=f"{_sid}_gemini_analysis_{_today_str}.txt",
                    mime="text/plain",
                    key="dl_gm",
                )
        elif has_openai_report:
            st.markdown("### 🔵 OpenAI o4-mini 分析報告")
            st.text_area("OpenAI Report", _rep_oa, height=600, key="ta_oa2")
            st.download_button(
                "⬇️ Download OpenAI Report｜下載報告",
                data=_rep_oa,
                file_name=f"{_sid}_openai_analysis_{_today_str}.txt",
                mime="text/plain",
                key="dl_oa2",
            )
        elif has_gemini_report:
            st.markdown("### 🟢 Gemini 2.5 Flash 分析報告")
            st.text_area("Gemini Report", _rep_gm, height=600, key="ta_gm2")
            st.download_button(
                "⬇️ Download Gemini Report｜下載報告",
                data=_rep_gm,
                file_name=f"{_sid}_gemini_analysis_{_today_str}.txt",
                mime="text/plain",
                key="dl_gm2",
            )
