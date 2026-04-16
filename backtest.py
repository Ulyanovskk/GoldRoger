"""
backtest.py — Backtesting de la stratégie FXBOT EUR/USD sur données historiques.
Utilise OpenRouter (DeepSeek) pour reproduire les décisions IA du bot live.

# BACKTEST-AI : Fichier autonome — ne modifie PAS bot.py, utils.py, config.py.
# Le DEEPSEEK_SYSTEM_PROMPT est importé depuis config.py pour cohérence live/backtest.

Usage :
    python backtest.py
    python backtest.py --symbol EURUSDm --days 90 --balance 50
"""

import os
import sys
import json
import time
import hashlib
import asyncio
import argparse
import datetime
import logging
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import ta
import MetaTrader5 as mt5

from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════
# BACKTEST-AI : Chargement de la configuration
# ══════════════════════════════════════════════════════════════

load_dotenv()

# Importer le prompt système depuis config.py (cohérence live/backtest)
import config
DEEPSEEK_SYSTEM_PROMPT = config.DEEPSEEK_SYSTEM_PROMPT

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION OPENROUTER — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

if not OPENROUTER_API_KEY:
    print("❌ ERREUR : OPENROUTER_API_KEY manquante dans .env")
    print("   Ajoutez : OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxx")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 3. RATE LIMITING — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

DELAY_BETWEEN_CALLS = 1.5  # secondes entre chaque appel OpenRouter

# ══════════════════════════════════════════════════════════════
# 4. CACHE DES DÉCISIONS — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

CACHE_FILE = "backtest_cache.json"


def load_cache() -> dict:
    """Charge le cache des décisions IA depuis le fichier JSON."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache: dict):
    """Sauvegarde le cache des décisions IA dans le fichier JSON."""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)


def get_cache_key(prompt_data: str) -> str:
    """Génère une clé de cache MD5 à partir du prompt."""
    return hashlib.md5(prompt_data.encode()).hexdigest()


# ══════════════════════════════════════════════════════════════
# LOGGING BACKTEST — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [BACKTEST] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest")


# ══════════════════════════════════════════════════════════════
# 2. FONCTION D'APPEL OPENROUTER — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

async def ai_decision(prompt_data: str) -> dict:
    """
    Envoie le prompt à OpenRouter (DeepSeek) et retourne le signal JSON.
    Reproduit exactement le format d'appel du bot live.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://goldbot.local",
        "X-Title": "GOLDBOT Backtest"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "max_tokens": 200,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_data}
        ]
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()

    # BACKTEST-AI : Extraire le contenu de la réponse
    content = data['choices'][0]['message']['content']

    # BACKTEST-AI : Parser le JSON retourné (nettoyer les blocs ```json```)
    clean = content.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]

    return json.loads(clean.strip())


# ══════════════════════════════════════════════════════════════
# DONNÉES HISTORIQUES MT5 — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

def get_historical_data(symbol: str, timeframe, days: int) -> Optional[pd.DataFrame]:
    """
    Récupère les données historiques M15 depuis MetaTrader 5.
    Retourne un DataFrame avec colonnes : Open, High, Low, Close, Volume, time.
    """
    utc_to = datetime.datetime.now(datetime.timezone.utc)
    utc_from = utc_to - datetime.timedelta(days=days)

    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        log.error("Aucune donnée MT5 pour %s (timeframe=%s, days=%d)", symbol, timeframe, days)
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low", "close": "Close",
        "tick_volume": "Volume"
    }, inplace=True)

    log.info("Données chargées : %d bougies M15 (%s → %s)",
             len(df), df["time"].iloc[0].strftime("%Y-%m-%d"),
             df["time"].iloc[-1].strftime("%Y-%m-%d"))
    return df


# ══════════════════════════════════════════════════════════════
# CALCUL DES INDICATEURS — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule tous les indicateurs techniques sur le DataFrame complet.
    Reproduit exactement les calculs de utils.compute_indicators().
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(close, window=config.RSI_PERIOD).rsi()

    # MACD
    macd_obj = ta.trend.MACD(
        close,
        window_fast=config.MACD_FAST,
        window_slow=config.MACD_SLOW,
        window_sign=config.MACD_SIGNAL,
    )
    df["MACD"]        = macd_obj.macd()
    df["MACD_signal"] = macd_obj.macd_signal()
    df["MACD_hist"]   = macd_obj.macd_diff()

    # EMAs
    df["EMA20"]  = ta.trend.EMAIndicator(close, window=config.EMA_FAST).ema_indicator()
    df["EMA50"]  = ta.trend.EMAIndicator(close, window=config.EMA_MID).ema_indicator()
    df["EMA200"] = ta.trend.EMAIndicator(close, window=config.EMA_SLOW).ema_indicator()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=config.BB_PERIOD, window_dev=config.BB_STD)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    # ATR
    df["ATR"] = ta.volatility.AverageTrueRange(
        high, low, close, window=config.ATR_PERIOD
    ).average_true_range()

    log.info("Indicateurs calculés : RSI, MACD, EMA20/50/200, BB, ATR")
    return df


# ══════════════════════════════════════════════════════════════
# FONCTIONS D'ANALYSE STRUCTURELLE — BACKTEST-AI
# Répliques locales pour le backtest (mêmes algos que utils.py)
# ══════════════════════════════════════════════════════════════

def detect_price_structure(df: pd.DataFrame, idx: int) -> tuple:
    """
    BACKTEST-AI : Détection par Régression Linéaire sur 30 bougies.
    Retourne (Tendance, Pente).
    """
    try:
        start = max(0, idx - 30)
        y = df["Close"].values[start:idx]
        if len(y) < 10:
            return "RANGE", 0.0
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)

        high_slice = df["High"].iloc[start:idx]
        low_slice  = df["Low"].iloc[start:idx]
        close_slice = df["Close"].iloc[start:idx]
        atr = ta.volatility.AverageTrueRange(
            high_slice, low_slice, close_slice
        ).average_true_range().iloc[-1]
        threshold = atr * 0.01

        if slope < -threshold:
            return "DOWNTREND", slope
        elif slope > threshold:
            return "UPTREND", slope
        else:
            return "RANGE", slope
    except Exception:
        return "RANGE", 0.0


def ema_slope(values: np.ndarray, atr_value: float) -> str:
    """BACKTEST-AI : Pente EMA20 relative à l'ATR."""
    try:
        if len(values) < 5 or atr_value <= 0:
            return "FLAT"
        delta = values[-1] - values[-5]
        threshold = atr_value * 0.1
        if delta < -threshold:
            return "DOWN"
        elif delta > threshold:
            return "UP"
        else:
            return "FLAT"
    except Exception:
        return "FLAT"


def channel_position(df: pd.DataFrame, idx: int, current_price: float) -> str:
    """BACKTEST-AI : Position du prix dans le range 20 bougies."""
    try:
        start = max(0, idx - 20)
        highs = df["High"].values[start:idx]
        lows  = df["Low"].values[start:idx]
        ch_high  = float(np.max(highs))
        ch_low   = float(np.min(lows))
        ch_range = ch_high - ch_low
        if ch_range == 0:
            return "UNKNOWN"
        position = (current_price - ch_low) / ch_range
        if position > 0.75:
            return "TOP"
        elif position < 0.25:
            return "BOTTOM"
        else:
            return "MID"
    except Exception:
        return "UNKNOWN"


def rsi_slope_bt(rsi_values: np.ndarray) -> str:
    """BACKTEST-AI : Accélération RSI."""
    try:
        if len(rsi_values) < 5:
            return "FLAT"
        diff = rsi_values[-1] - rsi_values[-5]
        return "ACCEL_UP" if diff > 5 else "ACCEL_DOWN" if diff < -5 else "FLAT"
    except Exception:
        return "FLAT"


# ══════════════════════════════════════════════════════════════
# 5. CONSTRUCTION DU PROMPT UTILISATEUR — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

def build_backtest_prompt(row, df: pd.DataFrame, idx: int) -> str:
    """
    BACKTEST-AI : Construit le prompt identique au bot live.
    Reprend exactement le même format que compress_data() dans utils.py.
    """
    # Tendance EMA
    ema_trend = "bull" if row["Close"] > row["EMA200"] else "bear"

    # MACD direction
    macd_dir = "+" if row["MACD_hist"] > 0 else "-"

    # Bollinger position
    bb_range = row["BB_upper"] - row["BB_lower"]
    if bb_range > 0:
        bb_pct = (row["Close"] - row["BB_lower"]) / bb_range
    else:
        bb_pct = 0.5
    if bb_pct > 0.8:
        bb_pos = "high"
    elif bb_pct < 0.2:
        bb_pos = "low"
    else:
        bb_pos = "mid"

    # Structure de prix (régression linéaire)
    struct, lrs = detect_price_structure(df, idx)

    # Pente EMA20 M15 (ATR-relative)
    start_slope = max(0, idx - 5)
    ema20_vals = df["EMA20"].values[start_slope:idx]
    atr_val = row["ATR"] if not pd.isna(row["ATR"]) else 0.0005
    slope_m15 = ema_slope(ema20_vals, atr_val)

    # Position dans le channel
    ch_pos = channel_position(df, idx, row["Close"])

    # RSI slope
    start_rsi = max(0, idx - 5)
    rsi_vals = df["RSI"].values[start_rsi:idx]
    r_slope = rsi_slope_bt(rsi_vals)

    # BACKTEST-AI : Format identique à compress_data() du bot live
    prompt = (
        f"EURUSD|"
        f"M15:R={row['RSI']:.1f},M={macd_dir},B={bb_pos},"
        f"E={ema_trend},A={row['ATR']:.5f}|"
        f"price={row['Close']:.5f}|"
        f"sess=backtest|news=N/A|"
        f"STRUCT={struct}|LRS={lrs:.6f}|SLOPE_M15={slope_m15}|"
        f"CH_POS={ch_pos}|R_SLOPE={r_slope}"
    )
    return prompt


# ══════════════════════════════════════════════════════════════
# SIMULATION DE TRADE — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

SPREAD_PIPS = 1.5  # Spread simulé en pips (EUR/USD typique)
SPREAD_PRICE = SPREAD_PIPS * 0.0001  # Conversion en prix


def simulate_trade(signal: dict, df: pd.DataFrame, idx: int, balance: float) -> Optional[dict]:
    """
    BACKTEST-AI : Simule un trade sur les bougies suivantes.
    Vérifie si le TP ou SL est touché en premier.
    Retourne un dict avec les résultats ou None si le trade ne peut pas être simulé.
    """
    try:
        direction = signal.get("DIR", "HOLD")
        if direction == "HOLD":
            return None

        tp = signal.get("TP", 0)
        sl = signal.get("SL", 0)
        if tp == 0 or sl == 0:
            return None

        entry_price = df["Close"].iloc[idx]

        # BACKTEST-AI : Ajout du spread au prix d'entrée
        if direction == "BUY":
            entry_price += SPREAD_PRICE / 2  # Ask = mid + spread/2
        else:
            entry_price -= SPREAD_PRICE / 2  # Bid = mid - spread/2

        # BACKTEST-AI : Calcul du lot (risque 1.5% du balance)
        sl_diff = abs(entry_price - sl)
        if sl_diff <= 0:
            return None

        # Lot simplifié : risque / (SL en pips * valeur pip pour 1 lot standard)
        # Pour EUR/USD : 1 lot = 100,000 unités, 1 pip = 10 USD
        risk_amount = balance * (config.RISK_PERCENT / 100)
        sl_pips = sl_diff / 0.0001
        pip_value = 10.0  # USD par pip pour 1 lot standard EUR/USD
        lot = risk_amount / (sl_pips * pip_value)
        lot = max(0.01, round(lot, 2))
        lot = min(lot, 1.0)  # Cap à 1 lot

        entry_time = df["time"].iloc[idx]

        # BACKTEST-AI : Parcourir les bougies suivantes pour vérifier TP/SL
        max_bars = min(96, len(df) - idx - 1)  # Max 96 bougies = 24h en M15
        for i in range(1, max_bars + 1):
            future_idx = idx + i
            if future_idx >= len(df):
                break

            bar_high = df["High"].iloc[future_idx]
            bar_low  = df["Low"].iloc[future_idx]

            if direction == "BUY":
                # SL touché ?
                if bar_low <= sl:
                    pnl_pips = (sl - entry_price) / 0.0001
                    pnl = pnl_pips * pip_value * lot
                    return {
                        "dir": direction, "entry": entry_price, "exit": sl,
                        "tp": tp, "sl": sl, "lot": lot,
                        "pnl": round(pnl, 2), "pnl_pips": round(pnl_pips, 1),
                        "result": "SL", "bars": i,
                        "entry_time": str(entry_time),
                        "exit_time": str(df["time"].iloc[future_idx]),
                        "conf": signal.get("CONF", 0),
                        "reason": signal.get("REASON", ""),
                    }
                # TP touché ?
                if bar_high >= tp:
                    pnl_pips = (tp - entry_price) / 0.0001
                    pnl = pnl_pips * pip_value * lot
                    return {
                        "dir": direction, "entry": entry_price, "exit": tp,
                        "tp": tp, "sl": sl, "lot": lot,
                        "pnl": round(pnl, 2), "pnl_pips": round(pnl_pips, 1),
                        "result": "TP", "bars": i,
                        "entry_time": str(entry_time),
                        "exit_time": str(df["time"].iloc[future_idx]),
                        "conf": signal.get("CONF", 0),
                        "reason": signal.get("REASON", ""),
                    }

            elif direction == "SELL":
                # SL touché ?
                if bar_high >= sl:
                    pnl_pips = (entry_price - sl) / 0.0001
                    pnl = pnl_pips * pip_value * lot
                    return {
                        "dir": direction, "entry": entry_price, "exit": sl,
                        "tp": tp, "sl": sl, "lot": lot,
                        "pnl": round(pnl, 2), "pnl_pips": round(pnl_pips, 1),
                        "result": "SL", "bars": i,
                        "entry_time": str(entry_time),
                        "exit_time": str(df["time"].iloc[future_idx]),
                        "conf": signal.get("CONF", 0),
                        "reason": signal.get("REASON", ""),
                    }
                # TP touché ?
                if bar_low <= tp:
                    pnl_pips = (entry_price - tp) / 0.0001
                    pnl = pnl_pips * pip_value * lot
                    return {
                        "dir": direction, "entry": entry_price, "exit": tp,
                        "tp": tp, "sl": sl, "lot": lot,
                        "pnl": round(pnl, 2), "pnl_pips": round(pnl_pips, 1),
                        "result": "TP", "bars": i,
                        "entry_time": str(entry_time),
                        "exit_time": str(df["time"].iloc[future_idx]),
                        "conf": signal.get("CONF", 0),
                        "reason": signal.get("REASON", ""),
                    }

        # BACKTEST-AI : Timeout — fermeture au prix de la dernière bougie
        last_idx = min(idx + max_bars, len(df) - 1)
        exit_price = df["Close"].iloc[last_idx]
        if direction == "BUY":
            pnl_pips = (exit_price - entry_price) / 0.0001
        else:
            pnl_pips = (entry_price - exit_price) / 0.0001
        pnl = pnl_pips * pip_value * lot

        return {
            "dir": direction, "entry": entry_price, "exit": exit_price,
            "tp": tp, "sl": sl, "lot": lot,
            "pnl": round(pnl, 2), "pnl_pips": round(pnl_pips, 1),
            "result": "TIMEOUT", "bars": max_bars,
            "entry_time": str(entry_time),
            "exit_time": str(df["time"].iloc[last_idx]),
            "conf": signal.get("CONF", 0),
            "reason": signal.get("REASON", ""),
        }

    except Exception as e:
        log.error("Erreur simulate_trade idx=%d : %s", idx, e)
        return None


# ══════════════════════════════════════════════════════════════
# RAPPORT DE PERFORMANCE — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

def print_report(trades: list, final_balance: float, initial_balance: float):
    """
    BACKTEST-AI : Affiche un rapport complet de performance du backtest.
    """
    print("\n" + "=" * 70)
    print("  📊 RAPPORT DE BACKTEST — FXBOT EUR/USD (OpenRouter/DeepSeek)")
    print("=" * 70)

    if not trades:
        print("\n  ⚠️  Aucun trade exécuté pendant la période.\n")
        return

    total     = len(trades)
    winners   = [t for t in trades if t["pnl"] > 0]
    losers    = [t for t in trades if t["pnl"] < 0]
    breakeven = [t for t in trades if t["pnl"] == 0]

    win_rate = len(winners) / total * 100 if total > 0 else 0
    total_pnl = sum(t["pnl"] for t in trades)

    # Drawdown max
    running_balance = initial_balance
    peak = initial_balance
    max_dd = 0
    max_dd_pct = 0
    for t in trades:
        running_balance += t["pnl"]
        if running_balance > peak:
            peak = running_balance
        dd = peak - running_balance
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd = dd

    # Profit factor
    gross_profit = sum(t["pnl"] for t in winners) if winners else 0
    gross_loss   = abs(sum(t["pnl"] for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Moyenne
    avg_win  = gross_profit / len(winners) if winners else 0
    avg_loss = gross_loss / len(losers) if losers else 0
    avg_pnl  = total_pnl / total

    # Stats par direction
    buys  = [t for t in trades if t["dir"] == "BUY"]
    sells = [t for t in trades if t["dir"] == "SELL"]
    buy_wr  = sum(1 for t in buys if t["pnl"] > 0) / len(buys) * 100 if buys else 0
    sell_wr = sum(1 for t in sells if t["pnl"] > 0) / len(sells) * 100 if sells else 0

    # Stats par résultat
    tp_count      = sum(1 for t in trades if t["result"] == "TP")
    sl_count      = sum(1 for t in trades if t["result"] == "SL")
    timeout_count = sum(1 for t in trades if t["result"] == "TIMEOUT")

    # Durée moyenne
    avg_bars = sum(t["bars"] for t in trades) / total
    avg_duration_h = avg_bars * 15 / 60  # M15 → heures

    # Séries
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for t in trades:
        if t["pnl"] > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        elif t["pnl"] < 0:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    pnl_pct = (final_balance - initial_balance) / initial_balance * 100

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  💰 RÉSULTATS FINANCIERS                                    │
  ├─────────────────────────────────────────────────────────────┤
  │  Balance initiale  : {initial_balance:>10.2f} $                        │
  │  Balance finale    : {final_balance:>10.2f} $                        │
  │  P&L total         : {total_pnl:>+10.2f} $ ({pnl_pct:+.1f}%)                │
  │  Drawdown max      : {max_dd:>10.2f} $ ({max_dd_pct:.1f}%)                │
  │  Profit Factor     : {profit_factor:>10.2f}                              │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │  📈 STATISTIQUES DE TRADING                                 │
  ├─────────────────────────────────────────────────────────────┤
  │  Trades totaux     : {total:>5}                                    │
  │  Gagnants          : {len(winners):>5}  ({win_rate:.1f}%)                        │
  │  Perdants          : {len(losers):>5}  ({100-win_rate:.1f}%)                        │
  │  Break-even        : {len(breakeven):>5}                                    │
  │  TP atteints       : {tp_count:>5}                                    │
  │  SL atteints       : {sl_count:>5}                                    │
  │  Timeout           : {timeout_count:>5}                                    │
  ├─────────────────────────────────────────────────────────────┤
  │  Gain moyen        : {avg_win:>+10.2f} $                             │
  │  Perte moyenne     : {-avg_loss:>+10.2f} $                             │
  │  P&L moyen/trade   : {avg_pnl:>+10.2f} $                             │
  │  Durée moy./trade  : {avg_duration_h:>6.1f} h ({avg_bars:.0f} bougies M15)       │
  ├─────────────────────────────────────────────────────────────┤
  │  Série gagnante max: {max_consec_wins:>5}                                    │
  │  Série perdante max: {max_consec_losses:>5}                                    │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │  🔄 STATS PAR DIRECTION                                     │
  ├─────────────────────────────────────────────────────────────┤
  │  BUY  : {len(buys):>4} trades | Win rate : {buy_wr:>5.1f}%                    │
  │  SELL : {len(sells):>4} trades | Win rate : {sell_wr:>5.1f}%                    │
  └─────────────────────────────────────────────────────────────┘
""")

    # BACKTEST-AI : Sauvegarder les résultats dans un fichier JSON
    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_pnl": total_pnl,
        "pnl_pct": round(pnl_pct, 2),
        "total_trades": total,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 999,
        "max_drawdown_pct": round(max_dd_pct, 1),
        "avg_trade_pnl": round(avg_pnl, 2),
        "trades": trades,
    }
    report_file = f"backtest_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"  💾 Rapport détaillé sauvegardé : {report_file}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════
# 6. BOUCLE PRINCIPALE ASYNC — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

async def run_backtest(symbol: str = "EURUSDm",
                       days: int = 90,
                       initial_balance: float = 50.0,
                       min_conf: int = 62):
    """
    BACKTEST-AI : Boucle principale du backtesting.
    Parcourt les données historiques, appelle l'IA via OpenRouter,
    et simule les trades.
    """
    cache   = load_cache()
    balance = initial_balance
    trades  = []
    api_calls = 0
    cache_hits = 0
    skipped = 0

    # BACKTEST-AI : Récupération des données historiques
    log.info("Chargement des données historiques %s (%d jours)...", symbol, days)
    df_m15 = get_historical_data(symbol, mt5.TIMEFRAME_M15, days)
    if df_m15 is None:
        log.error("Impossible de charger les données. Arrêt du backtest.")
        return [], initial_balance

    # BACKTEST-AI : Calcul des indicateurs
    log.info("Calcul des indicateurs techniques...")
    df_m15 = calculate_indicators(df_m15)

    # BACKTEST-AI : Vérifier qu'on a assez de données (200 bougies EMA200 + buffer)
    start_idx = 200
    if len(df_m15) <= start_idx:
        log.error("Pas assez de données (%d bougies, min 200 requises)", len(df_m15))
        return [], initial_balance

    total = len(df_m15) - start_idx
    done  = 0

    log.info("Démarrage du backtest : %d bougies à analyser", total)
    log.info("Balance initiale : %.2f $", initial_balance)
    log.info("Modèle IA : %s (via OpenRouter)", OPENROUTER_MODEL)
    print()

    for idx in range(start_idx, len(df_m15)):
        row = df_m15.iloc[idx]

        # BACKTEST-AI : Ignorer si les indicateurs ne sont pas calculés (NaN)
        if pd.isna(row["RSI"]) or pd.isna(row["ATR"]) or pd.isna(row["EMA200"]):
            done += 1
            continue

        # BACKTEST-AI : Filtrage technique pré-IA pour économiser des jetons
        # On n'appelle OpenRouter que si on a une configuration intéressante :
        # - RSI survendu (<35) ou suracheté (>65)
        # - MACD fort (histogramme significatif)
        # - Pente M15 (EMA20) clairement UP ou DOWN (par rapport à l'ATR)
        # - Prix en bordure de Bollinger (haut ou bas)
        
        atr = row["ATR"]
        macdh_strong = abs(row["MACD_hist"]) > (atr * 0.1) if not pd.isna(row["MACD_hist"]) else False
        rsi_extreme = row["RSI"] < 40 or row["RSI"] > 60
        bollinger_edge = (row["Close"] > row["BB_upper"] - atr*0.5) or (row["Close"] < row["BB_lower"] + atr*0.5)
        
        # Obtenir la pente locale pour filtrer (réutilisation code du prompt)
        start_slope = max(0, idx - 5)
        ema20_vals = df_m15["EMA20"].values[start_slope:idx]
        slope_m15 = ema_slope(ema20_vals, atr)
        trend_strong = slope_m15 != "FLAT"
        
        if not (rsi_extreme or macdh_strong or bollinger_edge or trend_strong):
            skipped += 1
            done += 1
            continue

        # BACKTEST-AI : Construction du prompt (identique au live)
        prompt_data = build_backtest_prompt(row, df_m15, idx)
        cache_key   = get_cache_key(prompt_data)

        # BACKTEST-AI : Utiliser le cache si disponible
        if cache_key in cache:
            signal = cache[cache_key]
            cache_hits += 1
        else:
            try:
                signal = await ai_decision(prompt_data)
                cache[cache_key] = signal
                save_cache(cache)
                api_calls += 1
                # BACKTEST-AI : Rate limiting entre les appels API
                await asyncio.sleep(DELAY_BETWEEN_CALLS)
            except Exception as e:
                log.error("Erreur API idx=%d : %s", idx, e)
                skipped += 1
                done += 1
                continue

        # BACKTEST-AI : Filtrage du signal (même logique que le bot live)
        if signal.get("DIR") == "HOLD":
            done += 1
            continue
        if signal.get("CONF", 0) < min_conf:
            done += 1
            continue

        # BACKTEST-AI : Vérification RR minimum
        if signal.get("RR", 0) < config.MIN_RR:
            done += 1
            continue

        # BACKTEST-AI : Simulation du trade
        trade = simulate_trade(signal, df_m15, idx, balance)
        if trade:
            balance += trade["pnl"]
            trades.append(trade)

            # Log du trade
            emoji = "✅" if trade["pnl"] > 0 else "❌" if trade["pnl"] < 0 else "⚪"
            log.info(
                "%s %s | Entry=%.5f | Exit=%.5f | P&L=%+.2f$ (%+.1f pips) | %s | Bal=%.2f$",
                emoji, trade["dir"], trade["entry"], trade["exit"],
                trade["pnl"], trade["pnl_pips"], trade["result"], balance
            )

        # BACKTEST-AI : Progression
        done += 1
        if done % 50 == 0:
            pct = done / total * 100
            print(
                f"  ⏳ Progression: {pct:.1f}% | "
                f"Trades: {len(trades)} | "
                f"Balance: {balance:.2f}$ | "
                f"API: {api_calls} | Cache: {cache_hits} | "
                f"Skipped: {skipped}"
            )

    # BACKTEST-AI : Résumé final
    log.info("Backtest terminé ! %d trades | Balance finale : %.2f$ | API calls : %d | Cache hits : %d",
             len(trades), balance, api_calls, cache_hits)

    return trades, balance


# ══════════════════════════════════════════════════════════════
# 7. POINT D'ENTRÉE — BACKTEST-AI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # BACKTEST-AI : Parsing des arguments CLI
    parser = argparse.ArgumentParser(description="FXBOT EUR/USD — Backtest avec OpenRouter/DeepSeek")
    parser.add_argument("--symbol",  type=str,   default="EURUSDm",  help="Symbole MT5 (défaut: EURUSDm)")
    parser.add_argument("--days",    type=int,   default=90,         help="Nombre de jours (défaut: 90)")
    parser.add_argument("--balance", type=float, default=50.0,       help="Balance initiale (défaut: 50$)")
    parser.add_argument("--conf",    type=int,   default=62,         help="Confiance min (défaut: 62)")
    args = parser.parse_args()

    print()
    print("=" * 64)
    print("🤖 FXBOT EUR/USD — BACKTEST (OpenRouter/DeepSeek)")
    print("=" * 64)
    print(f" Symbole  : {args.symbol:<47}")
    print(f" Période  : {args.days} jours")
    print(f" Balance  : {args.balance:.2f} $")
    print(f" Conf min : {args.conf}%")
    print(f" Modèle   : {OPENROUTER_MODEL:<47}")
    print("=" * 64)
    print()

    # BACKTEST-AI : Initialisation MT5
    if not mt5.initialize():
        print("❌ Impossible d'initialiser MetaTrader 5")
        print("   Vérifiez que MT5 est lancé et connecté.")
        sys.exit(1)

    log.info("MT5 initialisé — compte #%s", mt5.account_info().login if mt5.account_info() else "?")

    try:
        # BACKTEST-AI : Lancement du backtest
        trades, final_bal = asyncio.run(
            run_backtest(
                symbol=args.symbol,
                days=args.days,
                initial_balance=args.balance,
                min_conf=args.conf,
            )
        )

        # BACKTEST-AI : Rapport de performance
        print_report(trades, final_bal, args.balance)

    except KeyboardInterrupt:
        print("\n\n  ⚠️ Backtest interrompu par l'utilisateur.\n")

    finally:
        mt5.shutdown()
        log.info("MT5 déconnecté.")
