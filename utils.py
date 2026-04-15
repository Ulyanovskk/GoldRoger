"""
utils.py — Toutes les fonctions utilitaires du bot FXBOT EUR/USD :
    - Connexion / données MetaTrader 5
    - Calcul des indicateurs techniques
    - Compression des données pour DeepSeek
    - Filtre pré-IA
    - Appel DeepSeek + parsing
    - Garde-fous risque
    - Exécution des ordres MT5
    - Alertes et commandes Telegram
    - Logging

# MIGRATION-EURUSD : Migration complète XAU/USD → EUR/USD (2026-04-15)
"""

import os
import re
import json
import logging
import requests
import httpx
import asyncio
import datetime
import numpy as np
import pandas as pd
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

import ta
import MetaTrader5 as mt5

import config

# ══════════════════════════════════════════════════════════════
# 1. LOGGING
# ══════════════════════════════════════════════════════════════

def _make_handler(path: str) -> TimedRotatingFileHandler:
    """Crée un handler de log avec rotation journalière."""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    handler = TimedRotatingFileHandler(
        path, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    return handler


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Configure et retourne un logger avec rotation journalière + console."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Éviter les doublons de handlers
    logger.setLevel(level)
    logger.addHandler(_make_handler(log_file))
    # Console
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console)
    return logger


# État global pour l'arrêt propre
SHUTDOWN_MODE: bool = False
bot_log = setup_logger("bot", config.LOG_BOT)
trades_log = setup_logger("trades", config.LOG_TRADES)

# AUDIT-FIX #2 — Race Condition MT5
# Lock global pour sérialiser TOUS les appels MT5 (order_send, positions_get,
# account_info, copy_rates, symbol_info_tick) depuis plusieurs coroutines.
mt5_lock: asyncio.Lock = asyncio.Lock()


# ══════════════════════════════════════════════════════════════
# 2. CONNEXION METATRADER 5
# ══════════════════════════════════════════════════════════════

def mt5_connect() -> bool:
    """
    Initialise et connecte MetaTrader 5.
    Retourne True si la connexion est établie, False sinon.
    """
    try:
        if mt5.initialize(
            login=config.MT5_LOGIN,
            password=config.MT5_PASSWORD,
            server=config.MT5_SERVER,
        ):
            bot_log.info(
                "MT5 connecté — compte #%s | solde %.2f USD",
                config.MT5_LOGIN,
                mt5.account_info().balance,
            )
            return True
        bot_log.error("Échec connexion MT5 : %s", mt5.last_error())
        return False
    except Exception as exc:
        bot_log.error("Exception mt5_connect : %s", exc)
        return False


def mt5_disconnect() -> None:
    """Ferme proprement la connexion MT5."""
    try:
        mt5.shutdown()
        bot_log.info("MT5 déconnecté.")
    except Exception as exc:
        bot_log.error("Exception mt5_disconnect : %s", exc)


def mt5_ensure_connected() -> bool:
    """
    Vérifie l'état de la connexion MT5 et tente une reconnexion si nécessaire.
    Retourne True si connecté après vérification.
    Note : appelé hors contexte async — utilise mt5 directement (pas de lock nécessaire ici).
    """
    try:
        info = mt5.account_info()
        if info is not None:
            return True
        bot_log.warning("MT5 déconnecté, tentative de reconnexion…")
        mt5.shutdown()
        return mt5_connect()
    except Exception as exc:
        bot_log.error("Exception mt5_ensure_connected : %s", exc)
        return False


# ══════════════════════════════════════════════════════════════
# 3. COLLECTE DES DONNÉES OHLCV + INDICATEURS
# ══════════════════════════════════════════════════════════════

# Correspondance nom → constante MT5
_TF_MAP = {
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

# M6 — Cache des indicateurs {tf: {data, last_bar_time}}
_indicator_cache: dict = {}


async def get_ohlcv_async(timeframe_name: str, n: int = config.CANDLES_COUNT) -> Optional[pd.DataFrame]:
    """
    AUDIT-FIX #2 — Récupère n bougies OHLCV via mt5_lock (copy_rates protégé).
    Retourne un DataFrame pandas ou None en cas d'erreur.
    """
    try:
        tf_const = _TF_MAP.get(timeframe_name)
        if tf_const is None:
            bot_log.error("Timeframe inconnu : %s", timeframe_name)
            return None

        async with mt5_lock:  # AUDIT-FIX #2 — copy_rates sérialisé
            rates = mt5.copy_rates_from_pos(config.MT5_SYMBOL, tf_const, 0, n)

        if rates is None or len(rates) == 0:
            bot_log.error("Aucune donnée MT5 pour %s/%s", config.MT5_SYMBOL, timeframe_name)
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low", "close": "Close",
                            "tick_volume": "Volume"}, inplace=True)
        return df
    except Exception as exc:
        bot_log.error("Exception get_ohlcv_async(%s) : %s", timeframe_name, exc)
        return None


def get_ohlcv(timeframe_name: str, n: int = config.CANDLES_COUNT) -> Optional[pd.DataFrame]:
    """
    Version synchrone de get_ohlcv (utilisée dans collect_all_data via asyncio.run).
    Appelle mt5.copy_rates_from_pos directement — à n'utiliser que hors boucle async.
    """
    try:
        tf_const = _TF_MAP.get(timeframe_name)
        if tf_const is None:
            bot_log.error("Timeframe inconnu : %s", timeframe_name)
            return None

        rates = mt5.copy_rates_from_pos(config.MT5_SYMBOL, tf_const, 0, n)
        if rates is None or len(rates) == 0:
            bot_log.error("Aucune donnée MT5 pour %s/%s", config.MT5_SYMBOL, timeframe_name)
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low", "close": "Close",
                            "tick_volume": "Volume"}, inplace=True)
        return df
    except Exception as exc:
        bot_log.error("Exception get_ohlcv(%s) : %s", timeframe_name, exc)
        return None


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Calcule RSI, MACD, EMA 20/50/200, Bollinger Bands, ATR sur le DataFrame.
    Retourne un dictionnaire des valeurs de la dernière bougie.
    """
    try:
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=config.RSI_PERIOD).rsi()

        # MACD
        macd_obj = ta.trend.MACD(
            close,
            window_fast=config.MACD_FAST,
            window_slow=config.MACD_SLOW,
            window_sign=config.MACD_SIGNAL,
        )
        macd_line   = macd_obj.macd()
        macd_signal = macd_obj.macd_signal()
        macd_hist   = macd_obj.macd_diff()

        # EMAs
        ema20  = ta.trend.EMAIndicator(close, window=config.EMA_FAST).ema_indicator()
        ema50  = ta.trend.EMAIndicator(close, window=config.EMA_MID).ema_indicator()
        ema200 = ta.trend.EMAIndicator(close, window=config.EMA_SLOW).ema_indicator()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close, window=config.BB_PERIOD, window_dev=config.BB_STD
        )
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()

        # ATR
        atr = ta.volatility.AverageTrueRange(
            high, low, close, window=config.ATR_PERIOD
        ).average_true_range()

        # Extraction de la dernière valeur
        last_close  = close.iloc[-1]
        last_ema20  = ema20.iloc[-1]
        last_ema50  = ema50.iloc[-1]
        last_ema200 = ema200.iloc[-1]
        last_bb_up  = bb_upper.iloc[-1]
        last_bb_lo  = bb_lower.iloc[-1]

        # Position du prix dans les bandes de Bollinger
        bb_range = last_bb_up - last_bb_lo
        if bb_range > 0:
            bb_pct = (last_close - last_bb_lo) / bb_range  # 0=bas, 1=haut
        else:
            bb_pct = 0.5

        if bb_pct > 0.8:
            bb_pos = "high"
        elif bb_pct < 0.2:
            bb_pos = "low"
        else:
            bb_pos = "mid"

        # Tendance EMA (Simplifiée pour maximiser les tests)
        if last_close > last_ema200:
            ema_trend = "bull"
        elif last_close < last_ema200:
            ema_trend = "bear"
        else:
            ema_trend = "mix"

        return {
            "rsi":       round(rsi.iloc[-1], 1),
            "macd":      "+" if macd_hist.iloc[-1] > 0 else "-",
            "macd_val":  round(macd_hist.iloc[-1], 5),
            "ema_trend": ema_trend,
            "ema20":     round(last_ema20, 5),   # FIX-PRECISION : 5 décimales pour FX
            "ema50":     round(last_ema50, 5),
            "ema200":    round(last_ema200, 5),
            "bb_pos":    bb_pos,
            "atr":       round(atr.iloc[-1], 5), # FIX-PRECISION : Crucial pour EUR/USD
            "close":     round(last_close, 5),
        }
    except Exception as exc:
        bot_log.error("Exception compute_indicators : %s", exc)
        return {}


def find_support_resistance(df: pd.DataFrame, n_levels: int = 2) -> tuple[list, list]:
    """
    Identifie les niveaux de support et résistance clés sur H4/D1
    à partir des pivots locaux (hauts/bas significatifs).
    Retourne (supports, resistances) triés.
    """
    try:
        highs  = df["High"].values
        lows   = df["Low"].values
        window = 5  # Fenêtre pivot

        resistances = []
        supports    = []

        for i in range(window, len(highs) - window):
            # Résistance : sommet local
            if highs[i] == max(highs[i - window: i + window + 1]):
                resistances.append(round(highs[i], 2))
            # Support : creux local
            if lows[i] == min(lows[i - window: i + window + 1]):
                supports.append(round(lows[i], 2))

        # Regrouper les niveaux proches (tolérance 0.5%)
        def cluster(levels: list, tol: float = 0.005) -> list:
            if not levels:
                return []
            levels = sorted(set(levels))
            clustered = [levels[0]]
            for lvl in levels[1:]:
                if (lvl - clustered[-1]) / clustered[-1] > tol:
                    clustered.append(lvl)
            return clustered

        current_price = df["Close"].iloc[-1]

        # Conserver les résistances au-dessus du prix et les supports en dessous
        r_levels = [float(r) for r in cluster(resistances) if r > current_price]
        s_levels = [float(s) for s in cluster(supports)    if s < current_price]

        return (
            s_levels[-n_levels:] if s_levels else [],
            r_levels[:n_levels]  if r_levels else [],
        )
    except Exception as exc:
        bot_log.error("Exception find_support_resistance : %s", exc)
        return [], []


def get_cached_indicators(tf: str, df: pd.DataFrame) -> dict:
    """
    Retourne les indicateurs depuis le cache si la bougie n'a pas changé (M6).
    Invalide le cache uniquement à la clôture d'une nouvelle bougie.
    """
    last_bar_time = df["time"].iloc[-1]
    cached = _indicator_cache.get(tf)
    if cached and cached["last_bar_time"] == last_bar_time:
        bot_log.debug("Cache hit indicateurs [%s] — bougie %s", tf, last_bar_time)
        return cached["data"]
    result = compute_indicators(df)
    _indicator_cache[tf] = {"data": result, "last_bar_time": last_bar_time}
    bot_log.debug("Cache miss indicateurs [%s] — recalcul", tf)
    return result


async def collect_all_data() -> Optional[dict]:  # AUDIT-FIX #2 — async + mt5_lock
    """
    Collecte et calcule les indicateurs sur les 4 timeframes.
    Tous les appels MT5 sont sérialisés via mt5_lock.
    Retourne un dictionnaire structuré ou None en cas d'erreur critique.
    """
    try:
        result = {}
        for tf in ["M15", "H1", "H4", "D1"]:
            df = await get_ohlcv_async(tf)  # AUDIT-FIX #2 — copy_rates protégé
            if df is None or df.empty:
                bot_log.error("Données manquantes pour %s", tf)
                return None
            indicators = get_cached_indicators(tf, df)  # M6
            result[tf] = {"df": df, "ind": indicators}

        # Support/Résistance sur H4 et D1
        s_h4, r_h4 = find_support_resistance(result["H4"]["df"])
        s_d1, r_d1 = find_support_resistance(result["D1"]["df"])
        result["SR"] = {"H4": {"s": s_h4, "r": r_h4},
                        "D1": {"s": s_d1, "r": r_d1}}

        async with mt5_lock:  # AUDIT-FIX #2 — symbol_info_tick + account_info sérialisés
            tick    = mt5.symbol_info_tick(config.MT5_SYMBOL)
            account = mt5.account_info()

        result["current_price"] = tick.ask if tick else result["M15"]["ind"]["close"]
        result["balance"] = round(account.balance, 2) if account else 0.0
        result["equity"]  = round(account.equity,  2) if account else 0

        return result
    except Exception as exc:
        bot_log.error("Exception collect_all_data : %s", exc)
        return None


# ══════════════════════════════════════════════════════════════
# 4. COMPRESSION DES DONNÉES (format ultra-compact)
# ══════════════════════════════════════════════════════════════

def get_current_session() -> str:
    """Retourne la session active (London/NewYork/Off) selon l'heure UTC."""
    now_h = datetime.datetime.utcnow().hour
    sessions = []
    for name, hours in config.ACTIVE_SESSIONS.items():
        if hours["start"] <= now_h < hours["end"]:
            short = "London" if "London" in name else "NY"
            sessions.append(short)
    return "+".join(sessions) if sessions else "Off"


def fetch_dxy_data() -> str:
    """Récupère la tendance 24h de l'Indice Dollar (DXY)."""
    try:
        rates = mt5.copy_rates_from_pos(config.DXY_SYMBOL, mt5.TIMEFRAME_D1, 0, 1)
        if rates is not None and len(rates) > 0:
            change = (rates[0]['close'] - rates[0]['open']) / rates[0]['open'] * 100
            return f"DXY:{'+' if change > 0 else ''}{change:.2f}%"
        return "DXY:?"
    except:
        return "DXY:?"

def is_high_impact_news() -> tuple[bool, str]:
    """Vérifie si une news économique majeure approche ou vient de passer."""
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        start = now - datetime.timedelta(minutes=config.NEWS_CHECK_WINDOW_MINS)
        end = now + datetime.timedelta(minutes=config.NEWS_CHECK_WINDOW_MINS)
        
        # Vérification si la fonction existe dans cette version de la lib MT5
        if not hasattr(mt5, 'calendar_get'):
            return False, ""

        # MIGRATION-EURUSD : devises EUR/USD au lieu de USD/XAU
        events = mt5.calendar_get(time_from=int(start.timestamp()), time_to=int(end.timestamp()))
        if events:
            for ev in events:
                if ev.importance >= config.BLOCK_NEWS_IMPORTANCE and ev.currency in ("USD", "EUR"):
                    return True, ev.name
        return False, ""
    except Exception as e:
        bot_log.error("Erreur calendrier MT5 : %s", e)
        return False, ""


def fetch_market_news() -> str:
    """Récupère les prochains événements économiques majeurs via MT5 (USD/EUR). # MIGRATION-EURUSD"""
    try:
        if not hasattr(mt5, 'calendar_get'):
            return "news:N/A"

        now = datetime.datetime.now(datetime.timezone.utc)
        # On regarde les prochaines 24 heures
        end = now + datetime.timedelta(hours=24)
        
        events = mt5.calendar_get(time_from=int(now.timestamp()), time_to=int(end.timestamp()))
        if events:
            # MIGRATION-EURUSD : filtrer USD et EUR (EUR/USD impacté par les deux devises)
            important = [ev for ev in events if ev.importance >= 2 and ev.currency in ("USD", "EUR")]
            if important:
                lines = []
                for ev in important[:3]:
                    ev_time = datetime.datetime.fromtimestamp(ev.time, datetime.timezone.utc).strftime("%H:%M")
                    # Un format très court pour Telegram
                    lines.append(f"• {ev_time} {ev.name}")
                return "\n".join(lines)
        
        return "Aucune annonce majeure prévue (24h)."
    except Exception as e:
        bot_log.error("Erreur fetch_market_news : %s", e)
        return "news:?"


# PRICE-STRUCTURE — Fonctions d'analyse de la structure de prix
# Appelées depuis compress_data() pour enrichir le contexte envoyé à DeepSeek.

def detect_price_structure(df: pd.DataFrame) -> str:
    """
    # PRICE-STRUCTURE — Détecte la structure de prix sur les 15 dernières bougies.
    Utilise un vote majoritaire (≥65% des bougies) pour capter les canaux.
    """
    try:
        highs = df["High"].values[-15:]
        lows  = df["Low"].values[-15:]
        n     = len(highs) - 1
        if n <= 0:
            return "RANGE"

        lh_count = sum(1 for i in range(n) if highs[i] > highs[i + 1])
        ll_count = sum(1 for i in range(n) if lows[i]  > lows[i + 1])
        hh_count = sum(1 for i in range(n) if highs[i] < highs[i + 1])
        hl_count = sum(1 for i in range(n) if lows[i]  < lows[i + 1])

        threshold = 0.65  # Sensibilité augmentée à 65%

        if (lh_count / n) >= threshold and (ll_count / n) >= threshold:
            return "DOWNTREND"
        elif (hh_count / n) >= threshold and (hl_count / n) >= threshold:
            return "UPTREND"
        else:
            return "RANGE"
    except Exception as exc:
        bot_log.debug("detect_price_structure : %s", exc)
        return "RANGE"


def ema_slope(ema_series: pd.Series, atr_value: float) -> str:
    """
    # DYNAMIC-ADAPTATION — Calcule la pente de l'EMA20 relative à la volatilité.
    Au lieu d'un seuil fixe, on utilise 10% de l'ATR comme seuil de pente.
    Si le delta > 0.1 * ATR, la tendance est validée.
    """
    try:
        values = ema_series.dropna().values
        if len(values) < 5 or atr_value <= 0:
            return "FLAT"
        
        delta = values[-1] - values[-5]
        # Le seuil s'adapte dynamiquement à la volatilité actuelle
        threshold = atr_value * 0.1 
        
        if delta < -threshold:
            return "DOWN"
        elif delta > threshold:
            return "UP"
        else:
            return "FLAT"
    except Exception as exc:
        bot_log.debug("ema_slope : %s", exc)
        return "FLAT"


def channel_position(df: pd.DataFrame, current_price: float) -> str:
    """
    # PRICE-STRUCTURE — Position du prix dans le range des 20 dernières bougies.
    TOP  > 75% du range → proximité résistance
    BOTTOM < 25% du range → proximité support
    MID  : milieu du channel
    """
    try:
        highs    = df["High"].values[-20:]
        lows     = df["Low"].values[-20:]
        ch_high  = float(np.max(highs))
        ch_low   = float(np.min(lows))
        ch_range = ch_high - ch_low
        if ch_range == 0:
            return "UNKNOWN"
        position = (current_price - ch_low) / ch_range
        if position > 0.75:
            return "TOP"     # près de la résistance
        elif position < 0.25:
            return "BOTTOM"  # près du support
        else:
            return "MID"     # milieu du channel
    except Exception as exc:
        bot_log.debug("channel_position : %s", exc)
        return "UNKNOWN"


def compress_data(data: dict, context: dict = None) -> str:
    """
    AUDIT-FIX #1 — Compresse les données techniques avec les clés alignées sur le prompt système.
    Mapping : RSI→R, MACD→M, Bollinger→B, EMA_trend→E, ATR→A
    # PRICE-STRUCTURE — Enrichi avec : STRUCT, SLOPE_M15, SLOPE_H1, CH_POS
    (correspond exactement aux 'Data keys' du DEEPSEEK_SYSTEM_PROMPT dans config.py)
    """
    try:
        parts = []

        # M5 — Head context + Niveau 2 : biais adaptatif par direction
        if context:
            trades = context.get('last_trades', [])
            l3 = "".join(["W" if t['pnl'] > 0 else "L" for t in trades])
            wb = context.get('week_bias', 'NEUT')[:4]
            sr = context.get('persistent_sr', [])

            # Calcul du win rate par direction sur les trades récents
            buy_trades  = [t for t in trades if str(t.get('dir', '')).upper() == 'BUY']
            sell_trades = [t for t in trades if str(t.get('dir', '')).upper() == 'SELL']
            buy_wr  = round(sum(1 for t in buy_trades  if t.get('pnl', 0) > 0) / len(buy_trades)  * 100) if buy_trades  else 50
            sell_wr = round(sum(1 for t in sell_trades if t.get('pnl', 0) > 0) / len(sell_trades) * 100) if sell_trades else 50

            parts.append(f"CTX:L3={l3 or 'None'},WB={wb},SR={sr},BUY_wr={buy_wr}%,SELL_wr={sell_wr}%")

        parts.append("EURUSD")  # MIGRATION-EURUSD : XAU → EURUSD

        # AUDIT-FIX #1 — Clés alignées avec le dict du prompt système :
        # R=RSI, M=MACD, B=Bollinger, E=EMA_trend, A=ATR
        def tf_str(tf: str) -> str:
            ind = data[tf]["ind"]
            # FIX-PRECISION — On affiche l'ATR avec ses décimales pour l'IA
            return (
                f"{tf}:"
                f"R={ind['rsi']},"
                f"M={ind['macd']},"
                f"B={ind['bb_pos']},"
                f"E={ind['ema_trend']},"
                f"A={ind['atr']:.5f}" 
            )

        parts.append(tf_str("M15"))
        parts.append(tf_str("H1"))

        # H4 : ajout S/R — clés courtes également
        ind_h4 = data["H4"]["ind"]
        sr_h4  = data["SR"]["H4"]
        s_str  = str(sr_h4["s"][-1]) if sr_h4["s"] else "?"
        r_str  = str(sr_h4["r"][0])  if sr_h4["r"] else "?"
        parts.append(
            f"H4:"
            f"R={ind_h4['rsi']},"       # AUDIT-FIX #1
            f"M={ind_h4['macd']},"      # AUDIT-FIX #1
            f"E={ind_h4['ema_trend']}," # AUDIT-FIX #1
            f"SR={r_str}/{s_str}"
        )

        # D1 : biais directionnel
        ind_d1 = data["D1"]["ind"]
        bias = (
            "bull" if ind_d1["ema_trend"] == "bull"
            else "bear" if ind_d1["ema_trend"] == "bear"
            else "neut"
        )
        parts.append(f"D1:bias={bias},R={ind_d1['rsi']}")  # AUDIT-FIX #1

        parts.append(f"bal={int(data['balance'])}USD")
        parts.append(f"sess={get_current_session()}")

        # MIGRATION-EURUSD : DXY retiré comme indicateur principal — USD Index générale uniquement
        # parts.append(fetch_dxy_data())  # Supprimé pour EUR/USD
        parts.append(fetch_market_news())  # News EUR et USD

        parts.append(f"price={data['current_price']}")

        # ───────────────────────────────────────
        # PRICE-STRUCTURE — Analyse structurelle M15
        # ───────────────────────────────────────
        try:
            df_m15  = data["M15"]["df"]
            df_h1   = data["H1"]["df"]
            price   = data["current_price"]

            # DYNAMIC-ADAPTATION — Utilisation de l'ATR pour la pente
            atr_m15   = data["M15"]["ind"]["atr"]
            atr_h1    = data["H1"]["ind"]["atr"]
            
            # PRICE-STRUCTURE — 1. Structure de prix (Lower Highs/Lows ou Higher)
            struct = detect_price_structure(df_m15)

            # PRICE-STRUCTURE — 2. Pente EMA20 (Seuils dynamiques basés sur ATR)
            ema20_m15 = ta.trend.EMAIndicator(df_m15["Close"], window=config.EMA_FAST).ema_indicator()
            ema20_h1  = ta.trend.EMAIndicator(df_h1["Close"],  window=config.EMA_FAST).ema_indicator()
            slope_m15 = ema_slope(ema20_m15, atr_m15)
            slope_h1  = ema_slope(ema20_h1,  atr_h1)

            # PRICE-STRUCTURE — 3. Position dans le channel 20 bougies
            ch_pos = channel_position(df_m15, price)

            parts.append(
                f"STRUCT={struct}|SLOPE_M15={slope_m15}|"
                f"SLOPE_H1={slope_h1}|CH_POS={ch_pos}"
            )
            bot_log.debug(
                "PRICE-STRUCTURE | struct=%s | slope_m15=%s | slope_h1=%s | ch_pos=%s",
                struct, slope_m15, slope_h1, ch_pos
            )
        except Exception as ps_exc:
            bot_log.warning("PRICE-STRUCTURE calcul ignoré : %s", ps_exc)
        # ───────────────────────────────────────

        return "|".join(parts)
    except Exception as exc:
        bot_log.error("Exception compress_data : %s", exc)
        return ""


# ══════════════════════════════════════════════════════════════
# 5. FILTRE PRÉ-IA
# ══════════════════════════════════════════════════════════════

def get_open_trades_count() -> int:
    """Retourne le nombre de positions ouvertes par ce bot (magic number)."""
    try:
        positions = mt5.positions_get(symbol=config.MT5_SYMBOL)  # sync — appelé hors await
        if positions is None:
            return 0
        return sum(1 for p in positions if p.magic == config.MT5_MAGIC)
    except Exception as exc:
        bot_log.error("Exception get_open_trades_count : %s", exc)
        return 0


def get_daily_drawdown_pct(start_balance: float = 0.0) -> float:
    """
    AUDIT-FIX #3 — Calcule le drawdown journalier basé sur start_balance.
    Si start_balance fourni : drawdown_pct = (start_balance - balance_actuel) / start_balance * 100
    Sinon fallback sur equity vs balance (mode dégradé).
    Guard : si start_balance == 0.0 → log erreur + retourne 0.0 (jamais de faux déclenchement).
    """
    try:
        account = mt5.account_info()  # AUDIT-FIX #2 — sync, appelé hors boucle critique
        if account is None:
            return 0.0

        if start_balance > 0:
            # AUDIT-FIX #3 — Base de calcul fixée au début de journée
            dd = (start_balance - account.balance) / start_balance * 100
            bot_log.debug(
                "DD Check | start=%.2f | current=%.2f | dd=%.2f%% | max=%.1f%%",
                start_balance, account.balance, dd, config.MAX_DAILY_DRAWDOWN
            )
        elif start_balance == 0.0:
            # GUARD — start_balance pas encore initialisé : on ne déclenche PAS l'arrêt
            bot_log.error(
                "DD Check IGNORÉ — start_balance=0.0 (non initialisé). "
                "Drawdown non calculable, retour 0.0 pour éviter un faux arrêt."
            )
            return 0.0
        else:
            if account.balance <= 0:
                return 0.0
            dd = (account.balance - account.equity) / account.balance * 100
            bot_log.debug(
                "DD Check (fallback equity) | balance=%.2f | equity=%.2f | dd=%.2f%% | max=%.1f%%",
                account.balance, account.equity, dd, config.MAX_DAILY_DRAWDOWN
            )

        return round(dd, 2) if dd > 0 else 0.0
    except Exception as exc:
        bot_log.error("Exception get_daily_drawdown_pct : %s", exc)
        return 0.0


def is_active_session() -> bool:
    """Retourne True si on est dans une session London ou New York (UTC)."""
    now_h = datetime.datetime.utcnow().hour
    return any(
        h["start"] <= now_h < h["end"]
        for h in config.ACTIVE_SESSIONS.values()
    )


def are_timeframes_aligned(data: dict) -> bool:
    """
    Vérifie si H1 affiche une tendance directionnelle claire (bull ou bear).
    H4 est lu et loggé à titre informatif mais n'est plus une condition bloquante.
    Le M15 peut être en correction sans bloquer le bot.
    """
    try:
        h1_trend  = data["H1"]["ind"]["ema_trend"]
        h4_trend  = data["H4"]["ind"]["ema_trend"]   # Informatif uniquement
        m15_trend = data["M15"]["ind"]["ema_trend"]

        # Seul H1 est bloquant : il doit être bull ou bear (pas mix)
        if h1_trend not in ("bull", "bear"):
            bot_log.debug(
                "Alignement refusé : H1=%s (indirectionnel) | H4=%s (info) | M15=%s",
                h1_trend, h4_trend, m15_trend
            )
            return False

        # H1 est directionnel : on passe, on logue H4 pour information
        bot_log.debug(
            "Alignement validé : H1=%s (actif) | H4=%s (info, non-bloquant) | M15=%s",
            h1_trend, h4_trend, m15_trend
        )
        return True
    except Exception as exc:
        bot_log.error("Exception are_timeframes_aligned : %s", exc)
        return False


def is_news_window() -> tuple[bool, str]:
    """
    GARDE-FOU NEWS : Vérifie si on est dans une fenêtre de publication majeure (±20 min).
    """
    try:
        news_file = "news.json"
        if not os.path.exists(news_file):
            return False, ""
        
        with open(news_file, "r") as f:
            news_data = json.load(f)
        
        now = datetime.datetime.now()
        for event in news_data.get("major_events", []):
            try:
                event_time = datetime.datetime.strptime(event["time"], "%Y-%m-%d %H:%M")
                diff = abs((now - event_time).total_seconds()) / 60
                if diff <= config.NEWS_BLOCK_WINDOW:
                    return True, event["name"]
            except:
                continue
        return False, ""
    except Exception as e:
        bot_log.error("Erreur filtre news : %s", e)
        return False, ""


def get_trend_direction(data: dict) -> str:
    """
    GARDE-FOU TREND : Vérifie l'alignement EMA20 > EMA50 sur H1 ET H4.
    """
    try:
        # H1
        h1_ema20 = data["H1"]["ind"]["ema20"]
        h1_ema50 = data["H1"]["ind"]["ema50"]
        # H4
        h4_ema20 = data["H4"]["ind"]["ema20"]
        h4_ema50 = data["H4"]["ind"]["ema50"]
        
        if h1_ema20 > h1_ema50 and h4_ema20 > h4_ema50:
            return "BUY"
        elif h1_ema20 < h1_ema50 and h4_ema20 < h4_ema50:
            return "SELL"
        else:
            return "RANGE"
    except:
        return "UNKNOWN"


def pre_ia_filter(data: dict, mode: str = "normal", start_balance: float = 0.0) -> tuple[bool, str]:
    """
    AUDIT-FIX #6 — Filtre pré-IA avec comportement adapté au mode (aggro/safe/normal).
    AUDIT-FIX #3 — Utilise start_balance pour le calcul du drawdown journalier.

    # HOTFIX-2 — Seuils spread en PIPS (EUR/USD). Formule : (ask-bid)/(point*10)
    Règles par mode :
      aggro : NEWS_BLOCK actif | SPREAD_MAX=25 pips | filtre trend désactivé | MIN_CONFIDENCE=60
      safe  : SPREAD_MAX=10 pips | MIN_CONFIDENCE=68 | filtre trend H1+H4 actif
      normal: SPREAD_MAX=15 pips (config.MAX_SPREAD_POINTS)
    """
    now_str = datetime.datetime.now().strftime("%H:%M")

    # AUDIT-FIX #6 — Résolution des paramètres selon le mode
    mode = (mode or "normal").lower()
    if mode == "aggro":
        spread_max   = 25    # HOTFIX-2 : 25 pips en mode aggro
        min_conf     = 60
        trend_filter = False
        news_block   = True
    elif mode == "safe":
        spread_max   = 10    # HOTFIX-2 : 10 pips en mode safe
        min_conf     = 68
        trend_filter = True
        news_block   = True
    else:  # normal
        spread_max   = config.MAX_SPREAD_POINTS  # HOTFIX-2 : 15 pips (config.py)
        min_conf     = config.MIN_CONFIDENCE
        trend_filter = False
        news_block   = True

    bot_log.info("Filtre pré-IA | mode=%s | spread_max=%d | min_conf=%d | trend=%s",
                 mode, spread_max, min_conf, "ON" if trend_filter else "OFF")

    # 1. Session active
    if not is_active_session():
        return False, f"SKIP: Session OFF | Heure={now_str}"

    # 2. Sécurités vitales (Trades / Drawdown)
    open_trades = get_open_trades_count()
    if open_trades >= config.MAX_SIMULTANEOUS_TRADES:
        return False, f"SKIP: Max trades ({open_trades}) | Heure={now_str}"

    # AUDIT-FIX #3 — Guard start_balance == 0.0 : on ne bloque jamais si non initialisé
    if start_balance == 0.0:
        bot_log.error(
            "pre_ia_filter — start_balance=0.0 (non initialisé) : "
            "vérification drawdown ignorée pour éviter un faux SKIP."
        )
    else:
        dd = get_daily_drawdown_pct(start_balance=start_balance)
        # Log INFO uniquement lors du filtre pré-IA (toutes les 15 min)
        bot_log.info("Drawdown Check | dd=%.2f%% | max=%.1f%%", dd, config.MAX_DAILY_DRAWDOWN)
        if dd >= config.MAX_DAILY_DRAWDOWN:
            return False, f"SKIP: Max DD ({dd:.2f}% >= {config.MAX_DAILY_DRAWDOWN}%) | Heure={now_str}"

    # 3. GARDE-FOU SPREAD — calculé en PIPS  # HOTFIX-2
    tick = mt5.symbol_info_tick(config.MT5_SYMBOL)
    if tick:
        symbol_info_data = mt5.symbol_info(config.MT5_SYMBOL)
        # HOTFIX-2 — EUR/USD Exness 5 décimales : point=0.00001, pip=0.0001=10 points
        # (ask-bid)/point donne des points ; on divise par 10 pour obtenir des PIPS
        pip_size = symbol_info_data.point * 10
        spread = round((tick.ask - tick.bid) / pip_size, 1)  # en pips
        if spread > spread_max:
            bot_log.info("SKIP: spread élevé | Spread=%.1f pips > %d pips | mode=%s | Heure=%s",
                         spread, spread_max, mode, now_str)
            return False, f"SKIP: spread élevé ({spread} pips > {spread_max} pips) [mode={mode}]"
    else:
        spread = 0

    # 4. GARDE-FOU NEWS — toujours actif (y compris mode aggro)  # AUDIT-FIX #6
    if news_block:
        is_news, event_name = is_news_window()
        if is_news:
            bot_log.info("SKIP: fenêtre news [%s] | mode=%s | Heure=%s",
                         event_name, mode, now_str)
            return False, f"SKIP: fenêtre news [{event_name}] [mode={mode}]"

    # 5. GARDE-FOU TREND — actif en mode safe, informatif en normal, désactivé en aggro  # AUDIT-FIX #6
    trend = get_trend_direction(data)
    bot_log.info("Trend: %s | Spread=%.1f pips | mode=%s | Heure=%s", trend, spread, mode, now_str)  # HOTFIX-2

    if trend_filter:  # AUDIT-FIX #6 — bloquant uniquement si mode=safe
        if trend == "RANGE":
            return False, f"SKIP: Trend RANGE (filtre actif, mode={mode}) | Heure={now_str}"

    return True, ""


# ══════════════════════════════════════════════════════════════
# 5b. ALERTES PROACTIVES (M11)
# ══════════════════════════════════════════════════════════════

async def check_proactive_alerts(state_obj) -> None:
    """
    Vérifie et déclenche les alertes Telegram proactives (M11).
    Appelé depuis monitoring_loop toutes les 10s.
    """
    # 1. Win rate < 40% sur les 10 derniers trades
    trades = list(state_obj.last_trades)
    if len(trades) >= 5:
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        win_rate = wins / len(trades) * 100
        if win_rate < 40 and state_obj.is_active():
            await state_obj.set_status("pausé")
            send_telegram(
                f"⚠️ <b>Win rate faible : {win_rate:.0f}%</b>\n"
                f"Bot mis en pause préventive sur les {len(trades)} derniers trades."
            )
            bot_log.warning("Pause préventive : win rate %.0f%%", win_rate)

    # 2. Latence IA > 10s (3s était trop sensible pour DeepSeek)
    if state_obj.last_ia_latency_s > 10.0:
        send_telegram(f"⏱ <b>IA lente : {state_obj.last_ia_latency_s}s</b>")
        bot_log.warning("Latence IA élevée : %.2fs", state_obj.last_ia_latency_s)
        state_obj.last_ia_latency_s = 0.0  # Reset pour éviter spam

    # 3. Drawdown proactif — seuils depuis config.MAX_DAILY_DRAWDOWN (jamais hardcodés)
    # GUARD : si start_balance non initialisé, on ne déclenche PAS l'arrêt automatique
    start_bal = getattr(state_obj, 'start_balance', 0.0)
    if start_bal == 0.0:
        bot_log.error(
            "check_proactive_alerts — start_balance=0.0 : "
            "vérification drawdown proactif ignorée pour éviter un faux arrêt."
        )
    else:
        dd = get_daily_drawdown_pct(start_balance=start_bal)  # log DD Check inclus
        dd_stop   = config.MAX_DAILY_DRAWDOWN        # ex: 10.0% — lu depuis .env
        dd_reduce = config.MAX_DAILY_DRAWDOWN * 0.5  # ex: 5.0% — seuil réduction risque

        if dd >= dd_stop and state_obj.is_active():
            await state_obj.set_status("stoppé")
            close_all_positions()
            send_telegram(
                f"🛑 <b>Drawdown {dd:.2f}% ≥ seuil {dd_stop:.0f}%.</b> Arrêt du bot."
            )
            bot_log.error(
                "Arrêt automatique : drawdown %.2f%% >= seuil %.1f%%", dd, dd_stop
            )
        elif dd >= dd_reduce:
            if state_obj.max_risk_pct > config.MAX_RISK_PCT * 0.5:
                state_obj.max_risk_pct = round(config.MAX_RISK_PCT * 0.5, 2)
                send_telegram(
                    f"📉 <b>Drawdown {dd:.2f}% ≥ {dd_reduce:.0f}%.</b>\n"
                    f"Risque réduit à {state_obj.max_risk_pct}% (50% du max)."
                )





# ══════════════════════════════════════════════════════════════

class AISignal(BaseModel):
    DIR: Literal["BUY", "SELL", "HOLD"]
    LOT: float = Field(ge=0.0, le=10.0)  # Accept 0.0, recalibré dans validate_signal
    TP: float
    SL: float
    CONF: int = Field(ge=0, le=100)
    RR: float
    REASON: str  # Tronqué à 250 chars par le validator ci-dessous
    # MIGRATION-EURUSD : Limite portée à 250 pour plus de clarté

    @field_validator("REASON", mode="before")
    @classmethod
    def truncate_reason(cls, v: str) -> str:
        """Tronque silencieusement REASON à 250 chars au lieu de rejeter le signal."""
        if isinstance(v, str) and len(v) > 250:
            bot_log.warning(
                "REASON tronqué (%d→250 chars) : %s…", len(v), v[:60]
            )
            return v[:250]
        return v

async def call_deepseek(compressed_data: str, state_obj=None) -> Optional[dict]:
    """
    Envoie les données à DeepSeek avec retry exponentiel et circuit breaker (M3).
    """
    if state_obj and state_obj.circuit_open_until:
        if datetime.datetime.now() < state_obj.circuit_open_until:
            bot_log.warning("Circuit ouvert : Appel DeepSeek ignoré.")
            return None
        else:
            state_obj.circuit_open_until = None
            state_obj.ia_fail_count = 0

    retries = [2, 4, 8]
    for attempt, wait_time in enumerate(retries + [0, 0], 1):
        try:
            async with httpx.AsyncClient(timeout=config.DEEPSEEK_TIMEOUT) as client:
                _t0 = asyncio.get_event_loop().time()
                resp = await client.post(
                    config.DEEPSEEK_URL,
                    headers={
                        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": config.DEEPSEEK_MODEL,
                        "messages": [
                            {"role": "system", "content": config.DEEPSEEK_SYSTEM_PROMPT},
                            {"role": "user",   "content": compressed_data},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 150,
                        "response_format": {"type": "json_object"}
                    }
                )
                latency = round(asyncio.get_event_loop().time() - _t0, 2)
                if state_obj:
                    state_obj.last_ia_latency_s = latency
                resp.raise_for_status()
                raw_content = resp.json()["choices"][0]["message"]["content"]
                
                signal_obj = AISignal.model_validate_json(raw_content)
                if state_obj: state_obj.ia_fail_count = 0 
                return signal_obj.model_dump()

        except Exception as exc:
            bot_log.error("Tentative %d/5 échouée : %s", attempt, exc)
            if attempt <= 3:
                await asyncio.sleep(wait_time)
            else:
                if state_obj:
                    state_obj.ia_fail_count += 1
                    if state_obj.ia_fail_count >= 5:
                        state_obj.circuit_open_until = datetime.datetime.now() + datetime.timedelta(minutes=15)
                        alert_error(f"Circuit ouvert (IA). Pause 15min. Total échecs: {state_obj.ia_fail_count}")
                return None
    return None


async def generate_fallback_signal(data: dict) -> Optional[dict]:
    """
    Génère un signal technique de secours (M4).
    """
    try:
        # 1. Alignement EMA (M15+H1+H4)
        trends = [data[tf]["ind"]["ema_trend"] for tf in ["M15", "H1", "H4"]]
        if not (trends[0] == trends[1] == trends[2] and trends[0] in ("bull", "bear")):
            return None 

        direction = "BUY" if trends[0] == "bull" else "SELL"
        
        # 2. RSI Filter
        rsi_m15 = data["M15"]["ind"]["rsi"]
        if direction == "BUY" and rsi_m15 < 55: return None
        if direction == "SELL" and rsi_m15 > 45: return None

        # 3. Construction du signal
        current_price = data["current_price"]
        atr = data["M15"]["ind"]["atr"]
        
        # SL/TP basiques pour le fallback (1.5 ATR / 3.0 ATR pour RR=2.0)
        sl_dist = atr * 1.5
        tp_dist = sl_dist * 2.0
        
        sl = current_price - sl_dist if direction == "BUY" else current_price + sl_dist
        tp = current_price + tp_dist if direction == "BUY" else current_price - tp_dist

        # Calcul LOT (réduit de 50%)
        # On utilise une valeur temporaire pour LOT car validate_signal le recalibrera
        # mais on précisera à validate_signal de réduire le risque.
        
        fallback_signal = {
            "DIR":    direction,
            "LOT":    0.01, # Sera recalculé dans validate_signal
            "TP":     round(tp, 2),
            "SL":     round(sl, 2),
            "CONF":   60,
            "RR":     2.0,
            "REASON": "FALLBACK_TECH",
            "source": "fallback_technique"
        }
        
        bot_log.info("Signal Fallback généré : %s", fallback_signal)
        return fallback_signal

    except Exception as exc:
        bot_log.error("Erreur generate_fallback_signal : %s", exc)
        return None

def calculate_lot_size(balance: float, sl_pips: float, confidence: int = 85, risk_override: float = None) -> float:
    """
    Calcule la taille de lot avec risque dynamique basé sur la confiance.
    """
    try:
        # Calcul du pourcentage de risque dynamique
        if config.USE_DYNAMIC_RISK:
            # Interpolation linéaire entre MIN_RISK et MAX_RISK selon la confiance
            conf_range = 100 - config.MIN_CONFIDENCE
            if conf_range <= 0: conf_range = 1
            
            progress = (confidence - config.MIN_CONFIDENCE) / conf_range
            progress = max(0, min(1, progress)) # Borné entre 0 et 1
            
            risk_pct = config.MIN_RISK_PCT + (progress * (config.MAX_RISK_PCT - config.MIN_RISK_PCT))
        else:
            # Utilise le risque constant ou override
            risk_pct = risk_override if risk_override is not None else config.MAX_RISK_PCT

        risk_usd = balance * risk_pct / 100
        
        if sl_pips <= 0:
            return 0.01

        # Calcul générique : 1 lot = contract_size, 1 point = tick_size
        symbol_info = mt5.symbol_info(config.MT5_SYMBOL)
        if symbol_info is None:
            return 0.01
        
        # Valeur du risque par point pour 1 lot
        point_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size
        lot = risk_usd / (sl_pips * point_value)
        lot = max(0.01, round(lot, 2))
        
        # Log du risque appliqué
        bot_log.info("Risque calculé : %.2f%% (Conf=%d) -> Lot: %.2f", risk_pct, confidence, lot)
        
        return min(lot, 10.0)
    except Exception as exc:
        bot_log.error("Exception calculate_lot_size : %s", exc)
        return 0.01


def get_account_balance() -> float:
    """Récupère le solde réel actuel du compte MT5."""
    info = mt5.account_info()  # AUDIT-FIX #2 — sync, pas de boucle concurrente ici
    if info is None:
        bot_log.error("Impossible de récupérer le solde MT5")
        return 0.0
    return info.balance


def calculate_lot(balance: float, sl_pips: float, risk_pct: float = None) -> float:
    """
    # DYNAMIC-ADAPTATION — Calcul du lot 100% universel via MT5.
    Utilise la valeur réelle du tick fournie par le broker pour le symbole actuel.
    Fonctionne sur Forex, Or, Indices, Crypto sans modification.
    """
    try:
        effective_risk = risk_pct if risk_pct is not None else config.RISK_PERCENT
        risk_amount = balance * (effective_risk / 100)
        
        symbol_info = mt5.symbol_info(config.MT5_SYMBOL)
        if symbol_info is None:
            return 0.01

        # tick_value = profit en monnaie de compte pour 1 lot si le prix bouge de 1 tick
        tick_value = symbol_info.trade_tick_value
        tick_size  = symbol_info.point # ou tick_size
        
        if tick_value <= 0 or sl_pips <= 0:
            return 0.01

        # Nombre de points de SL (1 pip = 10 points sur la majorité des brokers 5 digits)
        # Mais pour être universel, on calcule le risque par point
        sl_points = sl_pips * 10 
        
        # Formule : Lot = Risque / (SL_points * Valeur_du_point_pour_1_lot)
        lot = risk_amount / (sl_points * tick_value)
        
        # Respecter les limites du broker
        lot = max(symbol_info.volume_min, lot)
        lot = min(symbol_info.volume_max, lot)
        step = symbol_info.volume_step
        
        final_lot = round(lot / step) * step
        return round(final_lot, 2)
        
    except Exception as e:
        bot_log.error("Erreur calculate_lot dynamique : %s", e)
        return 0.01


def validate_signal(signal: dict, balance: float, current_price: float, 
                    min_conf: int = None, max_risk: float = None, market_data: dict = None) -> tuple[bool, dict, str]:
    """
    Applique tous les garde-fous calibrés (PRO) sur le signal DeepSeek.
    """
    try:
        now_str = datetime.datetime.now().strftime("%H:%M")
        if signal["DIR"] == "HOLD":
            return False, signal, "Direction HOLD : aucun trade"

        # 1. Confiance minimale (Garde-fou #1)
        effective_min_conf = min_conf if min_conf is not None else config.MIN_CONFIDENCE
        if signal["CONF"] < effective_min_conf:
            return False, signal, f"SKIP Confiance faible | {signal['CONF']}% < {effective_min_conf}% | Heure={now_str}"

        # 2. Ratio risque/rendement (Garde-fou #1)
        if signal["RR"] < config.MIN_RR:
            return False, signal, f"SKIP RR insuffisant | {signal['RR']} < {config.MIN_RR} | Heure={now_str}"

        # 3. GARDE-FOU TREND (H1/H4 EMA20/50) — Non-bloquant pour prendre quelques risques
        if market_data:
            trend = get_trend_direction(market_data)
            bot_log.info("Analyse Trend: %s | Signal: %s", trend, signal["DIR"])
            # On ne bloque plus le trade, on logue simplement si on est contre la tendance
            if (signal["DIR"] == "BUY" and trend == "SELL") or (signal["DIR"] == "SELL" and trend == "BUY"):
                bot_log.warning("ATTENTION : Trade contre-tendance initié par l'IA.")

        # 4. Cohérence SL/TP
        if signal["DIR"] == "BUY":
            if signal["SL"] >= current_price:
                return False, signal, f"SL={signal['SL']} >= prix={current_price}"
            if signal["TP"] <= current_price:
                return False, signal, f"TP={signal['TP']} <= prix={current_price}"
        elif signal["DIR"] == "SELL":
            if signal["SL"] <= current_price:
                return False, signal, f"SL={signal['SL']} <= prix={current_price}"
            if signal["TP"] >= current_price:
                return False, signal, f"TP={signal['TP']} >= prix={current_price}"

        # 5. Calcul Lot & Sizing — Utilise le risque dynamique défini via Telegram (state.max_risk_pct)
        sl_pips = abs(current_price - signal["SL"])
        max_lot = calculate_lot(balance, sl_pips, risk_pct=max_risk)
        signal["LOT"] = max_lot

        # Log de validation finale
        bot_log.info("VALIDÉ | %s | Conf=%d%% | RR=%.1f | Lot=%.2f | Heure=%s", 
                     signal["DIR"], signal["CONF"], signal["RR"], max_lot, now_str)
        
        return True, signal, ""
    except Exception as exc:
        bot_log.error("Exception validate_signal : %s", exc)
        return False, signal, str(exc)
    except Exception as exc:
        bot_log.error("Exception validate_signal : %s", exc)
        return False, signal, str(exc)


# ══════════════════════════════════════════════════════════════
# 8. EXÉCUTION DES ORDRES MT5
# ══════════════════════════════════════════════════════════════

def get_symbol_info() -> Optional[object]:
    """Récupère les informations du symbole configuré."""
    try:
        info = mt5.symbol_info(config.MT5_SYMBOL)
        if info is None:
            bot_log.error("Symbole %s introuvable dans MT5", config.MT5_SYMBOL)
        return info
    except Exception as exc:
        bot_log.error("Exception get_symbol_info : %s", exc)
        return None


async def execute_trade_async(signal: dict) -> Optional[dict]:  # AUDIT-FIX #2
    """
    AUDIT-FIX #2 — Envoie l'ordre MT5 avec mt5_lock (order_send + symbol_info_tick protégés).
    Retourne un dict avec les informations de l'ordre ou None.
    """
    try:
        symbol_info = get_symbol_info()
        if symbol_info is None:
            return None

        async with mt5_lock:  # AUDIT-FIX #2 — symbol_info_tick + order_send atomiques
            tick = mt5.symbol_info_tick(config.MT5_SYMBOL)
            if tick is None:
                bot_log.error("Impossible de récupérer le tick %s", config.MT5_SYMBOL)
                return None

            order_type = mt5.ORDER_TYPE_BUY if signal["DIR"] == "BUY" else mt5.ORDER_TYPE_SELL
            price      = tick.ask if signal["DIR"] == "BUY" else tick.bid
            deviation  = 20  # pips de slippage maximum

            request = {
                "action":        mt5.TRADE_ACTION_DEAL,
                "symbol":        config.MT5_SYMBOL,
                "volume":        signal["LOT"],
                "type":          order_type,
                "price":         price,
                "sl":            signal["SL"],
                "tp":            signal["TP"],
                "deviation":     deviation,
                "magic":         config.MT5_MAGIC,
                "comment":       f"GB-{signal['CONF']}",
                "type_time":     mt5.ORDER_TIME_GTC,
                "type_filling":  mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)  # AUDIT-FIX #2 — order_send protégé

        if result is None:
            bot_log.error("order_send a retourné None — %s", mt5.last_error())
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            bot_log.error(
                "Ordre refusé retcode=%d : %s", result.retcode, result.comment
            )
            return None

        trade_info = {
            "ticket":    result.order,
            "dir":       signal["DIR"],
            "lot":       signal["LOT"],
            "price":     result.price,
            "sl":        signal["SL"],
            "tp":        signal["TP"],
            "conf":      signal["CONF"],
            "rr":        signal["RR"],
            "reason":    signal["REASON"],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        trades_log.info(
            "OPEN | %s | lot=%.2f | price=%.2f | SL=%.2f | TP=%.2f | "
            "CONF=%d | RR=%.1f | REASON=%s | ticket=%d",
            signal["DIR"], signal["LOT"], result.price,
            signal["SL"], signal["TP"],
            signal["CONF"], signal["RR"], signal["REASON"], result.order,
        )

        return trade_info
    except Exception as exc:
        bot_log.error("Exception execute_trade_async : %s", exc)
        return None


def execute_trade(signal: dict) -> Optional[dict]:
    """
    Version synchrone conservée pour compatibilité.
    Wrapper vers execute_trade_async — à appeler via await depuis le cycle async.
    """
    import warnings
    warnings.warn("execute_trade() sync is deprecated — use await execute_trade_async()", DeprecationWarning)
    return None  # Le bot.py doit appeler execute_trade_async


def get_open_positions() -> list:
    """Retourne la liste des positions ouvertes par le bot."""
    try:
        positions = mt5.positions_get(symbol=config.MT5_SYMBOL)  # AUDIT-FIX #2 — sync OK ici
        if positions is None:
            return []
        return [p for p in positions if p.magic == config.MT5_MAGIC]
    except Exception as exc:
        bot_log.error("Exception get_open_positions : %s", exc)
        return []


def close_all_positions() -> int:
    """
    Ferme toutes les positions ouvertes par le bot.
    Retourne le nombre de positions fermées.
    """
    closed = 0
    try:
        positions = get_open_positions()
        for pos in positions:
            tick = mt5.symbol_info_tick(config.MT5_SYMBOL)
            if tick is None:
                continue
            close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       config.MT5_SYMBOL,
                "volume":       pos.volume,
                "type":         close_type,
                "position":     pos.ticket,
                "price":        close_price,
                "deviation":    20,
                "magic":        config.MT5_MAGIC,
                "comment":      "FXBOT|STOP_CMD",  # MIGRATION-EURUSD : GOLDBOT → FXBOT
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                profit_pips = (
                    (close_price - pos.price_open) if pos.type == mt5.ORDER_TYPE_BUY
                    else (pos.price_open - close_price)
                )
                trades_log.info(
                    "CLOSE | ticket=%d | profit=%.2fUSD | pips=%.1f",
                    pos.ticket, pos.profit, profit_pips,
                )
                closed += 1
            else:
                bot_log.error(
                    "Échec fermeture ticket=%d : %s",
                    pos.ticket, result.comment if result else mt5.last_error()
                )
    except Exception as exc:
        bot_log.error("Exception close_all_positions : %s", exc)
    return closed


def modify_position_sl(ticket: int, new_sl: float) -> bool:
    """
    Modifie le Stop Loss d'une position ouverte.
    AUDIT-FIX #2 — positions_get + order_send via contexte synchrone du monitoring_loop.
    """
    try:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": round(new_sl, 2),
        }
        pos = mt5.positions_get(ticket=ticket)  # AUDIT-FIX #2 — positions_get protégé
        if pos:
            request["tp"] = pos[0].tp
            request["symbol"] = pos[0].symbol

        result = mt5.order_send(request)  # AUDIT-FIX #2 — order_send protégé
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            bot_log.info("SL modifié pour #%d -> %.2f", ticket, new_sl)
            return True
        return False
    except Exception as exc:
        bot_log.error("Exception modify_position_sl #%d : %s", ticket, exc)
        return False


def partial_close_position(ticket: int, pct: float = 0.5) -> bool:
    """Ferme un pourcentage du volume d'une position."""
    try:
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        pos = pos[0]
        
        symbol = pos.symbol
        volume = pos.volume
        close_vol = round(volume * pct, 2)
        if close_vol < 0.01:
            return False

        tick = mt5.symbol_info_tick(symbol)
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_vol,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": config.MT5_MAGIC,
            "comment": pos.comment + "|PC", # Marqueur Partial Close
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            bot_log.info("Clôture partielle effectuée pour #%d (%.2f lots)", ticket, close_vol)
            return True
        return False
    except Exception as exc:
        bot_log.error("Exception partial_close_position #%d : %s", ticket, exc)
        return False


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Retourne la valeur ADX sur la dernière bougie (M12)."""
    try:
        adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=period)
        return round(adx_ind.adx().iloc[-1], 2)
    except Exception:
        return 20.0  # Valeur neutre par défaut


def process_active_trade_management(pos) -> None:
    """Applique Breakeven, Partial Close (M12 dynamique) et Trailing Stop (M12 ADX)."""
    try:
        ticket = pos.ticket
        price_open = pos.price_open
        current_price = pos.price_current
        sl_current = pos.sl
        tp_current = pos.tp
        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        
        # 1. Calcul de la progression vers le TP
        dist_total = abs(tp_current - price_open)
        if dist_total == 0: return
        
        dist_actuelle = (current_price - price_open) if is_buy else (price_open - current_price)
        progression = dist_actuelle / dist_total

        # 2. Clôture Partielle + Break-even (M12 — TP1 dynamique)
        if config.USE_PARTIAL_CLOSE and "|PC" not in pos.comment:
            # Lecture CONF/RR depuis le commentaire (format FXBOT|REASON|CONF=XX) # MIGRATION-EURUSD
            _conf, _rr = 85, 2.0
            try:
                for part in pos.comment.split("|"):
                    if part.startswith("CONF="):
                        _conf = int(part.split("=")[1])
            except Exception:
                pass

            # TP1 dynamique selon confiance et RR
            if _conf >= 90:
                close_pct = 0.30  # laisser courir
            elif _conf >= 85:
                close_pct = 0.50  # comportement standard
            else:
                close_pct = 0.65  # sortie prudente

            if progression >= close_pct:
                if partial_close_position(ticket, pct=0.5):
                    send_telegram(f"✅ <b>TP1 ATTEINT — #{ticket}</b>\nClôture {int(close_pct*100)}% (CONF={_conf}).")
                    if config.USE_BREAK_EVEN:
                        new_sl = price_open + (0.5 if is_buy else -0.5)
                        if modify_position_sl(ticket, new_sl):
                            send_telegram(f"🛡 <b>BREAK-EVEN — #{ticket}</b>\nSL déplacé au prix d'entrée.")

        # 3. Trailing Stop ADX-adapté (M12)
        if config.USE_TRAILING_STOP and progression > 0.2:
            rates = mt5.copy_rates_from_pos(pos.symbol, mt5.TIMEFRAME_M15, 0, 30)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
                atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range().iloc[-1]
                adx = compute_adx(df)
                # M12 : trend fort (ADX>30) → multiplieur réduit (serrer le stop)
                atr_multiplier = 1.5 if adx > 30 else 2.5
                dist_trail = atr * atr_multiplier

                if is_buy:
                    target_sl = current_price - dist_trail
                    if target_sl > sl_current + (atr * 0.5):
                        modify_position_sl(ticket, target_sl)
                else:
                    target_sl = current_price + dist_trail
                    if target_sl < sl_current - (atr * 0.5) or sl_current == 0:
                        modify_position_sl(ticket, target_sl)

    except Exception as exc:
        bot_log.error("Erreur process_active_trade_management #%d : %s", pos.ticket, exc)


def monitor_open_trades() -> list:
    """
    Vérifie l'état des positions ouvertes.
    Retourne la liste des positions actuellement ouvertes.
    """
    try:
        return get_open_positions()
    except Exception as exc:
        bot_log.error("Exception monitor_open_trades : %s", exc)
        return []


def _fetch_trade_result(ticket: int) -> Optional[dict]:
    """Récupère le profit et les pips d'une position fermée depuis l'historique."""
    try:
        import MetaTrader5 as mt5
        # Fenêtre de 24h pour récupérer l'historique de la position
        from_date = datetime.datetime.now() - datetime.timedelta(days=1)
        deals = mt5.history_deals_get(from_date, datetime.datetime.now(), position=ticket)
        
        if deals is None or len(deals) < 2:
            return None
        
        # Identification deal entrée/sortie
        entry_deal = [d for d in deals if d.entry == mt5.DEAL_ENTRY_IN][0]
        exit_deal = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT][0]
        
        symbol_info = mt5.symbol_info(entry_deal.symbol)
        point = symbol_info.point if symbol_info else 0.01

        # Profit réel (Profit + Commission + Swap)
        total_profit = sum(d.profit + d.commission + d.swap for d in deals)
        
        # Calcul des pips selon la direction
        if entry_deal.type == mt5.ORDER_TYPE_BUY:
            pips = (exit_deal.price - entry_deal.price) / point
        else:
            pips = (entry_deal.price - exit_deal.price) / point
        
        # Calcul du % réel de gain/perte sur la balance
        acc = mt5.account_info()
        balance = acc.balance if acc else 100.0
        pnl_pct = (total_profit / balance) * 100 if balance > 0 else 0.0

        return {
            "ticket": ticket,
            "profit": round(float(total_profit), 2),
            "pips": round(float(pips), 1),
            "dir": "BUY" if entry_deal.type == mt5.ORDER_TYPE_BUY else "SELL",
            "pnl_pct": round(float(pnl_pct), 2),
            "entry": float(entry_deal.price),
            "exit": float(exit_deal.price),
            "lot": float(entry_deal.volume),
            "sl": float(entry_deal.sl) if hasattr(entry_deal, 'sl') else 0,
            "tp": float(entry_deal.tp) if hasattr(entry_deal, 'tp') else 0
        }
    except Exception as exc:
        bot_log.error("Exception _fetch_trade_result pour ticket #%d : %s", ticket, exc)
        return None


def check_and_alert_closed_trades(known_tickets: list[int]) -> tuple[list[int], list[dict]]:
    """
    Vérifie les clôtures et retourne (tickets_actifs, resultats_clotures).
    """
    closed_results = []
    try:
        current_positions = get_open_positions()
        current_tickets = [p.ticket for p in current_positions]
        
        still_open = []
        for ticket in known_tickets:
            if ticket not in current_tickets:
                res = _fetch_trade_result(ticket)
                if res:
                    alert_trade_close(ticket, res['profit'], res['pips'])
                    closed_results.append(res)
                else:
                    send_telegram(f"ℹ️ <b>POSITION FERMÉE</b>\nTicket : #{ticket}")
            else:
                still_open.append(ticket)
        
        for t in current_tickets:
            if t not in still_open: still_open.append(t)
                
        return still_open, closed_results
    except Exception as exc:
        bot_log.error("Exception check_and_alert_closed_trades : %s", exc)
        return known_tickets, []


def close_position_by_ticket(ticket: int) -> bool:
    """Ferme une position spécifique par son ticket."""
    try:
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        pos = pos[0]
        
        tick = mt5.symbol_info_tick(pos.symbol)
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     pos.ticket,
            "price":        close_price,
            "deviation":    20,
            "magic":        config.MT5_MAGIC,
            "comment":      "LOGBOT|MANUAL_CLOSE",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            bot_log.info("Position #%d fermée manuellement via Telegram.", ticket)
            return True
        return False
    except Exception as e:
        bot_log.error("Erreur close_position_by_ticket #%d : %s", ticket, e)
        return False


# ══════════════════════════════════════════════════════════════
# 9. TELEGRAM — ALERTES ASYNCHRONES
# ══════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    """
    Envoie un message texte sur Telegram via l'API HTTP (synchrone).
    Retourne True si succès.
    """
    try:
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            bot_log.warning("Telegram non configuré, message ignoré.")
            return False

        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id":    config.TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as exc:
        bot_log.error("Erreur envoi Telegram : %s", exc)
        return False


def alert_trade_open(trade: dict) -> None:
    """Alerte Telegram à l'ouverture d'un trade."""
    msg = (
        f"🟢 <b>TRADE OUVERT — {trade['dir']}</b>\n"
        f"📊 Lot : {trade['lot']} | Prix : {trade['price']}\n"
        f"🎯 TP : {trade['tp']} | 🛑 SL : {trade['sl']}\n"
        f"🤖 IA Confiance : {trade['conf']}% | RR : {trade['rr']}\n"
        f"📝 Raison : {trade['reason']}\n"
        f"🎟 Ticket : #{trade['ticket']}"
    )
    send_telegram(msg)


def alert_trade_close(ticket: int, profit_usd: float, pips: float) -> None:
    """Alerte Telegram à la fermeture d'un trade."""
    emoji = "💰" if profit_usd >= 0 else "🔴"
    msg = (
        f"{emoji} <b>TRADE FERMÉ #{ticket}</b>\n"
        f"Résultat : {profit_usd:+.2f} USD | {pips:+.1f} pips"
    )
    send_telegram(msg)


def alert_error(error_msg: str) -> None:
    """Alerte Telegram pour erreur critique."""
    msg = f"⚠️ <b>ERREUR CRITIQUE</b>\n{error_msg}"
    send_telegram(msg)


def alert_daily_summary(balance: float, equity: float, trades_today: int,
                         profit_today: float) -> None:
    """Résumé journalier envoyé à 23h00. # MIGRATION-EURUSD : header mis à jour"""
    dd = (balance - equity) / balance * 100 if balance > 0 else 0
    emoji = "📈" if profit_today >= 0 else "📉"
    msg = (
        f"{emoji} <b>RÉSUMÉ JOURNALIER FXBOT EUR/USD</b>\n"  # MIGRATION-EURUSD
        f"💰 Balance : {balance:.2f} USD\n"
        f"📊 Equity  : {equity:.2f} USD\n"
        f"📉 Drawdown: {dd:.2f}%\n"
        f"🔢 Trades  : {trades_today}\n"
        f"💵 P&L     : {profit_today:+.2f} USD"
    )
    send_telegram(msg)


def get_log_tail(n: int = config.LOG_MAX_LINES_TELEGRAM) -> str:
    """Retourne les n dernières lignes du fichier bot.log."""
    try:
        if not os.path.exists(config.LOG_BOT):
            return "Fichier log introuvable."
        with open(config.LOG_BOT, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception as exc:
        bot_log.error("Exception get_log_tail : %s", exc)
        return "Erreur lecture log."


def format_open_positions_message() -> str:
    """Formate la liste des positions ouvertes pour Telegram."""
    try:
        positions = get_open_positions()
        if not positions:
            return "Aucune position ouverte."
        lines = [f"📋 <b>Positions ouvertes ({len(positions)})</b>"]
        for p in positions:
            dir_str = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            lines.append(
                f"• #{p.ticket} {dir_str} {p.volume}L "
                f"@ {p.price_open:.2f} | SL={p.sl:.2f} | TP={p.tp:.2f} "
                f"| P&L: {p.profit:+.2f}$"
            )
        return "\n".join(lines)
    except Exception as exc:
        bot_log.error("Exception format_open_positions_message : %s", exc)
        return "Erreur récupération positions."
