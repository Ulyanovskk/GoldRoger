"""
utils.py — Toutes les fonctions utilitaires du bot GOLDBOT :
    - Connexion / données MetaTrader 5
    - Calcul des indicateurs techniques
    - Compression des données pour DeepSeek
    - Filtre pré-IA
    - Appel DeepSeek + parsing
    - Garde-fous risque
    - Exécution des ordres MT5
    - Alertes et commandes Telegram
    - Logging
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


# Loggers globaux
bot_log = setup_logger("bot", config.LOG_BOT)
trades_log = setup_logger("trades", config.LOG_TRADES)


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


def get_ohlcv(timeframe_name: str, n: int = config.CANDLES_COUNT) -> Optional[pd.DataFrame]:
    """
    Récupère n bougies OHLCV pour le symbole configuré (ex: XAUUSD) sur le timeframe donné.
    Retourne un DataFrame pandas ou None en cas d'erreur.
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

        # Tendance EMA
        if last_close > last_ema20 > last_ema50 > last_ema200:
            ema_trend = "bull"
        elif last_close < last_ema20 < last_ema50 < last_ema200:
            ema_trend = "bear"
        else:
            ema_trend = "mix"

        return {
            "rsi":       round(rsi.iloc[-1], 1),
            "macd":      "+" if macd_hist.iloc[-1] > 0 else "-",
            "macd_val":  round(macd_hist.iloc[-1], 4),
            "ema_trend": ema_trend,
            "ema20":     round(last_ema20, 2),
            "ema50":     round(last_ema50, 2),
            "ema200":    round(last_ema200, 2),
            "bb_pos":    bb_pos,
            "atr":       round(atr.iloc[-1], 2),
            "close":     round(last_close, 2),
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
        r_levels = [r for r in cluster(resistances) if r > current_price]
        s_levels = [s for s in cluster(supports)    if s < current_price]

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


def collect_all_data() -> Optional[dict]:
    """
    Collecte et calcule les indicateurs sur les 4 timeframes.
    Retourne un dictionnaire structuré ou None en cas d'erreur critique.
    """
    try:
        result = {}
        for tf in ["M15", "H1", "H4", "D1"]:
            df = get_ohlcv(tf)
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

        tick = mt5.symbol_info_tick(config.MT5_SYMBOL)
        result["current_price"] = tick.ask if tick else result["M15"]["ind"]["close"]

        account = mt5.account_info()
        result["balance"] = round(account.balance, 2) if account else 0
        result["equity"]  = round(account.equity, 2)  if account else 0

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
        
        # Récupération des événements du calendrier MT5
        events = mt5.calendar_get(time_from=int(start.timestamp()), time_to=int(end.timestamp()))
        if events:
            for ev in events:
                if ev.importance >= config.BLOCK_NEWS_IMPORTANCE and ev.currency in ("USD", "XAU"):
                    return True, ev.name
        return False, ""
    except Exception as e:
        bot_log.error("Erreur calendrier MT5 : %s", e)
        return False, ""


def fetch_market_news() -> str:
    """Récupère les prochains événements économiques majeurs via MT5 (USD/XAU)."""
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        # On regarde les prochaines 24 heures
        end = now + datetime.timedelta(hours=24)
        
        events = mt5.calendar_get(time_from=int(now.timestamp()), time_to=int(end.timestamp()))
        if events:
            # On filtre les news de moyenne et haute importance pour USD et Or
            important = [ev for ev in events if ev.importance >= 2 and ev.currency in ("USD", "XAU")]
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


def compress_data(data: dict, context: dict = None) -> str:
    """
    Compresse les données techniques + injection du contexte (M5).
    """
    try:
        parts = []
        
        # M5 — Head context
        if context:
            l3 = "".join(["W" if t['pnl'] > 0 else "L" for t in context.get('last_trades', [])])
            wb = context.get('week_bias', 'NEUT')[:4]
            sr = context.get('persistent_sr', [])
            parts.append(f"CTX:L3={l3 or 'None'},WB={wb},SR={sr}")

        parts.append("XAU")

        def tf_str(tf: str) -> str:
            ind = data[tf]["ind"]
            return (
                f"{tf}:RSI={ind['rsi']},"
                f"MACD={ind['macd']},"
                f"BB={ind['bb_pos']},"
                f"EMA={ind['ema_trend']},"
                f"ATR={ind['atr']}"
            )

        parts.append(tf_str("M15"))
        parts.append(tf_str("H1"))

        # H4 : ajout S/R
        ind_h4 = data["H4"]["ind"]
        sr_h4  = data["SR"]["H4"]
        s_str  = str(sr_h4["s"][-1]) if sr_h4["s"] else "?"
        r_str  = str(sr_h4["r"][0])  if sr_h4["r"] else "?"
        parts.append(
            f"H4:RSI={ind_h4['rsi']},"
            f"MACD={ind_h4['macd']},"
            f"EMA={ind_h4['ema_trend']},"
            f"SR={r_str}/{s_str}"
        )

        # D1 : biais directionnel
        ind_d1 = data["D1"]["ind"]
        bias = (
            "bull" if ind_d1["ema_trend"] == "bull"
            else "bear" if ind_d1["ema_trend"] == "bear"
            else "neut"
        )
        parts.append(f"D1:bias={bias},RSI={ind_d1['rsi']}")

        parts.append(f"bal={int(data['balance'])}USD")
        parts.append(f"sess={get_current_session()}")
        
        # Dollar Index + News (Crucial pour Gold)
        parts.append(fetch_dxy_data())
        parts.append(fetch_market_news())
        
        parts.append(f"price={data['current_price']}")

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
        positions = mt5.positions_get(symbol=config.MT5_SYMBOL)
        if positions is None:
            return 0
        return sum(1 for p in positions if p.magic == config.MT5_MAGIC)
    except Exception as exc:
        bot_log.error("Exception get_open_trades_count : %s", exc)
        return 0


def get_daily_drawdown_pct() -> float:
    """
    Calcule le drawdown journalier en pourcentage.
    (balance_départ - equity_actuelle) / balance_départ * 100
    """
    try:
        account = mt5.account_info()
        if account is None:
            return 0.0
        # Approximation : on utilise l'equity vs le solde comme proxy
        if account.balance <= 0:
            return 0.0
        dd = (account.balance - account.equity) / account.balance * 100
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
    Vérifie si H1 et H4 pointent dans la même direction générale (tendance de fond).
    Condition principale : H1 et H4 ont le même ema_trend (bull ou bear).
    Le M15 peut être en correction/respiration sans bloquer le bot.
    Un bonus est accordé si M15 est également aligné (loggé).
    """
    try:
        h1_trend  = data["H1"]["ind"]["ema_trend"]
        h4_trend  = data["H4"]["ind"]["ema_trend"]
        m15_trend = data["M15"]["ind"]["ema_trend"]

        if h1_trend == h4_trend and h1_trend in ("bull", "bear"):
            if m15_trend == h1_trend:
                bot_log.debug("Timeframes parfaitement alignés (M15+H1+H4 = %s)", h1_trend)
            else:
                bot_log.debug(
                    "Alignement partiel : H1+H4 = %s | M15 = %s (respiration tolérée)",
                    h1_trend, m15_trend
                )
            return True
        return False
    except Exception as exc:
        bot_log.error("Exception are_timeframes_aligned : %s", exc)
        return False


def pre_ia_filter(data: dict) -> tuple[bool, str]:
    """
    Applique tous les filtres pré-IA.
    Retourne (True, "") si tout est OK, (False, raison) sinon.
    """
    # Session active
    if not is_active_session():
        return False, "Session inactive (hors London/NY)"

    # Trades simultanés
    open_count = get_open_trades_count()
    if open_count >= config.MAX_SIMULTANEOUS_TRADES:
        return False, f"Max trades atteint ({open_count}/{config.MAX_SIMULTANEOUS_TRADES})"

    # Drawdown journalier
    dd = get_daily_drawdown_pct()
    if dd >= config.MAX_DAILY_DRAWDOWN:
        return False, f"Drawdown journalier atteint ({dd:.2f}% >= {config.MAX_DAILY_DRAWDOWN}%)"

    # Alignement des timeframes
    if not are_timeframes_aligned(data):
        return False, "Timeframes non alignés (M15/H1/H4)"

    # Volatilité ATR
    atr = data["M15"]["ind"]["atr"]
    if atr < config.ATR_MIN_THRESHOLD:
        return False, f"Volatilité trop faible (ATR={atr:.1f})"
    if atr > config.ATR_MAX_THRESHOLD:
        return False, f"Volatilité trop élevée (ATR={atr:.1f})"

    # FILTRES DE SÉCURITÉ SUPPLÉMENTAIRES (Or)
    # 1. Spread
    tick = mt5.symbol_info_tick(config.MT5_SYMBOL)
    if tick and tick.spread > config.MAX_SPREAD_POINTS:
        return False, f"Spread trop élevé ({tick.spread} pts)"

    # 2. Calendrier Économique
    is_news, news_name = is_high_impact_news()
    if is_news:
        return False, f"News majeure : {news_name}"

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

    # 2. Latence IA > 3s
    if state_obj.last_ia_latency_s > 3.0:
        send_telegram(f"⏱ <b>IA lente : {state_obj.last_ia_latency_s}s</b>")
        bot_log.warning("Latence IA élevée : %.2fs", state_obj.last_ia_latency_s)
        state_obj.last_ia_latency_s = 0.0  # Reset pour éviter spam

    # 3. Drawdown session > 2% → risque réduit
    dd = get_daily_drawdown_pct()
    if dd >= 3.0 and state_obj.is_active():
        await state_obj.set_status("stoppé")
        close_all_positions()
        send_telegram("🛑 <b>Drawdown 3% atteint.</b> Arrêt du bot.")
        bot_log.error("Arrêt automatique : drawdown %.2f%%", dd)
    elif dd >= 2.0:
        if state_obj.max_risk_pct > config.MAX_RISK_PCT * 0.5:
            state_obj.max_risk_pct = round(config.MAX_RISK_PCT * 0.5, 2)
            send_telegram(
                f"📉 <b>Drawdown {dd:.2f}% atteint.</b>\n"
                f"Risque réduit à {state_obj.max_risk_pct}% (50%)."
            )

    # 4. Spread anormal (déjà dans pre_ia_filter — log seul ici si hors session)
    try:
        tick = mt5.symbol_info_tick(config.MT5_SYMBOL)
        if tick and tick.spread > config.MAX_SPREAD_POINTS * 2:
            send_telegram(f"📊 <b>Spread anormal détecté : {tick.spread} pts.</b> Trade ignoré.")
    except Exception:
        pass



# ══════════════════════════════════════════════════════════════

class AISignal(BaseModel):
    DIR: Literal["BUY", "SELL", "HOLD"]
    LOT: float = Field(ge=0.01, le=10.0)
    TP: float
    SL: float
    CONF: int = Field(ge=0, le=100)
    RR: float
    REASON: str = Field(max_length=100) # max 5 words est une consigne prompt

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
        lot = round(lot, 2)
        
        # Log du risque appliqué
        bot_log.info("Risque calculé : %.2f%% (Conf=%d) -> Lot: %.2f", risk_pct, confidence, lot)
        
        return max(0.01, min(lot, 10.0))
    except Exception as exc:
        bot_log.error("Exception calculate_lot_size : %s", exc)
        return 0.01


def validate_signal(signal: dict, balance: float, current_price: float, 
                    min_conf: int = None, max_risk: float = None) -> tuple[bool, dict, str]:
    """
    Applique tous les garde-fous risque sur le signal DeepSeek.
    Supporte les overrides dynamiques min_conf et max_risk.
    """
    try:
        if signal["DIR"] == "HOLD":
            return False, signal, "Direction HOLD : aucun trade"

        # Confiance minimale (dynamique ou config)
        effective_min_conf = min_conf if min_conf is not None else config.MIN_CONFIDENCE
        if signal["CONF"] < effective_min_conf:
            return False, signal, f"Confiance insuffisante ({signal['CONF']} < {effective_min_conf})"

        # Ratio risque/rendement
        if signal["RR"] < config.MIN_RR:
            return False, signal, f"RR insuffisant ({signal['RR']} < {config.MIN_RR})"

        # Cohérence SL/TP avec la direction
        if signal["DIR"] == "BUY":
            if signal["SL"] >= current_price:
                return False, signal, f"SL={signal['SL']} >= prix={current_price} pour BUY"
            if signal["TP"] <= current_price:
                return False, signal, f"TP={signal['TP']} <= prix={current_price} pour BUY"
        elif signal["DIR"] == "SELL":
            if signal["SL"] <= current_price:
                return False, signal, f"SL={signal['SL']} <= prix={current_price} pour SELL"
            if signal["TP"] >= current_price:
                return False, signal, f"TP={signal['TP']} >= prix={current_price} pour SELL"

        # Recalcul du lot selon le risque réel (dynamique)
        sl_pips = abs(current_price - signal["SL"])
        
        # Override dynamique du risque max si présent
        effective_risk = max_risk if max_risk is not None else config.MAX_RISK_PCT
        
        # M4 — Sécurité fallback : réduction de 50% du risque
        if signal.get("source") == "fallback_technique":
            effective_risk *= 0.5
            
        max_lot = calculate_lot_size(balance, sl_pips, confidence=signal["CONF"], risk_override=effective_risk)



        if signal["LOT"] > max_lot:
            bot_log.warning(
                "LOT IA (%s) > max calculé (%s), recalibrage.", signal["LOT"], max_lot
            )
            signal["LOT"] = max_lot

        # Drawdown en temps réel
        dd = get_daily_drawdown_pct()
        if dd >= config.MAX_DAILY_DRAWDOWN:
            return False, signal, f"Drawdown journalier atteint ({dd:.2f}%)"

        return True, signal, ""
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


def execute_trade(signal: dict) -> Optional[dict]:
    """
    Envoie l'ordre MT5 à partir du signal validé.
    Retourne un dict avec les informations de l'ordre ou None.
    """
    try:
        symbol_info = get_symbol_info()
        if symbol_info is None:
            return None

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
            "comment":       f"GOLDBOT|{signal['REASON']}|CONF={signal['CONF']}",
            "type_time":     mt5.ORDER_TIME_GTC,
            "type_filling":  mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
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

        # Log trades.log
        trades_log.info(
            "OPEN | %s | lot=%.2f | price=%.2f | SL=%.2f | TP=%.2f | "
            "CONF=%d | RR=%.1f | REASON=%s | ticket=%d",
            signal["DIR"], signal["LOT"], result.price,
            signal["SL"], signal["TP"],
            signal["CONF"], signal["RR"], signal["REASON"], result.order,
        )

        return trade_info
    except Exception as exc:
        bot_log.error("Exception execute_trade : %s", exc)
        return None


def get_open_positions() -> list:
    """Retourne la liste des positions ouvertes par le bot."""
    try:
        positions = mt5.positions_get(symbol=config.MT5_SYMBOL)
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
                "comment":      "GOLDBOT|STOP_CMD",
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
    """Modifie le Stop Loss d'une position ouverte."""
    try:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": round(new_sl, 2),
        }
        # On garde le TP actuel s'il existe
        pos = mt5.positions_get(ticket=ticket)
        if pos:
            request["tp"] = pos[0].tp
            request["symbol"] = pos[0].symbol
        
        result = mt5.order_send(request)
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
            # Lecture CONF/RR depuis le commentaire (format GOLDBOT|REASON|CONF=XX)
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
        # On définit une fenêtre de temps large pour l'historique (dernières 24h)
        from_date = datetime.datetime.now() - datetime.timedelta(days=1)
        deals = mt5.history_deals_get(from_date, datetime.datetime.now(), position=ticket)
        
        if deals is None or len(deals) < 2:
            return None
        
        # Le deal d'entrée est le premier, le deal de sortie est le dernier
        # On calcule le profit total cumulé des deals pour cette position
        total_profit = sum(d.profit + d.commission + d.swap for d in deals)
        
        # Calcul des pips (différence entre deal d'entrée et de sortie)
        entry_deal = [d for d in deals if d.entry == mt5.DEAL_ENTRY_IN][0]
        exit_deal = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT][0]
        
        diff = abs(entry_deal.price - exit_deal.price)
        pips = diff 
        
        # M5 context data
        return {
            "ticket": ticket,
            "profit": total_profit, 
            "pips": pips,
            "dir": "BUY" if entry_deal.type == mt5.ORDER_TYPE_BUY else "SELL",
            "pnl_pct": (total_profit / entry_deal.volume / entry_deal.price) * 100 # Approx
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
    """Résumé journalier envoyé à 23h00."""
    dd = (balance - equity) / balance * 100 if balance > 0 else 0
    emoji = "📈" if profit_today >= 0 else "📉"
    msg = (
        f"{emoji} <b>RÉSUMÉ JOURNALIER GOLDBOT</b>\n"
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
