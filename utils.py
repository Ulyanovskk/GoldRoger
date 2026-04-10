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
import datetime
import numpy as np
import pandas as pd
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

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


def get_ohlcv(timeframe_name: str, n: int = config.CANDLES_COUNT) -> Optional[pd.DataFrame]:
    """
    Récupère n bougies OHLCV pour XAUUSD sur le timeframe donné.
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
            indicators = compute_indicators(df)
            result[tf] = {"df": df, "ind": indicators}

        # Support/Résistance sur H4 et D1
        s_h4, r_h4 = find_support_resistance(result["H4"]["df"])
        s_d1, r_d1 = find_support_resistance(result["D1"]["df"])
        result["SR"] = {"H4": {"s": s_h4, "r": r_h4},
                        "D1": {"s": s_d1, "r": r_d1}}

        # Prix actuel et info compte
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


def compress_data(data: dict) -> str:
    """
    Compresse toutes les données multi-timeframe en une chaîne < 120 tokens.
    Format : XAU|M15:...|H1:...|H4:...|D1:...|bal=...|sess=...
    """
    try:
        parts = ["XAU"]

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
    Vérifie si M15, H1 et H4 pointent dans la même direction générale.
    Condition : les 3 timeframes ont le même ema_trend (bull ou bear).
    """
    try:
        trends = [data[tf]["ind"]["ema_trend"] for tf in ["M15", "H1", "H4"]]
        return trends[0] == trends[1] == trends[2] and trends[0] in ("bull", "bear")
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

    return True, ""


# ══════════════════════════════════════════════════════════════
# 6. APPEL DEEPSEEK
# ══════════════════════════════════════════════════════════════

def call_deepseek(compressed_data: str) -> Optional[dict]:
    """
    Envoie les données compressées à DeepSeek et parse la réponse.
    Retourne un dict {DIR, LOT, TP, SL, CONF, RR, REASON} ou None.
    """
    try:
        headers = {
            "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": config.DEEPSEEK_SYSTEM_PROMPT},
                {"role": "user",   "content": compressed_data},
            ],
            "temperature": 0.1,
            "max_tokens": 80,
        }

        resp = requests.post(
            config.DEEPSEEK_URL,
            headers=headers,
            json=payload,
            timeout=config.DEEPSEEK_TIMEOUT,
        )
        resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        bot_log.info("Réponse DeepSeek brute : %s", raw)

        return _parse_deepseek_response(raw)
    except requests.exceptions.Timeout:
        bot_log.error("Timeout DeepSeek API après %ds", config.DEEPSEEK_TIMEOUT)
        return None
    except requests.exceptions.RequestException as exc:
        bot_log.error("Erreur HTTP DeepSeek : %s", exc)
        return None
    except Exception as exc:
        bot_log.error("Exception call_deepseek : %s", exc)
        return None


def _parse_deepseek_response(raw: str) -> Optional[dict]:
    """
    Parse la réponse au format :
    DIR=BUY|LOT=0.02|TP=2334|SL=2318|CONF=87|RR=1.8|REASON=3w
    Retourne un dict ou None si le parsing échoue.
    """
    try:
        pattern = (
            r"DIR=(?P<DIR>BUY|SELL|WAIT)"
            r"\|LOT=(?P<LOT>[\d.]+)"
            r"\|TP=(?P<TP>[\d.]+)"
            r"\|SL=(?P<SL>[\d.]+)"
            r"\|CONF=(?P<CONF>\d+)"
            r"\|RR=(?P<RR>[\d.]+)"
            r"\|REASON=(?P<REASON>[^\s|]+)"
        )
        m = re.search(pattern, raw)
        if not m:
            bot_log.error("Impossible de parser la réponse DeepSeek : '%s'", raw)
            return None

        return {
            "DIR":    m.group("DIR"),
            "LOT":    float(m.group("LOT")),
            "TP":     float(m.group("TP")),
            "SL":     float(m.group("SL")),
            "CONF":   int(m.group("CONF")),
            "RR":     float(m.group("RR")),
            "REASON": m.group("REASON"),
        }
    except Exception as exc:
        bot_log.error("Exception _parse_deepseek_response : %s", exc)
        return None


# ══════════════════════════════════════════════════════════════
# 7. GARDE-FOUS RISQUE
# ══════════════════════════════════════════════════════════════

def calculate_lot_size(balance: float, sl_pips: float) -> float:
    """
    Calcule la taille de lot basée sur le risque en pourcentage.
    Pour XAUUSD : 1 pip ≈ 0.1 USD par micro-lot (0.01).
    Valeur pip XAUUSD : 1 pip = 1 USD pour 1 lot standard.
    """
    try:
        risk_usd = balance * config.RISK_PERCENT / 100
        if sl_pips <= 0:
            return 0.01
        # Pour XAUUSD : 1 lot = 100 oz, 1 pip = $1 (si prix en USD)
        # Valeur pip = 1 USD/pip/lot
        lot = risk_usd / (sl_pips * 1.0)
        lot = round(lot, 2)
        lot = max(0.01, min(lot, 10.0))  # Entre 0.01 et 10 lots
        return lot
    except Exception as exc:
        bot_log.error("Exception calculate_lot_size : %s", exc)
        return 0.01


def validate_signal(signal: dict, balance: float, current_price: float) -> tuple[bool, dict, str]:
    """
    Applique tous les garde-fous risque sur le signal DeepSeek.
    Retourne (valide, signal_corrigé, raison_rejet).
    """
    try:
        if signal["DIR"] == "WAIT":
            return False, signal, "Direction WAIT : aucun trade"

        # Confiance minimale
        if signal["CONF"] < config.MIN_CONFIDENCE:
            return False, signal, f"Confiance insuffisante ({signal['CONF']} < {config.MIN_CONFIDENCE})"

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

        # Recalcul du lot selon le risque réel
        sl_pips = abs(current_price - signal["SL"])
        max_lot = calculate_lot_size(balance, sl_pips)
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
    """Récupère les informations du symbole XAUUSD."""
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
            bot_log.error("Impossible de récupérer le tick XAUUSD")
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
