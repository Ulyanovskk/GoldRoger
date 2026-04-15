"""
config.py — Chargement des variables d'environnement et constantes globales.
Toutes les valeurs sensibles viennent du fichier .env (jamais hardcodées).

# MIGRATION-EURUSD : Migration complète XAU/USD → EUR/USD (2026-04-15)
"""

import os
from dotenv import load_dotenv

# Chargement du fichier .env
load_dotenv()

# ──────────────────────────────────────────────
# DeepSeek API
# ──────────────────────────────────────────────
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL: str = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL: str = "deepseek-chat"
DEEPSEEK_TIMEOUT: int = 30  # secondes

# MIGRATION-EURUSD : DXY retiré comme indicateur principal — corrélation USD Index générale uniquement
# DXY_SYMBOL n'est plus utilisé activement pour EUR/USD
DXY_SYMBOL: str = "DXYm"  # Conservé pour compatibilité — non utilisé en mode EUR/USD

# MIGRATION-EURUSD + PRICE-STRUCTURE : Prompt système mis à jour pour EUR/USD M15
DEEPSEEK_SYSTEM_PROMPT: str = (
    "You are a professional EUR/USD Forex trader analyst. "
    "You specialize in M15 technical analysis on EUR/USD. "
    "Your ONLY job is to analyze technical data and return a trading signal. "

    "RULES: "
    "1. Only return BUY or SELL if at least 3 independent indicators align. "
    "2. Return HOLD if signals are mixed or contradictory. "
    "3. CONF must reflect real signal strength: "
    "   - 60-69: weak alignment, 1 conflicting indicator. "
    "   - 70-79: moderate, most indicators agree. "
    "   - 80+: strong, all timeframes and indicators aligned. "
    "4. Never return CONF > 80 during high volatility (ATR spike). "
    "5. TP and SL must be based on ATR value provided (EUR/USD ATR is typically 0.0005-0.002). "
    "6. Do NOT calculate LOT — leave it at 0.0. "
    "7. Consider general USD Index correlation (not DXY-specific) for directional bias. "

    "Data keys: R=RSI, M=MACD, B=Bollinger, "
    "E=EMA trend, A=ATR, Bwr/Swr=BUY/SELL win rates. "

    "Additional keys: "  # PRICE-STRUCTURE
    "STRUCT=price structure (UPTREND/DOWNTREND/RANGE), "
    "SLOPE_M15=EMA20 slope on M15 (UP/DOWN/FLAT), "
    "SLOPE_H1=EMA20 slope on H1 (UP/DOWN/FLAT), "
    "CH_POS=price position in channel (TOP/MID/BOTTOM). "

    "Additional keys: "
    "STRUCT=UPTREND/DOWNTREND/RANGE, LRS=Linear Regression Slope (float), "
    "SLOPE_M15/H1=EMA slope, CH_POS=channel pos, R_SLOPE=RSI accel. "

    "PRIORITY RULE (DYNAMIC REVOLUTION): "
    "1. LRS is the absolute truth for structure. If LRS < -0.00002 → DOWNTREND is active. "
    "2. If LRS and SLOPE_M15 agree, TRADE THE FLOW. "
    "3. Ignore CH_POS=BOTTOM if LRS is negative and R_SLOPE is FLAT/DOWN (Trend continuation). "
    "4. Ignore LT Bullish bias (D1/H4) if LRS M15 is strongly negative (Scalp/Intraday flow). "

    "Response MUST be strict JSON only, no extra text: "
    '{"DIR":"BUY|SELL|HOLD", "LOT":0.0, "TP":float, '
    '"SL":float, "CONF":int, "RR":float, '
    '"REASON":"indicator1 + indicator2 + context"}'
)

# ──────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID")

# ──────────────────────────────────────────────
# MetaTrader 5
# ──────────────────────────────────────────────
MT5_LOGIN: int = int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None
MT5_PASSWORD: str = os.getenv("MT5_PASSWORD")
MT5_SERVER: str = os.getenv("MT5_SERVER")
MT5_SYMBOL: str = "EURUSDm"  # MIGRATION-EURUSD : XAUUSDm → EURUSDm
MT5_MAGIC: int = 20240115  # Identifiant unique du bot

# Timeframes utilisés pour l'analyse multi-temporelle
MT5_TIMEFRAMES: dict = {
    "M15": None,  # rempli dans utils.py après import MetaTrader5
    "H1":  None,
    "H4":  None,
    "D1":  None,
}

# Nombre de bougies à récupérer par timeframe
CANDLES_COUNT: int = 300  # Min 200+100 buffer pour l'EMA 200 — ne pas descendre en dessous

# ──────────────────────────────────────────────
# Gestion du risque (STRICTE 1.5%)
# ──────────────────────────────────────────────
RISK_PERCENT: float = 1.5
MAX_DAILY_DRAWDOWN: float = float(os.getenv("MAX_DAILY_DRAWDOWN", "10.0"))  # Lu depuis .env — ne jamais hardcoder ailleurs
MIN_CONFIDENCE: int = int(os.getenv("MIN_CONFIDENCE", "55"))  # Abaissé à 55% pour plus d'activité
MAX_SIMULTANEOUS_TRADES: int = int(os.getenv("MAX_SIMULTANEOUS_TRADES", "2"))
MIN_RR: float = 1.1  # Ajusté 1.3 -> 1.1 pour EUR/USD (Flux dynamique)
NEWS_BLOCK_WINDOW: int = 20  # GARDE-FOU NEWS : ±20 min autour des news majoritairement impactantes

# ──────────────────────────────────────────────
# Risque Dynamique (Nouveau)
# ──────────────────────────────────────────────
USE_DYNAMIC_RISK: bool = False
MIN_RISK_PCT: float = 0.5   # Risque min pour CONF = MIN_CONFIDENCE
MAX_RISK_PCT: float = 2.5   # Risque max pour CONF = 100

# ──────────────────────────────────────────────
# Filtre Volatilité ATR — MIGRATION-EURUSD
# ──────────────────────────────────────────────
# EUR/USD : ATR typique 0.0005 à 0.002 (en unités de prix, pas en USD)
ATR_MIN_THRESHOLD: float = 0.0003   # MIGRATION-EURUSD : seuil min ATR EUR/USD
ATR_MAX_THRESHOLD: float = 0.005    # MIGRATION-EURUSD : seuil max ATR EUR/USD

# MIGRATION-EURUSD : Paramètres ATR/SL/TP pour EUR/USD
ATR_MULTIPLIER_TP: float = 1.5   # TP = 1.5x ATR
ATR_MULTIPLIER_SL: float = 1.0   # SL = 1.0x ATR
MIN_SL_PIPS: int = 10            # SL minimum 10 pips
MAX_SL_PIPS: int = 50            # SL maximum 50 pips

# ──────────────────────────────────────────────
# Filtres de Protection (Spread & Calendrier)
# ──────────────────────────────────────────────
# HOTFIX-2 — Spread en PIPS EUR/USD (formule utils.py : (ask-bid)/(point*10))
# Normal=15 pips, Aggro=25 pips, Safe=10 pips — 1 pip = 0.0001 sur EUR/USD Exness
MAX_SPREAD_POINTS: int = 15       # HOTFIX-2 : 15 pips (valeur directe en pips, pas en points)
BLOCK_NEWS_IMPORTANCE: int = 3    # 3 = Haute importance (NFP, CPI, Fed)
NEWS_CHECK_WINDOW_MINS: int = 30  # ±30 min avant/après une news majeure

# ──────────────────────────────────────────────
# Gestion Active des Positions (Nouveau)
# ──────────────────────────────────────────────
USE_PARTIAL_CLOSE: bool = True     # Fermer 50% au premier palier
PARTIAL_CLOSE_PCT: float = 0.5     # À 50% de l'objectif TP
USE_BREAK_EVEN: bool = True        # Déplacer SL au prix d'entrée après TP1
USE_TRAILING_STOP: bool = True     # Activer le stop suiveur
TRAILING_STOP_DISTANCE_ATR: float = 2.0  # Distance du stop (en multiples de l'ATR)

# ──────────────────────────────────────────────
# Cycle principal
# ──────────────────────────────────────────────
CYCLE_INTERVAL_MINUTES: int = 15  # Aligné sur M15
DAILY_SUMMARY_HOUR: int = 23     # Heure d'envoi du résumé Telegram

# ──────────────────────────────────────────────
# Sessions de trading actives (UTC)
# ──────────────────────────────────────────────
ACTIVE_SESSIONS: dict = {
    "London":   {"start": 7,  "end": 16},
    "New York": {"start": 12, "end": 21},
    "Asia/Test": {"start": 0,  "end": 24},   # Ouvert 24h/24 pour phase de test
}

# ──────────────────────────────────────────────
# Indicateurs techniques
# ──────────────────────────────────────────────
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_FAST: int = 20
EMA_MID: int = 50
EMA_SLOW: int = 200
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BB_PERIOD: int = 20
BB_STD: int = 2

# ──────────────────────────────────────────────
# Logs
# ──────────────────────────────────────────────
LOG_DIR: str = "logs"
LOG_BOT: str = f"{LOG_DIR}/bot.log"
LOG_TRADES: str = f"{LOG_DIR}/trades.log"
LOG_MAX_LINES_TELEGRAM: int = 20  # Lignes envoyées via /log
