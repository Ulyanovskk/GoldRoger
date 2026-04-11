"""
config.py — Chargement des variables d'environnement et constantes globales.
Toutes les valeurs sensibles viennent du fichier .env (jamais hardcodées).
"""

import os
from dotenv import load_dotenv

# Chargement du fichier .env
load_dotenv()

# ──────────────────────────────────────────────
# DeepSeek API
# ──────────────────────────────────────────────
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL: str = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL: str = "deepseek-chat"
DEEPSEEK_TIMEOUT: int = 30  # secondes

# Corrélation et Sentiment
ETH_SYMBOL: str = "ETHUSDm"
NEWS_FEED_URL: str = "https://cryptopanic.com/api/v1/posts/?kind=news&public=true"

# Prompt système fixe (optimisé tokens)
DEEPSEEK_SYSTEM_PROMPT: str = (
    "Expert BTCUSDm trader. Analyze technicals, ETH correlation and news. Respond ONLY in this exact format:\n"
    "DIR=BUY|LOT=0.01|TP=75000|SL=65000|CONF=87|RR=1.5|REASON=3w\n"
    "DIR: BUY/SELL/WAIT. CONF: 0-100. REASON: max 3 words. No other text."
)

# ──────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────
# MetaTrader 5
# ──────────────────────────────────────────────
MT5_LOGIN: int = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")
MT5_SERVER: str = os.getenv("MT5_SERVER", "")
MT5_SYMBOL: str = "BTCUSDm"
MT5_MAGIC: int = 20240115  # Identifiant unique du bot

# Timeframes utilisés pour l'analyse multi-temporelle
MT5_TIMEFRAMES: dict = {
    "M15": None,  # rempli dans utils.py après import MetaTrader5
    "H1":  None,
    "H4":  None,
    "D1":  None,
}

# Nombre de bougies à récupérer par timeframe
CANDLES_COUNT: int = 200

# ──────────────────────────────────────────────
# Gestion du risque
# ──────────────────────────────────────────────
RISK_PERCENT: float = float(os.getenv("RISK_PERCENT", "1.5"))
MAX_DAILY_DRAWDOWN: float = float(os.getenv("MAX_DAILY_DRAWDOWN", "3.0"))
MIN_CONFIDENCE: int = int(os.getenv("MIN_CONFIDENCE", "85"))
MAX_SIMULTANEOUS_TRADES: int = int(os.getenv("MAX_SIMULTANEOUS_TRADES", "2"))
MIN_RR: float = 1.5  # Ratio risque/rendement minimum accepté

# ──────────────────────────────────────────────
# Risque Dynamique (Nouveau)
# ──────────────────────────────────────────────
USE_DYNAMIC_RISK: bool = True
MIN_RISK_PCT: float = 0.5   # Risque min pour CONF = MIN_CONFIDENCE
MAX_RISK_PCT: float = 2.5   # Risque max pour CONF = 100

# ──────────────────────────────────────────────
# Filtre Volatilité ATR (Nouveau)
# ──────────────────────────────────────────────
# Le bot refuse de trader si l'ATR est hors de ces bornes (exprimé en USD pour BTC)
ATR_MIN_THRESHOLD: float = 50.0   # Marché trop plat
ATR_MAX_THRESHOLD: float = 2500.0 # Marché trop volatil (news)

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
    "London":   {"start": 7,  "end": 16},   # 07h00 – 16h00 UTC
    "New York": {"start": 12, "end": 21},   # 12h00 – 21h00 UTC
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
