"""
bot.py — Orchestrateur principal du GOLDBOT.
    - Boucle de trading toutes les 15 minutes (alignée M15)
    - Bot Telegram asynchrone (commandes /start, /stop, /pause, /status, etc.)
    - Watchdog : survie aux exceptions non gérées
    - Résumé journalier à 23h00
    - Gestion propre de l'arrêt (fermeture des trades)

    Compatible : python-telegram-bot >= 21.0
"""

import asyncio
import datetime
import signal
import sys
import threading
import traceback

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

import config
import utils

# ══════════════════════════════════════════════════════════════
# ÉTAT GLOBAL DU BOT
# ══════════════════════════════════════════════════════════════

class BotState:
    """Conteneur thread-safe de l'état du bot partagé entre la boucle et Telegram."""

    ACTIVE  = "actif"
    PAUSED  = "pausé"
    STOPPED = "stoppé"

    def __init__(self) -> None:
        self.status: str = self.STOPPED
        # Statistiques journalières
        self.trades_today:   int   = 0
        self.profit_today:   float = 0.0
        self.start_balance:  float = 0.0
        self.daily_summary_sent: bool = False
        self.known_tickets: list[int] = []
        # Réglages dynamiques (overrides de config.py)
        self.max_risk_pct: float = config.MAX_RISK_PCT
        self.min_confidence: int = config.MIN_CONFIDENCE
        self._lock = asyncio.Lock()

    async def set_status(self, new_status: str) -> None:
        async with self._lock:
            self.status = new_status

    def is_active(self) -> bool:
        return self.status == self.ACTIVE

    def is_paused(self) -> bool:
        return self.status == self.PAUSED

    def is_stopped(self) -> bool:
        return self.status == self.STOPPED


# Instance unique partagée
state = BotState()


# ══════════════════════════════════════════════════════════════
# CYCLE DE TRADING (exécuté toutes les 15 min)
# ══════════════════════════════════════════════════════════════

async def trading_cycle() -> None:
    """
    Un cycle complet de trading :
      1. Vérification connexion MT5
      2. Vérification état du bot
      3. Collecte des données
      4. Filtre pré-IA
      5. Appel DeepSeek (si filtre OK)
      6. Garde-fous + exécution (si signal valide)
      7. Surveillance des trades ouverts
    """
    utils.bot_log.info("═══ Début du cycle de trading ═══")

    # ── 1. Connexion MT5 ────────────────────────────────────────
    if not utils.mt5_ensure_connected():
        msg = "Impossible de se connecter à MT5. Cycle ignoré."
        utils.bot_log.error(msg)
        utils.alert_error(msg)
        return

    # ── 2. État du bot ──────────────────────────────────────────
    if state.is_stopped():
        utils.bot_log.info("Bot arrêté — cycle ignoré.")
        return
    if state.is_paused():
        utils.bot_log.info("Bot en pause — cycle ignoré (trades ouverts conservés).")
        return

    # ── 2b. Limite de positions simultanées ─────────────────────
    if utils.get_open_trades_count() >= config.MAX_SIMULTANEOUS_TRADES:
        utils.bot_log.info("Max trades atteint (%d/%d) — cycle ignoré pour économiser les tokens.",
                           utils.get_open_trades_count(), config.MAX_SIMULTANEOUS_TRADES)
        return

    # ── 3. Collecte des données ─────────────────────────────────
    data = utils.collect_all_data()
    if data is None:
        utils.bot_log.error("Collecte de données échouée. Cycle ignoré.")
        return

    utils.bot_log.info(
        "Données collectées — prix=%.2f | balance=%.2f$",
        data["current_price"], data["balance"],
    )

    # Initialisation de la balance de départ (première fois)
    if state.start_balance == 0.0:
        state.start_balance = data["balance"]

    # ── 4. Filtre pré-IA ────────────────────────────────────────
    ok, reason = utils.pre_ia_filter(data)
    if not ok:
        utils.bot_log.info("Filtre pré-IA → SKIP : %s", reason)
        return
    
    # On transmet les réglages dynamiques (modifiés via Telegram)
    data["min_confidence"] = state.min_confidence
    data["max_risk_pct"] = state.max_risk_pct

    utils.bot_log.info("Filtre pré-IA → OK")

    # ── 5. Compression + appel DeepSeek ─────────────────────────
    compressed = utils.compress_data(data)
    utils.bot_log.info("Données compressées : %s", compressed)

    signal_raw = utils.call_deepseek(compressed)
    if signal_raw is None:
        utils.bot_log.warning("Aucun signal DeepSeek reçu. Cycle ignoré.")
        return

    utils.bot_log.info(
        "Signal DeepSeek : DIR=%s | CONF=%d | RR=%.1f | REASON=%s",
        signal_raw["DIR"], signal_raw["CONF"], signal_raw["RR"], signal_raw["REASON"],
    )

    # ── 6. Garde-fous + exécution ────────────────────────────────
    valid, signal_checked, reject_reason = utils.validate_signal(
        signal_raw, data["balance"], data["current_price"],
        min_conf=state.min_confidence, max_risk=state.max_risk_pct
    )

    if not valid:
        utils.bot_log.info("Signal rejeté par garde-fous : %s", reject_reason)
        return

    utils.bot_log.info("Signal validé — exécution de l'ordre…")
    trade = utils.execute_trade(signal_checked)

    if trade:
        utils.alert_trade_open(trade)
        state.trades_today += 1
        utils.bot_log.info(
            "Ordre exécuté — ticket #%d | %s %.2f lots @ %.2f",
            trade["ticket"], trade["dir"], trade["lot"], trade["price"],
        )
    else:
        utils.bot_log.error("Échec de l'exécution de l'ordre.")
        utils.alert_error("Échec d'exécution d'un ordre MT5 validé.")

    # ── 7. Surveillance des trades ouverts ───────────────────────
    open_positions = utils.monitor_open_trades()
    utils.bot_log.info(
        "%d position(s) ouverte(s) par le bot.", len(open_positions)
    )

    # Mise à jour du P&L journalier (estimation via equity)
    import MetaTrader5 as mt5
    account = mt5.account_info()
    if account and state.start_balance > 0:
        state.profit_today = account.equity - state.start_balance

    utils.bot_log.info("═══ Fin du cycle ═══")


async def monitoring_loop() -> None:
    """
    Boucle haute fréquence (toutes les 10s) pour surveiller 
    la fermeture des trades (TP/SL) et la gestion active (BE/Trailing).
    """
    utils.bot_log.info("Boucle de surveillance (Gestion Active + TP/SL) démarrée.")
    while True:
        try:
            if state.is_active() and utils.mt5_ensure_connected():
                # 1. Récupérer les positions pour la gestion active
                positions = utils.get_open_positions()
                for p in positions:
                    utils.process_active_trade_management(p)

                # 2. Vérifier si des positions ont été fermées (TP/SL/Manuel)
                state.known_tickets = utils.check_and_alert_closed_trades(state.known_tickets)
        except Exception as exc:
            utils.bot_log.error("Erreur critique dans monitoring_loop : %s", exc)
            await asyncio.sleep(30) # Plus long délai en cas de crash répété
        
        await asyncio.sleep(10)  # Check toutes les 10 secondes


# ══════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE (tâche asyncio)
# ══════════════════════════════════════════════════════════════

async def main_loop() -> None:
    """
    Boucle infinie alignée sur les intervalles M15 (0, 15, 30, 45 min).
    Inclut un watchdog : toute exception est loggée et le cycle redémarre.
    """
    utils.bot_log.info("Boucle principale démarrée. Intervalle : %d min.",
                       config.CYCLE_INTERVAL_MINUTES)

    while True:
        try:
            # ── Résumé journalier à 23h00 ───────────────────────────
            now = datetime.datetime.now()
            if now.hour == config.DAILY_SUMMARY_HOUR and not state.daily_summary_sent:
                try:
                    import MetaTrader5 as mt5
                    account = mt5.account_info()
                    if account:
                        utils.alert_daily_summary(
                            account.balance, account.equity,
                            state.trades_today, state.profit_today
                        )
                    state.daily_summary_sent = True
                    state.trades_today  = 0
                    state.profit_today  = 0.0
                    state.start_balance = account.balance if account else 0.0
                except Exception as e:
                    utils.bot_log.error("Erreur lors de la génération du résumé journalier : %s", e)

            elif now.hour != config.DAILY_SUMMARY_HOUR:
                state.daily_summary_sent = False  # Réinitialisé pour le lendemain

            # ── Cycle de trading ─────────────────────────────────────
            await trading_cycle()

        except Exception as exc:
            tb = traceback.format_exc()
            utils.bot_log.error("ERREUR CRITIQUE dans main_loop :\n%s", tb)
            utils.alert_error(f"⚠️ Watchdog main_loop relancé après erreur :\n<code>{str(exc)[:200]}</code>")
            await asyncio.sleep(10) # Petit délai avant de reprendre

        # ── Attente jusqu'au prochain multiple de 15 min ─────────
        try:
            now    = datetime.datetime.now()
            minute = now.minute
            second = now.second
            wait_s = (config.CYCLE_INTERVAL_MINUTES - (minute % config.CYCLE_INTERVAL_MINUTES)) * 60 - second
            if wait_s <= 0:
                wait_s = config.CYCLE_INTERVAL_MINUTES * 60

            utils.bot_log.info("Prochain cycle dans %.0f secondes.", wait_s)
            await asyncio.sleep(wait_s)
        except Exception:
            await asyncio.sleep(60)


# ══════════════════════════════════════════════════════════════
# COMMANDES TELEGRAM
# ══════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start — Démarre le bot de trading."""
    if not _is_authorized(update):
        return
    if state.is_active():
        await update.message.reply_text("✅ Le bot est déjà actif.")
        return

    # Connexion MT5 au démarrage
    if not utils.mt5_ensure_connected():
        await update.message.reply_text("❌ Impossible de connecter MT5. Vérifiez les credentials.")
        return

    await state.set_status(BotState.ACTIVE)
    import MetaTrader5 as mt5
    account = mt5.account_info()
    if account:
        state.start_balance = account.balance

    msg = (
        f"🚀 <b>GOLDBOT démarré</b>\n"
        f"💰 Balance : {account.balance:.2f} USD\n"
        f"🏦 Compte   : #{config.MT5_LOGIN}\n"
        f"📡 Serveur  : {config.MT5_SERVER}\n"
        f"⏱ Cycle    : toutes les {config.CYCLE_INTERVAL_MINUTES} minutes"
    )
    await update.message.reply_text(msg, parse_mode="HTML")
    utils.bot_log.info("Bot démarré via commande Telegram /start.")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/stop — Stoppe le bot et ferme les trades ouverts."""
    if not _is_authorized(update):
        return
    if state.is_stopped():
        await update.message.reply_text("🛑 Le bot est déjà arrêté.")
        return

    await update.message.reply_text("⏳ Fermeture de toutes les positions en cours…")
    closed = utils.close_all_positions()
    await state.set_status(BotState.STOPPED)

    msg = (
        f"🛑 <b>GOLDBOT arrêté</b>\n"
        f"Positions fermées : {closed}"
    )
    await update.message.reply_text(msg, parse_mode="HTML")
    utils.bot_log.info("Bot arrêté via commande Telegram /stop. %d position(s) fermée(s).", closed)


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/pause — Suspend le bot (trades ouverts conservés)."""
    if not _is_authorized(update):
        return
    if state.is_paused():
        await update.message.reply_text("⏸ Le bot est déjà en pause.")
        return
    await state.set_status(BotState.PAUSED)
    await update.message.reply_text(
        "⏸ <b>Bot mis en pause.</b>\nLes trades ouverts sont conservés.\n"
        "Utilisez /start pour reprendre.",
        parse_mode="HTML"
    )
    utils.bot_log.info("Bot mis en pause via /pause.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/status — Affiche l'état actuel du bot."""
    if not _is_authorized(update):
        return

    emoji = {"actif": "✅", "pausé": "⏸", "stoppé": "🛑"}.get(state.status, "❓")
    session = utils.get_current_session()

    import MetaTrader5 as mt5
    connected_str = "✅ Connecté" if mt5.account_info() else "❌ Déconnecté"

    open_count = utils.get_open_trades_count()
    dd = utils.get_daily_drawdown_pct()

    msg = (
        f"📊 <b>STATUT GOLDBOT</b>\n"
        f"État      : {emoji} {state.status.capitalize()}\n"
        f"MT5       : {connected_str}\n"
        f"Session   : 🕐 {session}\n"
        f"Positions : {open_count}/{config.MAX_SIMULTANEOUS_TRADES}\n"
        f"Drawdown  : {dd:.2f}% (max {config.MAX_DAILY_DRAWDOWN}%)\n"
        f"Trades/j  : {state.trades_today}\n"
        f"P&L/j     : {state.profit_today:+.2f} USD"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/balance — Affiche le solde et le drawdown journalier."""
    if not _is_authorized(update):
        return

    import MetaTrader5 as mt5
    account = mt5.account_info()
    if account is None:
        await update.message.reply_text("❌ MT5 non connecté.")
        return

    dd = utils.get_daily_drawdown_pct()
    msg = (
        f"💰 <b>BALANCE</b>\n"
        f"Balance  : {account.balance:.2f} USD\n"
        f"Equity   : {account.equity:.2f} USD\n"
        f"Marge    : {account.margin:.2f} USD\n"
        f"Libre    : {account.margin_free:.2f} USD\n"
        f"Drawdown : {dd:.2f}% / {config.MAX_DAILY_DRAWDOWN}%"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help — Affiche l'aide et la liste des commandes."""
    if not _is_authorized(update): return
    msg = (
        "📖 <b>GUIDE DES COMMANDES</b>\n\n"
        "<b>🕹️ Base :</b>\n"
        "/start, /stop, /pause, /status, /help\n\n"
        "<b>📊 Trading :</b>\n"
        "/analyze - Analyser maintenant\n"
        "/trades - Positions ouvertes\n"
        "/balance - Solde MT5\n"
        "/close &lt;ticket&gt; - Fermer un trade\n"
        "/breakeven &lt;ticket&gt; - Sécuriser\n\n"
        "<b>⚙️ Tactique :</b>\n"
        "/mode &lt;safe|normal|aggro&gt;\n"
        "/setrisk &lt;%&gt;, /setconf &lt;%&gt;, /clearstats\n\n"
        "<b>🔍 Système :</b>\n"
        "/news, /log, /panic"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/trades — Liste les trades ouverts."""
    if not _is_authorized(update):
        return
    msg = utils.format_open_positions_message()
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_log(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/log — Envoie les 20 dernières lignes du fichier bot.log."""
    if not _is_authorized(update):
        return
    tail = utils.get_log_tail(config.LOG_MAX_LINES_TELEGRAM)
    if not tail.strip():
        await update.message.reply_text("Log vide ou fichier introuvable.")
        return
    # Découper si trop long pour Telegram (4096 chars max)
    max_len = 4000
    chunks = [tail[i:i + max_len] for i in range(0, len(tail), max_len)]
    for chunk in chunks:
        await update.message.reply_text(f"<pre>{chunk}</pre>", parse_mode="HTML")


async def cmd_setrisk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/setrisk <float> — Change le risque dynamique max."""
    if not _is_authorized(update) or not context.args: return
    try:
        new_risk = float(context.args[0])
        if 0.1 <= new_risk <= 5.0:
            state.max_risk_pct = new_risk
            await update.message.reply_text(f"🎯 Risque max mis à jour : <b>{new_risk}%</b>", parse_mode="HTML")
        else:
            await update.message.reply_text("❌ Valeur invalide (doit être entre 0.1 et 5.0)")
    except ValueError:
        await update.message.reply_text("❌ Format : /setrisk 1.5")

async def cmd_setconf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/setconf <int> — Change la confiance min."""
    if not _is_authorized(update) or not context.args: return
    try:
        new_conf = int(context.args[0])
        if 50 <= new_conf <= 100:
            state.min_confidence = new_conf
            await update.message.reply_text(f"🧠 Confiance min mise à jour : <b>{new_conf}%</b>", parse_mode="HTML")
        else:
            await update.message.reply_text("❌ Valeur invalide (entre 50 et 100)")
    except ValueError:
        await update.message.reply_text("❌ Format : /setconf 85")

async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/mode <safe|normal|aggro> — Profils de risque prédéfinis."""
    if not _is_authorized(update) or not context.args: return
    mode = context.args[0].lower()
    if mode == "safe":
        state.max_risk_pct, state.min_confidence = 0.5, 95
    elif mode == "normal":
        state.max_risk_pct, state.min_confidence = 1.5, 87
    elif mode == "aggro":
        state.max_risk_pct, state.min_confidence = 3.0, 80
    else:
        await update.message.reply_text("❌ Modes valides : safe, normal, aggro")
        return
    await update.message.reply_text(f"🎭 Mode <b>{mode.upper()}</b> activé\n(Risque {state.max_risk_pct}% | Confiance {state.min_confidence}%)", parse_mode="HTML")

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/analyze — Force un cycle de trading immédiat."""
    if not _is_authorized(update): return
    await update.message.reply_text("🔍 Analyse immédiate demandée...")
    await trading_cycle()

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/panic — Arrêt total, tout ferme, bot stoppé."""
    if not _is_authorized(update): return
    await state.set_status(BotState.STOPPED)
    closed = utils.close_all_positions()
    await update.message.reply_text(f"🚨 <b>PANIC BUTTON ACTIVATED</b>\nPositions fermées : {closed}\nBot stoppé.", parse_mode="HTML")

async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/close <ticket> — Ferme un trade spécifique."""
    if not _is_authorized(update) or not context.args: return
    try:
        ticket = int(context.args[0])
        if utils.close_position_by_ticket(ticket):
            await update.message.reply_text(f"✅ Position #{ticket} fermée.")
        else:
            await update.message.reply_text(f"❌ Impossible de fermer #{ticket}.")
    except ValueError:
        await update.message.reply_text("❌ Format : /close 123456")

async def cmd_breakeven(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/breakeven <ticket> — Passe un trade en sécurité."""
    if not _is_authorized(update) or not context.args: return
    try:
        ticket = int(context.args[0])
        import MetaTrader5 as mt5
        pos = mt5.positions_get(ticket=ticket)
        if pos:
            if utils.modify_position_sl(ticket, pos[0].price_open):
                await update.message.reply_text(f"🛡 Position #{ticket} mise à Break-even.")
            else:
                await update.message.reply_text(f"❌ Échec modification #{ticket}.")
        else:
            await update.message.reply_text("❌ Position introuvable.")
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur : {e}")

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/news — Affiche les dernières actualités suivies."""
    if not _is_authorized(update): return
    news = utils.fetch_market_news()
    await update.message.reply_text(f"🌍 <b>FIL D'ACTUALITÉ :</b>\n{news}", parse_mode="HTML")

async def cmd_clearstats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/clearstats — Reset les stats journalières."""
    if not _is_authorized(update): return
    state.trades_today = 0
    state.profit_today = 0.0
    await update.message.reply_text("🧹 Statistiques journalières réinitialisées.")


def _is_authorized(update: Update) -> bool:
    """
    Vérifie que la commande provient du TELEGRAM_CHAT_ID autorisé.
    Protège contre les commandes non sollicitées.
    """
    chat_id = str(update.effective_chat.id)
    if chat_id != str(config.TELEGRAM_CHAT_ID):
        utils.bot_log.warning(
            "Commande refusée depuis chat_id=%s (attendu=%s)",
            chat_id, config.TELEGRAM_CHAT_ID,
        )
        return False
    return True


# ══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

def _run_trading_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Lance la boucle de trading dans un thread séparé.
    PTB v21 occupe la boucle principale avec run_polling(),
    donc on exécute main_loop() dans son propre event loop.
    """
    asyncio.set_event_loop(loop)
    while True:
        try:
            # Lancement de la surveillance haute fréquence en arrière-plan
            loop.create_task(monitoring_loop())
            # Lancement de la boucle principale de trading (cycle 15 min)
            loop.run_until_complete(main_loop())
        except Exception as exc:
            utils.bot_log.critical("ERREUR FATALE SYSTEME : Redémarrage du thread trading dans 30s... %s", exc, exc_info=True)
            utils.alert_error(f"💀 Thread trading crashé : {str(exc)[:200]}\nTentative de redémarrage automatique...")
            import time
            time.sleep(30) # Attente avant restart total du thread logic


async def telegram_error_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les erreurs dans le bot Telegram pour éviter qu'il ne s'arrête."""
    utils.bot_log.error("Erreur Telegram non gérée : %s", context.error)
    # Envoi d'une alerte si possible, sans trop spammer
    try:
        if "Network is unreachable" not in str(context.error):
            utils.alert_error(f"⚠️ Erreur Telegram : <code>{str(context.error)[:100]}</code>")
    except: pass


def main() -> None:
    """
    Fonction principale :
      - Initialise MT5
      - Lance le bot Telegram (run_polling bloquant — PTB v21)
      - Lance la boucle de trading dans un thread parallèle
    """
    utils.bot_log.info("╔══════════════════════════════╗")
    utils.bot_log.info("║      GOLDBOT  démarrage      ║")
    utils.bot_log.info("╚══════════════════════════════╝")

    # ── Connexion MT5 initiale ──────────────────────────────────
    if not utils.mt5_connect():
        utils.bot_log.critical("Impossible de démarrer MT5. Arrêt.")
        utils.send_telegram("❌ <b>GOLDBOT</b> : Impossible de démarrer MT5. Vérifiez les credentials.")
        sys.exit(1)

    # ── Vérification token Telegram ─────────────────────────────
    if not config.TELEGRAM_BOT_TOKEN:
        utils.bot_log.critical("TELEGRAM_BOT_TOKEN manquant dans .env. Arrêt.")
        sys.exit(1)

    # ── Construction de l'application Telegram ──────────────────
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Enregistrement des commandes
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("stop",    cmd_stop))
    app.add_handler(CommandHandler("pause",   cmd_pause))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CommandHandler("balance", cmd_balance))
    app.add_handler(CommandHandler("trades",  cmd_trades))
    app.add_handler(CommandHandler("log",     cmd_log))
    
    # Nouvelles commandes tactiques
    app.add_handler(CommandHandler("setrisk",    cmd_setrisk))
    app.add_handler(CommandHandler("setconf",    cmd_setconf))
    app.add_handler(CommandHandler("mode",       cmd_mode))
    app.add_handler(CommandHandler("analyze",    cmd_analyze))
    app.add_handler(CommandHandler("panic",      cmd_panic))
    app.add_handler(CommandHandler("close",      cmd_close))
    app.add_handler(CommandHandler("breakeven",  cmd_breakeven))
    app.add_handler(CommandHandler("news",       cmd_news))
    app.add_handler(CommandHandler("clearstats", cmd_clearstats))

    # Gestionnaire d'erreurs global pour Telegram
    app.add_error_handler(telegram_error_handler)

    # ── Notification de démarrage (HTTP synchrone) ──────────────
    import MetaTrader5 as mt5
    account = mt5.account_info()
    bal_str = f"{account.balance:.2f} USD" if account else "N/A"
    utils.send_telegram(
        f"🤖 <b>GOLDBOT en ligne</b>\n"
        f"Compte : #{config.MT5_LOGIN} | Balance : {bal_str}\n"
        f"Envoyez /start pour lancer le trading."
    )

    # ── Thread de trading (event loop dédié) ────────────────────
    trading_loop = asyncio.new_event_loop()
    trading_thread = threading.Thread(
        target=_run_trading_loop,
        args=(trading_loop,),
        daemon=True,
        name="TradingLoop",
    )
    trading_thread.start()
    utils.bot_log.info("Thread de trading démarré.")

    # ── Polling Telegram (bloquant — PTB v21) ───────────────────
    utils.bot_log.info("Lancement du polling Telegram (PTB v21)…")
    try:
        app.run_polling(drop_pending_updates=True)
    finally:
        # Arrêt propre à la sortie du polling
        trading_loop.call_soon_threadsafe(trading_loop.stop)
        trading_thread.join(timeout=10)
        utils.mt5_disconnect()
        utils.bot_log.info("GOLDBOT arrêté proprement.")


def _handle_signal(sig, frame) -> None:
    """Gestionnaire de signaux OS (SIGINT, SIGTERM) pour arrêt propre."""
    utils.bot_log.info("Signal %s reçu — arrêt du bot…", sig)
    utils.send_telegram("🛑 <b>GOLDBOT</b> : Arrêt OS reçu — fermeture propre.")
    utils.close_all_positions()
    utils.mt5_disconnect()
    sys.exit(0)


if __name__ == "__main__":
    # Capture CTRL+C et SIGTERM
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Vérification que les variables critiques sont présentes
    missing = []
    for var in ("DEEPSEEK_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"):
        val = getattr(config, var, None)
        if not val or str(val) in ("", "0"):
            missing.append(var)

    if missing:
        print(f"[ERREUR] Variables manquantes dans .env : {', '.join(missing)}")
        print("Copiez .env.example vers .env et remplissez toutes les valeurs.")
        sys.exit(1)

    # Lancement (PTB v21 — main() est synchrone, run_polling gère son propre loop)
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        utils.bot_log.critical("Erreur fatale au démarrage : %s", e, exc_info=True)
        utils.send_telegram(f"💀 <b>GOLDBOT crash fatal</b> : {str(e)[:300]}")
        sys.exit(1)
