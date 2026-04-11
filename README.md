# 🏴‍☠️ GoldRoger — BTCUSDm AI Trading Bot

GoldRoger est un bot de trading automatique spécialisé sur le Bitcoin (BTCUSDm), propulsé par l'IA **DeepSeek** et intégré à **MetaTrader 5**.

## 🚀 Fonctionnalités
- **Analyse Multi-Timeframe** : M15, H1, H4, D1.
- **IA DeepSeek** : Analyse technique compressée pour une prise de décision intelligente.
- **Contrôleur Telegram** : Commandes `/start`, `/stop`, `/status`, `/balance`, `/trades`, `/log`.
- **Gestion du Risque** : Drawdown journalier max, calcul de lot dynamique, alignement des tendances.
- **Session Trading** : Focus sur les sessions de Londres et New York.

## 🛠 Installation

1. **Cloner le projet** :
   ```bash
   git clone https://github.com/Ulyanovskk/GoldRoger.git
   cd GoldRoger
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurer les secrets** :
   - Copiez `.env.example` vers `.env`.
   - Remplissez vos clés API (DeepSeek, Telegram) et vos accès MT5.

4. **Lancer le bot** :
   - Sur Windows : `.\run.ps1` ou `python bot.py`

## 📱 Commandes Telegram
- `/start` : Lancer le bot.
- `/stop` : Arrêter et fermer les positions.
- `/status` : État du bot et performance.
- `/log` : Voir les derniers logs.

## ⚠️ Avertissement
Le trading comporte des risques. Testez toujours ce bot sur un **compte Démo** avant de passer en réel.
