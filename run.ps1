# FXBOT EUR/USD — Script de démarrage  # MIGRATION-EURUSD
# Usage : .\run.ps1

Write-Host "=== FXBOT EUR/USD ===" -ForegroundColor Cyan  # MIGRATION-EURUSD

# Vérification .env
if (-not (Test-Path ".env")) {
    Write-Host "[ERREUR] Fichier .env manquant. Copiez .env.example et remplissez les valeurs." -ForegroundColor Red
    exit 1
}

# Vérification que les clés essentielles ne sont pas vides
$envContent = Get-Content ".env"
$requiredKeys = @("DEEPSEEK_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER")
$missing = @()

foreach ($key in $requiredKeys) {
    $line = $envContent | Where-Object { $_ -match "^$key=(.+)$" }
    if (-not $line) {
        $missing += $key
    }
}

if ($missing.Count -gt 0) {
    Write-Host "[ERREUR] Variables manquantes dans .env :" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    exit 1
}

Write-Host "[OK] Configuration .env validée" -ForegroundColor Green
Write-Host "[OK] Démarrage du bot..." -ForegroundColor Green
Write-Host "     Appuyez sur CTRL+C pour arrêter proprement" -ForegroundColor Gray
Write-Host ""

python bot.py
