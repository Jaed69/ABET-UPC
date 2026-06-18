#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# health-check.sh — Healthcheck periódico del backend UPC ABET.
#
# Se ejecuta vía cron cada 1 minuto. Si /api/health falla 3 veces
# seguidas, reinicia el servicio automáticamente.
#
# Instalar en cron:
#   * * * * * /opt/upc-abet/deploy/health-check.sh >> /opt/upc-abet/backend/logs/health-check.log 2>&1
#
# O con systemd timer (recomendado):
#   sudo cp deploy/upc-abet-health.timer /etc/systemd/system/
#   sudo systemctl enable --now upc-abet-health.timer
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_DIR/backend/logs/health.jsonl"
FAIL_FILE="/tmp/upc-abet-health-fails"
SERVICE="upc-abet-backend"
HEALTH_URL="http://127.0.0.1:8000/api/health"
MAX_FAILS=3

mkdir -p "$(dirname "$LOG_FILE")"

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_MS=$(date +%s%3N)

RESP=$(curl -sf --max-time 10 "$HEALTH_URL" 2>/dev/null) || RESP=""
END_MS=$(date +%s%3N)
LATENCY=$((END_MS - START_MS))

if [[ -n "$RESP" ]]; then
    STATUS="ok"
    FAILS=0
    echo "$FAILS" > "$FAIL_FILE" 2>/dev/null || true
else
    FAILS=$(($(cat "$FAIL_FILE" 2>/dev/null || echo 0) + 1))
    echo "$FAILS" > "$FAIL_FILE"
    STATUS="fail"
fi

# Log JSONL (1 línea por check)
echo "{\"ts\":\"$TS\",\"status\":\"$STATUS\",\"latency_ms\":$LATENCY,\"fails\":$FAILS}" >> "$LOG_FILE"

# Auto-restart si falla demasiado
if [[ "$STATUS" == "fail" && "$FAILS" -ge "$MAX_FAILS" ]]; then
    echo "{\"ts\":\"$TS\",\"event\":\"auto_restart\",\"reason\":\"${MAX_FAILS}_consecutive_fails\"}" >> "$LOG_FILE"
    sudo systemctl restart "$SERVICE" 2>/dev/null || true
    echo 0 > "$FAIL_FILE"
fi

# Exit code para cron (0 = OK, 1 = warn pero no crítico)
[[ "$STATUS" == "ok" ]] && exit 0 || exit 1
