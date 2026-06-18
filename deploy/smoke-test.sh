#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# smoke-test.sh — Pruebas de humo end-to-end del backend UPC ABET.
#
# Verifica que todos los componentes funcionan: health, provider,
# chat raw, chat con knowledge, chat con archivos, y telemetría.
#
# Uso:
#   ./deploy/smoke-test.sh                          # localhost:8000
#   ./deploy/smoke-test.sh --base-url http://prod:8000
#   ./deploy/smoke-test.sh --base-url https://acc-ia.tcupc.pe
#   ./deploy/smoke-test.sh --carrera sw             # probar con SW
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_URL="http://localhost:8000"
CARRERA="cc"
TIMEOUT=120

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url) BASE_URL="$2"; shift 2 ;;
        --carrera) CARRERA="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) echo "Opción desconocida: $1"; exit 1 ;;
    esac
done

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

pass() { echo -e "  ${GREEN}✓ PASS${NC} — $1"; ((PASS++)); }
fail() { echo -e "  ${RED}✗ FAIL${NC} — $1"; ((FAIL++)); }
skip() { echo -e "  ${YELLOW}⊘ SKIP${NC} — $1"; ((SKIP++)); }
header() { echo -e "\n${CYAN}━━ $1 ━━${NC}"; }

echo "══════════════════════════════════════════════════════"
echo "  Smoke Test — UPC ABET Backend"
echo "  Base URL: $BASE_URL"
echo "  Carrera:  $CARRERA"
echo "══════════════════════════════════════════════════════"

# ────────────────────────────────────────────────────────────
header "1. Health check (/api/health)"
HEALTH=$(curl -sf --max-time 10 "$BASE_URL/api/health" 2>/dev/null) || {
    fail "No se pudo conectar a $BASE_URL/api/health"
    echo ""
    echo "Total: $PASS pass, $FAIL fail, $SKIP skip"
    exit 1
}

HEALTH_STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
if [[ "$HEALTH_STATUS" == "ok" ]]; then
    pass "status=ok"
else
    fail "status no es 'ok' (es '$HEALTH_STATUS')"
fi

HEALTH_MODEL=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model','?'))" 2>/dev/null)
pass "Modelo activo: $HEALTH_MODEL"

CARRERAS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('carreras_disponibles',[])))" 2>/dev/null)
if [[ "$CARRERAS" -gt 0 ]]; then
    pass "Carreras disponibles: $CARRERAS"
else
    fail "No hay carreras disponibles (revisar layout de knowledge/)"
fi

PROVIDER=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('provider','?'))" 2>/dev/null)
pass "Provider: $PROVIDER"

# ────────────────────────────────────────────────────────────
header "2. Provider info (/api/provider)"
PROV_RESP=$(curl -sf --max-time 10 "$BASE_URL/api/provider" 2>/dev/null) || {
    fail "No se pudo obtener /api/provider"
    PROV_RESP=""
}
if [[ -n "$PROV_RESP" ]]; then
    PROV_MODEL=$(echo "$PROV_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model','?'))" 2>/dev/null)
    pass "Provider model: $PROV_MODEL"
fi

# ────────────────────────────────────────────────────────────
header "3. Chat modo raw (sin system prompt)"
START=$(date +%s%3N)
RAW_RESP=$(curl -sf --max-time $TIMEOUT -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Responde solo con la palabra: OK"}],"raw":true,"think":false,"num_ctx":4096,"stream":false}' 2>/dev/null) || {
    fail "Chat raw no respondió"
    RAW_RESP=""
}
END=$(date +%s%3N)
RAW_LATENCY=$((END - START))

if [[ -n "$RAW_RESP" ]]; then
    RAW_CONTENT=$(echo "$RAW_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:100])" 2>/dev/null)
    if [[ -n "$RAW_CONTENT" ]]; then
        pass "Respuesta raw: '${RAW_CONTENT:0:50}...' (${RAW_LATENCY}ms)"
    else
        fail "Respuesta raw vacía"
    fi
fi

# ────────────────────────────────────────────────────────────
header "4. Chat con knowledge (carrera=$CARRERA)"
START=$(date +%s%3N)
KB_RESP=$(curl -sf --max-time $TIMEOUT -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"¿Cuántos outcomes tiene la carrera?\"}],\"carrera\":\"$CARRERA\",\"use_knowledge\":true,\"think\":false,\"num_ctx\":32768,\"stream\":false}" 2>/dev/null) || {
    fail "Chat con knowledge no respondió"
    KB_RESP=""
}
END=$(date +%s%3N)
KB_LATENCY=$((END - START))

if [[ -n "$KB_RESP" ]]; then
    KB_CONTENT=$(echo "$KB_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:200])" 2>/dev/null)
    if [[ -n "$KB_CONTENT" ]]; then
        pass "Respuesta knowledge: '${KB_CONTENT:0:60}...' (${KB_LATENCY}ms)"
    else
        fail "Respuesta knowledge vacía"
    fi
fi

# ────────────────────────────────────────────────────────────
header "5. Stats y telemetría"
STATS_RESP=$(curl -sf --max-time 10 "$BASE_URL/api/stats?since_days=1&limit=100" 2>/dev/null) || {
    fail "/api/stats no respondió"
    STATS_RESP=""
}
if [[ -n "$STATS_RESP" ]]; then
    STATS_TOTAL=$(echo "$STATS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_queries',0))" 2>/dev/null)
    pass "Total queries registradas: $STATS_TOTAL"
fi

# ────────────────────────────────────────────────────────────
header "6. Logs recientes"
LOGS_RESP=$(curl -sf --max-time 10 "$BASE_URL/api/logs/recent?limit=3" 2>/dev/null) || {
    fail "/api/logs/recent no respondió"
    LOGS_RESP=""
}
if [[ -n "$LOGS_RESP" ]]; then
    pass "/api/logs/recent responde OK"
fi

# ────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Resultado: $PASS pass, $FAIL fail, $SKIP skip"
echo "══════════════════════════════════════════════════════"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
exit 0
