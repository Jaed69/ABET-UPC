#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# switch-model.sh — Cambia el modelo LLM de Ollama en caliente.
#
# Uso:
#   ./deploy/switch-model.sh <modelo>           # cambia y reinicia
#   ./deploy/switch-model.sh <modelo> --no-pull # sin descargar
#   ./deploy/switch-model.sh --list             # lista modelos locales
#   ./deploy/switch-model.sh                    # muestra ayuda
#
# Ejemplos:
#   ./deploy/switch-model.sh qwen2.5:7b
#   ./deploy/switch-model.sh gemma4:12b
#   ./deploy/switch-model.sh llama3.1:8b
#
# Qué hace:
#   1. (opcional) ollama pull <modelo> — descarga si no está
#   2. Edita OPENROUTER_MODEL en backend/.env
#   3. systemctl restart upc-abet-backend
#   4. Smoke test: verifica que /api/health responde con el modelo nuevo
#
# Requisitos: ejecutar como usuario con sudo para systemctl.
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_DIR/backend/.env"
SERVICE="upc-abet-backend"
HEALTH_URL="http://127.0.0.1:8000/api/health"

# ── Colores ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
info() { echo -e "${CYAN}[..]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*" >&2; }

# ── Ayuda / lista ──
if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Uso: $0 <modelo> [--no-pull]"
    echo "     $0 --list   (lista modelos locales de Ollama)"
    echo ""
    echo "Modelos recomendados (GPU 8GB):"
    echo "  qwen2.5:7b     — rápido, 4.5GB VRAM"
    echo "  llama3.1:8b    — balance, 5GB VRAM"
    echo "  gemma4:12b     — máxima calidad, 7GB VRAM"
    exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
    info "Modelos locales de Ollama:"
    ollama list 2>/dev/null || { err "No se pudo conectar a Ollama"; exit 1; }
    exit 0
fi

MODEL="$1"
DO_PULL=true
if [[ "${2:-}" == "--no-pull" ]]; then
    DO_PULL=false
fi

# ── Validaciones ──
if [[ ! -f "$ENV_FILE" ]]; then
    err "No se encontró $ENV_FILE"
    exit 1
fi

if ! command -v ollama &>/dev/null; then
    err "Ollama no está instalado o no está en PATH"
    exit 1
fi

# ── 1. Pull (opcional) ──
if $DO_PULL; then
    info "Descargando modelo '$MODEL' (si no está)..."
    if ollama pull "$MODEL"; then
        ok "Modelo '$MODEL' disponible localmente"
    else
        err "Falló ollama pull '$MODEL'"
        exit 1
    fi
else
    info "Saltando pull (--no-pull)"
fi

# ── 2. Editar .env ──
CURRENT=$(grep '^OPENROUTER_MODEL=' "$ENV_FILE" | head -1 | cut -d= -f2-)
info "Modelo actual en .env: $CURRENT"
info "Cambiando a: $MODEL"

# Reemplazar la línea OPENROUTER_MODEL=...
if grep -q '^OPENROUTER_MODEL=' "$ENV_FILE"; then
    sed -i "s|^OPENROUTER_MODEL=.*|OPENROUTER_MODEL=$MODEL|" "$ENV_FILE"
else
    echo "OPENROUTER_MODEL=$MODEL" >> "$ENV_FILE"
fi
ok ".env actualizado: OPENROUTER_MODEL=$MODEL"

# ── 3. Reiniciar backend ──
info "Reiniciando servicio $SERVICE..."
if sudo systemctl restart "$SERVICE" 2>/dev/null; then
    ok "Servicio reiniciado"
else
    warn "No se pudo reiniciar via systemctl (¿no es prod?). Continuando..."
fi

# ── 4. Smoke test ──
info "Esperando a que el backend responda..."
sleep 2
for i in $(seq 1 10); do
    if RESP=$(curl -sf --max-time 5 "$HEALTH_URL" 2>/dev/null); then
        HEALTH_MODEL=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model','?'))" 2>/dev/null || echo "?")
        if [[ "$HEALTH_MODEL" == "$MODEL" ]]; then
            ok "Smoke test OK — /api/health reporta modelo='$HEALTH_MODEL'"
            echo ""
            echo -e "${GREEN}✓ Cambio de modelo completado: $CURRENT → $MODEL${NC}"
            exit 0
        else
            warn "/api/health reporta modelo='$HEALTH_MODEL' (esperado '$MODEL')"
        fi
    fi
    sleep 1
done

err "El backend no respondió correctamente tras 10s"
err "Revisa: journalctl -u $SERVICE -f"
exit 1
