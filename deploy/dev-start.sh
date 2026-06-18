#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# dev-start.sh — Arranque del backend en modo desarrollo (local).
#
# Verifica Ollama, activa venv, arranca uvicorn con --reload y
# espera a que /api/health responda.
#
# Uso:
#   ./deploy/dev-start.sh                    # arranca en localhost:8000
#   ./deploy/dev-start.sh --port 9000        # puerto custom
#   ./deploy/dev-start.sh --no-ollama-check  # saltar check de Ollama
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
HEALTH_URL="http://localhost:${PORT:-8000}/api/health"

PORT=8000
CHECK_OLLAMA=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --no-ollama-check) CHECK_OLLAMA=false; shift ;;
        *) echo "Opción desconocida: $1"; exit 1 ;;
    esac
done

HEALTH_URL="http://localhost:${PORT}/api/health"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
info() { echo -e "${CYAN}[..]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*" >&2; }

# ── 1. Verificar Ollama ──
if $CHECK_OLLAMA; then
    info "Verificando Ollama..."
    if curl -sf --max-time 3 http://localhost:11434/api/tags &>/dev/null; then
        MODELS=$(curl -sf http://localhost:11434/api/tags 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(', '.join(m['name'] for m in d.get('models',[])))" 2>/dev/null || echo "?")
        ok "Ollama activo. Modelos: $MODELS"
    else
        warn "Ollama no responde en localhost:11434"
        warn "Arrancar con: ollama serve"
        warn "Continuando de todas formas (--no-ollama-check para suprimir)..."
    fi
fi

# ── 2. Activar venv ──
if [[ -d "$BACKEND_DIR/venv" ]]; then
    info "Activando venv..."
    source "$BACKEND_DIR/venv/bin/activate"
    ok "venv activado: $(python --version)"
else
    warn "venv no encontrado en $BACKEND_DIR/venv"
    info "Creando venv..."
    python3 -m venv "$BACKEND_DIR/venv"
    source "$BACKEND_DIR/venv/bin/activate"
    pip install --upgrade pip
    pip install -r "$BACKEND_DIR/requirements.txt"
    ok "venv creado y dependencias instaladas"
fi

# ── 3. Verificar .env ──
if [[ ! -f "$BACKEND_DIR/.env" ]]; then
    warn ".env no encontrado. Creando con defaults (Ollama local, gemma4:12b)..."
    cat > "$BACKEND_DIR/.env" << 'ENVEOF'
LLM_PROVIDER=local
OPENROUTER_BASE_URL=http://localhost:11434/v1
OPENROUTER_MODEL=gemma4:12b
OPENROUTER_API_KEY=ollama
MAX_FILE_SIZE_MB=20
RAG_K=10
RAG_MIN_SCORE=0.10
RAG_MAX_CONTEXT_CHARS=12000
ENVEOF
    ok ".env creado"
fi

# ── 4. Arrancar uvicorn ──
cd "$BACKEND_DIR"
info "Arrancando uvicorn en http://localhost:$PORT (reload=true)..."
info "Ctrl+C para detener"
echo ""

export OLLAMA_NUM_CTX=${OLLAMA_NUM_CTX:-32768}
exec python -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
