#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# ollama-tune.sh — Tuning automático de Ollama según VRAM de la GPU.
#
# Detecta la VRAM disponible y aplica parámetros óptimos:
#   - OLLAMA_KEEP_ALIVE: tiempo que el modelo stays cargado
#   - OLLAMA_NUM_PARALLEL: requests paralelos (1 = máx VRAM por request)
#   - OLLAMA_MAX_LOADED_MODELS: modelos simultáneos en VRAM
#   - OLLAMA_FLASH_ATTENTION: aceleración si la GPU lo soporta
#
# Crea un override de systemd para ollama.service.
#
# Uso:
#   ./deploy/ollama-tune.sh                # detecta y aplica
#   ./deploy/ollama-tune.sh --vram 24000   # forzar VRAM (MB)
#   ./deploy/ollama-tune.sh --show         # solo mostrar config actual
#   ./deploy/ollama-tune.sh --reset        # quitar override
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
OVERRIDE_FILE="$OVERRIDE_DIR/override.conf"

VRAM_MB=""
SHOW_ONLY=false
RESET=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vram) VRAM_MB="$2"; shift 2 ;;
        --show) SHOW_ONLY=true; shift ;;
        --reset) RESET=true; shift ;;
        *) echo "Opción desconocida: $1"; exit 1 ;;
    esac
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
info() { echo -e "${CYAN}[..]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC}  $*"; }

# ── Reset ──
if $RESET; then
    info "Quitando override de Ollama..."
    sudo rm -rf "$OVERRIDE_DIR" 2>/dev/null || true
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    ok "Override eliminado. Ollama reiniciado con config default."
    exit 0
fi

# ── Detectar VRAM ──
if [[ -z "$VRAM_MB" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [[ -z "$VRAM_MB" || "$VRAM_MB" == "null" ]]; then
            warn "No se detectó GPU NVIDIA. Asumiendo modo CPU (16GB RAM)."
            VRAM_MB=0
        fi
    else
        warn "nvidia-smi no disponible. Asumiendo modo CPU."
        VRAM_MB=0
    fi
fi

echo "══════════════════════════════════════════════════════"
echo "  Ollama Tuning — Detección de Hardware"
echo "══════════════════════════════════════════════════════"

if [[ "$VRAM_MB" -gt 0 ]]; then
    VRAM_GB=$((VRAM_MB / 1024))
    echo "  GPU VRAM detectada: ${VRAM_GB} GB (${VRAM_MB} MB)"
else
    echo "  Modo: CPU (sin GPU NVIDIA)"
fi

# ── Mostrar config actual ──
if $SHOW_ONLY; then
    echo ""
    echo "Config actual de Ollama:"
    systemctl show ollama --property=Environment 2>/dev/null || echo "  (no se pudo leer)"
    echo ""
    echo "Modelos cargados actualmente:"
    ollama ps 2>/dev/null || echo "  (ollama no responde)"
    exit 0
fi

# ── Determinar parámetros según VRAM ──
if [[ "$VRAM_MB" -ge 20000 ]]; then
    # 20GB+ (L4 24GB, A10, etc.) — premium
    KEEP_ALIVE="30m"
    NUM_PARALLEL="1"
    MAX_MODELS="2"
    FLASH="1"
    PROFILE="premium (20GB+)"
elif [[ "$VRAM_MB" -ge 12000 ]]; then
    # 12-20GB (T4 16GB, etc.) — standard
    KEEP_ALIVE="30m"
    NUM_PARALLEL="1"
    MAX_MODELS="1"
    FLASH="1"
    PROFILE="standard (12-20GB)"
elif [[ "$VRAM_MB" -ge 6000 ]]; then
    # 6-12GB (RTX 3060, etc.) — compact
    KEEP_ALIVE="15m"
    NUM_PARALLEL="1"
    MAX_MODELS="1"
    FLASH="0"
    PROFILE="compact (6-12GB)"
else
    # CPU o GPU pequeña
    KEEP_ALIVE="5m"
    NUM_PARALLEL="1"
    MAX_MODELS="1"
    FLASH="0"
    PROFILE="cpu/small-gpu"
fi

echo "  Perfil: $PROFILE"
echo ""
echo "Parámetros a aplicar:"
echo "  OLLAMA_KEEP_ALIVE=$KEEP_ALIVE"
echo "  OLLAMA_NUM_PARALLEL=$NUM_PARALLEL"
echo "  OLLAMA_MAX_LOADED_MODELS=$MAX_MODELS"
echo "  OLLAMA_FLASH_ATTENTION=$FLASH"
echo ""

# ── Crear override de systemd ──
info "Creando override de systemd..."
sudo mkdir -p "$OVERRIDE_DIR"
sudo tee "$OVERRIDE_FILE" > /dev/null << EOF
# Override generado por deploy/ollama-tune.sh
# Perfil: $PROFILE (VRAM: ${VRAM_MB}MB)
[Service]
Environment="OLLAMA_KEEP_ALIVE=$KEEP_ALIVE"
Environment="OLLAMA_NUM_PARALLEL=$NUM_PARALLEL"
Environment="OLLAMA_MAX_LOADED_MODELS=$MAX_MODELS"
Environment="OLLAMA_FLASH_ATTENTION=$FLASH"
EOF

ok "Override creado en $OVERRIDE_FILE"

# ── Recargar y reiniciar ──
info "Recargando systemd y reiniciando Ollama..."
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 2

# Verificar
if systemctl is-active --quiet ollama; then
    ok "Ollama reiniciado correctamente"
    echo ""
    echo "Config activa:"
    systemctl show ollama --property=Environment 2>/dev/null | tr ' ' '\n' | grep OLLAMA || echo "(verificar con systemctl show ollama)"
else
    warn "Ollama no arrancó correctamente. Verificar: journalctl -u ollama -f"
    exit 1
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Tuning completado."
echo "  Para revertir: ./deploy/ollama-tune.sh --reset"
echo "  Para ver config: ./deploy/ollama-tune.sh --show"
echo "══════════════════════════════════════════════════════"
