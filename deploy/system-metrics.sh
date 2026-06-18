#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# system-metrics.sh — Captura métricas de sistema cada 5 minutos.
#
# Registra RAM, CPU, disco, GPU VRAM y modelo Ollama activo en
# logs/system-metrics.jsonl para telemetría de recursos.
#
# Instalar en cron:
#   */5 * * * * /opt/upc-abet/deploy/system-metrics.sh
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_DIR/backend/logs/system-metrics.jsonl"

mkdir -p "$(dirname "$LOG_FILE")"

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# ── RAM (MB) ──
RAM_TOTAL=$(free -m | awk '/^Mem:/ {print $2}')
RAM_USED=$(free -m | awk '/^Mem:/ {print $3}')
RAM_AVAIL=$(free -m | awk '/^Mem:/ {print $7}')

# ── Swap (MB) ──
SWAP_TOTAL=$(free -m | awk '/^Swap:/ {print $2}')
SWAP_USED=$(free -m | awk '/^Swap:/ {print $3}')

# ── CPU (%) ──
CPU_USAGE=$(top -bn1 | awk '/^%Cpu/ {print 100 - $8}' | cut -d. -f1)

# ── Disco (%) ──
DISK_PCT=$(df -h /opt | awk 'NR==2 {gsub("%",""); print $5}')
DISK_FREE=$(df -h /opt | awk 'NR==2 {print $4}')

# ── GPU VRAM (NVIDIA) ──
GPU_VRAM_USED=""
GPU_VRAM_TOTAL=""
GPU_UTIL=""
if command -v nvidia-smi &>/dev/null; then
    GPU_VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "")
    GPU_VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "")
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "")
fi

# ── Modelo Ollama activo ──
OLLAMA_MODELS=""
if command -v ollama &>/dev/null; then
    OLLAMA_MODELS=$(ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | paste -sd "," || echo "")
fi

# ── Output JSONL ──
echo "{\"ts\":\"$TS\",\"ram_mb\":{\"total\":$RAM_TOTAL,\"used\":$RAM_USED,\"avail\":$RAM_AVAIL},\"swap_mb\":{\"total\":${SWAP_TOTAL:-0},\"used\":${SWAP_USED:-0}},\"cpu_pct\":${CPU_USAGE:-0},\"disk\":{\"pct_used\":${DISK_PCT:-0},\"free\":\"$DISK_FREE\"},\"gpu\":{\"vram_used_mb\":${GPU_VRAM_USED:-null},\"vram_total_mb\":${GPU_VRAM_TOTAL:-null},\"util_pct\":${GPU_UTIL:-null}},\"ollama_loaded\":\"$OLLAMA_MODELS\"}" >> "$LOG_FILE"
