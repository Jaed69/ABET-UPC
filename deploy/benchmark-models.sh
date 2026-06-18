#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# benchmark-models.sh — Benchmark comparativo de modelos LLM para
# extracción JSON estructurada (mallas COCO UPC).
#
# Prueba cada modelo con el mismo prompt (malla COCO de SI) y mide:
#   - Tiempo total (latencia)
#   - Tokens/seg de generación
#   - Tokens de input
#   - VRAM usada
#   - Si el JSON parsea correctamente
#
# Modelos probados (configurables):
#   - qwen2.5:7b   (rápido, buena calidad JSON)
#   - qwen2.5:14b  (precisión quirúrgica JSON)
#   - llama3.1:8b  (buen seguimiento de instrucciones)
#   - gemma4:12b   (baseline, texto libre)
#
# Uso:
#   ./deploy/benchmark-models.sh                    # todos los modelos
#   ./deploy/benchmark-models.sh --models qwen2.5:7b,llama3.1:8b
#   ./deploy/benchmark-models.sh --prompt deploy/malla-test.txt
#   ./deploy/benchmark-models.sh --json             # forzar json_output
#
# Resultados: backend/logs/benchmark-<timestamp>.jsonl
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="$SCRIPT_DIR/malla-test.txt"
LOG_DIR="$PROJECT_DIR/backend/logs"
OLLAMA_URL="http://localhost:11434/v1"
TIMEOUT=300

DEFAULT_MODELS="qwen2.5:7b,qwen2.5:14b,llama3.1:8b,gemma4:12b"
MODELS="$DEFAULT_MODELS"
FORCE_JSON=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --prompt) PROMPT_FILE="$2"; shift 2 ;;
        --json) FORCE_JSON=true; shift ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) echo "Opción desconocida: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TS_FILE=$(date -u +"%Y%m%dT%H%M%SZ")
LOG_FILE="$LOG_DIR/benchmark-${TS_FILE}.jsonl"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
info() { echo -e "${CYAN}[..]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*" >&2; }

# ── Verificaciones ──
if [[ ! -f "$PROMPT_FILE" ]]; then
    err "No se encontró el prompt: $PROMPT_FILE"
    exit 1
fi

if ! curl -sf --max-time 5 "http://localhost:11434/api/tags" &>/dev/null; then
    err "Ollama no responde en http://localhost:11434"
    err "Verificar con: sudo systemctl status ollama"
    exit 1
fi

PROMPT=$(cat "$PROMPT_FILE")
PROMPT_CHARS=${#PROMPT}
INPUT_TOKENS=$((PROMPT_CHARS / 4))

echo "══════════════════════════════════════════════════════"
echo "  Benchmark de Modelos — Extracción JSON (Malla COCO)"
echo "  Fecha: $TS"
echo "  Prompt: $PROMPT_FILE ($PROMPT_CHARS chars, ~$INPUT_TOKENS tokens)"
echo "  JSON forzado: $FORCE_JSON"
echo "  Log: $LOG_FILE"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Función para obtener VRAM ──
get_vram() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "null"
    else
        echo "null"
    fi
}

# ── Función para validar JSON ──
validate_json() {
    local content="$1"
    python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    keys = len(d.keys()) if isinstance(d, dict) else 0
    courses = len(d.get('courses', [])) if isinstance(d, dict) else 0
    print(f'valid|{keys}|{courses}')
except:
    print('invalid|0|0')
" <<< "$content" 2>/dev/null
}

# ── Benchmark de cada modelo ──
IFS=',' read -ra MODEL_LIST <<< "$MODELS"
RESULTS=""

for MODEL in "${MODEL_LIST[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "Modelo: $MODEL"

    # Verificar que el modelo está disponible
    if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
        warn "Modelo '$MODEL' no instalado. Ejecutar: ollama pull $MODEL"
        warn "Saltando..."
        echo ""
        continue
    fi

    # VRAM antes
    VRAM_BEFORE=$(get_vram)

    # Construir payload
    PAYLOAD=$(python3 -c "
import json, sys
payload = {
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': sys.stdin.read()}],
    'stream': False,
    'temperature': 0.1,
    'options': {'num_ctx': 16384, 'num_predict': 4096},
}
$(if $FORCE_JSON; then echo "payload['response_format'] = {'type': 'json_object'}"; fi)
print(json.dumps(payload))
" <<< "$PROMPT")

    # Llamar al modelo y medir tiempo
    info "Ejecutando (timeout ${TIMEOUT}s)..."
    START_MS=$(date +%s%3N)

    RESP=$(echo "$PAYLOAD" | curl -sf --max-time "$TIMEOUT" \
        -X POST "$OLLAMA_URL/chat/completions" \
        -H "Content-Type: application/json" \
        -d @- 2>/dev/null) || {
        err "Request falló o timeout para $MODEL"
        VRAM_AFTER=$(get_vram)
        RESULT="{\"ts\":\"$TS\",\"model\":\"$MODEL\",\"status\":\"error\",\"latency_ms\":null,\"input_tokens\":$INPUT_TOKENS,\"output_tokens\":0,\"tokens_per_sec\":0,\"vram_before_mb\":${VRAM_BEFORE:-null},\"vram_after_mb\":${VRAM_AFTER:-null},\"json_valid\":false,\"json_keys\":0,\"courses_extracted\":0}"
        echo "$RESULT" >> "$LOG_FILE"
        echo ""
        continue
    }

    END_MS=$(date +%s%3N)
    LATENCY=$((END_MS - START_MS))

    # VRAM después
    VRAM_AFTER=$(get_vram)

    # Extraer métricas de la respuesta
    METRICS=$(echo "$RESP" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    content = d.get('choices', [{}])[0].get('message', {}).get('content', '')
    usage = d.get('usage', {})
    pt = usage.get('prompt_tokens', 0)
    ct = usage.get('completion_tokens', 0)
    # validar JSON
    try:
        parsed = json.loads(content)
        valid = 'true'
        keys = len(parsed.keys()) if isinstance(parsed, dict) else 0
        courses = len(parsed.get('courses', [])) if isinstance(parsed, dict) else 0
    except:
        valid = 'false'
        keys = 0
        courses = 0
    print(f'{pt}|{ct}|{len(content)}|{valid}|{keys}|{courses}')
except Exception as e:
    print(f'0|0|0|false|0|0')
" 2>/dev/null)

    IFS='|' read -r OUT_PT OUT_CT OUT_CHARS JSON_VALID JSON_KEYS COURSES <<< "$METRICS"

    # Tokens/seg (output_tokens / latency_seg)
    if [[ "$LATENCY" -gt 0 && "$OUT_CT" -gt 0 ]]; then
        TPS=$(python3 -c "print(round($OUT_CT / ($LATENCY / 1000), 1))" 2>/dev/null)
    else
        TPS=0
    fi

    # Resultado
    STATUS="ok"
    if [[ "$JSON_VALID" == "false" ]]; then
        warn "JSON inválido para $MODEL"
        STATUS="json_invalid"
    fi

    RESULT="{\"ts\":\"$TS\",\"model\":\"$MODEL\",\"status\":\"$STATUS\",\"latency_ms\":$LATENCY,\"input_tokens\":${OUT_PT:-$INPUT_TOKENS},\"output_tokens\":${OUT_CT:-0},\"tokens_per_sec\":${TPS:-0},\"vram_before_mb\":${VRAM_BEFORE:-null},\"vram_after_mb\":${VRAM_AFTER:-null},\"json_valid\":${JSON_VALID},\"json_keys\":${JSON_KEYS:-0},\"courses_extracted\":${COURSES:-0}}"
    echo "$RESULT" >> "$LOG_FILE"

    # Reporte en consola
    if [[ "$JSON_VALID" == "true" ]]; then
        ok "$MODEL — ${LATENCY}ms, ${TPS} tok/s, JSON válido, ${COURSES} cursos, VRAM ${VRAM_AFTER}MB"
    else
        warn "$MODEL — ${LATENCY}ms, ${TPS} tok/s, JSON INVÁLIDO, VRAM ${VRAM_AFTER}MB"
    fi
    echo ""
done

# ── Resumen ──
echo "══════════════════════════════════════════════════════"
echo "  Resumen del Benchmark"
echo "══════════════════════════════════════════════════════"
python3 -c "
import json, sys
results = []
with open('$LOG_FILE') as f:
    for line in f:
        try:
            results.append(json.loads(line.strip()))
        except:
            pass
if not results:
    print('Sin resultados.')
    sys.exit(0)
print(f'{\"Modelo\":<20} {\"Latencia\":>10} {\"Tok/seg\":>10} {\"JSON\":>6} {\"Cursos\":>8} {\"VRAM MB\":>10}')
print('─' * 70)
for r in sorted(results, key=lambda x: x.get('tokens_per_sec', 0), reverse=True):
    model = r.get('model', '?')
    lat = f\"{r.get('latency_ms', 0)}ms\" if r.get('latency_ms') else 'N/A'
    tps = f\"{r.get('tokens_per_sec', 0)}\"
    valid = '✓' if r.get('json_valid') else '✗'
    courses = str(r.get('courses_extracted', 0))
    vram = str(r.get('vram_after_mb', 'N/A'))
    print(f'{model:<20} {lat:>10} {tps:>10} {valid:>6} {courses:>8} {vram:>10}')
"
echo ""
echo "Resultados guardados en: $LOG_FILE"
