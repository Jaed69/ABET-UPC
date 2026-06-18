#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# fix-ctx.sh — Crea un modelo derivado con num_ctx ampliado (32768)
#
# Problema: Los modelos HF (hf.co/...) en Ollama traen num_ctx=4096
# grabado en su Modelfile. Aunque el backend envía options.num_ctx
# via API, Ollama prioriza el valor del Modelfile.
#
# Solución: Crear un modelo derivado (FROM original + PARAMETER num_ctx)
# que hereda todo pero con la ventana de contexto corregida.
#
# Uso:
#   sudo -u upc ./deploy/fix-ctx.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

ORIGINAL="hf.co/unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL"
DERIVED="gemma4-12b-32k"
NUM_CTX="${OLLAMA_NUM_CTX:-32768}"

echo "┌─────────────────────────────────────────────┐"
echo "│  fix-ctx.sh — Ampliar ventana de contexto   │"
echo "└─────────────────────────────────────────────┘"
echo ""
echo "Modelo original:  $ORIGINAL"
echo "Modelo derivado:  $DERIVED"
echo "num_ctx:          $NUM_CTX"
echo ""

# ── 1. Verificar que Ollama está corriendo ──────────────────────────
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama no responde en http://localhost:11434"
    echo "   Verifica: sudo systemctl status ollama"
    exit 1
fi
echo "✓ Ollama activo"

# ── 2. Verificar que el modelo original existe ──────────────────────
if ! ollama list 2>/dev/null | grep -q "$ORIGINAL"; then
    echo "⚠ El modelo original no está descargado todavía."
    echo "  Descargando (puede tardar varios minutos)…"
    ollama pull "$ORIGINAL"
fi
echo "✓ Modelo original disponible"

# ── 3. Crear Modelfile temporal ─────────────────────────────────────
TMPFILE=$(mktemp /tmp/Modelfile.XXXXXX)
trap 'rm -f "$TMPFILE"' EXIT

cat > "$TMPFILE" <<EOF
FROM $ORIGINAL

PARAMETER num_ctx $NUM_CTX
PARAMETER num_predict -1
EOF

echo "✓ Modelfile temporal creado: $TMPFILE"
echo ""
echo "── Contenido ──"
cat "$TMPFILE"
echo "──────────────"
echo ""

# ── 4. Construir modelo derivado ────────────────────────────────────
echo "Construyendo modelo derivado (copia los pesos, no re-descarga)…"
ollama create "$DERIVED" -f "$TMPFILE"
echo "✓ Modelo '$DERIVED' creado"

# ── 5. Verificar ────────────────────────────────────────────────────
echo ""
echo "── Verificación ──"
ollama show "$DERIVED" | grep -i "num_ctx\|num_predict" || true
echo ""

# ── 6. Actualizar .env ──────────────────────────────────────────────
ENV_FILE="/opt/upc-abet/backend/.env"
if [ -f "$ENV_FILE" ]; then
    if grep -q "OPENROUTER_MODEL=" "$ENV_FILE"; then
        sed -i "s|^OPENROUTER_MODEL=.*|OPENROUTER_MODEL=$DERIVED|" "$ENV_FILE"
        echo "✓ .env actualizado: OPENROUTER_MODEL=$DERIVED"
    else
        echo "OPENROUTER_MODEL=$DERIVED" >> "$ENV_FILE"
        echo "✓ .env: línea añadida OPENROUTER_MODEL=$DERIVED"
    fi
else
    echo "⚠ No se encontró $ENV_FILE — actualiza manualmente:"
    echo "  OPENROUTER_MODEL=$DERIVED"
fi

# ── 7. Reiniciar backend ────────────────────────────────────────────
echo ""
echo "Reiniciando backend…"
sudo systemctl restart upc-abet-backend
sleep 2

if curl -sf http://localhost:8000/api/health >/dev/null 2>&1; then
    echo "✓ Backend activo"
else
    echo "⚠ Backend no responde aún — revisa:"
    echo "  sudo journalctl -u upc-abet-backend --since '30s ago' -f"
fi

echo ""
echo "┌─────────────────────────────────────────────────┐"
echo "│  ✅ Listo. Modelo activo: $DERIVED"
echo "│  num_ctx=$NUM_CTX  num_predict=-1               │"
echo "└─────────────────────────────────────────────────┘"
