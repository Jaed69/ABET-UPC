# ABET-UPC — Asistente IA de Verificación y Control de Outcomes

Asistente de IA institucional para la **verificación y control de outcomes ABET** en carreras de la UPC. Permite a coordinadores académicos consultar las Mallas COCOS, Reportes de Control (RC) y Reportes de Verificación (RV) de cada carrera, y **auditar documentación adjunta** contra la malla oficial.

---

## Tabla de contenidos

1. [Arquitectura](#arquitectura)
2. [Requisitos](#requisitos)
3. [Estructura del código](#estructura-del-código)
4. [Ejecución local (desarrollo)](#ejecución-local-desarrollo)
5. [Despliegue en producción (Ubuntu)](#despliegue-en-producción-ubuntu)
6. [Cambio de modelo LLM](#cambio-de-modelo-llm)
7. [Mantenimiento y telemetría](#mantenimiento-y-telemetría)
8. [API — endpoints](#api--endpoints)
9. [Modos de operación](#modos-de-operación)
10. [Troubleshooting](#troubleshooting)
11. [Activación opcional del RAG](#activación-opcional-del-rag)

---

## Arquitectura

```
                    Internet
                       │
                       ▼
              ┌─────────────────┐
              │  Nginx (HTTPS)  │  acc-ia.tcupc.pe
              │  + Let's Encrypt │
              └────────┬────────┘
                       │  reverse proxy (proxy_buffering off para SSE)
                       ▼
              ┌─────────────────┐
              │  FastAPI        │  uvicorn :8000
              │  (backend)      │  - 12 endpoints /api/*
              │                 │  - router de relevancia
              │                 │  - system prompt builder
              └────────┬────────┘
                       │  httpx (OpenAI-compatible API)
                       ▼
              ┌─────────────────┐
              │  Ollama         │  :11434
              │  (LLM local)    │  gemma4:12b / qwen2.5:7b / ...
              └─────────────────┘

              ┌─────────────────┐
              │  Frontend SPA   │  servida por el mismo backend
              │  (vanilla JS)   │  frontend/static/index.html
              └─────────────────┘

              ┌─────────────────┐
              │  Knowledge      │  backend/knowledge/*.md
              │  (mallas COCOS) │  11 carreras + _base
              └─────────────────┘
```

**Stack técnico:**
- **Backend**: FastAPI + uvicorn (Python 3.12)
- **LLM**: Ollama (local) — API OpenAI-compatible
- **Frontend**: SPA vanilla JS (un único `index.html`, sin frameworks)
- **Reverse proxy**: Nginx + Let's Encrypt (HTTPS)
- **Gestión**: systemd (servicios), cron (telemetría)

---

## Requisitos

### Servidor de producción
- **OS**: Ubuntu 22.04+ (o equivalente Linux)
- **RAM**: 16 GB mínimo
- **GPU**: NVIDIA 8GB+ VRAM (recomendado para modelo 12B; sin GPU usar modelo 7B)
- **Disco**: 20 GB libres (modelos Ollama + ChromaDB opcional + logs)
- **Software**:
  - Python 3.12
  - Ollama
  - Nginx
  - certbot (Let's Encrypt)

### Desarrollo local
- Python 3.12
- Ollama arrancado (`ollama serve`)
- Un modelo descargado (`ollama pull gemma4:12b`)

---

## Estructura del código

```
ABET-UPC/
├── README.md                    # Este archivo
├── GUIA.md                      # Guía de integración API (raw, stream, with-files)
├── backend/
│   ├── main.py                  # App FastAPI: 12 endpoints, router de relevancia, system prompt
│   ├── config.py                # Config centralizada (DRY): proveedor LLM, rutas, embeddings
│   ├── logging_utils.py         # Logging JSONL con privacidad + stats agregadas
│   ├── requirements.txt         # 8 deps core (pinneadas ==)
│   ├── requirements-dev.txt     # Deps opcionales (RAG + tools)
│   ├── .env                     # Config activa (proveedor, modelo, límites)
│   ├── knowledge/               # Base de conocimiento .md
│   │   ├── _base/               #   system_prompt.md + glosario_simbolos.md
│   │   └── <carrera>/malla.md   #   11 carreras (cc, sw, si, civil, ...)
│   ├── rag/                     # Sistema RAG (OPCIONAL, no activo por defecto)
│   │   ├── ingest.py            #   Indexa knowledge/ → ChromaDB
│   │   ├── retriever.py         #   Búsqueda híbrida (densa + BM25 + RRF)
│   │   ├── diagnose.py          #   Diagnóstico de ChromaDB
│   │   └── search_test.py       #   Suite de tests del retriever
│   ├── tools/                   # Scripts offline
│   │   ├── build_audit_context.py  # PDF/DOCX/XLSX → .md (MarkItDown, sin LLM)
│   │   ├── pdf_to_knowledge.py     # PDFs UPC → .md compactos (vía LLM)
│   │   └── eval.py                 # Evaluación LLM-as-judge
│   └── logs/                    # Telemetría
│       └── queries.jsonl        #   1 JSON por request (privacidad por diseño)
├── frontend/
│   └── static/
│       ├── index.html           # SPA completa (2847 líneas, vanilla JS)
│       ├── favicon.ico
│       └── upc-logo-white.png
└── deploy/                      # Artefactos de despliegue
    ├── switch-model.sh          # Cambio de modelo Ollama en caliente
    ├── upc-abet-backend.service # Systemd unit del backend
    ├── nginx-acc-ia.conf        # Nginx reverse proxy + SSE
    ├── health-check.sh          # Healthcheck periódico + auto-restart
    ├── system-metrics.sh        # Métricas RAM/CPU/GPU/disco
    ├── logrotate-upc.conf       # Rotación de logs
    ├── dev-start.sh             # Arranque local (desarrollo)
    └── smoke-test.sh            # Pruebas de humo end-to-end
```

### Archivos clave

| Archivo | Función | Línea clave |
|---|---|---|
| `backend/main.py` | `build_full_system_prompt()` — arma system prompt según modo | `main.py:646` |
| `backend/main.py` | `detect_relevant_docs()` — router de relevancia (qué .md cargar) | `main.py:418` |
| `backend/main.py` | `discover_carreras()` — escanea knowledge/ (tolera ambos layouts) | `main.py:244` |
| `backend/config.py` | `build_embedding_model()` — modelo de embeddings (RAG) | `config.py:169` |
| `backend/logging_utils.py` | `QueryLog` — registra cada request sin guardar texto | `logging_utils.py:86` |
| `backend/rag/retriever.py` | `search_docs()` — búsqueda híbrida densa+BM25+RRF | `retriever.py:190` |

---

## Ejecución local (desarrollo)

### Opción 1: Script automático

```bash
# Clonar y arrancar (verifica Ollama, crea venv, instala deps, arranca uvicorn)
cd ABET-UPC
chmod +x deploy/*.sh
./deploy/dev-start.sh
```

El backend queda en `http://localhost:8000`. Abrir en navegador.

### Opción 2: Manual

```bash
# 1. Arrancar Ollama (en otra terminal)
ollama serve

# 2. Descargar un modelo
ollama pull gemma4:12b

# 3. Crear venv e instalar deps
cd backend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Arrancar backend (con reload para dev)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Verificar que todo funciona

```bash
./deploy/smoke-test.sh
# o contra otra URL:
./deploy/smoke-test.sh --base-url http://localhost:8000 --carrera cc
```

El smoke test verifica: health, provider, chat raw, chat con knowledge, stats y logs.

---

## Despliegue en producción (Ubuntu)

### Paso 1: Preparar el servidor

```bash
# Actualizar e instalar dependencias del sistema
sudo apt update && sudo apt install -y \
    python3.12 python3.12-venv \
    nginx certbot python3-certbot-nginx

# Verificar GPU (si aplica)
nvidia-smi
```

### Paso 2: Instalar Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama

# Descargar modelos
ollama pull gemma4:12b     # default (calidad, 7GB VRAM)
ollama pull qwen2.5:7b     # rápido (4.5GB VRAM)
ollama pull llama3.1:8b    # alternativa (5GB VRAM)
```

### Paso 3: Clonar repo y preparar backend

```bash
sudo mkdir -p /opt/upc-abet
sudo chown $USER:$USER /opt/upc-abet
git clone <repo-url> /opt/upc-abet

cd /opt/upc-abet/backend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Paso 4: Configurar .env

```bash
# Editar /opt/upc-abet/backend/.env
LLM_PROVIDER=local
OPENROUTER_BASE_URL=http://localhost:11434/v1
OPENROUTER_MODEL=gemma4:12b
OPENROUTER_API_KEY=ollama
MAX_FILE_SIZE_MB=20
RAG_K=10
RAG_MIN_SCORE=0.10
RAG_MAX_CONTEXT_CHARS=12000
```

### Paso 5: Crear usuario y instalar servicio systemd

```bash
# Usuario dedicado
sudo useradd -r -s /bin/false -d /opt/upc-abet upc
sudo chown -R upc:upc /opt/upc-abet

# Instalar servicio
sudo cp deploy/upc-abet-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now upc-abet-backend

# Verificar
sudo journalctl -u upc-abet-backend -f
# Debe mostrar: "Config: provider=local model=gemma4:12b ..."
```

### Paso 6: Nginx + HTTPS

```bash
# Instalar config de Nginx
sudo cp deploy/nginx-acc-ia.conf /etc/nginx/sites-available/acc-ia
sudo ln -s /etc/nginx/sites-available/acc-ia /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Obtener certificado HTTPS
sudo certbot --nginx -d acc-ia.tcupc.pe
```

### Paso 7: Telemetría (cron)

```bash
# Hacer scripts ejecutables
chmod +x /opt/upc-abet/deploy/*.sh

# Instalar cron jobs
crontab -e
# Añadir:
* * * * * /opt/upc-abet/deploy/health-check.sh
*/5 * * * * /opt/upc-abet/deploy/system-metrics.sh

# Instalar logrotate
sudo cp deploy/logrotate-upc.conf /etc/logrotate.d/upc-abet
```

### Paso 8: Verificación final

```bash
./deploy/smoke-test.sh --base-url https://acc-ia.tcupc.pe
```

Abrir `https://acc-ia.tcupc.pe` en navegador, hacer una consulta real.

---

## Cambio de modelo LLM

El backend no soporta cambiar de modelo en runtime — siempre requiere reiniciar el servicio. El script `switch-model.sh` automatiza todo:

```bash
# Cambiar a qwen2.5:7b (rápido)
./deploy/switch-model.sh qwen2.5:7b

# Cambiar a gemma4:12b (calidad)
./deploy/switch-model.sh gemma4:12b

# Cambiar sin descargar (--no-pull, si ya está local)
./deploy/switch-model.sh llama3.1:8b --no-pull

# Listar modelos locales
./deploy/switch-model.sh --list
```

**Qué hace el script:**
1. `ollama pull <modelo>` (descarga si no está)
2. Edita `OPENROUTER_MODEL` en `backend/.env`
3. `systemctl restart upc-abet-backend`
4. Smoke test: verifica `/api/health` reporta el modelo nuevo

### Modelos recomendados

| Modelo | VRAM (Q4) | Velocidad | Calidad | Cuándo usar |
|---|---|---|---|---|
| `qwen2.5:7b` | ~4.5 GB | Rápida | Buena | Consultas rápidas, respuestas cortas |
| `llama3.1:8b` | ~5 GB | Rápida | Buena | Alternativa balanceada |
| `gemma4:12b` | ~7 GB | Media | Alta | Default, auditoría, respuestas detalladas |

> **Sin GPU**: usar `qwen2.5:7b` o modelos menores. 16GB RAM sin GPU no soporta 12B cómodamente.

### Cambio manual (sin script)

```bash
# 1. Editar .env
nano /opt/upc-abet/backend/.env
# Cambiar: OPENROUTER_MODEL=nuevo_modelo

# 2. Reiniciar
sudo systemctl restart upc-abet-backend

# 3. Verificar
curl https://acc-ia.tcupc.pe/api/health | python3 -m json.tool
```

---

## Mantenimiento y telemetría

### Logs existentes

| Archivo/Endpoint | Contenido | Privacidad |
|---|---|---|
| `backend/logs/queries.jsonl` | 1 JSON por request: carrera, modo, modelo, tokens, latencia, status | Hash SHA-256 (sin texto del query), IP anonimizada |
| `backend/logs/health.jsonl` | Healthcheck cada 1 min: status, latencia, auto-restarts | Sin datos sensibles |
| `backend/logs/system-metrics.jsonl` | Métricas cada 5 min: RAM, CPU, GPU, disco, modelo activo | Sin datos sensibles |
| `journalctl -u upc-abet-backend` | Logs de uvicorn (stderr) | Logs de app |
| `journalctl -u ollama` | Logs de Ollama | Logs de Ollama |

### Endpoints de monitoreo

```bash
# Estado general del backend
curl https://acc-ia.tcupc.pe/api/health | python3 -m json.tool

# Estadísticas agregadas (últimos 7 días)
curl "https://acc-ia.tcupc.pe/api/stats?since_days=7" | python3 -m json.tool

# Últimos 10 requests
curl "https://acc-ia.tcupc.pe/api/logs/recent?limit=10" | python3 -m json.tool

# Info del proveedor LLM activo
curl https://acc-ia.tcupc.pe/api/provider | python3 -m json.tool

# Estado del RAG (si está activo)
curl https://acc-ia.tcupc.pe/api/rag/status | python3 -m json.tool
```

### Operaciones de mantenimiento

```bash
# Reiniciar backend
sudo systemctl restart upc-abet-backend

# Reiniciar Ollama
sudo systemctl restart ollama

# Recargar Nginx (tras cambio de config)
sudo systemctl reload nginx

# Ver logs en vivo
sudo journalctl -u upc-abet-backend -f
sudo journalctl -u ollama -f

# Ver métricas de sistema recientes
tail -20 /opt/upc-abet/backend/logs/system-metrics.jsonl | python3 -m json.tool

# Rotación de logs (manual, si hace falta)
sudo logrotate -f /etc/logrotate.d/upc-abet

# Limpiar modelos Ollama no usados
ollama rm <modelo-no-usado>
```

### Auto-restart

`health-check.sh` (cron cada 1 min) reinicia el backend automáticamente si `/api/health` falla 3 veces seguidas. El evento queda registrado en `logs/health.jsonl` con `"event":"auto_restart"`.

---

## API — endpoints

| Método | Ruta | Propósito |
|---|---|---|
| GET | `/api/health` | Estado: modelo, provider, carreras, RAG, capabilities |
| GET | `/api/provider` | Info del proveedor LLM activo |
| GET | `/api/models` | Lista de modelos del motor LLM |
| GET | `/api/knowledge` | Inventario del knowledge con tokens por archivo |
| GET | `/api/carreras` | Lista de carreras con periodos y comisiones |
| GET | `/api/rag/status` | Diagnóstico del RAG (ChromaDB) |
| GET | `/api/audit/status` | Estado del modo auditoría |
| GET | `/api/preview-prompt` | Debug: ve qué system prompt se armaría |
| GET | `/api/stats` | Agregados del log (totales, latencia p50/p95/p99) |
| GET | `/api/logs/recent` | Últimos N registros del log |
| POST | `/api/chat` | **Chat principal** (JSON, stream o no-stream) |
| POST | `/api/chat/with-files` | Chat con archivos adjuntos (multipart) |

### Ejemplos básicos

```bash
# Health
curl https://acc-ia.tcupc.pe/api/health

# Chat raw (sin system prompt, respuesta directa)
curl -X POST https://acc-ia.tcupc.pe/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hola"}],"raw":true,"stream":false}'

# Chat con knowledge (carrera CC)
curl -X POST https://acc-ia.tcupc.pe/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"¿Cuántos outcomes tiene CC?"}],"carrera":"cc","use_knowledge":true,"stream":false}'
```

Para integración detallada (streaming SSE, archivos, modo raw, auditoría), ver [`GUIA.md`](GUIA.md).

---

## Modos de operación

| Modo | Activación | Descripción |
|---|---|---|
| **Knowledge clásico** | `use_knowledge=true` (default) | Carga `.md` completos de la carrera como system prompt |
| **RAG** | `use_rag=true` | Recupera chunks relevantes vía ChromaDB (búsqueda híbrida densa+BM25+RRF) |
| **Auditoría** | `audit=true` | Compara adjuntos vs malla oficial; salida texto o JSON (`veredicto`, `hallazgos`, `campos_correctos`) |
| **Raw** | `raw=true` | Sin system prompt (modelo crudo, para integraciones) |
| **General** | `use_knowledge=false` | Asistente sin contexto institucional |
| **Razonamiento** | `think=true` | Activa `reasoning_effort` (más lento, más profundo) |
| **Búsqueda web** | `web_search=true` | Internet en tiempo real (solo OpenRouter, no Ollama) |

**Prioridad de modos**: `raw` > `audit` > `use_knowledge + use_rag` > `general`.

---

## Troubleshooting

| Problema | Solución |
|---|---|
| Backend no arranca | `journalctl -u upc-abet-backend -f` — buscar errores de import o config |
| `/api/health` reporta 0 carreras | Verificar layout de `knowledge/` — cada carrera debe tener `malla.md` (directo o bajo `<periodo>/`) |
| Respuestas vacías del LLM | Verificar Ollama arriba: `curl localhost:11434/api/tags` y `ollama ps` (modelo cargado) |
| Respuestas cortadas | Aumentar `OLLAMA_NUM_CTX` (default 32768 en prod) |
| SSE/streaming cortado | Nginx debe tener `proxy_buffering off` y `proxy_read_timeout 600s` |
| OOM (out of memory) | Cambiar a modelo 7B: `./deploy/switch-model.sh qwen2.5:7b` |
| GPU no usada | `nvidia-smi` verificar driver + `ollama ps` verificar que Ollama detecta GPU |
| Error 502 Bad Gateway | Backend caído: `sudo systemctl restart upc-abet-backend` |
| Error 413 Request Entity Too Large | `client_max_body_size` en Nginx (actual: 25M) |
| Certificado HTTPS vencido | `sudo certbot renew --dry-run` luego `sudo certbot renew` |

---

## Activación opcional del RAG

El RAG (ChromaDB + embeddings + BM25) está **desactivado por defecto** en producción para ahorrar ~2GB de RAM. Para activarlo:

```bash
# 1. Instalar deps adicionales
cd /opt/upc-abet/backend
source venv/bin/activate
pip install -r requirements-dev.txt

# 2. Indexar knowledge/ en ChromaDB
python -m rag.ingest

# 3. Verificar
python -m rag.diagnose
python -m rag.search_test

# 4. Reiniciar backend (detecta RAG automáticamente)
sudo systemctl restart upc-abet-backend

# 5. Verificar
curl https://acc-ia.tcupc.pe/api/rag/status | python3 -m json.tool
```

Una vez activo, el toggle "RAG" en el frontend o el flag `use_rag=true` en la API lo utiliza. Si RAG no devuelve contexto útil, cae automáticamente a knowledge clásico.

---

## Referencias

- [`GUIA.md`](GUIA.md) — Guía de integración API (modo raw, streaming SSE, archivos adjuntos)
- `backend/config.py` — Configuración centralizada (proveedor, rutas, límites, embeddings)
- `backend/logging_utils.py` — Sistema de telemetría (QueryLog, compute_stats)
- `deploy/` — Artefactos de despliegue y mantenimiento
