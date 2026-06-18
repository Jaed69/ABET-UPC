# Guía rápida — API UPC ABET (modo raw / integración)

## 1. Comandos del servidor

### Iniciar la API
```bash
cd ~/upc-abet/backend
source venv/bin/activate          # activar entorno virtual (venv)
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
La API queda en `http://localhost:8000` (internamente) y, vía Nginx,
en `https://acc-ia.tcupc.pe`.

### Modificar el .env
```bash
cd ~/upc-abet/backend
nano .env                         # está oculto; se ve con: ls -a
```
Variables principales:
```
LLM_PROVIDER=local
OPENROUTER_BASE_URL=http://localhost:11434/v1
OPENROUTER_MODEL=gemma4:12b
OPENROUTER_API_KEY=ollama
```
Tras editar el .env, reiniciar la API (Ctrl+C y volver a arrancar).

### Reiniciar limpio (si algo quedó colgado)
```bash
pkill -f uvicorn
sleep 2
cd ~/upc-abet/backend && source venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 2. Conexión a la API (modo raw)

**Endpoint:** `POST https://acc-ia.tcupc.pe/api/chat`
**Content-Type:** `application/json`

### Modo raw — qué es
El modo raw envía el prompt al modelo SIN ningún system prompt
(sin contexto institucional, sin "asistente", sin saludos). El modelo
responde solo a lo que está en el mensaje. Ideal para integrar y
controlar el comportamiento 100% desde el prompt que envías.

### Parámetros (JSON)
| Campo | Valor | Significado |
|---|---|---|
| `messages` | `[{"role":"user","content":"..."}]` | el prompt |
| `raw` | `true` | sin system prompt (modelo crudo) |
| `think` | `false` | sin razonamiento (respuesta directa, más rápida) |
| `num_ctx` | `32768` | ventana de contexto amplia (evita cortes con inputs grandes) |
| `stream` | `true` o `false` | ver sección 3 |

El formato de la respuesta (JSON, markdown, etc.) lo decide el prompt
que envíes, NO la API.

---

## 3. Stream vs Normal — cuál usar

### Normal (`stream: false`)
La API espera a generar TODA la respuesta y la devuelve de una sola vez,
en un único JSON. Más simple de procesar.
- Bueno para: respuestas cortas/medianas, scripts batch.
- Riesgo: si la respuesta tarda mucho (varios minutos), la conexión
  puede dar timeout esperando.

```python
import requests

r = requests.post(
    "https://acc-ia.tcupc.pe/api/chat",
    json={
        "messages": [{"role": "user", "content": "tu prompt"}],
        "raw": True, "think": False, "num_ctx": 32768,
        "stream": False,
    },
    timeout=600,
)
print(r.json()["choices"][0]["message"]["content"])
```

### Stream (`stream: true`)
La API devuelve la respuesta poco a poco (token por token) vía SSE.
- Bueno para: respuestas largas, mostrar en vivo en una UI.
- Ventaja: la conexión nunca queda inactiva → NO da timeout aunque tarde.
- Recomendado para inputs/respuestas grandes.

```python
import requests, json

with requests.post(
    "https://acc-ia.tcupc.pe/api/chat",
    json={
        "messages": [{"role": "user", "content": "tu prompt"}],
        "raw": True, "think": False, "num_ctx": 32768,
        "stream": True,
    },
    stream=True, timeout=600,
) as r:
    contenido = ""
    for line in r.iter_lines(chunk_size=1, decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            break
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            continue
        delta = chunk.get("choices",[{}])[0].get("delta",{}).get("content","")
        if delta:
            print(delta, end="", flush=True)
            contenido += delta
```

**Formato del stream (SSE):** líneas `data: {...}`. Ignorar las que no
empiezan con `data:`. El texto está en `choices[0].delta.content`.
Fin con `data: [DONE]`.

**Regla práctica:** para integración en una web con respuestas largas,
usar **stream**. Para llamadas rápidas internas, **normal** es más simple.

---

## 4. Endpoint con archivos adjuntos (`/api/chat/with-files`)

Para enviar archivos (PDF, DOCX, XLSX) además del texto. Usa
`multipart/form-data` en vez de JSON.

**Endpoint:** `POST https://acc-ia.tcupc.pe/api/chat/with-files`
**Content-Type:** `multipart/form-data`

### Diferencias clave con /api/chat
- Los datos van como **form-data** (`data=`), NO como JSON.
- Los valores van como **strings**: `"true"`/`"false"`, no booleanos.
- El texto va en el campo `message` (no en `messages`).
- Los archivos van en el campo `files`.

### Campos
| Campo | Valor | Significado |
|---|---|---|
| `message` | texto | instrucción/prompt |
| `files` | archivo(s) | documentos a procesar (PDF, DOCX, XLSX) |
| `raw` | `"true"`/`"false"` | sin system prompt |
| `think` | `"false"` | sin razonamiento |
| `num_ctx` | `"32768"` | ventana de contexto |
| `stream` | `"true"`/`"false"` | streaming o normal |
| `carrera` | ej. `"cc"` | carrera (para auditoría) |
| `periodo` | ej. `"2025-1"` | periodo (para auditoría) |
| `audit` | `"true"`/`"false"` | modo auditoría |

### Ejemplo — modo raw con archivo (normal, stream=false)
```python
import requests

data = {
    "message": "Resume este documento",
    "raw":     "true",
    "think":   "false",
    "num_ctx": "32768",
    "stream":  "false",
}
files = [("files", ("doc.pdf", open("doc.pdf", "rb"), "application/pdf"))]

r = requests.post(
    "https://acc-ia.tcupc.pe/api/chat/with-files",
    data=data, files=files, timeout=600,
)
print(r.json()["choices"][0]["message"]["content"])
```

### Ejemplo — auditoría con archivo (stream=true)
```python
import requests, json

data = {
    "message": "Audita este documento contra la malla",
    "carrera": "cc",
    "periodo": "2025-1",
    "audit":   "true",
    "think":   "false",
    "num_ctx": "32768",
    "stream":  "true",
}
files = [("files", ("reporte.pdf", open("reporte.pdf", "rb"), "application/pdf"))]

with requests.post(
    "https://acc-ia.tcupc.pe/api/chat/with-files",
    data=data, files=files, stream=True, timeout=600,
) as r:
    for line in r.iter_lines(chunk_size=1, decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            break
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            continue
        delta = chunk.get("choices",[{}])[0].get("delta",{}).get("content","")
        if delta:
            print(delta, end="", flush=True)
```

### Varios archivos
```python
files = [
    ("files", ("a.pdf", open("a.pdf","rb"), "application/pdf")),
    ("files", ("b.docx", open("b.docx","rb"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
]
```

---

## 5. Formato de la respuesta (ambos modos)
Compatible con OpenAI:
- Normal: `choices[0].message.content`
- Stream: `choices[0].delta.content` (por fragmento)
