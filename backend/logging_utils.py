"""
Logging estructurado de queries y stats agregados.

Guarda una línea JSONL por request a /api/chat* en:
    backend/logs/queries.jsonl

NO guarda el contenido literal del mensaje (privacidad) — solo:
- Hash SHA-256 del query (para deduplicación / análisis de patrones)
- Longitud del query en chars
- Carrera, modo (knowledge/rag/general), modelo usado
- Tokens consumidos (sys, in, out — los reales que reportó OpenRouter)
- Latencia en ms
- Resultado: success / error / aborted
- Si se usó web_search, si hubo archivos adjuntos (cuántos)
- IP del cliente (anonimizada al /24 si es IPv4)

El endpoint /api/stats lee este archivo y devuelve agregados:
- Queries totales / por día / por carrera / por modo
- Tokens consumidos totales
- Latencia p50 / p95 / p99
- % de errores
- Top de modelos usados

El log es append-safe (cada línea es un JSON independiente, no hay parser
que mantenga estado), así que se puede leer mientras se escribe.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
LOGS_DIR    = BACKEND_DIR / "logs"
LOG_FILE    = LOGS_DIR / "queries.jsonl"

LOGS_DIR.mkdir(exist_ok=True)

# Lock para escribir al archivo (multiproceso/multi-thread safe).
# Append a un archivo es atómico en POSIX hasta 4KB, pero usamos lock
# por las dudas (Windows no garantiza atomicidad).
_write_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────
# Helpers de privacidad
# ──────────────────────────────────────────────────────────────
def hash_query(text: str) -> str:
    """Hash corto del query (12 chars) para deduplicación."""
    if not text:
        return ""
    h = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
    return h[:12]


def anonymize_ip(ip: str) -> str:
    """Anonimiza IPv4 al /24 (último octeto a 0) y IPv6 al /64.
    Si no se puede parsear, devuelve 'unknown'."""
    if not ip:
        return "unknown"
    try:
        addr = ipaddress.ip_address(ip)
        if isinstance(addr, ipaddress.IPv4Address):
            parts = str(addr).split(".")
            return f"{parts[0]}.{parts[1]}.{parts[2]}.0"
        # IPv6: dejar solo los primeros 4 segmentos
        return ":".join(str(addr).split(":")[:4]) + "::/64"
    except (ValueError, AttributeError):
        return "unknown"


# ──────────────────────────────────────────────────────────────
# QueryLog: contexto del request en curso
# ──────────────────────────────────────────────────────────────
class QueryLog:
    """Contiene los datos de una request en curso. Se construye al iniciar
    el handler, se va llenando durante el procesamiento, y se persiste al
    final con .finish() (success/error/aborted).

    Para streams, esto se maneja desde el wrapper async (ver log_stream)
    que llama a .finish() cuando termina o falla el stream.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        carrera: Optional[str],
        mode: str,
        model: str,
        use_knowledge: bool,
        use_rag: bool,
        web_search: bool,
        query_text: str,
        sys_tokens: int,
        loaded_docs: list[str] | str,
        files_count: int,
        client_ip: str,
        stream: bool,
    ):
        self.t0          = time.perf_counter()
        self.ts_iso      = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.endpoint    = endpoint
        self.carrera     = carrera
        self.mode        = mode
        self.model       = model
        self.use_knowledge = use_knowledge
        self.use_rag       = use_rag
        self.web_search    = web_search
        self.query_hash    = hash_query(query_text)
        self.query_len     = len(query_text or "")
        self.sys_tokens    = sys_tokens
        # Normalizar loaded_docs a string compacto para JSONL
        if isinstance(loaded_docs, list):
            self.loaded_docs = ",".join(loaded_docs) if loaded_docs else "none"
        else:
            self.loaded_docs = str(loaded_docs or "none")
        self.files_count   = files_count
        self.client_ip     = anonymize_ip(client_ip)
        self.stream        = stream

        # Datos que se llenan al final
        self.prompt_tokens:     Optional[int] = None
        self.completion_tokens: Optional[int] = None
        self.status: str = "pending"  # success | error | aborted
        self.error_message: Optional[str] = None
        self.rag_chunks: int = 0
        self.rag_top_score: float = 0.0

    def set_rag_info(self, chunks: int, top_score: float) -> None:
        self.rag_chunks    = chunks
        self.rag_top_score = top_score

    def set_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens     = prompt_tokens
        self.completion_tokens = completion_tokens

    def finish(self, status: str = "success", error: Optional[str] = None) -> None:
        """Cierra el log y lo persiste."""
        latency_ms = int((time.perf_counter() - self.t0) * 1000)
        record = {
            "ts":               self.ts_iso,
            "endpoint":         self.endpoint,
            "carrera":          self.carrera,
            "mode":             self.mode,
            "model":            self.model,
            "use_knowledge":    self.use_knowledge,
            "use_rag":          self.use_rag,
            "web_search":       self.web_search,
            "stream":           self.stream,
            "query_hash":       self.query_hash,
            "query_len":        self.query_len,
            "files_count":      self.files_count,
            "sys_tokens":       self.sys_tokens,
            "prompt_tokens":    self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "loaded_docs":      self.loaded_docs,
            "rag_chunks":       self.rag_chunks,
            "rag_top_score":    self.rag_top_score,
            "latency_ms":       latency_ms,
            "status":           status,
            "error":            error,
            "client_ip":        self.client_ip,
        }
        _write_log_record(record)


def _write_log_record(record: dict) -> None:
    """Append-safe write al JSONL. Una falla al loggear NUNCA debe
    romper el request — capturamos todo."""
    try:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _write_lock:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        # Log a stderr para no perder el evento pero NO levantar
        print(f"[WARN] No pude escribir el query log: {e}")


# ──────────────────────────────────────────────────────────────
# Lectura y agregados
# ──────────────────────────────────────────────────────────────
def _read_records(limit: Optional[int] = None) -> list[dict]:
    """Lee el JSONL. Tolerante a líneas corruptas (las saltea)."""
    if not LOG_FILE.exists():
        return []
    records: list[dict] = []
    try:
        with LOG_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if limit is not None:
            records = records[-limit:]
    except Exception as e:
        print(f"[WARN] Error leyendo log: {e}")
    return records


def _percentile(values: list[float], p: float) -> float:
    """Percentil simple sin numpy. p en [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def compute_stats(
    since_days: Optional[int] = None,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Lee el log y devuelve agregados listos para JSON.

    Args:
        since_days: si se pasa, filtra a los últimos N días.
        limit:      tope de registros a procesar (los más recientes).
    """
    records = _read_records(limit=limit)

    if since_days is not None:
        cutoff = time.time() - since_days * 86400
        filtered = []
        for r in records:
            try:
                ts = datetime.fromisoformat(r["ts"]).timestamp()
                if ts >= cutoff:
                    filtered.append(r)
            except (ValueError, KeyError):
                continue
        records = filtered

    total = len(records)
    if total == 0:
        return {
            "total_queries":   0,
            "since_days":      since_days,
            "log_file":        str(LOG_FILE),
            "log_exists":      LOG_FILE.exists(),
        }

    # Contadores
    by_status   = Counter(r.get("status", "unknown") for r in records)
    by_mode     = Counter(r.get("mode", "unknown") for r in records)
    by_carrera  = Counter(r.get("carrera") or "none" for r in records)
    by_model    = Counter(r.get("model", "unknown") for r in records)
    by_endpoint = Counter(r.get("endpoint", "unknown") for r in records)

    # Por día (yyyy-mm-dd)
    by_day: dict[str, int] = defaultdict(int)
    for r in records:
        ts = r.get("ts", "")
        day = ts[:10] if len(ts) >= 10 else "unknown"
        by_day[day] += 1

    # Tokens totales
    sum_sys = sum(r.get("sys_tokens") or 0 for r in records)
    sum_in  = sum(r.get("prompt_tokens") or 0 for r in records)
    sum_out = sum(r.get("completion_tokens") or 0 for r in records)

    # Latencias (solo de exitosos)
    latencies = [
        r["latency_ms"] for r in records
        if r.get("status") == "success" and isinstance(r.get("latency_ms"), (int, float))
    ]

    # Hashes de queries únicos (estimación de unicidad)
    unique_queries = len({r.get("query_hash") for r in records if r.get("query_hash")})

    # Top 5 queries más repetidos
    hash_counts = Counter(r.get("query_hash") for r in records if r.get("query_hash"))
    top_repeated = [
        {"query_hash": h, "count": c}
        for h, c in hash_counts.most_common(5)
        if c > 1
    ]

    return {
        "total_queries":      total,
        "since_days":         since_days,
        "unique_query_hashes": unique_queries,
        "repeat_rate":        round(1.0 - unique_queries / total, 3) if total else 0,

        "by_status":   dict(by_status),
        "by_mode":     dict(by_mode),
        "by_carrera":  dict(by_carrera),
        "by_endpoint": dict(by_endpoint),
        "by_model":    dict(by_model.most_common(10)),
        "by_day":      dict(sorted(by_day.items())),

        "tokens": {
            "sum_sys":        sum_sys,
            "sum_prompt_in":  sum_in,
            "sum_completion": sum_out,
            "sum_total":      sum_in + sum_out,
            "avg_sys":        round(sum_sys / total, 1) if total else 0,
            "avg_prompt_in":  round(sum_in / total, 1)  if total else 0,
            "avg_completion": round(sum_out / total, 1) if total else 0,
        },

        "latency_ms": {
            "p50": int(_percentile(latencies, 50)) if latencies else None,
            "p95": int(_percentile(latencies, 95)) if latencies else None,
            "p99": int(_percentile(latencies, 99)) if latencies else None,
            "max": max(latencies) if latencies else None,
            "avg": int(sum(latencies) / len(latencies)) if latencies else None,
        },

        "top_repeated_queries": top_repeated,
        "log_file":             str(LOG_FILE),
    }