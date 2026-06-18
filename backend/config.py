"""
Configuración centralizada del proyecto.

Este es el ÚNICO lugar donde se lee la configuración del proveedor LLM.
main.py, tools/eval.py y tools/pdf_to_knowledge.py importan de acá, para que
no haya tres copias distintas de la misma lógica (DRY).

El código habla con cualquier API OpenAI-compatible. Con LLM_PROVIDER=openrouter
usa OpenRouter (la nube); con LLM_PROVIDER=local apunta a un vLLM o llama.cpp
corriendo en el mismo servidor. Lo único que cambia entre ambos es la URL base,
la key (opcional en local) y el plugin de búsqueda web (solo OpenRouter).

Variables de entorno (todas opcionales salvo la key en modo openrouter):

    LLM_PROVIDER          "openrouter" (default) | "local"
    OPENROUTER_API_KEY    clave real para OpenRouter; "EMPTY" o cualquier cosa
                          para vLLM/llama.cpp (no la validan)
    OPENROUTER_BASE_URL   si no se setea, se autoconfigura según el provider
    OPENROUTER_MODEL      slug del modelo por defecto
    MAX_FILE_SIZE_MB      tope de tamaño de archivos subidos (default 20)
    RAG_K                 chunks a recuperar en RAG (default 6)
    RAG_MIN_SCORE         score mínimo para incluir un chunk (default 0.15)
    RAG_MAX_CONTEXT_CHARS presupuesto de chars del contexto RAG (default 6000)
    DEFAULT_CARRERA       carrera por defecto si el request no especifica (default "cc")
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

# Cargar .env desde la carpeta backend/ (este archivo vive en backend/)
BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

logger = logging.getLogger("upc")


# ──────────────────────────────────────────────────────────────
# Proveedor LLM
# ──────────────────────────────────────────────────────────────
VALID_PROVIDERS = ("openrouter", "local")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower().strip()
if LLM_PROVIDER not in VALID_PROVIDERS:
    logger.warning("LLM_PROVIDER=%r inválido, usando 'openrouter'", LLM_PROVIDER)
    LLM_PROVIDER = "openrouter"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# URL base: si el .env no la setea, default según provider.
# local → vLLM en :8001. Para llama.cpp poné OPENROUTER_BASE_URL=...:8080/v1
_DEFAULT_BASE_URL = {
    "openrouter": "https://openrouter.ai/api/v1",
    "local":      "http://localhost:11434/v1",
}[LLM_PROVIDER]
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")

# Modelo por defecto, con fallback según provider.
_DEFAULT_MODEL_FALLBACK = {
    "openrouter": "deepseek/deepseek-chat-v3:free",
    "local":      "Qwen/Qwen2.5-32B-Instruct-AWQ",
}[LLM_PROVIDER]
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL_FALLBACK)

# Identificación opcional para OpenRouter (aparece en su dashboard).
# vLLM/llama.cpp ignoran estos headers sin error.
HTTP_REFERER = os.getenv("APP_REFERER", "https://qa-accreditation.upc.edu.pe")
APP_TITLE    = os.getenv("APP_TITLE", "UPC Outcomes Verification")


def llm_headers() -> dict:
    """Headers para hablar con el motor LLM (OpenRouter o local)."""
    key = OPENROUTER_API_KEY or "EMPTY"
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  HTTP_REFERER,
        "X-Title":       APP_TITLE,
    }


def require_api_key() -> None:
    """Lanza si falta la key cuando es obligatoria (modo openrouter).
    En local la key es opcional (vLLM no la valida)."""
    if LLM_PROVIDER == "local":
        return
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY no configurada (requerida en modo openrouter)")


def supports_web_search() -> bool:
    """El plugin de búsqueda web solo existe en OpenRouter."""
    return LLM_PROVIDER == "openrouter"


# ──────────────────────────────────────────────────────────────
# Rutas del proyecto
# ──────────────────────────────────────────────────────────────
KNOWLEDGE_DIR       = BACKEND_DIR / "knowledge"
BASE_DIR            = KNOWLEDGE_DIR / "_base"
AUDIT_KNOWLEDGE_DIR = BACKEND_DIR / "audit_knowledge"
AUDIT_SOURCES_DIR   = BACKEND_DIR / "audit_sources"
LOGS_DIR            = BACKEND_DIR / "logs"


# ──────────────────────────────────────────────────────────────
# Límites y parámetros
# ──────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB      = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
RAG_K                 = int(os.getenv("RAG_K", "6"))
RAG_MIN_SCORE         = float(os.getenv("RAG_MIN_SCORE", "0.15"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))

# Carrera por defecto cuando un request no especifica una. Se usa como último
# recurso; lo normal es que el frontend siempre mande la carrera elegida.
DEFAULT_CARRERA = os.getenv("DEFAULT_CARRERA", "cc").lower()

# Umbral de tokens a partir del cual una carrera de auditoría se considera
# "pesada" (puede no entrar en modelos de contexto chico).
AUDIT_HEAVY_TOKEN_THRESHOLD = int(os.getenv("AUDIT_HEAVY_TOKEN_THRESHOLD", "100000"))


# ──────────────────────────────────────────────────────────────
# RAG
# ──────────────────────────────────────────────────────────────
# Modelo de embeddings. DEBE ser el mismo al indexar (ingest) y al buscar
# (retriever); por eso vive acá, en un solo lugar.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Carpeta de caché del modelo de embeddings. Tanto ingest como retriever
# usan ESTA misma carpeta, para descargar el modelo una sola vez y que
# ambos lo encuentren en el mismo lugar.
HF_CACHE_DIR = BACKEND_DIR / "rag" / "hf_cache"


def setup_hf_cache() -> None:
    """Configura las variables de entorno de HuggingFace para usar la caché
    local del proyecto. Debe llamarse ANTES de importar/crear cualquier
    modelo de sentence-transformers. Activa modo offline solo si el modelo
    ya está descargado (evita fallar por falta de internet en el primer uso)."""
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Offline solo si ya hay un snapshot del modelo cacheado.
    model_tag = EMBEDDING_MODEL_NAME.split("/")[-1].lower()
    cached = False
    for base in (HF_CACHE_DIR, Path.home() / ".cache" / "huggingface"):
        if base.exists():
            for cfg in base.rglob("config.json"):
                if model_tag in str(cfg).lower():
                    cached = True
                    break
        if cached:
            break
    if cached:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def build_embedding_model():
    """Crea el modelo de embeddings con la config correcta (normalización L2
    + caché local). Único lugar donde se construye, para que ingest y
    retriever sean idénticos. Importa langchain perezosamente para no
    cargar torch si no se usa RAG."""
    setup_hf_cache()
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=str(HF_CACHE_DIR),
    )


def summary() -> str:
    """Resumen legible de la config activa (para loguear al arrancar)."""
    return (
        f"provider={LLM_PROVIDER} base_url={OPENROUTER_BASE_URL} "
        f"model={DEFAULT_MODEL} key_set={bool(OPENROUTER_API_KEY)}"
    )