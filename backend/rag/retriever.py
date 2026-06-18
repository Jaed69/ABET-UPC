"""
Retriever sobre la ChromaDB. Singleton para no recargar el modelo en cada request.

Uso típico desde main.py:

    from rag.retriever import search_docs, build_rag_context

    chunks = search_docs("¿cursos críticos?", carrera="cc", doc_types=["control"], k=5)
    context = build_rag_context(chunks)
    # Pasar `context` al system prompt del LLM.
"""
from __future__ import annotations

import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# Silenciar el FutureWarning interno de huggingface_hub sobre `resume_download`.
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.*")

# Config centralizada (config.py está en backend/, un nivel arriba de rag/)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Configurar la caché de HuggingFace ANTES de importar langchain/torch.
# Esto define HF_HOME, activa offline si el modelo ya está cacheado, etc.
# Es la misma caché que usa ingest.py (config.HF_CACHE_DIR).
config.setup_hf_cache()

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Config centralizada (config.py está en backend/, un nivel arriba de rag/)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ──────────────────────────────────────────────────────────────
# Config — el embedding viene de config (debe coincidir con ingest.py)
# ──────────────────────────────────────────────────────────────
BACKEND_DIR = config.BACKEND_DIR
CHROMA_PATH = BACKEND_DIR / "rag" / "chroma_db"
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME

# Carreras que el código entiende — siempre se incluye "base" en la búsqueda
ALWAYS_INCLUDE_DOC_TYPES = {"base"}


# ──────────────────────────────────────────────────────────────
# Singleton thread-safe
# ──────────────────────────────────────────────────────────────
_db_lock = threading.Lock()
_db_instance: Chroma | None = None


def get_db() -> Chroma:
    """Carga ChromaDB una sola vez por proceso."""
    global _db_instance
    if _db_instance is not None:
        return _db_instance

    with _db_lock:
        if _db_instance is None:
            if not CHROMA_PATH.exists():
                raise RuntimeError(
                    f"ChromaDB no existe en {CHROMA_PATH}. "
                    f"Corre `python -m rag.ingest` primero."
                )
            import time as _t
            _t0 = _t.perf_counter()
            print("[RAG] Cargando modelo de embeddings (primera vez)...")
            # El modelo se construye desde config para que sea idéntico al
            # que usó ingest (misma normalización L2 y misma caché).
            embedding_model = config.build_embedding_model()
            _db_instance = Chroma(
                persist_directory=str(CHROMA_PATH),
                embedding_function=embedding_model,
            )
            _elapsed = _t.perf_counter() - _t0
            print(f"[RAG] Modelo + Chroma listos en {_elapsed:.1f}s (esto pasa UNA vez por proceso)")
    return _db_instance


# ──────────────────────────────────────────────────────────────
# Índice BM25 (búsqueda léxica) — para búsqueda híbrida
# ──────────────────────────────────────────────────────────────
# El embedding denso es bueno en semántica pero falla con keywords
# exactas (códigos como "1ASI0644", siglas como "RE1", "outcomes").
# BM25 (Robertson & Zaragoza 2009) cubre justo ese hueco: ranking léxico
# por frecuencia de términos. Combinamos ambos con Reciprocal Rank Fusion.
#
# Cargamos TODOS los chunks de Chroma a memoria una vez y construimos el
# índice BM25. Como el corpus es chico (cientos de chunks) esto es trivial
# en RAM y velocísimo. Para corpus enormes se usaría un índice invertido
# en disco (Elasticsearch/OpenSearch), pero a esta escala BM25 in-memory
# es lo correcto.
_bm25_instance = None          # BM25Okapi
_bm25_docs: list[Document] = []  # documentos en el MISMO orden que el índice
_bm25_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    """Tokenización simple para BM25: minúsculas, separa por no-alfanumérico,
    conserva tokens tipo '1asi0644' y 're1'. Suficiente para español técnico."""
    import re as _re
    return _re.findall(r"[a-záéíóúñ0-9]+", text.lower())


def get_bm25():
    """Construye (una vez) el índice BM25 sobre todos los chunks de Chroma.
    Si rank_bm25 no está instalado, devuelve (None, []) y search_docs cae
    a búsqueda solo-densa sin romperse."""
    global _bm25_instance, _bm25_docs
    if _bm25_instance is not None:
        return _bm25_instance, _bm25_docs

    with _bm25_lock:
        if _bm25_instance is None:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                print("[RAG] rank_bm25 no instalado → búsqueda solo-densa "
                      "(instala 'rank-bm25' para activar búsqueda híbrida)")
                return None, []
            db = get_db()
            col = db._collection
            data = col.get(include=["documents", "metadatas"])
            texts = data.get("documents", []) or []
            metas = data.get("metadatas", []) or []
            _bm25_docs = [
                Document(page_content=t, metadata=m or {})
                for t, m in zip(texts, metas)
            ]
            tokenized = [_tokenize(d.page_content) for d in _bm25_docs]
            _bm25_instance = BM25Okapi(tokenized) if tokenized else None
            print(f"[RAG] Índice BM25 construido sobre {len(_bm25_docs)} chunks")
    return _bm25_instance, _bm25_docs


def _passes_filter(meta: dict, carrera: str | None,
                   doc_types: Iterable[str] | None) -> bool:
    """Replica el filtro de Chroma para los resultados BM25 (que no pasan
    por el filtro de la DB). Misma lógica: carrera + 'base', doc_type + 'base'."""
    if carrera:
        if meta.get("carrera") not in (carrera.lower(), "base"):
            return False
    if doc_types:
        allowed = set(doc_types) | {"base"}
        if meta.get("doc_type") not in allowed:
            return False
    return True


# ──────────────────────────────────────────────────────────────
# Filtros
# ──────────────────────────────────────────────────────────────
def _build_filter(
    carrera: str | None,
    doc_types: Iterable[str] | None,
) -> dict | None:
    """
    Genera el filtro Chroma. Lógica:

    - Siempre incluye chunks de "base" (system prompt, glosario).
    - Si especifican carrera, incluye SOLO esa carrera + base.
    - Si especifican doc_types, además filtra por esos tipos (más "base").
    """
    where_carrera = None
    if carrera:
        where_carrera = {"carrera": {"$in": [carrera.lower(), "base"]}}

    where_doctype = None
    if doc_types:
        types = list(set(doc_types) | ALWAYS_INCLUDE_DOC_TYPES)
        where_doctype = {"doc_type": {"$in": types}}

    if where_carrera and where_doctype:
        return {"$and": [where_carrera, where_doctype]}
    return where_carrera or where_doctype


# ──────────────────────────────────────────────────────────────
# Búsqueda
# ──────────────────────────────────────────────────────────────
def search_docs(
    query: str,
    *,
    carrera: str | None = None,
    doc_types: Iterable[str] | None = None,
    k: int = 5,
) -> list[tuple[Document, float]]:
    """
    Búsqueda HÍBRIDA: combina recuperación densa (embeddings) + léxica (BM25)
    y fusiona ambos rankings con Reciprocal Rank Fusion (RRF).

    Por qué híbrida:
      - Densa (embeddings): capta semántica ("desempeño" ~ "rendimiento").
      - BM25 (léxica): capta keywords exactas que el denso suele perder,
        como "RE1", "outcomes", códigos "1ASI0644".
    Juntas cubren los dos modos de fallo. Es el estándar de la industria
    (lo usan Elasticsearch, Weaviate, etc.).

    Devuelve lista de (Document, score) ordenada por relevancia. El score
    devuelto es la similitud coseno del candidato (informativo); el ORDEN
    lo decide el RRF combinado.
    """
    if not query or not query.strip():
        return []

    import time as _t
    _t0 = _t.perf_counter()

    # Recuperamos un pool más amplio de cada método (candidate_k) y luego
    # fusionamos y recortamos a k. Un pool ~4x k es lo habitual.
    candidate_k = max(k * 4, 20)

    # ── 1. Recuperación DENSA (Chroma) ─────────────────────────────
    db = get_db()
    where = _build_filter(carrera, doc_types)
    dense_raw = db.similarity_search_with_score(
        query=query, k=candidate_k, filter=where,
    )
    # lista de (doc, similarity) y un id estable por chunk para fusionar
    dense_ranked: list[tuple[str, Document, float]] = []
    for doc, dist in dense_raw:
        sim = max(0.0, min(1.0, 1.0 - float(dist)))
        dense_ranked.append((_doc_id(doc), doc, sim))

    # ── 2. Recuperación LÉXICA (BM25) ──────────────────────────────
    bm25, bm25_docs = get_bm25()
    bm25_ranked: list[tuple[str, Document, float]] = []
    if bm25 is not None:
        scores = bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        taken = 0
        for i in order:
            # Si el score BM25 es ~0 el chunk no comparte NINGÚN término con
            # la query: no es un candidato léxico real, no lo aportamos al
            # fusion (evita que chunks genéricos entren "gratis" por RRF).
            if scores[i] <= 0.0:
                break
            d = bm25_docs[i]
            if not _passes_filter(d.metadata or {}, carrera, doc_types):
                continue
            bm25_ranked.append((_doc_id(d), d, float(scores[i])))
            taken += 1
            if taken >= candidate_k:
                break

    # ── 3. Weighted Reciprocal Rank Fusion ─────────────────────────
    # RRF (Cormack et al. 2009): score = Σ peso/(rrf_k + rank). Robusto
    # porque usa la POSICIÓN, no la escala (coseno vs BM25 no son comparables).
    # Ponderamos el denso por encima del léxico: para preguntas en lenguaje
    # natural el embedding semántico es más confiable; BM25 actúa como
    # "rescatador" de chunks con keywords exactas que el denso pierde.
    RRF_K = 60          # constante estándar del paper
    W_DENSE = 1.0       # peso del ranking denso
    W_BM25  = 0.5       # peso del ranking léxico (la mitad: es complemento)
    fused: dict[str, float] = {}
    doc_by_id: dict[str, Document] = {}
    sim_by_id: dict[str, float] = {}

    for rank, (cid, doc, sim) in enumerate(dense_ranked):
        fused[cid] = fused.get(cid, 0.0) + W_DENSE / (RRF_K + rank)
        doc_by_id[cid] = doc
        sim_by_id[cid] = sim
    for rank, (cid, doc, _s) in enumerate(bm25_ranked):
        fused[cid] = fused.get(cid, 0.0) + W_BM25 / (RRF_K + rank)
        doc_by_id.setdefault(cid, doc)
        sim_by_id.setdefault(cid, 0.0)

    # Ordenar por RRF; desempate por similitud densa (más interpretable).
    ranked_ids = sorted(
        fused,
        key=lambda c: (fused[c], sim_by_id.get(c, 0.0)),
        reverse=True,
    )[:k]
    out = [(doc_by_id[cid], sim_by_id[cid]) for cid in ranked_ids]

    _elapsed = _t.perf_counter() - _t0
    print(f"[RAG] hybrid '{query[:40]}' carrera={carrera} → "
          f"{len(out)} chunks (denso={len(dense_ranked)}, bm25={len(bm25_ranked)}) "
          f"en {_elapsed*1000:.0f}ms (filtro={where})")
    return out


def _doc_id(doc: Document) -> str:
    """ID estable de un chunk para fusionar rankings. Usa carrera+doc_type+
    sección+prefijo del contenido (suficiente para distinguir chunks)."""
    m = doc.metadata or {}
    head = doc.page_content[:60].strip()
    return f"{m.get('carrera','?')}|{m.get('doc_type','?')}|{m.get('section','?')}|{head}"


# ──────────────────────────────────────────────────────────────
# Formato para LLM
# ──────────────────────────────────────────────────────────────
def build_rag_context(
    results: list[tuple[Document, float]],
    *,
    min_score: float = 0.0,
    max_total_chars: int = 8000,
) -> str:
    """
    Convierte los chunks recuperados en un bloque de texto listo para
    inyectar al system prompt del LLM. Hace dos cosas:
    - Filtra resultados muy de baja relevancia (min_score).
    - Trunca para no pasarse de max_total_chars (presupuesto de tokens).
    """
    if not results:
        return ""

    out: list[str] = ["# Conocimiento recuperado (RAG)\n"]
    total = len(out[0])

    for i, (doc, score) in enumerate(results, 1):
        if score < min_score:
            continue

        meta = doc.metadata or {}
        header = (
            f"\n## Fragmento {i} "
            f"(carrera={meta.get('carrera','?')}, "
            f"doc={meta.get('doc_type','?')}, "
            f"sección={meta.get('section','?')}, "
            f"score={score:.2f})\n"
        )
        body = doc.page_content.strip() + "\n"

        if total + len(header) + len(body) > max_total_chars:
            out.append("\n[…contexto truncado por presupuesto de tokens…]\n")
            break

        out.append(header)
        out.append(body)
        total += len(header) + len(body)

    return "".join(out)


# ──────────────────────────────────────────────────────────────
# Stats / debug
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_stats() -> dict:
    """Útil para mostrar en /api/health o en la UI."""
    try:
        db = get_db()
        col = db._collection  # acceso al chromadb.Collection subyacente
        total = col.count()
        return {
            "available":    True,
            "total_chunks": total,
            "embedding":    EMBEDDING_MODEL_NAME,
            "chroma_path":  str(CHROMA_PATH),
            "metric":       "cosine",
            "carreras":     list_indexed_carreras(),
        }
    except Exception as e:
        return {
            "available": False,
            "error":     str(e),
        }


@lru_cache(maxsize=1)
def list_indexed_carreras() -> list[str]:
    """Lista las carreras realmente presentes en la ChromaDB (leyendo la
    metadata de todos los chunks). Excluye 'base'. Esto permite que el
    frontend ofrezca en el selector las carreras que el RAG conoce, aunque
    no estén como carpetas .md en knowledge/.

    Cacheado: solo se calcula una vez por proceso (reiniciar para refrescar
    tras un reindexado)."""
    try:
        db = get_db()
        col = db._collection
        data = col.get(include=["metadatas"])
        metas = data.get("metadatas", []) or []
        carreras = set()
        for m in metas:
            c = (m or {}).get("carrera")
            if c and c != "base":
                carreras.add(c)
        return sorted(carreras)
    except Exception as e:
        print(f"[RAG] No pude listar carreras indexadas: {e}")
        return []