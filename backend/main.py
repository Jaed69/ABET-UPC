import io
import os
import re
import json
import logging
import httpx
import PyPDF2
import docx2txt
from pathlib import Path
from typing import Optional, AsyncIterator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# Configuración centralizada (única fuente de verdad — ver config.py)
import config
from config import (
    LLM_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_MODEL,
    MAX_FILE_SIZE_MB,
    RAG_K,
    RAG_MIN_SCORE,
    RAG_MAX_CONTEXT_CHARS,
    DEFAULT_CARRERA,
    AUDIT_HEAVY_TOKEN_THRESHOLD,
    KNOWLEDGE_DIR,
    BASE_DIR,
    AUDIT_KNOWLEDGE_DIR,
    llm_headers,
    require_api_key,
    supports_web_search,
)

# Logging estructurado de queries (módulo local)
from logging_utils import QueryLog, compute_stats

# Ventana de contexto por defecto para Ollama (num_ctx). Por defecto Ollama usa
# 2048, muy poco para auditoría (system prompt + malla + documento). 16384 da
# margen para inputs grandes sin que el modelo devuelva vacío. Configurable por
# entorno (OLLAMA_NUM_CTX) sin tocar código.
DEFAULT_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("upc")
logger.info("Config: %s", config.summary())

# Crear directorios necesarios (idempotente)
KNOWLEDGE_DIR.mkdir(exist_ok=True)
BASE_DIR.mkdir(exist_ok=True)
AUDIT_KNOWLEDGE_DIR.mkdir(exist_ok=True)

# Tipos de documento que se esperan dentro de cada carpeta de carrera
CARRERA_DOC_TYPES = {
    "malla":        "malla.md",
    "control":      "reportes_control.md",
    "verificacion": "reporte_verificacion.md",
}

# ──────────────────────────────────────────────────────────────
# Integración RAG (opcional)
# ──────────────────────────────────────────────────────────────
# Si el módulo `rag/` está disponible y ChromaDB indexado, el flag use_rag
# del request hará que el system prompt se arme con chunks recuperados por
# embeddings en vez de cargar los .md completos.
# Si falla la import o no hay BD, el sistema sigue funcionando con el
# router clásico (el flag se ignora).
# ──────────────────────────────────────────────────────────────
# Integración RAG (encapsulada en una sola función)
# ──────────────────────────────────────────────────────────────
# El acoplamiento entre main.py y el módulo rag/ está limitado a DOS
# funciones públicas: get_rag_context() y rag_status().
# Todo lo demás (qué retriever se usa, qué embedding, cómo se chunkea,
# si se hace reranking, query expansion, búsqueda híbrida, etc.) vive
# dentro de rag/ y se puede cambiar sin tocar main.py.
#
# Contrato:
#   - get_rag_context(query, carrera, doc_types_hint) → (context_text, meta)
#       context_text: string listo para inyectar al system prompt (puede ser "")
#       meta: dict con {chunks, top_score, used_filters, ...}
#   - rag_status() → dict para /api/rag/status
#
# Si el módulo no carga (ChromaDB no indexado, deps no instaladas, etc.),
# RAG_AVAILABLE=False y el flag use_rag del request se ignora silenciosamente
# (fallback automático al router clásico).
try:
    from rag.retriever import search_docs as _rag_search_docs
    from rag.retriever import build_rag_context as _rag_build_context
    from rag.retriever import get_stats as _rag_get_stats
    from rag.retriever import list_indexed_carreras as _rag_list_carreras
    RAG_AVAILABLE = True
    logger.info("RAG module loaded.")
except Exception as _rag_err:
    _rag_search_docs   = None  # type: ignore
    _rag_build_context = None  # type: ignore
    _rag_get_stats     = lambda: {"available": False, "error": str(_rag_err)}
    _rag_list_carreras = lambda: []  # type: ignore
    RAG_AVAILABLE = False
    logger.warning("RAG no disponible: %s", _rag_err)


# Los parámetros del RAG (RAG_K, RAG_MIN_SCORE, RAG_MAX_CONTEXT_CHARS)
# se importan de config.py — ver arriba.


def get_rag_context(
    query: str,
    *,
    carrera: Optional[str] = None,
    doc_types_hint: Optional[list[str]] = None,
) -> tuple[str, dict]:
    """API estable que main.py usa para hablar con el módulo RAG.

    Devuelve (context_text, meta). Si RAG no está disponible o falla,
    devuelve ("", {"available": False, ...}) y el caller decide qué hacer
    (típicamente caer al modo knowledge clásico).

    Argumentos:
        query:           la pregunta del usuario.
        carrera:         filtro opcional ("cc", "civil", etc.).
        doc_types_hint:  filtro opcional (["malla","control"]). Si es None,
                         el RAG busca en todos los tipos.

    El retriever bajo el capó decide cómo recuperar; este wrapper sólo
    aplica los defaults y maneja errores.
    """
    if not RAG_AVAILABLE:
        return "", {
            "available":  False,
            "rag_used":   False,
            "rag_chunks": 0,
            "rag_top_score": 0.0,
            "reason": "RAG module not loaded",
        }

    try:
        results = _rag_search_docs(
            query,
            carrera=carrera,
            doc_types=doc_types_hint,
            k=RAG_K,
        )
    except Exception as e:
        logger.warning("RAG search falló: %s", e)
        return "", {
            "available":  False,
            "rag_used":   False,
            "rag_chunks": 0,
            "rag_top_score": 0.0,
            "reason": f"search error: {e}",
        }

    if not results:
        return "", {
            "available":  True,
            "rag_used":   True,
            "rag_chunks": 0,
            "rag_top_score": 0.0,
            "reason": "no results",
        }

    context_text = _rag_build_context(
        results,
        min_score=RAG_MIN_SCORE,
        max_total_chars=RAG_MAX_CONTEXT_CHARS,
    )

    top_score = round(results[0][1], 3) if results else 0.0
    return context_text, {
        "available":     True,
        "rag_used":      True,
        "rag_chunks":    len(results),
        "rag_top_score": top_score,
        "filters":       {"carrera": carrera, "doc_types": doc_types_hint},
    }


def rag_status() -> dict:
    """Estado del módulo RAG para diagnóstico (expuesto por /api/rag/status)."""
    return _rag_get_stats()


# ──────────────────────────────────────────────────────────────
# Caché en memoria del contenido de los .md (invalida por mtime)
# ──────────────────────────────────────────────────────────────
_kb_cache: dict[str, str]   = {}
_kb_mtime: dict[str, float] = {}

def _read_cached(fpath: Path) -> str:
    """Lee un archivo con caché. Si cambia el mtime en disco, recarga."""
    if not fpath.exists():
        return ""
    key   = str(fpath)
    mtime = fpath.stat().st_mtime
    if _kb_mtime.get(key) != mtime:
        _kb_cache[key] = fpath.read_text(encoding="utf-8").strip()
        _kb_mtime[key] = mtime
    return _kb_cache[key]


def _scan_comisiones(base_dir: Path) -> dict[str, dict[str, Path]]:
    """Dada una carpeta (la de un periodo), detecta sus comisiones.

    - Si tiene .md DIRECTAMENTE (malla.md) → comisión única "_default".
    - Si tiene SUBCARPETAS → cada una es una comisión con su nombre.

    Devuelve { comision_key: { doc_type: Path } }.
    """
    # .md directos → comisión única "_default"
    direct_files: dict[str, Path] = {}
    for doc_type, fname in CARRERA_DOC_TYPES.items():
        fpath = base_dir / fname
        if fpath.exists():
            direct_files[doc_type] = fpath

    subdirs = [
        d for d in sorted(base_dir.iterdir())
        if d.is_dir() and not d.name.startswith((".", "_"))
    ]

    comisiones: dict[str, dict[str, Path]] = {}
    if direct_files:
        comisiones["_default"] = direct_files

    for sub in subdirs:
        files: dict[str, Path] = {}
        for doc_type, fname in CARRERA_DOC_TYPES.items():
            fpath = sub / fname
            if fpath.exists():
                files[doc_type] = fpath
        if files:
            comisiones[sub.name.lower()] = files

    return comisiones


def discover_carreras() -> dict[str, dict[str, dict[str, dict[str, Path]]]]:
    """Escanea knowledge/ y devuelve carreras, PERIODOS y comisiones.

    Estructura devuelta (4 niveles):
        { carrera_key: { periodo_key: { comision_key: { doc_type: Path } } } }

    Soporta DOS layouts en disco (tolera ambos, incluso mezclados por carrera):

      A) Con nivel de periodo (jerarquía completa):
         knowledge/<carrera>/<periodo>/malla.md                  (comisión única)
         knowledge/<carrera>/<periodo>/<comision>/malla.md       (con comisiones)
         → knowledge/cc/2025-1/malla.md        → cc, 2025-1, "_default"
         → knowledge/sw/2025-1/eac_sw/malla.md → sw, 2025-1, eac_sw

      B) Sin nivel de periodo (malla directamente en la carpeta de carrera):
         knowledge/<carrera>/malla.md
         → se trata como periodo virtual "_default", comisión "_default".
         → knowledge/cc/malla.md → cc, "_default", "_default"

    Carpetas que empiezan con '_' (como _base) se ignoran. Si una carrera
    tiene .md directos Y subcarpetas de periodo, ambos se incluyen (los
    directos bajo "_default", las subcarpetas bajo su nombre).
    """
    out: dict[str, dict[str, dict[str, dict[str, Path]]]] = {}

    for carrera_dir in sorted(KNOWLEDGE_DIR.iterdir()):
        if not carrera_dir.is_dir() or carrera_dir.name.startswith("_"):
            continue

        carrera_key = carrera_dir.name.lower()
        periodos: dict[str, dict[str, dict[str, Path]]] = {}

        # Layout B: .md directos en la carpeta de carrera (sin nivel de
        # periodo) → periodo virtual "_default", comisión "_default".
        # OJO: solo miramos .md directos, NO subcarpetas (en layout A las
        # subcarpetas son periodos, no comisiones).
        direct_files: dict[str, Path] = {}
        for doc_type, fname in CARRERA_DOC_TYPES.items():
            fpath = carrera_dir / fname
            if fpath.exists():
                direct_files[doc_type] = fpath
        if direct_files:
            periodos["_default"] = {"_default": direct_files}

        # Layout A: cada subcarpeta de la carrera es un PERIODO.
        for periodo_dir in sorted(carrera_dir.iterdir()):
            if not periodo_dir.is_dir() or periodo_dir.name.startswith((".", "_")):
                continue
            comisiones = _scan_comisiones(periodo_dir)
            if comisiones:
                periodos[periodo_dir.name.lower()] = comisiones

        if periodos:
            out[carrera_key] = periodos

    return out


def listar_periodos(carrera: str) -> list[str]:
    """Lista los periodos disponibles para una carrera (ordenados)."""
    return sorted(discover_carreras().get((carrera or "").lower(), {}).keys())


def resolve_periodo(carrera: str, periodo: Optional[str]) -> Optional[str]:
    """Resuelve qué periodo usar para una carrera.

    - Si el periodo pedido existe → ese.
    - Si se pidió uno que NO existe → None (caller decide).
    - Si NO se pide periodo → el más reciente disponible (último al ordenar).
    - Si no hay periodos → None.
    """
    periodos = discover_carreras().get((carrera or "").lower(), {})
    if not periodos:
        return None
    if periodo and periodo.lower() in periodos:
        return periodo.lower()
    if periodo:
        return None  # se pidió uno que no existe
    return sorted(periodos.keys())[-1]  # más reciente por defecto


def carrera_tiene_comisiones(carrera: str, periodo: Optional[str] = None) -> bool:
    """True si la carrera (en ese periodo) tiene comisiones explícitas
    (no solo "_default")."""
    carreras = discover_carreras()
    periodos = carreras.get((carrera or "").lower(), {})
    per_key = resolve_periodo(carrera, periodo)
    if per_key is None or per_key not in periodos:
        return False
    coms = periodos[per_key]
    return any(c != "_default" for c in coms)


def resolve_comision(carrera: str, comision: Optional[str],
                     periodo: Optional[str] = None) -> Optional[str]:
    """Resuelve qué comisión usar para una carrera y periodo dados.

    - Carrera/periodo de comisión única → devuelve "_default".
    - Con comisiones y el usuario especifica una válida → esa.
    - Con comisiones y NO especifica → None (el caller decide).
    """
    per_key = resolve_periodo(carrera, periodo)
    periodos = discover_carreras().get((carrera or "").lower(), {})
    if per_key is None or per_key not in periodos:
        return None
    coms = periodos[per_key]
    if not coms:
        return None
    if list(coms.keys()) == ["_default"]:
        return "_default"
    if comision and comision.lower() in coms:
        return comision.lower()
    return None


# ═══════════════════════════════════════════════════════════════
# ROUTER DE RELEVANCIA v2
# Construido a partir del vocabulario REAL de los .md compactados.
# ═══════════════════════════════════════════════════════════════

# Regex para detectar códigos de curso UPC (1ACC0054, 1ASI0644, 1ACI0578, 1AIG0032, etc.)
COURSE_CODE_RE = re.compile(r"\b1(?:ACC|ACI|AHU|AMA|ASI|AAD|AIN|AIG)\d{3,4}\b", re.I)

# Regex para identificadores de outcome:
#   "RE1", "re3", "outcome 4", "outcome1", "outcome  6", "SO2"
OUTCOME_RE = re.compile(r"\b(?:re\s*\d+|outcome\s*\d+|so\s*\d+)\b", re.I)

# Palabras que aparecen en MALLA.md (estructura curricular)
MALLA_KEYWORDS = {
    "malla", "ciclo", "nivel", "niveles", "cf", "formativo", "formativos",
    "curso formativo", "cursos formativos", "electivo", "electivos",
    "qué cursos", "que cursos", "qué materias", "que materias", "asignatura",
    "asignaturas", "prerequisito", "plan curricular", "blended",
    "cocos", "leyenda", "minerva",
    "primer ciclo", "segundo ciclo", "tercer ciclo", "cuarto ciclo",
    "quinto ciclo", "sexto ciclo", "séptimo ciclo", "septimo ciclo",
    "octavo ciclo", "noveno ciclo", "décimo ciclo", "decimo ciclo",
    "qué nivel", "que nivel",
}

# Palabras que aparecen en REPORTES_CONTROL.md (datos cuantitativos por curso)
CONTROL_KEYWORDS = {
    "control", "reporte de control", "reportes de control",
    "necesita mejora", "esperado", "sobresaliente",
    "desempeño", "desempeno", "rendimiento",
    "crítico", "critico", "críticos", "criticos",
    "porcentaje", "porcentajes", "%",
    "alumnos", "estudiantes", "matriculados", "evaluados",
    "distribución", "distribucion", "nota", "notas",
    "aprobados", "desaprobados", "reprobados",
    "estadística del curso", "estadistica del curso",
    "cuántos alumnos", "cuantos alumnos",
}

# Palabras que aparecen en REPORTE_VERIFICACION.md (outcomes consolidados)
VERIFICACION_KEYWORDS = {
    "verificación", "verificacion",
    "reporte de verificación", "reporte de verificacion",
    "consolidado", "consolidados",
    "global", "globales",
    "outcomes de la carrera", "outcomes del programa",
    "definición de outcome", "definicion de outcome",
    "definiciones oficiales", "descripción oficial", "descripcion oficial",
    "logro final", "logro del programa",
    " rv ", "el rv", "del rv",
}

# Detectores de intención comparativa/analítica
COMPARATIVO_WORDS = {
    "peor", "mejor", "comparar", "comparación", "comparacion",
    "comparativa", "ranking", "más bajo", "mas bajo",
    "más alto", "mas alto", "diagnóstico", "diagnostico",
    "todos los outcomes", "cada outcome", "lista los outcomes",
    "qué outcomes", "que outcomes", "arrastra", "afecta",
    "más críticos", "mas criticos", "más crítico", "mas critico",
}

# Detectores de DEFINICIÓN PURA (resuelta solo con el glosario base)
DEFINITION_ONLY = (
    "qué significa cf", "que significa cf",
    "qué significan cf", "que significan cf",
    "qué significa que un curso tenga", "que significa que un curso tenga",
    "diferencia entre cf",
    "diferencia entre c y v",
    "qué es cf", "que es cf",
    "explica cf",
    "qué son cf", "que son cf",
    "símbolo cf", "simbolo cf",
    "leyenda de la malla",
)


def detect_relevant_docs(message: str) -> set[str]:
    """Decide qué doc_types cargar para una pregunta.

    Estrategia (en orden):
      1. Pregunta de DEFINICIÓN pura → set vacío (basta el glosario)
      2. Códigos de curso explícitos (1ACC0054) → malla + control
      3. IDs de outcome explícitos (RE1, outcome 3) → control + verificación
      4. Keywords temáticas por archivo → carga selectiva
      5. "outcome" genérico sin número → verificación
      6. Intención comparativa → control + verificación
      7. Cursos integradores (Taller, Seminario) → malla + control
      8. Fallback: pregunta larga sin keywords → carga todo
    """
    if not message or not message.strip():
        return set()

    q = " " + message.lower() + " "
    relevant: set[str] = set()

    # 1) Pregunta de DEFINICIÓN PURA → solo base (el glosario alcanza)
    if any(p in q for p in DEFINITION_ONLY):
        return set()

    # 2) Códigos de curso explícitos → necesita malla (ubicación) + control (datos)
    if COURSE_CODE_RE.search(message):
        relevant.update({"malla", "control"})

    # 3) IDs de outcome explícitos (RE1, outcome 3, etc.)
    if OUTCOME_RE.search(message):
        relevant.update({"control", "verificacion"})

    # 4) Keywords temáticas (palabras reales que viven en cada archivo)
    if any(kw in q for kw in MALLA_KEYWORDS):
        relevant.add("malla")
    if any(kw in q for kw in CONTROL_KEYWORDS):
        relevant.add("control")
    if any(kw in q for kw in VERIFICACION_KEYWORDS):
        relevant.add("verificacion")

    # 5) "outcome" genérico sin número → cargar RV (tiene definiciones oficiales)
    if "outcome" in q and not relevant:
        relevant.add("verificacion")

    # 6) Intención COMPARATIVA → RC (datos) + RV (definiciones)
    if any(t in q for t in COMPARATIVO_WORDS):
        relevant.update({"control", "verificacion"})

    # 7) Cursos integradores que viven entre malla y RC
    if "taller de proyecto" in q or "taller de desempeño" in q or \
       "taller de desempeno" in q or "seminario" in q:
        relevant.update({"malla", "control"})

    # 8) Fallback: pregunta de >3 palabras sin keywords → carga todo
    word_count = len([w for w in q.split() if len(w) > 2])
    if not relevant and word_count > 3:
        relevant = set(CARRERA_DOC_TYPES.keys())

    return relevant


# ──────────────────────────────────────────────────────────────
# Construcción del system prompt
# ──────────────────────────────────────────────────────────────
def load_base() -> str:
    """Carga el contenido de _base/ (system_prompt + glosario)."""
    parts: list[str] = []
    for fname in ("system_prompt.md", "glosario_simbolos.md"):
        content = _read_cached(BASE_DIR / fname)
        if content:
            parts.append(content)
    return "\n\n---\n\n".join(parts)


def load_carrera_docs(carrera: str, doc_types: set[str],
                      comision: Optional[str] = None,
                      periodo: Optional[str] = None) -> tuple[str, list[str]]:
    """Devuelve (texto_concatenado, lista_de_doc_types_realmente_cargados).

    Carga los .md de la carrera, en el PERIODO indicado (o el más reciente
    si no se indica), y la comisión indicada (o "_default" / la primera).
    """
    carreras = discover_carreras()
    key = (carrera or DEFAULT_CARRERA).lower()
    if key not in carreras:
        return "", []

    per_key = resolve_periodo(key, periodo)
    if per_key is None:
        return "", []
    periodos = carreras[key]
    coms = periodos.get(per_key, {})
    if not coms:
        return "", []

    # Resolver comisión: la pedida, o _default, o la primera que haya.
    com_key = resolve_comision(key, comision, per_key)
    if com_key is None:
        com_key = next(iter(coms))  # primera comisión como fallback
    docs = coms.get(com_key, {})

    parts: list[str] = []
    loaded: list[str] = []
    # Cargamos en orden estable (malla → control → verificacion) para que
    # el modelo vea primero el "qué" estructural y después los "datos"
    for doc_type in ("malla", "control", "verificacion"):
        if doc_type not in doc_types:
            continue
        fpath = docs.get(doc_type)
        if not fpath:
            continue
        content = _read_cached(fpath)
        if content:
            parts.append(f"### [{doc_type}]\n\n{content}")
            loaded.append(doc_type)
    return "\n\n---\n\n".join(parts), loaded


# ──────────────────────────────────────────────────────────────
# Modo auditoría
# ──────────────────────────────────────────────────────────────
def discover_audit_carreras() -> dict[str, list[Path]]:
    """Descubre las carreras disponibles para auditar y sus documentos de
    referencia. Combina DOS fuentes:

      1. audit_knowledge/<carrera>/   → crudos convertidos por build_audit_context
         (PDF/Excel/Word completos de la carrera). Tienen prioridad.
      2. knowledge/<carrera>/         → los .md que ya alimentan el RAG/knowledge
         (malla, reportes de control, verificación).

    Así el modo auditoría reutiliza automáticamente lo que ya cargaste en
    knowledge/ sin tener que duplicar archivos. Si una carrera tiene contexto
    en ambas carpetas, se usan los dos (audit_knowledge primero).

    Devuelve {carrera: [lista de .md, sin contar índices]}.
    """
    result: dict[str, list[Path]] = {}

    # Fuente 1: audit_knowledge/ (crudos convertidos)
    if AUDIT_KNOWLEDGE_DIR.exists():
        for sub in sorted(AUDIT_KNOWLEDGE_DIR.iterdir()):
            if not sub.is_dir() or sub.name.startswith((".", "_")):
                continue
            mds = sorted(f for f in sub.glob("*.md") if f.name != "_index.md")
            if mds:
                result[sub.name] = list(mds)

    # Fuente 2: knowledge/<carrera>/ (lo que ya usa el RAG/knowledge). Se AÑADE
    # a lo que ya haya, sin pisarlo. Excluye _base (no es una carrera).
    # Soporta comisiones: si la carrera tiene subcarpetas, las mallas están
    # dentro de cada comisión. Para auditoría juntamos las mallas de TODAS las
    # comisiones bajo la clave de la carrera (la validación contra la comisión
    # específica la maneja load_audit_context con el parámetro comision).
    if KNOWLEDGE_DIR.exists():
        for sub in sorted(KNOWLEDGE_DIR.iterdir()):
            if not sub.is_dir() or sub.name.startswith((".", "_")):
                continue
            # .md directos (comisión única)
            mds = sorted(
                f for f in sub.glob("*.md")
                if f.name not in ("_index.md", "system_prompt.md")
            )
            # .md dentro de subcarpetas (comisiones)
            for comdir in sorted(d for d in sub.iterdir()
                                 if d.is_dir() and not d.name.startswith((".", "_"))):
                mds.extend(sorted(
                    f for f in comdir.glob("*.md")
                    if f.name not in ("_index.md", "system_prompt.md")
                ))
            if mds:
                result.setdefault(sub.name, [])
                existing = {p for p in result[sub.name]}
                for m in mds:
                    if m not in existing:
                        result[sub.name].append(m)

    return result


def load_audit_context(carrera: str, comision: Optional[str] = None,
                       periodo: Optional[str] = None) -> tuple[str, dict]:
    """Carga la documentación de referencia de una carrera para auditar.

    Usa el PERIODO indicado (o el más reciente). Si la carrera/periodo tiene
    comisiones, carga SOLO la malla de la comisión indicada. Si es comisión
    única o no se indica, carga lo disponible.

    Devuelve (texto, meta) con la lista de documentos y tokens aprox.
    """
    key = (carrera or "").lower()

    per_key = resolve_periodo(key, periodo)
    com_key = resolve_comision(key, comision, per_key)
    carreras_knowledge = discover_carreras()

    if (key in carreras_knowledge and per_key
            and per_key in carreras_knowledge[key]
            and com_key and com_key in carreras_knowledge[key][per_key]):
        docs_map = carreras_knowledge[key][per_key][com_key]
        parts, doc_names = [], []
        for doc_type, fpath in docs_map.items():
            content = _read_cached(fpath)
            if content:
                parts.append(content)
                doc_names.append(fpath.stem)
        if parts:
            full = "\n\n---\n\n".join(parts)
            return full, {
                "available":    True,
                "carrera":      key,
                "periodo":      per_key,
                "comision":     com_key if com_key != "_default" else None,
                "docs":         doc_names,
                "doc_count":    len(doc_names),
                "tokens_aprox": len(full) // 4,
            }

    # Fallback: no se encontró en la estructura nueva.
    return "", {
        "available":  False,
        "carrera":    key,
        "periodo":    per_key,
        "comision":   None,
        "docs":       [],
        "doc_count":  0,
        "tokens_aprox": 0,
    }


def build_full_system_prompt(
    user_msg: str = "",
    user_system_prompt: Optional[str] = None,
    carrera: Optional[str] = None,
    comision: Optional[str] = None,
    periodo: Optional[str] = None,
    has_attached_files: bool = False,
    use_knowledge: bool = True,
    use_rag: bool = False,
    audit: bool = False,
    audit_format: str = "text",
    raw: bool = False,
) -> tuple[str, dict]:
    """Arma el system prompt y devuelve metadata para debugging.

    Modos posibles (en orden de prioridad):
      - audit=True                        → modo auditoría: carga TODO el
                                            contexto de la carrera desde
                                            audit_knowledge/ (ignora knowledge/RAG)
      - use_knowledge=False               → modo asistente general (~30 tk sys)
      - use_knowledge=True, use_rag=True  → RAG sobre ChromaDB (chunks recuperados)
      - use_knowledge=True, use_rag=False → router clásico (carga .md enteros)
    """

    # ── Modo CRUDO (raw): sin NINGÚN system prompt ──────────────
    # Cuando raw=True el modelo no recibe instrucciones de sistema de ningún
    # tipo (ni base, ni "asistente útil", ni knowledge). Responde como Ollama
    # puro. Útil para pasarle instrucciones propias en el mensaje del usuario
    # sin que el system prompt base interfiera (saludos, emojis, etc.).
    if raw:
        return "", {
            "loaded_docs":   [],
            "chars":         0,
            "tokens_aprox":  0,
            "carrera":       None,
            "mode":          "raw",
            "rag_used":      False,
            "rag_chunks":    0,
            "rag_top_score": 0.0,
        }

    # ── Modo 0: AUDITORÍA (prioridad máxima) ────────────────────
    # Carga toda la documentación de la carrera (convertida con MarkItDown)
    # para que el modelo audite los archivos que el usuario adjunte.
    # No usa el knowledge clásico ni el RAG: el objetivo es contexto completo.
    if audit:
        carrera_key = (carrera or "").lower()
        audit_ctx, audit_meta = load_audit_context(carrera_key, comision, periodo)

        if not audit_meta.get("available") or not audit_ctx:
            # No hay contexto de auditoría para esa carrera → avisamos en el
            # prompt para que el modelo no invente, pero no rompemos.
            fallback_prompt = (
                "Eres un asistente de auditoría académica para la UPC. "
                "El usuario activó el modo auditoría para la carrera "
                f"'{carrera_key}', pero NO hay documentación de contexto "
                "cargada para esa carrera. Indícale que debe generar el "
                "contexto con `python -m tools.build_audit_context` y que "
                "mientras tanto solo puedes analizar los archivos que adjunte "
                "directamente, sin el contexto de la carrera."
            )
            if user_system_prompt and user_system_prompt.strip():
                fallback_prompt += "\n\n## Instrucciones del coordinador\n\n" + user_system_prompt.strip()
            return fallback_prompt, {
                "loaded_docs":   [],
                "chars":         len(fallback_prompt),
                "tokens_aprox":  len(fallback_prompt) // 4,
                "carrera":       carrera_key,
                "mode":          "audit",
                "audit_available": False,
                "audit_docs":    [],
                "rag_used":      False,
                "rag_chunks":    0,
                "rag_top_score": 0.0,
            }

        # Instrucción de salida según el formato pedido
        if audit_format == "json":
            format_block = (
                "## Formato de salida OBLIGATORIO\n\n"
                "Responde ÚNICAMENTE con un objeto JSON válido (sin markdown, sin "
                "backticks, sin texto antes ni después) con esta estructura exacta:\n\n"
                "{\n"
                '  "veredicto": "correcto" | "con_observaciones" | "incorrecto",\n'
                '  "resumen": "<1-2 oraciones del estado general>",\n'
                '  "hallazgos": [\n'
                "    {\n"
                '      "severidad": "alta" | "media" | "baja",\n'
                '      "ubicacion": "<dónde está el problema: campo, fila, sección>",\n'
                '      "problema": "<qué está mal>",\n'
                '      "referencia": "<documento oficial contra el que se contrastó, si aplica>",\n'
                '      "sugerencia": "<cómo corregirlo>"\n'
                "    }\n"
                "  ],\n"
                '  "campos_correctos": ["<lista de lo que SÍ está bien>"]\n'
                "}\n\n"
                "Si todo está correcto, `hallazgos` es un array vacío y `veredicto` es "
                '"correcto". No inventes hallazgos para llenar.\n\n'
            )
        else:
            format_block = (
                "## Formato de salida\n\n"
                "Responde en texto claro y estructurado. Usa encabezados y listas. "
                "Empieza con un veredicto general (correcto / con observaciones / "
                "incorrecto), luego detalla cada hallazgo con su ubicación, el "
                "problema, la referencia oficial y la sugerencia de corrección. "
                "Termina mencionando explícitamente lo que está correcto.\n\n"
            )

        # Etiqueta de comisión y periodo para el prompt
        _com = audit_meta.get("comision")
        _per = audit_meta.get("periodo")
        carrera_label = carrera_key.upper()
        if _per:
            carrera_label += f" — periodo {_per}"
        if _com:
            carrera_label += f" — comisión {_com.upper()}"

        # Instrucción de sistema específica de auditoría
        audit_instructions = (
            "# Rol: Auditor académico UPC\n\n"
            "Eres un asistente especializado en auditar documentación académica "
            f"de la carrera **{carrera_label}** para acreditación ABET. "
            "A continuación tienes la documentación oficial de la carrera "
            "(la malla de la comisión correspondiente) como contexto de referencia.\n\n"
            "Cuando el usuario te pida revisar o auditar archivos (que adjuntará), "
            "tu tarea es:\n"
            "1. Comparar los archivos adjuntos contra la documentación oficial de la carrera.\n"
            "2. Verificar consistencia (códigos de curso, outcomes, porcentajes, formatos).\n"
            "3. Verificar que el archivo adjunto realmente corresponde a esta carrera"
            + (" y comisión" if _com else "") +
            "; si parece de otra, señálalo como hallazgo.\n"
            "4. Señalar errores, inconsistencias o datos faltantes de forma concreta.\n"
            "5. Citar el documento de referencia cuando detectes una discrepancia "
            "(los documentos indican su fuente en el encabezado).\n"
            "6. Si algo está correcto, decirlo explícitamente.\n\n"
            "No inventes datos. Si la información no está ni en el contexto ni en "
            "los archivos adjuntos, dilo claramente.\n\n"
            + format_block +
            "---\n\n"
            f"# Documentación oficial de {carrera_label} (contexto de referencia)\n\n"
            f"{audit_ctx}"
        )

        if user_system_prompt and user_system_prompt.strip():
            audit_instructions += (
                "\n\n---\n\n## Instrucciones adicionales del coordinador\n\n"
                + user_system_prompt.strip()
            )

        return audit_instructions, {
            "loaded_docs":     audit_meta["docs"],
            "chars":           len(audit_instructions),
            "tokens_aprox":    len(audit_instructions) // 4,
            "carrera":         carrera_key,
            "periodo":         audit_meta.get("periodo"),
            "comision":        audit_meta.get("comision"),
            "mode":            "audit",
            "audit_available": True,
            "audit_docs":      audit_meta["docs"],
            "audit_doc_count": audit_meta["doc_count"],
            "audit_format":    audit_format,
            "rag_used":        False,
            "rag_chunks":      0,
            "rag_top_score":   0.0,
        }

    # ── Modo 1: asistente general (sin knowledge) ───────────────
    if not use_knowledge:
        parts: list[str] = [
            "Eres un asistente útil, claro y honesto. Responde en el "
            "mismo idioma que el usuario y sé conciso a menos que pidan detalle."
        ]
        if user_system_prompt and user_system_prompt.strip():
            parts.append("## Instrucciones del usuario\n\n" + user_system_prompt.strip())
        full = "\n\n".join(parts)
        return full, {
            "loaded_docs":   [],
            "chars":         len(full),
            "tokens_aprox":  len(full) // 4,
            "carrera":       None,
            "mode":          "general",
            "rag_used":      False,
            "rag_chunks":    0,
            "rag_top_score": 0.0,
        }

    # ── Modo 2: RAG con ChromaDB ────────────────────────────────
    # Toda la complejidad del retrieval vive detrás de get_rag_context().
    # Si más adelante el módulo rag/ cambia (otro retriever, reranking,
    # búsqueda híbrida, etc.), este bloque NO debería tener que cambiar.
    if use_rag and RAG_AVAILABLE:
        carrera_key = (carrera or DEFAULT_CARRERA).lower()
        # Le pasamos al RAG el "hint" del router para filtrar la búsqueda
        # (si la pregunta menciona códigos → malla+control, etc.).
        # Si el router no detecta nada, el RAG busca en todos los tipos.
        suggested = detect_relevant_docs(user_msg)
        doc_types_hint: Optional[list[str]] = (
            list(suggested) if suggested else None
        )

        rag_context, rag_meta = get_rag_context(
            user_msg,
            carrera=carrera_key,
            doc_types_hint=doc_types_hint,
        )

        # Si el RAG falló o no devolvió contexto útil, fallback a knowledge clásico
        if not rag_meta.get("available") or not rag_context:
            logger.info("RAG sin contexto utilizable (%s), fallback a knowledge clásico.",
                        rag_meta.get('reason', '?'))
            return build_full_system_prompt(
                user_msg=user_msg,
                user_system_prompt=user_system_prompt,
                carrera=carrera,
                comision=comision,
                periodo=periodo,
                has_attached_files=has_attached_files,
                use_knowledge=True,
                use_rag=False,
            )

        # Armar el system prompt: base + RAG context + instrucción opcional
        parts: list[str] = []
        base = load_base()
        if base:
            parts.append(base)
        parts.append(
            f"## Contexto recuperado vía RAG ({carrera_key.upper()})\n\n"
            "Estos son los fragmentos más relevantes para la pregunta del "
            "usuario, recuperados mediante búsqueda semántica sobre el "
            "conocimiento institucional. No inventes información que no "
            "esté aquí.\n\n" + rag_context
        )
        if user_system_prompt and user_system_prompt.strip():
            parts.append("## Instrucciones del coordinador\n\n" + user_system_prompt.strip())

        full = "\n\n".join(parts)

        return full, {
            "loaded_docs":   doc_types_hint or ["auto"],
            "chars":         len(full),
            "tokens_aprox":  len(full) // 4,
            "carrera":       carrera_key,
            "mode":          "rag",
            "rag_used":      True,
            "rag_chunks":    rag_meta.get("rag_chunks", 0),
            "rag_top_score": rag_meta.get("rag_top_score", 0.0),
        }

    # ── Modo 3: knowledge clásico (router carga .md enteros) ────
    base = load_base()
    relevant = detect_relevant_docs(user_msg)

    # Si hay archivos adjuntos y la pregunta es vaga, no cargues docs de carrera:
    # probablemente el contenido viene del adjunto
    if has_attached_files and not relevant:
        relevant = set()

    carrera_text, loaded = load_carrera_docs(carrera or DEFAULT_CARRERA, relevant, comision, periodo)

    parts: list[str] = []
    if base:
        parts.append(base)
    if carrera_text:
        parts.append(
            f"## Datos de la carrera ({(carrera or DEFAULT_CARRERA).upper()})\n\n"
            "Esta es tu fuente de verdad para datos curriculares. "
            "No inventes información que no esté aquí.\n\n" + carrera_text
        )
    if user_system_prompt and user_system_prompt.strip():
        parts.append("## Instrucciones del coordinador\n\n" + user_system_prompt.strip())

    full = "\n\n".join(parts) if parts else (
        "Eres un asistente de verificación de outcomes para la UPC."
    )

    return full, {
        "loaded_docs":   loaded,
        "chars":         len(full),
        "tokens_aprox":  len(full) // 4,
        "carrera":       (carrera or DEFAULT_CARRERA).lower(),
        "mode":          "knowledge",
        "rag_used":      False,
        "rag_chunks":    0,
        "rag_top_score": 0.0,
    }


# ──────────────────────────────────────────────────────────────
# Extracción de archivos adjuntos
# ──────────────────────────────────────────────────────────────
def extract_pdf(data: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(p for p in pages if p.strip())
        return text or "[PDF sin texto extraíble — puede ser imagen escaneada]"
    except Exception as e:
        return f"[Error leyendo PDF: {e}]"

def extract_docx(data: bytes) -> str:
    try:
        text = docx2txt.process(io.BytesIO(data))
        return text.strip() or "[DOCX vacío]"
    except Exception as e:
        return f"[Error leyendo DOCX: {e}]"

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".html",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".yaml", ".yml",
    ".toml", ".ini", ".sh", ".sql", ".rst", ".tex",
}

def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(data)
    if ext in (".docx", ".doc"):
        return extract_docx(data)
    if ext in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace")
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return f"[No se pudo leer: {filename}]"


# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "system"):
            raise ValueError(f"role inválido: {v!r}")
        return v

class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str]         = None
    stream: bool                 = True
    temperature: float           = 0.3
    max_tokens: int              = 4096
    system_prompt: Optional[str] = None
    carrera: Optional[str]       = None
    comision: Optional[str]      = None   # ← comisión dentro de la carrera (si tiene varias)
    periodo: Optional[str]       = None   # ← periodo académico (ej. 2025-1); si no se indica, el más reciente
    use_knowledge: bool          = True   # ← cargar o no los .md institucionales
    web_search:    bool          = False  # ← activar plugin web de OpenRouter
    use_rag:       bool          = False  # ← usar ChromaDB para recuperar contexto
    audit:         bool          = False  # ← modo auditoría: contexto completo de la carrera
    audit_format:  str           = "text" # ← "text" | "json" (veredicto estructurado)
    think:         Optional[bool] = None  # ← None=default del modelo; False=sin razonamiento (rápido); True=con razonamiento
    raw:           bool          = False  # ← True=sin ningún system prompt (modelo crudo, como Ollama directo)
    num_ctx:       Optional[int] = None   # ← ventana de contexto de Ollama; None usa el default (16384)
    json_output:   bool          = False  # ← forzar salida JSON válida (response_format json_object)


# ──────────────────────────────────────────────────────────────
# Helpers de comunicación con el motor LLM
# ──────────────────────────────────────────────────────────────
def _check_key() -> None:
    """Verifica que haya API key cuando es obligatoria. Traduce el error
    de config a una HTTPException 503 apropiada para la API."""
    try:
        require_api_key()
    except RuntimeError as e:
        raise HTTPException(503, str(e))

def _build_payload(messages: list[dict], model: str, stream: bool,
                   temperature: float, max_tokens: int,
                   web_search: bool = False,
                   web_max_results: int = 5,
                   json_output: bool = False,
                   think: Optional[bool] = None,
                   num_ctx: Optional[int] = None) -> dict:
    payload = {
        "model":       model,
        "messages":    messages,
        "stream":      stream,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    # ── Ventana de contexto (num_ctx) para Ollama ───────────────
    # Por defecto Ollama usa una ventana pequeña (2048 tokens). Si el input
    # (system prompt + malla + documento a auditar) supera ese tamaño, el
    # modelo se queda sin espacio y puede devolver una respuesta VACÍA.
    # Subir num_ctx evita eso. Se pasa dentro de "options", que es como Ollama
    # recibe sus parámetros nativos en el endpoint OpenAI-compatible.
    # Solo aplica en local (Ollama); OpenRouter lo ignora.
    if LLM_PROVIDER == "local":
        ctx = num_ctx or DEFAULT_NUM_CTX
        payload["options"] = {
            "num_ctx": ctx,
            # num_predict = máximo de tokens de SALIDA en Ollama. -1 = sin
            # límite (genera hasta terminar naturalmente). Evita que la
            # respuesta se corte a mitad (ej. listados largos de cursos).
            "num_predict": -1,
        }
        # En el endpoint OpenAI-compat de Ollama, dejar también max_tokens en el
        # nivel superior puede entrar en conflicto con num_predict y cortar la
        # respuesta antes de tiempo. Como ya controlamos la salida con
        # num_predict, quitamos max_tokens del payload en modo local.
        payload.pop("max_tokens", None)

    # ── Control del "thinking mode" (modelos de razonamiento como Qwen 3.5) ──
    # El razonamiento genera muchos tokens internos antes de responder, lo que
    # ralentiza la respuesta y, con inputs grandes, puede dejar el `content`
    # VACÍO (todo el presupuesto se va al razonamiento). Para auditoría no lo
    # necesitamos.
    # `think` puede venir en la request:
    #   - think=False → desactiva el razonamiento
    #   - think=True  → lo deja activo
    #   - think=None  → no se toca (comportamiento por defecto del modelo)
    # En el endpoint OpenAI-compatible de Ollama, el control que SÍ funciona es
    # "reasoning_effort": "none" (probado con gemma/qwen). Los campos "think" y
    # "chat_template_kwargs" no surten efecto por esta vía, así que usamos
    # reasoning_effort. Solo en local; OpenRouter usa su propia semántica.
    if think is not None and LLM_PROVIDER == "local":
        payload["reasoning_effort"] = "low" if think else "none"

    # Pedirle al motor LLM que incluya `usage` en el stream.
    # Tanto OpenRouter como vLLM y llama.cpp respetan stream_options.
    if stream:
        payload["stream_options"] = {"include_usage": True}

    # Forzar salida JSON (para el veredicto estructurado de auditoría).
    # OpenRouter, vLLM y llama.cpp soportan response_format json_object.
    if json_output:
        payload["response_format"] = {"type": "json_object"}

    # Plugin de búsqueda web — SOLO existe en OpenRouter. En local lo
    # omitimos porque vLLM puede tirar 400 ante campos desconocidos.
    if web_search and supports_web_search():
        payload["plugins"] = [
            {"id": "web", "max_results": web_max_results}
        ]

    return payload

def _openrouter_headers() -> dict:
    """Headers para hablar con el motor LLM. Delegado a config.llm_headers()."""
    return llm_headers()

async def _stream_openrouter(
    payload: dict,
    query_log: Optional[QueryLog] = None,
) -> AsyncIterator[str]:
    """Stream del backend → frontend, pasando los chunks de OpenRouter tal cual.
    Adicional:
      - Detecta el objeto `usage` y lo reenvía como evento SSE 'event: usage'.
      - Detecta `annotations` (citas web search) y las reenvía como
        evento SSE 'event: citations' para que el frontend las muestre.
      - Si se pasa `query_log`, guarda usage en él y lo finaliza al cierre.
    """
    sent_citations = False
    finished = False
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST",
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=_openrouter_headers(),
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    err = body.decode(errors='replace')[:500]
                    yield f"data: [ERROR] {err}\n\n"
                    if query_log:
                        query_log.finish(status="error", error=f"HTTP {resp.status_code}: {err[:200]}")
                        finished = True
                    return

                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue

                    # Parsear chunk para detectar metadata extra
                    if raw_line.startswith("data: ") and raw_line != "data: [DONE]":
                        raw = raw_line[6:].strip()
                        try:
                            obj = json.loads(raw)

                            # ── usage (tokens reales) ──
                            usage = obj.get("usage")
                            if usage and isinstance(usage, dict):
                                pt = usage.get("prompt_tokens", 0)
                                ct = usage.get("completion_tokens", 0)
                                usage_payload = json.dumps({
                                    "prompt_tokens":     pt,
                                    "completion_tokens": ct,
                                    "total_tokens":      usage.get("total_tokens", 0),
                                })
                                yield f"event: usage\ndata: {usage_payload}\n\n"
                                if query_log:
                                    query_log.set_usage(pt, ct)

                            # ── annotations / citaciones web ──
                            if not sent_citations:
                                choices = obj.get("choices", [])
                                for choice in choices:
                                    anns = (choice.get("delta", {}).get("annotations")
                                            or choice.get("message", {}).get("annotations"))
                                    if anns and isinstance(anns, list):
                                        citations = []
                                        for a in anns:
                                            if a.get("type") == "url_citation":
                                                uc = a.get("url_citation", {})
                                                citations.append({
                                                    "url":   uc.get("url", ""),
                                                    "title": uc.get("title", ""),
                                                })
                                        if citations:
                                            yield (f"event: citations\n"
                                                   f"data: {json.dumps(citations)}\n\n")
                                            sent_citations = True
                                            break
                        except (json.JSONDecodeError, AttributeError):
                            pass

                    yield f"{raw_line}\n\n"

        # Stream terminó OK
        if query_log and not finished:
            query_log.finish(status="success")
            finished = True

    except Exception as e:
        # El cliente abortó (Disconnect) o algo más falló
        if query_log and not finished:
            # Diferenciar abort de error real es difícil; usamos "aborted" si
            # el error es de tipo CancelledError o ClientDisconnect
            err_name = type(e).__name__
            status = "aborted" if "Cancelled" in err_name or "Disconnect" in err_name else "error"
            query_log.finish(status=status, error=f"{err_name}: {str(e)[:200]}")
        raise


async def _call_openrouter_sync(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=600) as client:
        r = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=_openrouter_headers(),
            json=payload,
        )
    if r.status_code != 200:
        try:    detail = r.json()
        except: detail = r.text
        raise HTTPException(r.status_code, detail=detail)
    return r.json()


# ──────────────────────────────────────────────────────────────
# Headers comunes de respuesta
# ──────────────────────────────────────────────────────────────
def _build_meta_headers(meta: dict, web_search: bool) -> dict:
    """Construye los headers X-* que el frontend lee para mostrar metadata.
    Los strings se serializan a str() porque algunos servidores ASGI son
    estrictos con tipos no-string en headers."""
    loaded = meta.get("loaded_docs") or []
    return {
        "X-Accel-Buffering":     "no",
        "Cache-Control":         "no-cache",
        "X-Loaded-Docs":         ",".join(loaded) if loaded else "none",
        "X-System-Tokens-Aprox": str(meta.get("tokens_aprox", 0)),
        "X-Mode":                meta.get("mode", "knowledge"),
        "X-Web-Search":          "1" if web_search else "0",
        "X-Rag-Used":            "1" if meta.get("rag_used") else "0",
        "X-Rag-Chunks":          str(meta.get("rag_chunks", 0)),
        "X-Rag-Top-Score":       str(meta.get("rag_top_score", 0.0)),
    }


# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="UPC ABET API", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # CRÍTICO: sin esto el navegador oculta headers custom al JS,
    # y el frontend no puede leer X-System-Tokens-Aprox ni X-Loaded-Docs.
    expose_headers=[
        "X-System-Tokens-Aprox",
        "X-Loaded-Docs",
        "X-Mode",
        "X-Web-Search",
        "X-Rag-Used",
        "X-Rag-Chunks",
        "X-Rag-Top-Score",
        "X-Accel-Buffering",
    ],
)


# ──────────────────────────────────────────────────────────────
# No-cache para endpoints de estado (GET /api/*)
# ──────────────────────────────────────────────────────────────
# El frontend pide /api/knowledge, /api/carreras, /api/provider, etc. al
# cargar. Si el navegador cachea esas respuestas, al volver a entrar a la
# página puede servir una versión vieja/incompleta y el frontend se queda
# "buscando" en bucle (había que limpiar cookies para arreglarlo). Forzamos
# que estas respuestas NUNCA se cacheen.
@app.middleware("http")
async def no_cache_status_endpoints(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    # No cachear los endpoints de estado (/api/*) NI el HTML de la página.
    # El index.html se servía con caché del navegador (StaticFiles manda
    # ETag/Last-Modified), así que al recargar se usaba una versión vieja del
    # frontend y había que limpiar cookies/caché para ver los cambios. Forzamos
    # que el HTML y la raíz tampoco se cacheen. Los assets con hash (logos,
    # etc.) sí pueden cachearse, por eso solo apuntamos a HTML y la raíz.
    is_api = path.startswith("/api/") and request.method == "GET"
    is_html = path == "/" or path.endswith(".html")
    if is_api or is_html:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    carreras = discover_carreras()
    base_chars = len(load_base())
    # En modo local la key es opcional, así que reportamos key_set=True
    # para no confundir al frontend.
    key_set = (LLM_PROVIDER == "local") or bool(OPENROUTER_API_KEY)
    return {
        "status":   "ok",
        "model":    DEFAULT_MODEL,
        "key_set":  key_set,
        "provider": LLM_PROVIDER,
        "base_url": OPENROUTER_BASE_URL,
        "base_tokens_aprox":    base_chars // 4,
        "carreras_disponibles": list(carreras.keys()),
        "carreras_detalle": {
            # Por carrera: lista sus periodos y, dentro de cada uno, las
            # comisiones (o "_default" si es comisión única).
            k: {
                "periodos": {
                    per: (
                        {"doc_types": list(coms["_default"].keys())}
                        if list(coms.keys()) == ["_default"]
                        else {"comisiones": {ck: list(cv.keys()) for ck, cv in coms.items()}}
                    )
                    for per, coms in periodos.items()
                }
            }
            for k, periodos in carreras.items()
        },
        "rag":       RAG_AVAILABLE,
        "rag_stats": rag_status() if RAG_AVAILABLE else None,
        # Capacidades según provider — el frontend puede usarlo para
        # ocultar el toggle de web search cuando estamos en local.
        "capabilities": {
            "web_search": LLM_PROVIDER == "openrouter",
            "rag":        RAG_AVAILABLE,
        },
    }

@app.get("/api/provider")
def get_provider():
    """Info del proveedor LLM activo. El frontend lo consulta al
    cargar para adaptar la UI (esconder web_search si es local, etc.)."""
    return {
        "provider":   LLM_PROVIDER,
        "base_url":   OPENROUTER_BASE_URL,
        "model":      DEFAULT_MODEL,
        "web_search": LLM_PROVIDER == "openrouter",
    }

@app.get("/api/models")
async def list_models():
    """Lista de modelos disponibles.
    - En OpenRouter: pide la lista completa al endpoint /models.
    - En local: vLLM y llama.cpp también exponen /models, pero solo
      devuelven el modelo que están sirviendo actualmente (lo que es
      correcto: no podés cambiar de modelo en runtime sin reiniciar)."""
    _check_key()
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{OPENROUTER_BASE_URL}/models",
                headers=_openrouter_headers(),
            )
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text)
        return r.json()
    except httpx.ConnectError as e:
        # vLLM/llama.cpp pueden no estar arrancados todavía
        raise HTTPException(503, f"No se pudo contactar al motor LLM en {OPENROUTER_BASE_URL}: {e}")

@app.get("/api/knowledge")
def list_knowledge():
    """Lista todo el knowledge cargado con su peso en tokens."""
    carreras = discover_carreras()
    return {
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "base": [
            {
                "name":         f.name,
                "chars":        len(_read_cached(f)),
                "tokens_aprox": len(_read_cached(f)) // 4,
            }
            for f in sorted(BASE_DIR.glob("*.md"))
        ],
        "carreras": {
            k: {
                periodo_key: {
                    com_key: {
                        doc_type: {
                            "path":         str(fpath.relative_to(KNOWLEDGE_DIR)),
                            "chars":        len(_read_cached(fpath)),
                            "tokens_aprox": len(_read_cached(fpath)) // 4,
                        }
                        for doc_type, fpath in docs.items()
                    }
                    for com_key, docs in coms.items()
                }
                for periodo_key, coms in periodos.items()
            }
            for k, periodos in carreras.items()
        },
    }

@app.get("/api/carreras")
def list_carreras():
    carreras = discover_carreras()
    out = []
    for k, periodos in carreras.items():
        # Comisiones por periodo (para que el frontend sepa, al elegir periodo,
        # qué comisiones mostrar). "_default" se traduce a lista vacía.
        comisiones_por_periodo = {}
        for per_key, coms in periodos.items():
            reales = [c for c in coms.keys() if c != "_default"]
            comisiones_por_periodo[per_key] = reales
        entry = {
            "key":      k,
            "periodos": sorted(periodos.keys()),
            "comisiones_por_periodo": comisiones_por_periodo,
        }
        out.append(entry)
    return {"carreras": out}

@app.get("/api/rag/status")
def get_rag_status():
    """Diagnóstico del módulo RAG. Útil para debugging desde la UI."""
    if not RAG_AVAILABLE:
        return {"available": False, "reason": "RAG module not loaded"}
    return rag_status()

@app.get("/api/audit/status")
def audit_status():
    """Estado del modo auditoría: qué carreras/periodos tienen malla cargada
    y cuántos tokens aprox suman. El frontend lo usa para poblar los
    selectores (carrera → periodo → comisión) y advertir si es pesada."""
    knowledge = discover_carreras()
    detail = {}
    for carrera, periodos in knowledge.items():
        periodos_detalle = {}
        for per_key, coms in periodos.items():
            comisiones = [c for c in coms.keys() if c != "_default"]
            # Para estimar el peso del periodo: si tiene comisiones, usamos la
            # primera; si es comisión única, None resuelve a "_default".
            com_para_peso = comisiones[0] if comisiones else None
            _, meta = load_audit_context(carrera, com_para_peso, per_key)
            periodos_detalle[per_key] = {
                "doc_count":    meta["doc_count"],
                "docs":         meta["docs"],
                "tokens_aprox": meta["tokens_aprox"],
                "heavy":        meta["tokens_aprox"] > AUDIT_HEAVY_TOKEN_THRESHOLD,
                "comisiones":   comisiones,
            }
        detail[carrera] = {
            "periodos":          sorted(periodos.keys()),
            "periodos_detalle":  periodos_detalle,
        }
    return {
        "available":        len(knowledge) > 0,
        "carreras":         list(knowledge.keys()),
        "carreras_detalle": detail,
    }

@app.get("/api/preview-prompt")
def preview_prompt(
    message: str = "",
    carrera: Optional[str] = None,
    comision: Optional[str] = None,
    periodo: Optional[str] = None,
    use_rag: bool = False,
    audit: bool = False,
    audit_format: str = "text",
):
    """Útil para debugging: ve qué se cargaría para una pregunta dada
    SIN gastar tokens en el modelo."""
    full, meta = build_full_system_prompt(
        user_msg=message,
        carrera=carrera or DEFAULT_CARRERA,
        comision=comision,
        periodo=periodo,
        use_rag=use_rag,
        audit=audit,
        audit_format=audit_format,
    )
    return {
        "question":     message,
        "would_load":   list(detect_relevant_docs(message)),
        **meta,
        "preview_first_500": full[:500] + ("..." if len(full) > 500 else ""),
    }


@app.get("/api/stats")
def get_stats(
    since_days: Optional[int] = None,
    limit:      Optional[int] = None,
):
    """Agregados del log estructurado de queries.

    Query params:
      - since_days: filtrar a los últimos N días (None = todo el histórico)
      - limit:      tope de registros a procesar (los más recientes)

    Útil para:
      - Auditoría institucional (cuántas queries, qué carreras, qué modelos)
      - Decidir RAG vs router clásico con datos reales
      - Monitoreo de latencia y errores
    """
    return compute_stats(since_days=since_days, limit=limit)


@app.get("/api/logs/recent")
def get_logs_recent(limit: int = 50):
    """Devuelve los últimos N registros del log (no agregados).
    Solo metadatos — no contiene el texto literal de los queries.
    """
    from logging_utils import _read_records
    records = _read_records(limit=max(1, min(limit, 500)))
    return {
        "count":   len(records),
        "records": records,
    }


@app.get("/api/benchmark/results")
def get_benchmark_results(latest: bool = True):
    """Devuelve los resultados del benchmark de modelos.

    Lee los archivos logs/benchmark-*.jsonl generados por
    deploy/benchmark-models.sh. Si latest=True (default), devuelve
    solo el benchmark más reciente; si False, devuelve todos.

    El frontend lo usa para mostrar el panel de rendimiento comparativo
    de modelos (tokens/seg, latencia, validez JSON, VRAM).
    """
    import glob
    bench_dir = LOGS_DIR
    files = sorted(glob.glob(str(bench_dir / "benchmark-*.jsonl")))
    if not files:
        return {
            "available": False,
            "reason": "No hay benchmarks. Ejecutar: ./deploy/benchmark-models.sh",
            "results": [],
        }

    results = []
    if latest:
        files = files[-1:]

    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        import json as _json
                        results.append(_json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass

    return {
        "available":   len(results) > 0,
        "count":       len(results),
        "results":     results,
        "source_file": files[-1].split("/")[-1] if files else None,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    _check_key()
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    full_system, meta = build_full_system_prompt(
        user_msg=last_user,
        user_system_prompt=req.system_prompt,
        carrera=req.carrera,
        comision=req.comision,
        periodo=req.periodo,
        use_knowledge=req.use_knowledge,
        use_rag=req.use_rag,
        audit=req.audit,
        audit_format=req.audit_format,
        raw=req.raw,
    )

    # ── Iniciar logging del request ──
    chosen_model = req.model or DEFAULT_MODEL
    query_log = QueryLog(
        endpoint=      "/api/chat",
        carrera=       req.carrera,
        mode=          meta.get("mode", "unknown"),
        model=         chosen_model,
        use_knowledge= req.use_knowledge,
        use_rag=       req.use_rag,
        web_search=    req.web_search,
        query_text=    last_user,
        sys_tokens=    meta.get("tokens_aprox", 0),
        loaded_docs=   meta.get("loaded_docs", []),
        files_count=   0,
        client_ip=     (request.client.host if request.client else ""),
        stream=        req.stream,
    )
    query_log.set_rag_info(
        chunks=    meta.get("rag_chunks", 0),
        top_score= meta.get("rag_top_score", 0.0),
    )

    final_messages = [{"role": "system", "content": full_system}] if full_system else []
    if req.audit:
        # Modo auditoría es STATELESS: cada auditoría es independiente, no
        # arrastra historial. Esto evita que la ventana de contexto se llene
        # turno a turno — el contexto de la carrera (fijo) + el último mensaje
        # del usuario es todo lo que se manda. Así nunca se desborda mientras
        # quepa una sola vez.
        last_user_msg = next(
            (m for m in reversed(req.messages) if m.role == "user"), None
        )
        if last_user_msg:
            final_messages.append(last_user_msg.model_dump())
    else:
        final_messages += [m.model_dump() for m in req.messages]

    # Decidir el thinking: si la request lo especifica, se respeta. Si no,
    # en modo auditoría lo desactivamos por defecto (respuestas más rápidas,
    # no necesitamos el razonamiento expuesto).
    effective_think = req.think
    if effective_think is None and req.audit:
        effective_think = False

    payload = _build_payload(
        final_messages, chosen_model,
        req.stream, req.temperature, req.max_tokens,
        web_search=req.web_search,
        json_output=(req.json_output or (req.audit and req.audit_format == "json")),
        think=effective_think,
        num_ctx=req.num_ctx,
    )

    extra_headers = _build_meta_headers(meta, req.web_search)

    if req.stream:
        return StreamingResponse(
            _stream_openrouter(payload, query_log=query_log),
            media_type="text/event-stream",
            headers=extra_headers,
        )

    # En modo no-stream también devolvemos headers para que clientes API
    # (curl -i, integraciones) puedan leer la metadata.
    try:
        result = await _call_openrouter_sync(payload)
        # Capturar usage del response sincrónico
        usage = result.get("usage") or {}
        query_log.set_usage(
            prompt_tokens=     usage.get("prompt_tokens", 0),
            completion_tokens= usage.get("completion_tokens", 0),
        )
        query_log.finish(status="success")
        return JSONResponse(content=result, headers=extra_headers)
    except HTTPException as e:
        query_log.finish(status="error", error=f"HTTP {e.status_code}: {str(e.detail)[:200]}")
        raise
    except Exception as e:
        query_log.finish(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
        raise


@app.post("/api/chat/with-files")
async def chat_with_files(
    message: str                  = Form(default=""),
    history: str                  = Form(default="[]"),
    model: Optional[str]          = Form(default=None),
    system_prompt: Optional[str]  = Form(default=None),
    carrera: Optional[str]        = Form(default=None),
    comision: Optional[str]       = Form(default=None),
    periodo: Optional[str]        = Form(default=None),
    temperature: float            = Form(default=0.3),
    max_tokens: int               = Form(default=4096),
    stream: bool                  = Form(default=True),
    use_knowledge: bool           = Form(default=True),
    web_search: bool              = Form(default=False),
    use_rag: bool                 = Form(default=False),
    audit: bool                   = Form(default=False),
    audit_format: str             = Form(default="text"),
    think: Optional[bool]         = Form(default=None),
    raw: bool                     = Form(default=False),
    num_ctx: Optional[int]        = Form(default=None),
    json_output: bool             = Form(default=False),
    files: list[UploadFile]       = File(default=[]),
    request: Request              = None,  # type: ignore[assignment]
):
    _check_key()

    try:
        raw_history = json.loads(history)
        if not isinstance(raw_history, list):
            raw_history = []
    except Exception:
        raw_history = []

    VALID_ROLES = {"user", "assistant", "system"}
    clean_history = [
        {"role": m["role"], "content": str(m.get("content", ""))}
        for m in raw_history
        if isinstance(m, dict) and m.get("role") in VALID_ROLES
    ]

    file_blocks: list[str] = []
    for f in files:
        if not f.filename:
            continue
        data = await f.read()
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                413,
                f"'{f.filename}' ({size_mb:.1f} MB) supera el límite de {MAX_FILE_SIZE_MB} MB",
            )
        extracted = extract_text(f.filename, data)
        file_blocks.append(f"### Archivo: {f.filename}\n\n{extracted}")

    parts: list[str] = []
    if message.strip():
        parts.append(message.strip())
    if file_blocks:
        parts.append("**Archivos adjuntos:**\n\n" + "\n\n---\n\n".join(file_blocks))

    if not parts:
        raise HTTPException(422, "Debes enviar un mensaje de texto, archivos, o ambos.")

    user_content = "\n\n".join(parts)

    full_system, meta = build_full_system_prompt(
        user_msg=message,
        user_system_prompt=system_prompt,
        carrera=carrera,
        comision=comision,
        periodo=periodo,
        has_attached_files=bool(file_blocks),
        use_knowledge=use_knowledge,
        use_rag=use_rag,
        audit=audit,
        audit_format=audit_format,
        raw=raw,
    )

    # ── Iniciar logging del request ──
    chosen_model = model or DEFAULT_MODEL
    client_ip = (request.client.host if request and request.client else "")
    query_log = QueryLog(
        endpoint=      "/api/chat/with-files",
        carrera=       carrera,
        mode=          meta.get("mode", "unknown"),
        model=         chosen_model,
        use_knowledge= use_knowledge,
        use_rag=       use_rag,
        web_search=    web_search,
        query_text=    message,
        sys_tokens=    meta.get("tokens_aprox", 0),
        loaded_docs=   meta.get("loaded_docs", []),
        files_count=   len(file_blocks),
        client_ip=     client_ip,
        stream=        stream,
    )
    query_log.set_rag_info(
        chunks=    meta.get("rag_chunks", 0),
        top_score= meta.get("rag_top_score", 0.0),
    )

    final_messages: list[dict] = [{"role": "system", "content": full_system}] if full_system else []
    # Modo auditoría es STATELESS: no arrastra historial (ver explicación en /api/chat).
    if not audit:
        final_messages.extend(clean_history)
    final_messages.append({"role": "user", "content": user_content})

    # Decidir thinking (igual que /api/chat): request manda, o auditoría lo
    # desactiva por defecto.
    effective_think = think
    if effective_think is None and audit:
        effective_think = False

    payload = _build_payload(
        final_messages, chosen_model,
        stream, temperature, max_tokens,
        web_search=web_search,
        json_output=(json_output or (audit and audit_format == "json")),
        think=effective_think,
        num_ctx=num_ctx,
    )

    extra_headers = _build_meta_headers(meta, web_search)

    if stream:
        return StreamingResponse(
            _stream_openrouter(payload, query_log=query_log),
            media_type="text/event-stream",
            headers=extra_headers,
        )

    try:
        result = await _call_openrouter_sync(payload)
        usage = result.get("usage") or {}
        query_log.set_usage(
            prompt_tokens=     usage.get("prompt_tokens", 0),
            completion_tokens= usage.get("completion_tokens", 0),
        )
        query_log.finish(status="success")
        return JSONResponse(content=result, headers=extra_headers)
    except HTTPException as e:
        query_log.finish(status="error", error=f"HTTP {e.status_code}: {str(e.detail)[:200]}")
        raise
    except Exception as e:
        query_log.finish(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
        raise


# ──────────────────────────────────────────────────────────────
# Frontend estático
# Sirve el frontend desde frontend/static/ (junto al index.html, logos
# y favicon). Si la carpeta no existe (modo dev con frontend aparte),
# simplemente no se monta y el backend funciona igual como API pura.
# ──────────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent.parent / "frontend" / "static"
if (_static_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)