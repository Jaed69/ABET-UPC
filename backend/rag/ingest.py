"""
Ingesta del knowledge para ChromaDB.

Estrategia:
- Chunking por sección (## ...) en vez de tamaño fijo.
- Cada chunk lleva header contextual (carrera + doc_type + sección)
  para que el embedding capture de QUÉ está hablando, no solo el contenido.
- Embeddings multilingües NORMALIZADOS (necesario para similitud coseno).
- Distancia coseno en ChromaDB (estándar para sentence embeddings).
- Metadata rica para filtros precisos.

Uso:
    python -m rag.ingest

Cuando edites los .md de knowledge, vuelve a correr esto para reindexar.
"""
from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path

# Silenciar el FutureWarning interno de huggingface_hub sobre `resume_download`.
# Es un aviso de una API interna deprecada en hf_hub que langchain-huggingface
# todavía usa. No afecta el funcionamiento ni la calidad de los embeddings.
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.*")

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Config centralizada (config.py está en backend/, un nivel arriba de rag/)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
BACKEND_DIR    = config.BACKEND_DIR
CHROMA_PATH    = BACKEND_DIR / "rag" / "chroma_db"
KNOWLEDGE_PATH = config.KNOWLEDGE_DIR

# El modelo de embeddings viene de config (debe coincidir con retriever.py).
# Multilingüe, soporta español, embeddings normalizados L2 para métrica coseno.
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME

# Tope de chars por chunk (1 token ≈ 4 chars en español)
MAX_CHUNK_CHARS = 1800

CARRERA_DISPLAY = {
    "cc":        "Ciencias de la Computación",
    "si":        "Ingeniería de Sistemas de Información",
    "sw":        "Ingeniería de Software",
    "civil":     "Ingeniería Civil",
    "ambiental": "Ingeniería Ambiental",
}

DOC_TYPE_DISPLAY = {
    "malla":        "Malla COCOS",
    "control":      "Reportes de Control",
    "verificacion": "Reporte de Verificación",
    "base":         "Conocimiento Base",
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def detect_doc_type(filename: str) -> str:
    name = filename.lower()
    if "malla" in name:
        return "malla"
    if "control" in name:
        return "control"
    if "verificacion" in name:
        return "verificacion"
    return "base"


def split_by_h2_sections(text: str) -> list[tuple[str, str]]:
    """
    Parte el .md por headers ## . Devuelve lista de (titulo_seccion, contenido).
    El contenido antes del primer ## se asocia a "Introducción".

    Mantiene cada outcome/sección como una unidad semántica indivisible.
    Si una sección excede MAX_CHUNK_CHARS, se sub-parte por items de lista.
    """
    parts: list[tuple[str, str]] = []

    # Split conservando el delimitador
    blocks = re.split(r'(?m)^##\s+(.+)$', text)
    # blocks = [pre_intro, h1, contenido1, h2, contenido2, ...]

    if blocks and blocks[0].strip():
        parts.append(("Introducción", blocks[0].strip()))

    for i in range(1, len(blocks), 2):
        title = blocks[i].strip()
        content = blocks[i + 1].strip() if i + 1 < len(blocks) else ""
        if content:
            parts.append((title, content))

    return parts


def maybe_subsplit(title: str, content: str) -> list[tuple[str, str]]:
    """
    Si una sección es demasiado grande, la divide por bloques de items
    (líneas que empiezan con `-`), respetando que cada item quede completo.
    """
    if len(content) <= MAX_CHUNK_CHARS:
        return [(title, content)]

    # Detectar items de lista
    lines = content.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # +1 por el \n
        # Si agregar esta línea pasa el límite Y current ya tiene contenido,
        # cerramos chunk actual
        if current_size + line_size > MAX_CHUNK_CHARS and current:
            chunks.append("\n".join(current))
            current = [line]
            current_size = line_size
        else:
            current.append(line)
            current_size += line_size

    if current:
        chunks.append("\n".join(current))

    if len(chunks) == 1:
        return [(title, chunks[0])]

    return [
        (f"{title} (parte {i+1}/{len(chunks)})", c)
        for i, c in enumerate(chunks)
    ]


def build_contextual_chunk(
    *,
    carrera_key: str,
    doc_type: str,
    section_title: str,
    content: str,
) -> str:
    """
    Antepone un header contextual al chunk. Esto es lo que más mejora la precisión.
    El embedding va a "ver" la carrera + tipo de documento + sección,
    no solo el contenido suelto.
    """
    carrera_name  = CARRERA_DISPLAY.get(carrera_key, carrera_key.upper())
    doc_name      = DOC_TYPE_DISPLAY.get(doc_type, doc_type)

    header = (
        f"[Carrera: {carrera_name}] "
        f"[Documento: {doc_name}] "
        f"[Sección: {section_title}]\n\n"
    )
    return header + content


def build_embedding_model() -> HuggingFaceEmbeddings:
    """
    Construye el modelo de embeddings con normalización L2 y caché local.
    Delega a config.build_embedding_model() para que ingest y retriever
    usen EXACTAMENTE la misma configuración y la misma carpeta de caché
    (si no, el modelo se descargaría dos veces en rutas distintas).
    """
    return config.build_embedding_model()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print(f"Knowledge: {KNOWLEDGE_PATH}")
    print(f"Chroma:    {CHROMA_PATH}")
    print(f"Embedding: {EMBEDDING_MODEL_NAME}")
    print()

    # Reiniciar la BD para evitar duplicados si se corre varias veces
    if CHROMA_PATH.exists():
        print(f"Limpiando ChromaDB anterior...")
        shutil.rmtree(CHROMA_PATH)

    documents: list[Document] = []
    files_scanned = 0

    # Archivos que NO van al índice RAG: el system_prompt es la INSTRUCCIÓN
    # del asistente (se carga aparte vía load_base()), no conocimiento factual
    # buscable. Indexarlo contamina las búsquedas: su texto genérico
    # ("outcomes", "malla", "reporte"...) matchea casi cualquier query y se
    # cuela como falso positivo. El glosario sí se indexa (es referencia útil).
    SKIP_FROM_INDEX = {"system_prompt.md"}

    for file in sorted(KNOWLEDGE_PATH.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in (".md", ".txt"):
            continue
        if file.name.lower() in SKIP_FROM_INDEX:
            print(f"  · {file.relative_to(KNOWLEDGE_PATH)}  →  (omitido del índice: instrucción del sistema)")
            continue

        files_scanned += 1
        rel = file.relative_to(KNOWLEDGE_PATH)
        # carpeta "_base" → carrera="base"; resto → nombre carpeta
        parent_name = file.parent.name.lower()
        carrera = "base" if parent_name.startswith("_") else parent_name
        doc_type = detect_doc_type(file.name)

        print(f"  · {rel}  →  carrera={carrera}  doc_type={doc_type}")

        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"    ERROR leyendo: {e}")
            continue

        sections = split_by_h2_sections(text)
        chunk_idx = 0

        for title, content in sections:
            for subtitle, subcontent in maybe_subsplit(title, content):
                contextual = build_contextual_chunk(
                    carrera_key=carrera,
                    doc_type=doc_type,
                    section_title=subtitle,
                    content=subcontent,
                )

                documents.append(Document(
                    page_content=contextual,
                    metadata={
                        "source":      str(rel),
                        "carrera":     carrera,
                        "doc_type":    doc_type,
                        "section":     subtitle,
                        "chunk_idx":   chunk_idx,
                        "chunk_chars": len(contextual),
                    },
                ))
                chunk_idx += 1

    print()
    print(f"Archivos procesados: {files_scanned}")
    print(f"Chunks generados:    {len(documents)}")

    if not documents:
        print("No hay nada que indexar. Salir.")
        return

    print()
    print("Cargando modelo de embeddings (con normalización L2)...")
    embedding_model = build_embedding_model()

    print("Indexando en ChromaDB con métrica COSENO...")
    # collection_metadata={"hnsw:space": "cosine"} le dice a Chroma que use
    # similitud coseno en vez de L2 (default). Combinado con embeddings
    # normalizados, los scores quedan en rango interpretable [0, 1].
    Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(CHROMA_PATH),
        collection_metadata={"hnsw:space": "cosine"},
    )

    print()
    print(f"✓ ChromaDB persistida en {CHROMA_PATH}")
    print(f"✓ {len(documents)} chunks indexados.")
    print(f"✓ Métrica: coseno · Embeddings normalizados L2")


if __name__ == "__main__":
    main()