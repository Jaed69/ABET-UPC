"""
Diagnóstico de la ChromaDB del RAG. NO carga el modelo de embeddings
(es rápido), solo lee la metadata de los chunks indexados.

Uso:
    cd backend
    python -m rag.diagnose
"""
from pathlib import Path
from collections import Counter

BACKEND_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BACKEND_DIR / "rag" / "chroma_db"

print("=" * 60)
print("DIAGNÓSTICO RAG")
print("=" * 60)

print(f"\n1. ChromaDB en: {CHROMA_PATH}")
print(f"   Existe: {CHROMA_PATH.exists()}")
if not CHROMA_PATH.exists():
    print("\n   ❌ No hay ChromaDB. Corre: python -m rag.ingest")
    raise SystemExit(1)

import chromadb
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
cols = client.list_collections()
print(f"\n2. Colecciones: {[c.name for c in cols]}")

for col in cols:
    c = client.get_collection(col.name)
    total = c.count()
    print(f"\n   Colección '{col.name}': {total} chunks")
    data = c.get(include=['metadatas'])
    metas = data['metadatas']
    carreras = Counter(m.get('carrera', '?') for m in metas)
    doctypes = Counter(m.get('doc_type', '?') for m in metas)
    print(f"   Por CARRERA:  {dict(carreras)}")
    print(f"   Por DOC_TYPE: {dict(doctypes)}")

print("\n" + "=" * 60)
print("Si arriba ves SOLO 'cc' (+ 'base') → el problema es la INGESTA:")
print("   las otras carreras no se indexaron. Revisa que existan las")
print("   carpetas knowledge/si, knowledge/civil, etc. y reindexá.")
print("Si ves varias carreras → el RAG está bien, el problema era el")
print("   selector del front (ya corregido para incluirlas).")
print("=" * 60)