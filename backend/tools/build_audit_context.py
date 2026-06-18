"""
Pre-procesador del MODO AUDITORÍA.

Convierte TODOS los archivos crudos de cada carrera (PDF, Excel, Word, etc.)
a Markdown usando MarkItDown (Microsoft), y los guarda cacheados para que el
backend los cargue rápido en runtime.

A diferencia de tools/pdf_to_knowledge.py (que usa un LLM para REFORMATEAR
los PDFs al formato compacto del knowledge clásico), este script hace una
conversión DIRECTA y fiel (sin LLM, sin costo, sin alucinaciones). El objetivo
es distinto: acá queremos TODO el contenido de la carrera disponible como
contexto para que el modelo audite, no un resumen.

Flujo:
    audit_sources/<carrera>/*.{pdf,xlsx,docx,pptx,...}
        │
        │  MarkItDown (conversión directa, sin LLM)
        ▼
    audit_knowledge/<carrera>/*.md   +   _index.md (inventario)

Uso:
    cd backend
    python -m tools.build_audit_context                 # convierte todo
    python -m tools.build_audit_context --carrera sistemas
    python -m tools.build_audit_context --force         # reconvierte aunque exista
    python -m tools.build_audit_context --dry-run       # solo lista qué haría

Después de correrlo, el backend ya puede usar el modo auditoría sin más pasos.

Formatos soportados por MarkItDown:
    .pdf .docx .doc .xlsx .xls .pptx .ppt .csv .json .xml .html .txt .md
    (también imágenes con OCR y audio con transcripción, pero acá nos
     enfocamos en documentos)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Config centralizada (config.py está en backend/, un nivel arriba de tools/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
BACKEND_DIR     = config.BACKEND_DIR
SOURCES_DIR     = config.AUDIT_SOURCES_DIR
KNOWLEDGE_DIR   = config.AUDIT_KNOWLEDGE_DIR

# Extensiones que vamos a convertir
SUPPORTED_EXTS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".csv", ".json", ".xml", ".html", ".htm", ".txt", ".md",
}

# Umbral de advertencia: si una carrera supera estos tokens, avisamos
# que puede no entrar en modelos de contexto chico. (viene de config)
TOKEN_WARN_THRESHOLD = config.AUDIT_HEAVY_TOKEN_THRESHOLD


def estimate_tokens(text: str) -> int:
    """Estimación rápida: ~4 chars por token (regla de dedo para español/inglés)."""
    return len(text) // 4


# ──────────────────────────────────────────────────────────────
# Conversión
# ──────────────────────────────────────────────────────────────
def convert_file(md_converter, src_path: Path) -> Optional[str]:
    """Convierte un archivo a Markdown. Devuelve el texto o None si falla."""
    try:
        result = md_converter.convert(str(src_path))
        return result.text_content or ""
    except Exception as e:
        print(f"      [ERROR] No se pudo convertir {src_path.name}: {e}")
        return None


def build_carrera(
    md_converter,
    carrera: str,
    src_dir: Path,
    out_dir: Path,
    *,
    force: bool,
    dry_run: bool,
) -> dict:
    """Convierte todos los archivos de una carrera. Devuelve métricas."""
    print(f"\n━━━ Carrera: {carrera} ━━━")

    files = sorted(
        f for f in src_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        print(f"  (sin archivos soportados en {src_dir})")
        return {"carrera": carrera, "files": 0, "tokens": 0, "converted": 0}

    out_dir.mkdir(parents=True, exist_ok=True)

    index_entries: list[dict] = []
    total_tokens = 0
    converted = 0

    for src in files:
        out_md = out_dir / (src.stem + ".md")

        # Skip si ya existe y no forzamos (y el fuente no es más nuevo)
        if out_md.exists() and not force:
            if out_md.stat().st_mtime >= src.stat().st_mtime:
                existing = out_md.read_text(encoding="utf-8")
                tk = estimate_tokens(existing)
                total_tokens += tk
                index_entries.append({
                    "source": src.name, "md": out_md.name,
                    "tokens": tk, "status": "cached",
                })
                print(f"  · {src.name:45} → cache ({tk} tk)")
                continue

        print(f"  · {src.name:45} → convirtiendo...", end=" ", flush=True)

        if dry_run:
            print("[DRY-RUN]")
            index_entries.append({
                "source": src.name, "md": out_md.name,
                "tokens": 0, "status": "dry-run",
            })
            continue

        md_text = convert_file(md_converter, src)
        if md_text is None:
            index_entries.append({
                "source": src.name, "md": None,
                "tokens": 0, "status": "error",
            })
            continue

        # Header contextual: deja claro de qué archivo viene cada bloque
        # (útil para que el modelo pueda citar la fuente al auditar)
        header = (
            f"<!-- Fuente: {src.name} | Carrera: {carrera} -->\n"
            f"# Documento: {src.stem}\n\n"
            f"> Archivo fuente: `{src.name}`\n\n"
        )
        full_md = header + md_text

        out_md.write_text(full_md, encoding="utf-8")
        tk = estimate_tokens(full_md)
        total_tokens += tk
        converted += 1
        index_entries.append({
            "source": src.name, "md": out_md.name,
            "tokens": tk, "status": "converted",
        })
        print(f"OK ({tk} tk)")

    # Escribir el _index.md de la carrera (inventario legible + para el backend)
    if not dry_run and index_entries:
        index_md = _build_index_md(carrera, index_entries, total_tokens)
        (out_dir / "_index.md").write_text(index_md, encoding="utf-8")

    # Advertencia de tokens
    warn = ""
    if total_tokens > TOKEN_WARN_THRESHOLD:
        warn = f"  ⚠ {total_tokens} tokens — puede no entrar en modelos de contexto chico"
    print(f"  Total carrera: {total_tokens} tokens ({converted} convertidos){warn}")

    return {
        "carrera":   carrera,
        "files":     len(files),
        "tokens":    total_tokens,
        "converted": converted,
    }


def _build_index_md(carrera: str, entries: list[dict], total_tokens: int) -> str:
    """Genera el inventario _index.md de una carrera."""
    lines = [
        f"<!-- AUTO-GENERADO por build_audit_context.py. No editar a mano. -->",
        f"# Índice de auditoría — {carrera}",
        "",
        f"**Total de documentos:** {len([e for e in entries if e['status'] in ('converted','cached')])}",
        f"**Tokens aproximados:** {total_tokens}",
        "",
        "| Documento | Archivo fuente | Tokens | Estado |",
        "|---|---|---|---|",
    ]
    for e in entries:
        md_name = e["md"] or "—"
        lines.append(f"| {md_name} | {e['source']} | {e['tokens']} | {e['status']} |")
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convierte audit_sources/ a Markdown (modo auditoría)."
    )
    parser.add_argument("--carrera", type=str, default=None,
                        help="Procesar solo una carrera (default: todas)")
    parser.add_argument("--force", action="store_true",
                        help="Reconvertir aunque el .md ya exista y esté actualizado")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo listar qué haría, sin convertir")
    parser.add_argument("--sources-dir", type=Path, default=SOURCES_DIR,
                        help=f"Carpeta de archivos crudos (default: {SOURCES_DIR})")
    args = parser.parse_args()

    sources_dir: Path = args.sources_dir
    if not sources_dir.exists():
        print(f"❌ No existe la carpeta de fuentes: {sources_dir}")
        print(f"   Crea la estructura: {sources_dir}/<carrera>/<archivos>")
        sys.exit(1)

    # Importar MarkItDown solo cuando se va a usar (no en dry-run)
    md_converter = None
    if not args.dry_run:
        try:
            from markitdown import MarkItDown
            md_converter = MarkItDown()
        except ImportError:
            print("❌ MarkItDown no está instalado.")
            print("   Instalalo con: pip install markitdown")
            sys.exit(1)

    # Detectar carreras (subcarpetas de audit_sources/)
    carreras = sorted(
        d.name for d in sources_dir.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    )
    if args.carrera:
        carreras = [c for c in carreras if c == args.carrera]
        if not carreras:
            print(f"❌ Carrera '{args.carrera}' no encontrada en {sources_dir}")
            sys.exit(1)

    if not carreras:
        print(f"⚠ No hay subcarpetas de carrera en {sources_dir}")
        print(f"   Estructura esperada: {sources_dir}/sistemas/, {sources_dir}/civil/, ...")
        return

    print("=" * 70)
    print("🔍 Build Audit Context (MarkItDown)")
    print("=" * 70)
    print(f"  Fuentes:  {sources_dir}")
    print(f"  Salida:   {KNOWLEDGE_DIR}")
    print(f"  Carreras: {carreras}")
    print(f"  Force:    {args.force}   Dry-run: {args.dry_run}")

    t0 = time.perf_counter()
    results = []
    for carrera in carreras:
        src_dir = sources_dir / carrera
        out_dir = KNOWLEDGE_DIR / carrera
        results.append(build_carrera(
            md_converter, carrera, src_dir, out_dir,
            force=args.force, dry_run=args.dry_run,
        ))

    # Resumen
    elapsed = time.perf_counter() - t0
    print()
    print("=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)
    print(f"  {'Carrera':<20} {'Archivos':>9} {'Convertidos':>12} {'Tokens':>10}")
    print(f"  {'-'*20} {'-'*9} {'-'*12} {'-'*10}")
    grand_total = 0
    for r in results:
        grand_total += r["tokens"]
        flag = " ⚠" if r["tokens"] > TOKEN_WARN_THRESHOLD else ""
        print(f"  {r['carrera']:<20} {r['files']:>9} {r['converted']:>12} {r['tokens']:>10}{flag}")
    print(f"  {'-'*20} {'-'*9} {'-'*12} {'-'*10}")
    print(f"  {'TOTAL':<20} {'':<9} {'':<12} {grand_total:>10}")
    print(f"\n  Tiempo: {elapsed:.1f}s")

    if grand_total > TOKEN_WARN_THRESHOLD and not args.dry_run:
        print()
        print("  ⚠ ADVERTENCIA DE CONTEXTO:")
        print("    Alguna carrera supera los 100k tokens. Para auditar toda la")
        print("    carrera en una sola llamada necesitás un modelo de contexto")
        print("    grande (Gemini 1.5/2.x: 1M-2M tokens, Claude: 200k, GPT-4o: 128k).")
        print("    Si usás un modelo chico, considerá auditar por sub-conjuntos.")

    if not args.dry_run:
        print()
        print("  ✅ Listo. El backend ya puede usar el modo auditoría.")
        print("     Activá el toggle 'Auditoría' en el front y elegí la carrera.")


if __name__ == "__main__":
    main()