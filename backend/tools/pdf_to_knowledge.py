"""
Conversor de PDFs UPC a .md institucionales (formato compacto para RAG).

Flujo:
  1. Lee todos los PDFs de una carpeta plana (default: backend/pdfs/).
  2. Para cada PDF:
     - Detecta la CARRERA por el nombre del archivo (cc, si, sw, civil, ambiental).
     - Detecta el TIPO de documento por el CONTENIDO del PDF
       (malla COCOS / Reporte de Control / Reporte de Verificación).
     - Extrae el texto crudo con PyPDF2.
     - Envía el texto a un LLM de OpenRouter (modelo FREE por defecto)
       con instrucciones para reformatear en el formato compacto que el
       ingest/RAG ya entiende.
     - Guarda en knowledge/<carrera>/<archivo>.md (sobreescribe).
  3. Para Reportes de Control (7 PDFs por carrera, uno por outcome) los
     procesa individualmente y luego los CONCATENA en un único
     `reportes_control.md` con la nota de cada outcome ordenada.

Uso:
    # 1) Poner los PDFs en backend/pdfs/ (carpeta plana, cualquier nombre que
    #    contenga al menos la palabra de la carrera: "civil", "sw", etc.)
    # 2) Configurar OPENROUTER_API_KEY en el .env
    # 3) Ejecutar:
    python -m tools.pdf_to_knowledge
    python -m tools.pdf_to_knowledge --pdfs-dir /otra/ruta
    python -m tools.pdf_to_knowledge --carrera civil  # solo una carrera
    python -m tools.pdf_to_knowledge --dry-run        # ver qué haría sin escribir

Convenciones de nombres aceptadas (en el filename, case-insensitive):
    cc            → Ciencias de la Computación
    si | isi      → Ingeniería de Sistemas de Información
    sw | software → Ingeniería de Software
    civil         → Ingeniería Civil
    ambiental | ia → Ingeniería Ambiental

Tipos detectados por contenido:
    malla            ← contiene "MALLA DE COCOS" o "PLAN CURRICULAR"
    control          ← contiene "Reporte de control por Outcome" o tabla con NM/Esp/Sob
    verificacion     ← contiene "Reporte de Verificación Consolidado"
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import PyPDF2

# Importar la config centralizada (config.py está en backend/, un nivel arriba)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
BACKEND_DIR    = config.BACKEND_DIR
PDFS_DIR       = BACKEND_DIR / "pdfs"
KNOWLEDGE_DIR  = config.KNOWLEDGE_DIR

# Estas dos vienen de config.py (respetan LLM_PROVIDER, así que este script
# también funciona contra un LLM local, no solo OpenRouter).
OPENROUTER_API_KEY  = config.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = config.OPENROUTER_BASE_URL

# Modelos FREE de OpenRouter que han funcionado bien para reformateo estructurado.
# El primero es el que se usa; si falla, intenta el siguiente.
# Si querés cambiarlos, edita esta lista o pasá --model.
DEFAULT_FREE_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
]

# Detección de carrera por nombre de archivo (orden importa: más específico primero)
CARRERA_PATTERNS = [
    ("civil",     re.compile(r"\bcivil\b",                re.I)),
    ("ambiental", re.compile(r"\bambiental\b|\bia\b",     re.I)),
    ("sw",        re.compile(r"\bsw\b|software",          re.I)),
    ("si",        re.compile(r"\bisi\b|\bsi\b|sist",      re.I)),
    ("cc",        re.compile(r"\bcc\b|computaci",         re.I)),
]

# Detección de tipo por contenido del PDF
TYPE_PATTERNS = {
    "malla":        [
        re.compile(r"MALLA\s+DE\s+COCOS",          re.I),
        re.compile(r"PLAN\s+CURRICULAR",           re.I),
        re.compile(r"STUDENT\s+OUTCOMES",          re.I),
    ],
    "verificacion": [
        re.compile(r"Reporte\s+de\s+Verificaci[oó]n", re.I),
        re.compile(r"Verificaci[oó]n\s+Consolidado",  re.I),
        re.compile(r"NIVEL\s+DE\s+ACEPTACI[OÓ]N",     re.I),
    ],
    "control":      [
        re.compile(r"Reporte\s+de\s+control\s+por\s+Outcome", re.I),
        re.compile(r"OUTCOME\s+COMISI[OÓ]N",                  re.I),
    ],
}

DOC_FILENAMES = {
    "malla":        "malla.md",
    "control":      "reportes_control.md",
    "verificacion": "reporte_verificacion.md",
}

CARRERA_DISPLAY = {
    "cc":        "Ciencias de la Computación (CC)",
    "si":        "Ingeniería de Sistemas de Información (SI)",
    "sw":        "Ingeniería de Software (SW)",
    "civil":     "Ingeniería Civil",
    "ambiental": "Ingeniería Ambiental",
}


# ──────────────────────────────────────────────────────────────
# Extracción de texto del PDF
# ──────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: Path) -> str:
    """Extrae todo el texto del PDF. Si falla, devuelve string vacío."""
    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
                if t.strip():
                    pages.append(f"--- Página {i+1} ---\n{t.strip()}")
            except Exception as e:
                print(f"    [WARN] Página {i+1} falló: {e}")
        return "\n\n".join(pages)
    except Exception as e:
        print(f"    [ERROR] No se pudo leer {pdf_path.name}: {e}")
        return ""


def detect_carrera(filename: str) -> Optional[str]:
    """Devuelve la key de la carrera o None si no se reconoce el nombre."""
    for key, pattern in CARRERA_PATTERNS:
        if pattern.search(filename):
            return key
    return None


def detect_doc_type(text: str) -> Optional[str]:
    """Inspecciona los primeros ~3000 chars del PDF para clasificarlo."""
    head = text[:3000]
    # Orden: malla → verificacion → control (el más restrictivo primero)
    for doc_type in ("malla", "verificacion", "control"):
        patterns = TYPE_PATTERNS[doc_type]
        if any(p.search(head) for p in patterns):
            return doc_type
    return None


def extract_outcome_number(text: str) -> Optional[int]:
    """Para un RC, intenta detectar el número de outcome (1..7) del PDF.
    Busca patrones tipo 'OUTCOME 3' o 'OUTCOME  6' en la cabecera."""
    head = text[:2000]
    # En el header del RC aparece algo como:
    #   "SEDE MODALIDAD CICLO CARRERA ... OUTCOME COMISIÓN"
    #   "TODAS PREGRADO REGULAR 202510 INGENIERÍA CIVIL ... 3 INGENIERÍA"
    # Más confiable buscar "OUTCOME\s+N" en cualquier parte del header
    m = re.search(r"OUTCOME[^\d]{0,80}(\d{1,2})", head, re.I)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 10:
            return n
    return None


# ──────────────────────────────────────────────────────────────
# Prompts por tipo de documento
# ──────────────────────────────────────────────────────────────
def build_prompt_malla(carrera_display: str, raw_text: str) -> tuple[str, str]:
    system = (
        "Eres un asistente que convierte mallas curriculares (PDFs académicos UPC) "
        "a un formato Markdown compacto y consistente para indexar en un sistema RAG. "
        "Sigue EXACTAMENTE el formato que se te pide. "
        "No inventes datos: si algo no está en el PDF, omítelo. "
        "Responde SOLO con el Markdown final, sin explicaciones ni preámbulo."
    )

    user = f"""Convierte el siguiente PDF de Malla COCOS UPC al formato compacto.

CARRERA: {carrera_display}

Formato OBLIGATORIO de salida:

```
# Malla COCOS — <Nombre Carrera> (<KEY>)

Comisión <EAC-XX>. Ciclo referencia <YYYYNN>. **N outcomes** (RE1..REN).

## Notación

- `CF` = Curso Formativo
- `C:RE#` = Control para ese outcome
- `V:RE#` = Verificación para ese outcome
- Sin marca = solo formativo o sin actividad de outcome

## Nivel 1

- <CODIGO> <Nombre Curso> · CF
- <CODIGO> <Nombre Curso> · C:RE1
- ...

## Nivel 2

...

## Nivel 10

...

## Resumen totales por outcome (según malla)

CF=N · RE1=N · RE2=N · ...

## Vista cruzada por actividad

**Verificación (V) — alimenta el RV:**
- <CODIGO> <Nombre> → RE# (nivel N)
...

**Control (C) por outcome — alimenta los RC (cursos obligatorios):**
- RE1: <CODIGO>, <CODIGO>, ...
- RE2: ...
...

## Electivos con actividad de outcome

- <CODIGO> <Nombre> · C:RE#
- ...
```

Reglas:
1. Códigos de curso vienen con prefijo '1' (ejemplo: 1ACI0578, 1ASI0644). Conservarlos tal cual.
2. En el PDF la columna 'CF' marca curso formativo (rombo rojo ◆ → CF).
3. Las columnas 1..7 (o 1..6) marcan actividad por outcome: rombo negro ◆ → Control; check verde ✓ → Verificación.
4. Si un curso tiene varios outcomes (típico: Taller de Proyecto), pone todos: `· V:RE1 · C:RE2 · C:RE3 · ...`
5. Los electivos van AL FINAL en su propia sección.
6. Si el PDF dice "ciclo 202402" pero contexto sugiere 202501, usa 202501.

Texto crudo del PDF:

\"\"\"
{raw_text[:18000]}
\"\"\"

Responde SOLO con el Markdown."""
    return system, user


def build_prompt_control(carrera_display: str, outcome_n: Optional[int], raw_text: str) -> tuple[str, str]:
    system = (
        "Eres un asistente que convierte Reportes de Control UPC (tablas de "
        "rendimiento por outcome) a Markdown compacto para indexar en un RAG. "
        "Sigue EXACTAMENTE el formato. No inventes datos. "
        "Responde SOLO con el Markdown final."
    )

    outcome_hint = f"Outcome detectado: RE{outcome_n}" if outcome_n else "Outcome: detéctalo del PDF"

    user = f"""Convierte el siguiente Reporte de Control UPC al formato compacto.

CARRERA: {carrera_display}
{outcome_hint}

Formato OBLIGATORIO de salida (UNA SOLA SECCIÓN, la del outcome que corresponde):

```
## RC RE<N> — <Título corto del outcome>

> <Descripción oficial del outcome tomada del PDF, en una línea>

- <CODIGO_CURSO> <NOMBRE_CURSO> · NM=<n>(<p>%) Esp=<n>(<p>%) Sob=<n>(<p>%) n=<total>
- <CODIGO_CURSO> <NOMBRE_CURSO> · NM=<n>(<p>%) Esp=<n>(<p>%) Sob=<n>(<p>%) n=<total>
- ...

TOTAL RE<N>: NM=<n> Esp=<n> Sob=<n> n=<total>
CRÍTICO: <curso con mayor % NM>. <Comentario breve si hay algo notable>.
```

Reglas:
1. NM = "Necesita mejora" / Esp = "Esperado" / Sob = "Sobresaliente".
2. Los números entre paréntesis son cantidad absoluta, los % son el porcentaje.
3. Conserva los códigos de curso tal cual aparecen (ejemplo: 1ACI0578, 1ASI0644).
4. Conserva los nombres de curso en mayúsculas tal cual aparecen en el PDF.
5. Después de la lista, agrega línea "TOTAL RE<N>" con los totales del PDF.
6. Después agrega 1 línea "CRÍTICO:" con el curso de mayor % NM y un comentario breve si algo destaca (0 NM, curso integrador, etc.).
7. NO agregues título de archivo ni cabecera de carrera (eso se concatena después).

Texto crudo del PDF:

\"\"\"
{raw_text[:14000]}
\"\"\"

Responde SOLO con el Markdown de UNA sola sección RC RE<N>."""
    return system, user


def build_prompt_verificacion(carrera_display: str, raw_text: str) -> tuple[str, str]:
    system = (
        "Eres un asistente que convierte Reportes de Verificación UPC "
        "(consolidado de outcomes del programa) a Markdown compacto para "
        "indexar en un RAG. Sigue EXACTAMENTE el formato. No inventes datos. "
        "Responde SOLO con el Markdown final."
    )

    user = f"""Convierte el siguiente Reporte de Verificación UPC al formato compacto.

CARRERA: {carrera_display}

Formato OBLIGATORIO de salida:

```
# Reporte de Verificación (RV) — <Nombre Carrera> (<KEY>)

**Carrera:** <Nombre Carrera>
**Comisión:** <Comisión>
**Ciclo:** <YYYYNN>
**Sede:** Todas
**Modalidad:** Pregrado Regular

> **IMPORTANTE**: Esta carrera tiene **N outcomes** (RE1..REN). Comisión <EAC-XX>.

## Los N outcomes (definiciones oficiales) con resultado consolidado

**RE1**: <Descripción oficial del outcome 1 desde el PDF>
→ NM=<n>(<p>%) Esp=<n>(<p>%) Sob=<n>(<p>%) n=<total>

**RE2**: <Descripción oficial del outcome 2 desde el PDF>
→ NM=<n>(<p>%) Esp=<n>(<p>%) Sob=<n>(<p>%) n=<total>

... (uno por outcome)

## Totales del RV consolidado

NM=<n> · Esp=<n> · Sob=<n> · TOTAL=<n>

## Lectura del RV

- <Comentario breve: ciclo, mejor outcome, peor outcome, patrones llamativos>
- <Otro hallazgo>
- <Otro hallazgo>

## Cursos que alimentan el RV

Según la malla COCOS, los cursos con actividad de Verificación (✓):
- **<CODIGO> <Nombre>** (nivel N) → verifica RE<N>, RE<N>, ...
```

Reglas:
1. Conserva las descripciones oficiales de los outcomes TAL CUAL aparecen en el PDF.
2. NM = "Necesita mejora" / Esp = "Esperado" / Sob = "Sobresaliente".
3. La sección "Lectura del RV" es interpretativa breve (3-5 puntos): mejor/peor outcome, % global de NM, patrones.
4. La sección "Cursos que alimentan el RV" lista solo los cursos con ✓ (no los con ◆).

Texto crudo del PDF:

\"\"\"
{raw_text[:14000]}
\"\"\"

Responde SOLO con el Markdown."""
    return system, user


# ──────────────────────────────────────────────────────────────
# Cliente OpenRouter
# ──────────────────────────────────────────────────────────────
def call_openrouter(
    system: str,
    user: str,
    *,
    models: list[str],
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> str:
    """Llama al LLM con la lista de modelos free como fallback.
    Retorna el contenido o lanza una excepción si todos fallan."""
    config.require_api_key()

    headers = config.llm_headers()

    last_error = None
    for model in models:
        payload = {
            "model":       model,
            "messages":    [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":      False,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        try:
            with httpx.Client(timeout=180) as client:
                r = client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}: {r.text[:300]}"
                print(f"    [WARN] {model} → {last_error}")
                # Si es rate limit, esperar un poco antes de saltar al siguiente
                if r.status_code == 429:
                    time.sleep(2)
                continue

            data = r.json()
            content = data["choices"][0]["message"]["content"]
            # Limpiar fences ```markdown ... ``` si vinieron
            content = re.sub(r"^```(?:markdown|md)?\s*", "", content.strip())
            content = re.sub(r"\s*```\s*$", "", content)
            return content.strip()

        except Exception as e:
            last_error = str(e)
            print(f"    [WARN] {model} → excepción: {e}")
            continue

    raise RuntimeError(f"Todos los modelos fallaron. Último error: {last_error}")


# ──────────────────────────────────────────────────────────────
# Procesamiento principal
# ──────────────────────────────────────────────────────────────
def process_malla(pdf_path: Path, carrera: str, models: list[str], dry_run: bool) -> Optional[str]:
    print(f"  → Procesando MALLA: {pdf_path.name}")
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        print(f"    [SKIP] PDF vacío")
        return None

    system, user = build_prompt_malla(CARRERA_DISPLAY[carrera], text)
    if dry_run:
        print(f"    [DRY-RUN] {len(text)} chars de texto extraído, no llamo al LLM")
        return None

    print(f"    Llamando LLM ({len(text)} chars de input)...")
    md = call_openrouter(system, user, models=models)
    print(f"    LLM devolvió {len(md)} chars de Markdown")
    return md


def process_verificacion(pdf_path: Path, carrera: str, models: list[str], dry_run: bool) -> Optional[str]:
    print(f"  → Procesando VERIFICACIÓN: {pdf_path.name}")
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        print(f"    [SKIP] PDF vacío")
        return None

    system, user = build_prompt_verificacion(CARRERA_DISPLAY[carrera], text)
    if dry_run:
        print(f"    [DRY-RUN] {len(text)} chars de texto extraído, no llamo al LLM")
        return None

    print(f"    Llamando LLM ({len(text)} chars de input)...")
    md = call_openrouter(system, user, models=models)
    print(f"    LLM devolvió {len(md)} chars de Markdown")
    return md


def process_control_pdfs(
    pdfs: list[Path],
    carrera: str,
    models: list[str],
    dry_run: bool,
) -> Optional[str]:
    """Procesa los N PDFs de Control de una carrera y los concatena.

    Cada PDF es un outcome. El LLM devuelve una sección `## RC RE<N>` por
    PDF. Las concatenamos ordenadas por número de outcome.
    """
    print(f"  → Procesando {len(pdfs)} PDFs de CONTROL:")

    sections: dict[int, str] = {}
    for pdf_path in sorted(pdfs):
        text = extract_pdf_text(pdf_path)
        if not text.strip():
            print(f"    [SKIP] {pdf_path.name} → vacío")
            continue

        outcome_n = extract_outcome_number(text)
        if outcome_n is None:
            print(f"    [WARN] {pdf_path.name} → no detecté número de outcome, lo salto")
            continue

        print(f"    · {pdf_path.name} → RE{outcome_n}")
        if dry_run:
            sections[outcome_n] = f"## RC RE{outcome_n} — [DRY-RUN] {pdf_path.name}\n"
            continue

        system, user = build_prompt_control(CARRERA_DISPLAY[carrera], outcome_n, text)
        try:
            md = call_openrouter(system, user, models=models)
            sections[outcome_n] = md
            print(f"      → {len(md)} chars de MD")
        except Exception as e:
            print(f"      [ERROR] LLM falló: {e}")

    if not sections:
        return None

    # Header del archivo final + concatenación ordenada de las secciones
    header = f"""# Reportes de Control (RC) — {CARRERA_DISPLAY[carrera]}

**Carrera:** {CARRERA_DISPLAY[carrera]}
**Sede:** Todas
**Modalidad:** Pregrado Regular

Notación por curso: `código Nombre · NM=n(p%) Esp=n(p%) Sob=n(p%) n=total`

"""
    ordered = [sections[k] for k in sorted(sections.keys())]
    body = "\n\n".join(ordered)
    return header + body


def save_md(content: str, carrera: str, doc_type: str) -> Path:
    out_dir = KNOWLEDGE_DIR / carrera
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / DOC_FILENAMES[doc_type]
    out_path.write_text(content, encoding="utf-8")
    return out_path


# ──────────────────────────────────────────────────────────────
# Clasificación de PDFs por (carrera, doc_type)
# ──────────────────────────────────────────────────────────────
def classify_pdfs(pdfs_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """Agrupa los PDFs por (carrera, doc_type)."""
    grouped: dict[tuple[str, str], list[Path]] = {}
    unknown: list[Path] = []

    pdfs = sorted(pdfs_dir.glob("*.pdf"))
    print(f"Encontrados {len(pdfs)} PDFs en {pdfs_dir}")
    print()

    for p in pdfs:
        carrera = detect_carrera(p.name)
        if not carrera:
            unknown.append(p)
            print(f"  ✗ {p.name} → no detecté carrera del nombre")
            continue

        # Detectar tipo por contenido
        text = extract_pdf_text(p)
        if not text.strip():
            print(f"  ✗ {p.name} → PDF vacío")
            continue
        doc_type = detect_doc_type(text)
        if not doc_type:
            print(f"  ✗ {p.name} → no detecté tipo (¿malla/control/verificacion?)")
            continue

        print(f"  ✓ {p.name:50}  →  {carrera:10}  {doc_type}")
        grouped.setdefault((carrera, doc_type), []).append(p)

    if unknown:
        print()
        print(f"⚠ {len(unknown)} PDFs sin carrera detectada — renombrar incluyendo 'cc', 'civil', etc.")

    return grouped


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convierte PDFs UPC a .md institucionales para el RAG.",
    )
    parser.add_argument("--pdfs-dir", type=Path, default=PDFS_DIR,
                        help=f"Carpeta plana con los PDFs (default: {PDFS_DIR}).")
    parser.add_argument("--carrera", type=str, default=None,
                        help="Procesar solo una carrera (cc|si|sw|civil|ambiental).")
    parser.add_argument("--model", type=str, action="append", default=None,
                        help="Modelo de OpenRouter a usar. Se puede pasar varias veces "
                             "para fallback. Si no se pasa, usa los FREE defaults.")
    parser.add_argument("--dry-run", action="store_true",
                        help="No llama al LLM ni escribe archivos. Solo muestra la clasificación.")
    parser.add_argument("--only-type", type=str, default=None,
                        choices=["malla", "control", "verificacion"],
                        help="Procesar solo un tipo de documento.")
    args = parser.parse_args()

    pdfs_dir: Path = args.pdfs_dir
    if not pdfs_dir.exists():
        print(f"❌ Carpeta no existe: {pdfs_dir}")
        print(f"   Crea la carpeta y pone los PDFs adentro.")
        sys.exit(1)

    models = args.model or DEFAULT_FREE_MODELS

    if not args.dry_run and not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY no está configurada en .env")
        print("   Configurala o corre con --dry-run para previsualizar.")
        sys.exit(1)

    print("=" * 70)
    print("📚 PDF → Knowledge Converter")
    print("=" * 70)
    print(f"  Input:    {pdfs_dir}")
    print(f"  Output:   {KNOWLEDGE_DIR}")
    print(f"  Modelos:  {models}")
    print(f"  Dry-run:  {args.dry_run}")
    if args.carrera:    print(f"  Carrera:  {args.carrera}")
    if args.only_type:  print(f"  Tipo:     {args.only_type}")
    print()

    grouped = classify_pdfs(pdfs_dir)
    if not grouped:
        print("⚠ Nada que procesar.")
        return

    # Filtros
    if args.carrera:
        grouped = {k: v for k, v in grouped.items() if k[0] == args.carrera}
    if args.only_type:
        grouped = {k: v for k, v in grouped.items() if k[1] == args.only_type}

    print()
    print("=" * 70)
    print(f"Plan: {len(grouped)} (carrera, tipo) a procesar")
    print("=" * 70)

    success = 0
    failed = 0

    for (carrera, doc_type), pdfs in sorted(grouped.items()):
        print()
        print(f"━━━ {CARRERA_DISPLAY[carrera]} / {doc_type} ━━━")
        try:
            if doc_type == "control":
                md = process_control_pdfs(pdfs, carrera, models, args.dry_run)
            elif doc_type == "malla":
                # Si hay varios, usamos el primero (suele ser uno)
                md = process_malla(pdfs[0], carrera, models, args.dry_run)
            elif doc_type == "verificacion":
                md = process_verificacion(pdfs[0], carrera, models, args.dry_run)
            else:
                continue

            if md and not args.dry_run:
                out_path = save_md(md, carrera, doc_type)
                rel = out_path.relative_to(KNOWLEDGE_DIR.parent)
                print(f"  ✅ Guardado en {rel}")
                success += 1
            elif args.dry_run:
                print(f"  [DRY-RUN] No se escribió")
            else:
                print(f"  ⚠ Sin contenido")
                failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"✅ Éxito: {success}   ❌ Fallos: {failed}")
    print("=" * 70)

    if success > 0 and not args.dry_run:
        print()
        print("Próximo paso: reindexar la BD ChromaDB:")
        print("    python -m rag.ingest")


if __name__ == "__main__":
    main()