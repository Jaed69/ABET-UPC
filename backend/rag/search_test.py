"""
Script de prueba para validar la calidad del retriever.

Ejecuta consultas típicas y muestra qué chunks recupera + scores.
Útil para iterar sobre el chunking sin tener que levantar el backend.

Uso:
    python -m rag.search_test
    python -m rag.search_test --verbose       # imprime los chunks completos
    python -m rag.search_test --case 3        # solo el caso N
    python -m rag.search_test --query "..."   # query ad-hoc

Interpretación de scores (con similitud coseno + embeddings normalizados):
    > 0.6   = excelente match
    0.4-0.6 = bueno
    0.3-0.4 = regular (top-1 suele estar acá en este corpus)
    0.2-0.3 = débil pero todavía útil
    < 0.2   = probablemente ruido
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from retriever import search_docs, get_stats


# ──────────────────────────────────────────────────────────────
# Casos de prueba
# Cada caso: (descripción, query, carrera, doc_types, expected_hits)
# expected_hits es una lista de keywords que DEBERÍAN aparecer en el
# page_content del top-1 si el retrieval funciona bien. Sirve para
# medir precisión automáticamente.
# ──────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "desc":      "Código de curso específico (Civil)",
        "query":     "datos del curso 1ACI0578 Proyecto de Tesis",
        "carrera":   "civil",
        "doc_types": ["control"],
        "expected_hits": ["1ACI0578", "TESIS"],
    },
    {
        "desc":      "% Necesita Mejora por outcome (Civil)",
        "query":     "qué outcome tiene mayor porcentaje de Necesita Mejora",
        "carrera":   "civil",
        "doc_types": ["control"],
        "expected_hits": ["NM", "outcome", "%"],
    },
    {
        "desc":      "Cursos críticos en CC",
        "query":     "qué cursos tienen peor desempeño",
        "carrera":   "cc",
        "doc_types": ["control"],
        "expected_hits": ["NM", "crítico"],
    },
    {
        "desc":      "Definición de CF (solo base)",
        "query":     "qué significa CF Curso Formativo en la malla",
        "carrera":   None,
        "doc_types": ["base"],
        "expected_hits": ["CF", "Formativo"],
    },
    {
        "desc":      "Outcomes verificados en RV de SW",
        "query":     "outcomes verificados en el reporte de verificación de Software",
        "carrera":   "sw",
        "doc_types": ["verificacion"],
        "expected_hits": ["RE", "verificación"],
    },
    {
        "desc":      "Comparación 6 vs 7 outcomes",
        "query":     "qué carreras tienen 6 outcomes y cuáles 7",
        "carrera":   None,
        "doc_types": None,
        "expected_hits": ["outcome"],
    },
    {
        "desc":      "Curso integrador SW",
        "query":     "Taller de Proyecto I Ingeniería de Software",
        "carrera":   "sw",
        "doc_types": ["malla", "control"],
        "expected_hits": ["1ASI0644", "Taller"],
    },
    {
        "desc":      "RV de Ingeniería Ambiental",
        "query":     "resultados consolidados del RV de Ambiental",
        "carrera":   "ambiental",
        "doc_types": ["verificacion"],
        "expected_hits": ["AMBIENTAL", "outcome"],
    },
    {
        "desc":      "Curso de un outcome específico (CC, RE2)",
        "query":     "qué cursos controlan el outcome 2 en Ciencias de la Computación",
        "carrera":   "cc",
        "doc_types": ["control"],
        "expected_hits": ["RE2"],
    },
    {
        "desc":      "Pregunta vaga sin contexto",
        "query":     "comportamiento académico",
        "carrera":   None,
        "doc_types": None,
        "expected_hits": [],  # caso difícil — solo medimos top score
    },
]


# ──────────────────────────────────────────────────────────────
# Helpers de evaluación
# ──────────────────────────────────────────────────────────────
def check_expected_hits(text: str, expected: list[str]) -> tuple[int, int]:
    """Cuenta cuántas keywords esperadas aparecen en el text. Case-insensitive."""
    if not expected:
        return 0, 0
    lower = text.lower()
    hit = sum(1 for kw in expected if kw.lower() in lower)
    return hit, len(expected)


def color(text: str, code: str) -> str:
    """Colorea texto si la salida es terminal. ANSI escape codes."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def grade_score(score: float) -> str:
    """Etiqueta cualitativa basada en el score coseno."""
    if score >= 0.6:  return color("EXCELENTE", "32;1")
    if score >= 0.4:  return color("BUENO",     "32")
    if score >= 0.3:  return color("REGULAR",   "33")
    if score >= 0.2:  return color("DÉBIL",     "33;2")
    return                color("RUIDO",     "31")


# ──────────────────────────────────────────────────────────────
# Ejecución de un caso
# ──────────────────────────────────────────────────────────────
def run_case(case: dict, *, k: int = 4, verbose: bool = False) -> dict:
    """Ejecuta un caso y devuelve métricas para agregar al final."""
    print()
    print("=" * 80)
    print(f"[{case['desc']}]")
    print(f"  Query:     {case['query']!r}")
    print(f"  Carrera:   {case['carrera']}")
    print(f"  Doc types: {case['doc_types']}")
    if case.get("expected_hits"):
        print(f"  Espera:    {case['expected_hits']}")
    print("-" * 80)

    results = search_docs(
        case["query"],
        carrera=case["carrera"],
        doc_types=case["doc_types"],
        k=k,
    )

    if not results:
        print("  (sin resultados)")
        return {
            "desc":      case["desc"],
            "top_score": 0.0,
            "hits":      0,
            "expected":  len(case.get("expected_hits", [])),
            "n_results": 0,
        }

    top_doc, top_score = results[0]
    expected = case.get("expected_hits", [])
    hits, total = check_expected_hits(top_doc.page_content, expected)

    print(f"  Top score: {top_score:.3f}  [{grade_score(top_score)}]", end="")
    if expected:
        ok = "✓" if hits == total else ("~" if hits > 0 else "✗")
        print(f"  ·  expected_hits: {ok} {hits}/{total}")
    else:
        print()
    print()

    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        section = (meta.get("section") or "")[:50]
        preview_len = 600 if verbose else 200
        preview = doc.page_content[:preview_len].replace("\n", " ")
        marker = "★" if i == 1 else " "
        print(
            f"  {marker} #{i} score={score:.3f} "
            f"[{meta.get('carrera')}/{meta.get('doc_type')}] "
            f"sec={section!r}"
        )
        print(f"       {preview}…")
        if verbose:
            print()

    return {
        "desc":      case["desc"],
        "top_score": top_score,
        "hits":      hits,
        "expected":  total,
        "n_results": len(results),
    }


# ──────────────────────────────────────────────────────────────
# Reporte agregado
# ──────────────────────────────────────────────────────────────
def print_summary(metrics: list[dict]) -> None:
    print()
    print("=" * 80)
    print(color(" REPORTE AGREGADO ", "1;7"))
    print("=" * 80)
    print()
    print(f"  {'Caso':<48}  {'Top':>6}   {'Hits':>6}")
    print(f"  {'-'*48}  {'-'*6}   {'-'*6}")
    for m in metrics:
        desc = m["desc"][:48]
        score_s = f"{m['top_score']:.3f}"
        if m["expected"]:
            hits_s = f"{m['hits']}/{m['expected']}"
        else:
            hits_s = "  —  "
        print(f"  {desc:<48}  {score_s:>6}   {hits_s:>6}")
    print()

    # Estadísticas globales
    scores = [m["top_score"] for m in metrics if m["n_results"] > 0]
    if scores:
        avg = sum(scores) / len(scores)
        mn  = min(scores)
        mx  = max(scores)
        print(f"  Score top-1 promedio: {avg:.3f}")
        print(f"  Score top-1 min/max:  {mn:.3f} / {mx:.3f}")

    # Precisión vs expected_hits
    cases_with_expected = [m for m in metrics if m["expected"] > 0]
    if cases_with_expected:
        full_hit  = sum(1 for m in cases_with_expected if m["hits"] == m["expected"])
        partial   = sum(1 for m in cases_with_expected if 0 < m["hits"] < m["expected"])
        misses    = sum(1 for m in cases_with_expected if m["hits"] == 0)
        total     = len(cases_with_expected)
        print(f"  Precisión expected_hits:")
        print(f"    ✓ Acierto total:  {full_hit}/{total}")
        print(f"    ~ Acierto parcial: {partial}/{total}")
        print(f"    ✗ Fallo total:     {misses}/{total}")
    print()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tester del retriever RAG sobre la knowledge base UPC."
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Muestra preview más largo de los chunks.")
    parser.add_argument("--case", "-c", type=int, default=None,
                        help="Ejecuta solo el caso N (1-based).")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Ejecuta una query ad-hoc en vez de los casos predefinidos.")
    parser.add_argument("--carrera", type=str, default=None,
                        help="Filtro de carrera para --query.")
    parser.add_argument("--doc-types", nargs="*", default=None,
                        help="Filtro de doc_types para --query.")
    parser.add_argument("--k", type=int, default=4,
                        help="Cantidad de resultados a recuperar (default 4).")
    args = parser.parse_args()

    stats = get_stats()
    print("=" * 80)
    print("ChromaDB stats:", stats)
    print("=" * 80)

    if not stats.get("available"):
        print()
        print("⚠ ChromaDB no disponible. Corre `python -m rag.ingest` primero.")
        return

    # Query ad-hoc
    if args.query:
        case = {
            "desc":      "Query ad-hoc",
            "query":     args.query,
            "carrera":   args.carrera,
            "doc_types": args.doc_types,
            "expected_hits": [],
        }
        run_case(case, k=args.k, verbose=args.verbose)
        return

    # Casos predefinidos
    cases = TEST_CASES
    if args.case:
        if 1 <= args.case <= len(cases):
            cases = [TEST_CASES[args.case - 1]]
        else:
            print(f"  --case fuera de rango. Hay {len(TEST_CASES)} casos.")
            return

    metrics = []
    for case in cases:
        metrics.append(run_case(case, k=args.k, verbose=args.verbose))

    # Solo mostrar reporte agregado si se corrieron todos los casos
    if len(cases) > 1:
        print_summary(metrics)


if __name__ == "__main__":
    main()