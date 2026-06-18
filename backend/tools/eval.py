"""
Evaluación automatizada de respuestas del backend (LLM-as-judge).

Para cada caso en eval_questions.yaml:
  1) Manda la pregunta al backend en cada modo (knowledge, rag, general).
  2) Captura la respuesta + tokens + latencia + modo real usado.
  3) Verifica heurísticas básicas (must_include / must_not_include).
  4) Le pasa la respuesta a un LLM juez con la rúbrica → score 0-10 + razonamiento.

Output: CSV en results/eval_<timestamp>.csv con todas las columnas + resumen
en stdout con promedios por modo.

Uso:
    # Asegurate de tener el backend corriendo en localhost:8000
    cd backend
    python -m tools.eval

    # Opciones
    python -m tools.eval --case cc-outcomes-list      # solo un caso
    python -m tools.eval --judge claude-3.5-sonnet    # otro juez
    python -m tools.eval --no-judge                    # solo heurísticas (gratis)
    python -m tools.eval --base-url http://localhost:8001
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

# Config centralizada (config.py está en backend/, un nivel arriba)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
BACKEND_DIR = config.BACKEND_DIR
TOOLS_DIR   = Path(__file__).resolve().parent
CASES_FILE  = TOOLS_DIR / "eval_questions.yaml"
RESULTS_DIR = TOOLS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Vienen de config.py (respetan LLM_PROVIDER → el juez puede ser local también)
OPENROUTER_API_KEY  = config.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = config.OPENROUTER_BASE_URL

# Modelo del backend a probar. Por defecto usa el mismo que tiene configurado
# el proyecto (config.DEFAULT_MODEL); se puede sobreescribir con --model.
DEFAULT_BACKEND_MODEL = config.DEFAULT_MODEL

# Modelo juez. Idealmente distinto al del backend para evitar sesgo.
# Configurable por env JUDGE_MODEL o por --judge.
import os
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct:free")


# ──────────────────────────────────────────────────────────────
# Llamadas al backend
# ──────────────────────────────────────────────────────────────
def query_backend(
    base_url: str,
    question: str,
    carrera: Optional[str],
    mode: str,                       # knowledge | rag | general
    model: str,
    timeout: int = 120,
) -> dict:
    """Manda la pregunta al backend y devuelve respuesta + metadata.
    mode → traduce a los flags use_knowledge/use_rag del backend.
    """
    use_knowledge = mode in ("knowledge", "rag")
    use_rag       = mode == "rag"

    payload = {
        "messages":      [{"role": "user", "content": question}],
        "model":         model,
        "stream":        False,
        "carrera":       carrera,
        "use_knowledge": use_knowledge,
        "use_rag":       use_rag,
        "web_search":    False,
        "temperature":   0.3,
        "max_tokens":    2048,
    }

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(f"{base_url}/api/chat", json=payload)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        if r.status_code != 200:
            return {
                "ok":         False,
                "answer":     "",
                "error":      f"HTTP {r.status_code}: {r.text[:300]}",
                "latency_ms": latency_ms,
            }

        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage   = data.get("usage", {}) or {}

        return {
            "ok":             True,
            "answer":         content,
            "latency_ms":     latency_ms,
            "prompt_tokens":  usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "headers": {
                "mode":          r.headers.get("X-Mode", ""),
                "loaded_docs":   r.headers.get("X-Loaded-Docs", ""),
                "sys_tokens":    r.headers.get("X-System-Tokens-Aprox", ""),
                "rag_used":      r.headers.get("X-Rag-Used", ""),
                "rag_chunks":    r.headers.get("X-Rag-Chunks", ""),
                "rag_top_score": r.headers.get("X-Rag-Top-Score", ""),
            },
        }
    except Exception as e:
        return {
            "ok":         False,
            "answer":     "",
            "error":      f"{type(e).__name__}: {e}",
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }


# ──────────────────────────────────────────────────────────────
# Heurísticas locales (gratis, instantáneas)
# ──────────────────────────────────────────────────────────────
def evaluate_heuristics(answer: str, expected: dict) -> dict:
    """Verifica must_include / must_not_include. Devuelve dict con métricas."""
    if not answer:
        return {
            "hits":     0,
            "total":    len(expected.get("must_include", [])),
            "misses":   [],
            "violations": [],
            "heuristic_pass": False,
        }

    lower = answer.lower()
    must_inc = expected.get("must_include", []) or []
    must_not = expected.get("must_not_include", []) or []

    hits = []
    misses = []
    for kw in must_inc:
        if kw.lower() in lower:
            hits.append(kw)
        else:
            misses.append(kw)

    violations = [kw for kw in must_not if kw.lower() in lower]

    return {
        "hits":     len(hits),
        "total":    len(must_inc),
        "misses":   misses,
        "violations": violations,
        "heuristic_pass": (len(misses) == 0 and len(violations) == 0),
    }


# ──────────────────────────────────────────────────────────────
# LLM-as-judge
# ──────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """Eres un evaluador estricto pero justo de respuestas de un asistente académico UPC.

Tu única tarea es evaluar QUÉ TAN BIEN la respuesta del asistente cumple la rúbrica que te paso. Vas a recibir:
1) Pregunta
2) Rúbrica de evaluación
3) Respuesta del asistente

Tu output debe ser SIEMPRE un JSON válido con este formato exacto:
{
  "score": <0-10>,
  "reasoning": "<1-2 oraciones explicando el score>",
  "strengths": ["..."],
  "weaknesses": ["..."]
}

Criterios de scoring:
- 9-10: respuesta excelente, cumple la rúbrica completamente
- 7-8:  cumple lo esencial, con omisiones menores
- 5-6:  parcialmente correcta
- 3-4:  con errores importantes
- 0-2:  incorrecta o no responde

NO incluyas texto fuera del JSON. NO uses markdown. NO uses backticks.
"""


def llm_judge(
    question: str,
    rubric: str,
    answer: str,
    judge_model: str,
) -> dict:
    """Llama al LLM juez y parsea su veredicto JSON."""
    if not OPENROUTER_API_KEY:
        return {"score": None, "reasoning": "OPENROUTER_API_KEY no configurada", "error": "no_key"}

    user_msg = f"""## Pregunta
{question}

## Rúbrica
{rubric}

## Respuesta del asistente
{answer}

Evalúa la respuesta y devuelve el JSON con score, reasoning, strengths, weaknesses."""

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "temperature":     0.0,
        "max_tokens":      600,
        "response_format": {"type": "json_object"},
    }
    headers = config.llm_headers()

    try:
        with httpx.Client(timeout=120) as client:
            r = client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers, json=payload,
            )
        if r.status_code != 200:
            return {
                "score":   None,
                "reasoning": f"juez HTTP {r.status_code}: {r.text[:200]}",
                "error":   "judge_http_error",
            }

        content = r.json()["choices"][0]["message"]["content"].strip()
        # Algunos modelos meten ```json ... ``` aunque pidamos JSON puro
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            verdict = json.loads(content)
        except json.JSONDecodeError:
            # Intento de parseo último: buscar el primer { y el último }
            i, j = content.find("{"), content.rfind("}")
            if i >= 0 and j > i:
                try:
                    verdict = json.loads(content[i:j+1])
                except Exception:
                    return {"score": None, "reasoning": f"juez devolvió no-JSON: {content[:200]}", "error": "judge_parse"}
            else:
                return {"score": None, "reasoning": f"juez devolvió no-JSON: {content[:200]}", "error": "judge_parse"}

        return {
            "score":      verdict.get("score"),
            "reasoning":  verdict.get("reasoning", ""),
            "strengths":  verdict.get("strengths", []),
            "weaknesses": verdict.get("weaknesses", []),
        }
    except Exception as e:
        return {"score": None, "reasoning": f"juez excepción: {e}", "error": "judge_exception"}


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def fmt_score(v) -> str:
    if v is None:
        return color("  —  ", "37")
    if v >= 9:
        return color(f"{v:>4.1f}", "32;1")
    if v >= 7:
        return color(f"{v:>4.1f}", "32")
    if v >= 5:
        return color(f"{v:>4.1f}", "33")
    return color(f"{v:>4.1f}", "31")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el backend UPC ABET en los 3 modos.")
    parser.add_argument("--base-url",  default="http://localhost:8000",
                        help="URL base del backend (default localhost:8000)")
    parser.add_argument("--model",     default=DEFAULT_BACKEND_MODEL,
                        help=f"Modelo del backend a probar (default {DEFAULT_BACKEND_MODEL})")
    parser.add_argument("--judge",     default=DEFAULT_JUDGE_MODEL,
                        help=f"Modelo juez (default {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--case",      default=None,
                        help="ID del caso a evaluar (default: todos)")
    parser.add_argument("--no-judge",  action="store_true",
                        help="No usar LLM juez — solo heurísticas (gratis)")
    parser.add_argument("--cases-file", default=str(CASES_FILE),
                        help=f"YAML con casos (default {CASES_FILE})")
    args = parser.parse_args()

    # Cargar casos
    cases_path = Path(args.cases_file)
    if not cases_path.exists():
        print(f"❌ Archivo de casos no encontrado: {cases_path}")
        sys.exit(1)

    with cases_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cases = data.get("cases", [])
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"❌ Caso no encontrado: {args.case}")
            sys.exit(1)

    # Verificar backend
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{args.base_url}/api/health")
            if r.status_code != 200:
                print(f"❌ Backend no responde 200 en {args.base_url}/api/health")
                sys.exit(1)
            health = r.json()
            rag_available = health.get("rag", False)
    except Exception as e:
        print(f"❌ No pude conectar al backend en {args.base_url}: {e}")
        sys.exit(1)

    print("=" * 80)
    print(f"🔬 Eval start")
    print(f"   Backend:  {args.base_url}")
    print(f"   Modelo:   {args.model}")
    print(f"   Juez:     {'(deshabilitado)' if args.no_judge else args.judge}")
    print(f"   RAG:      {'disponible ✓' if rag_available else 'NO disponible (los casos con mode=rag van a fallar)'}")
    print(f"   Casos:    {len(cases)}")
    print("=" * 80)

    rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        for mode in case["modes"]:
            print(f"\n[{i}/{len(cases)}] {case['id']} · mode={mode}")

            # Skip si pide RAG y no está disponible
            if mode == "rag" and not rag_available:
                print(f"   ⏭  SKIP (RAG no disponible)")
                continue

            res = query_backend(
                base_url= args.base_url,
                question= case["question"],
                carrera=  case.get("carrera"),
                mode=     mode,
                model=    args.model,
            )

            if not res["ok"]:
                print(f"   ❌ {res['error']}")
                rows.append({
                    "case_id":   case["id"],
                    "mode":      mode,
                    "status":    "error",
                    "error":     res["error"],
                    "answer":    "",
                    "latency_ms": res["latency_ms"],
                    "score":     None,
                })
                continue

            # Heurísticas
            heur = evaluate_heuristics(res["answer"], case.get("expected", {}))
            heur_str = "✓" if heur["heuristic_pass"] else f"{heur['hits']}/{heur['total']}"
            print(f"   ⏱  {res['latency_ms']}ms  · sys={res['headers']['sys_tokens']} tk · heur={heur_str}")
            print(f"   📝 {res['answer'][:160].replace(chr(10),' ')}…")

            # Juez
            verdict = {}
            if not args.no_judge:
                rubric = case.get("expected", {}).get("rubric", "")
                if rubric:
                    verdict = llm_judge(
                        question=    case["question"],
                        rubric=      rubric,
                        answer=      res["answer"],
                        judge_model= args.judge,
                    )
                    print(f"   ⚖️  score={fmt_score(verdict.get('score'))}  {verdict.get('reasoning','')[:120]}")
                else:
                    verdict = {"score": None, "reasoning": "Sin rúbrica en YAML"}

            rows.append({
                "case_id":    case["id"],
                "question":   case["question"],
                "carrera":    case.get("carrera"),
                "mode":       mode,
                "status":     "success",
                "answer":     res["answer"],
                "latency_ms": res["latency_ms"],
                "sys_tokens": res["headers"]["sys_tokens"],
                "prompt_tokens":     res.get("prompt_tokens"),
                "completion_tokens": res.get("completion_tokens"),
                "loaded_docs":  res["headers"]["loaded_docs"],
                "rag_used":     res["headers"]["rag_used"],
                "rag_chunks":   res["headers"]["rag_chunks"],
                "rag_top_score": res["headers"]["rag_top_score"],
                "heur_hits":    heur["hits"],
                "heur_total":   heur["total"],
                "heur_misses":  ",".join(heur["misses"]),
                "heur_violations": ",".join(heur["violations"]),
                "heur_pass":    heur["heuristic_pass"],
                "score":        verdict.get("score"),
                "reasoning":    verdict.get("reasoning", ""),
                "strengths":    json.dumps(verdict.get("strengths", []), ensure_ascii=False),
                "weaknesses":   json.dumps(verdict.get("weaknesses", []), ensure_ascii=False),
            })

    # ── Resumen por modo ──
    print()
    print("=" * 80)
    print(color("📊 RESUMEN POR MODO", "1;7"))
    print("=" * 80)

    by_mode: dict[str, list[dict]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)

    print(f"\n  {'Modo':<12} {'Casos':>6} {'Avg score':>10} {'Heur pass':>11} {'Avg latency':>13} {'Avg sys tk':>11}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*11} {'-'*13} {'-'*11}")
    for mode in ("general", "knowledge", "rag"):
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        scores = [r["score"] for r in rs if r.get("score") is not None]
        avg_score = sum(scores) / len(scores) if scores else None
        passes = sum(1 for r in rs if r.get("heur_pass"))
        latencies = [r["latency_ms"] for r in rs if r.get("status") == "success"]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        sys_toks = [int(r["sys_tokens"]) for r in rs
                    if r.get("sys_tokens") and str(r["sys_tokens"]).isdigit()]
        avg_sys = sum(sys_toks) / len(sys_toks) if sys_toks else 0

        score_s = f"{avg_score:.2f}" if avg_score is not None else "  —  "
        print(f"  {mode:<12} {len(rs):>6} {score_s:>10} {passes}/{len(rs):>9} {int(avg_lat):>10}ms {int(avg_sys):>11}")

    # ── Guardar CSV ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"eval_{ts}.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n💾 Resultados guardados en: {csv_path}")
    print()


if __name__ == "__main__":
    main()