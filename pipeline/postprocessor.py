"""
Fase 4: Rule-Based Post-Processing
- fix_numbering: pastikan penomoran sequential tanpa gap
- validate_format: cek action verb di awal tiap step
- validate_entities: cek mandatory entities ada di output (threshold ≥85%)
- quality_score: format 30% + entity 50% + step_count 20%
"""

import re
from typing import Dict, Any, List, Tuple


# ─────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────

def parse_steps(output_text: str) -> List[str]:
    """
    Parse output LLM menjadi list of steps.
    Tangani berbagai format: "1.", "1)", "Step 1:", dll.
    """
    if not output_text:
        return []

    # Coba split berdasarkan pola penomoran
    patterns = [
        r'^\d+\.\s+',       # "1. "
        r'^\d+\)\s+',       # "1) "
        r'^Step\s+\d+:\s+', # "Step 1: "
        r'^\*\s+',          # "* "
        r'^-\s+',           # "- "
    ]

    lines = output_text.strip().split('\n')
    steps = []
    current_step = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_new_step = any(re.match(p, line, re.IGNORECASE) for p in patterns)

        if is_new_step:
            if current_step:
                steps.append(current_step.strip())
            # Hapus nomor/bullet di awal
            cleaned = re.sub(r'^(\d+[\.\)]|Step\s+\d+:|[\*\-])\s+', '', line, flags=re.IGNORECASE)
            current_step = cleaned
        else:
            if current_step:
                current_step += " " + line
            elif line:
                current_step = line

    if current_step:
        steps.append(current_step.strip())

    # Filter steps yang terlalu pendek (noise)
    steps = [s for s in steps if len(s.split()) >= 3]

    return steps


# ─────────────────────────────────────────────
# VALIDASI
# ─────────────────────────────────────────────

def fix_numbering(steps: List[str]) -> Tuple[List[str], List[str]]:
    """
    Pastikan penomoran sequential.
    Return: (renumbered_steps, corrections)
    """
    corrections = []
    renumbered = []

    for i, step in enumerate(steps, 1):
        renumbered.append(f"{i}. {step}")

    if len(steps) > 0:
        corrections.append(f"Renumbered {len(steps)} steps sequentially (1–{len(steps)})")

    return renumbered, corrections


def validate_format(steps: List[str]) -> Tuple[int, List[str]]:
    """
    Cek apakah setiap step dimulai dengan action verb (kata kerja).
    Return: (violations_count, details)
    """
    violations = []

    # List kata kerja umum dalam konteks berita
    common_verbs = {
        "announced", "launched", "released", "developed", "signed", "acquired",
        "merged", "expanded", "opened", "closed", "sold", "bought", "invested",
        "published", "reported", "confirmed", "revealed", "stated", "declared",
        "approved", "rejected", "filed", "submitted", "completed", "started",
        "ended", "began", "finished", "increased", "decreased", "raised", "cut",
        "hired", "fired", "appointed", "resigned", "founded", "established",
        "partnered", "collaborated", "introduced", "implemented", "deployed",
        "upgraded", "updated", "created", "built", "designed", "won", "lost",
        "received", "awarded", "achieved", "reached", "exceeded", "failed",
        "entered", "exited", "joined", "left", "moved", "transferred", "produced",
        "the", "a", "an"  # step dimulai dengan artikel = mungkin tidak ada verb
    }

    for i, step in enumerate(steps, 1):
        words = step.split()
        if not words:
            violations.append(f"Step {i}: empty step")
            continue

        first_word = words[0].lower().rstrip('.,')
        # Heuristic: jika kata pertama adalah artikel atau pronoun
        if first_word in ("the", "a", "an", "this", "that", "it", "they", "he", "she"):
            violations.append(f"Step {i}: starts with '{words[0]}' (possible missing action verb)")

    return len(violations), violations


def validate_entities(
    steps: List[str],
    mandatory_entities: List[str]
) -> Tuple[float, List[str], List[str]]:
    """
    Cek apakah mandatory entities ada di output (threshold ≥85%).
    Return: (pass_rate, found, missing)
    """
    if not mandatory_entities:
        return 1.0, [], []

    output_text = " ".join(steps).lower()
    found = []
    missing = []

    for ent in mandatory_entities:
        # Ambil nama entitas saja (tanpa label seperti "(ORG)")
        ent_name = re.sub(r'\s*\([A-Z]+\)\s*$', '', ent).strip().lower()
        if ent_name in output_text:
            found.append(ent)
        else:
            missing.append(ent)

    pass_rate = len(found) / len(mandatory_entities) if mandatory_entities else 1.0
    return pass_rate, found, missing


def compute_quality_score(
    steps: List[str],
    format_violations: int,
    entity_pass_rate: float
) -> Dict[str, float]:
    """
    Hitung quality score:
    - Format score: 30% (berdasarkan format violations)
    - Entity score: 50% (entity pass rate)
    - Step count score: 20% (ideal 4-8 steps)
    """
    # Format score
    if not steps:
        format_score = 0.0
    else:
        violation_rate = format_violations / len(steps)
        format_score = max(0.0, 1.0 - violation_rate)

    # Entity score
    entity_score = entity_pass_rate

    # Step count score (ideal 4-8 steps)
    n = len(steps)
    if 4 <= n <= 8:
        step_score = 1.0
    elif n < 4:
        step_score = n / 4
    else:
        step_score = max(0.0, 1.0 - (n - 8) * 0.1)

    total = (format_score * 0.30) + (entity_score * 0.50) + (step_score * 0.20)

    return {
        "total": round(total * 100, 1),
        "format_score": round(format_score * 100, 1),
        "entity_score": round(entity_score * 100, 1),
        "step_count_score": round(step_score * 100, 1),
        "step_count": n
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_postprocessing(
    llm_output: str,
    mandatory_entities: List[str]
) -> Dict[str, Any]:
    """
    Main function: jalankan seluruh Fase 4.
    """
    corrections = []

    # Parse output menjadi steps
    steps = parse_steps(llm_output)

    if not steps:
        return {
            "steps_raw": [],
            "steps_numbered": [],
            "step_count": 0,
            "format_violations": 0,
            "format_violation_details": [],
            "entity_pass_rate": 0.0,
            "entities_found": [],
            "entities_missing": mandatory_entities,
            "quality_score": {
                "total": 0.0,
                "format_score": 0.0,
                "entity_score": 0.0,
                "step_count_score": 0.0,
                "step_count": 0
            },
            "corrections": ["No steps could be parsed from output"],
            "validated_text": ""
        }

    # Step 1: Fix numbering
    steps_numbered, num_corrections = fix_numbering(steps)
    corrections.extend(num_corrections)

    # Step 2: Validate format
    format_violations, violation_details = validate_format(steps)
    if format_violations > 0:
        corrections.append(f"Found {format_violations} format violation(s)")

    # Step 3: Validate entities
    entity_pass_rate, entities_found, entities_missing = validate_entities(
        steps, mandatory_entities
    )
    if entities_missing:
        corrections.append(f"Missing entities: {', '.join(entities_missing[:3])}")

    # Step 4: Quality score
    quality = compute_quality_score(steps, format_violations, entity_pass_rate)

    # Gabungkan steps menjadi teks final
    validated_text = "\n".join(steps_numbered)

    return {
        "steps_raw": steps,
        "steps_numbered": steps_numbered,
        "step_count": len(steps),
        "format_violations": format_violations,
        "format_violation_details": violation_details,
        "entity_pass_rate": round(entity_pass_rate * 100, 1),
        "entities_found": entities_found,
        "entities_missing": entities_missing,
        "quality_score": quality,
        "corrections": corrections,
        "validated_text": validated_text
    }
