"""
Fase 3: Constrained LLM Conversion
- Groq (llama-3.3-70b-versatile)
- GPT (gpt-4o-mini)
- Claude (claude-haiku-4-5-20251001)

Semua model menggunakan prompt template yang sama
dengan constraints dari Fase 2.
"""

import os
import time
import requests
from typing import Dict, Any, Optional

# Import SDK — pastikan sudah terinstall
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None


def build_prompt(text: str, constraints: Dict[str, Any]) -> tuple[str, str]:
    """
    Bangun system prompt + user prompt dengan constraints terintegrasi.
    Digunakan oleh semua 3 model (prompt yang sama = perbandingan fair).
    """
    entity_str = ", ".join(constraints.get("mandatory_entities", []))
    if not entity_str:
        entity_str = "all named entities, organizations, and dates mentioned"

    event_str = ""
    if constraints.get("event_sequence"):
        event_str = "\n".join(
            f"  Step {i+1}: {e}"
            for i, e in enumerate(constraints["event_sequence"])
        )
        event_str = f"\nSuggested event sequence:\n{event_str}"

    system_prompt = (
        "You are an expert text converter specializing in transforming "
        "narrative news articles into clear, procedural step-by-step formats. "
        "Your output must be factually accurate and structurally consistent."
    )

    user_prompt = f"""Convert the following news article into a numbered procedural list.

CRITICAL RULES:
1. PRESERVE ALL ENTITIES — you MUST include: {entity_str}
2. FORMAT: Numbered list only (1., 2., 3., ...)
3. VERB TENSE: Past tense only for each step
4. STRUCTURE: Each step = Subject + Action Verb + Object (max 50 words per step)
5. ORDER: Strictly chronological sequence
6. NO introduction sentence, NO conclusion — only numbered steps
{event_str}

NEWS ARTICLE:
{text}

OUTPUT (numbered steps only):"""

    return system_prompt, user_prompt


# ─────────────────────────────────────────────
# GROQ
# ─────────────────────────────────────────────

def convert_with_groq(
    text: str,
    constraints: Dict[str, Any],
    api_key: str
) -> Dict[str, Any]:
    """Konversi menggunakan Groq API (HTTP langsung)."""
    start = time.time()
    system_prompt, user_prompt = build_prompt(text, constraints)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        output_text = data["choices"][0]["message"]["content"].strip()
        elapsed = time.time() - start

        return {
            "model": "Groq (Llama 3.3 70B)",
            "model_id": "llama-3.3-70b-versatile",
            "output": output_text,
            "elapsed_seconds": round(elapsed, 2),
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "model": "Groq (Llama 3.3 70B)",
            "model_id": "llama-3.3-70b-versatile",
            "output": "",
            "elapsed_seconds": round(time.time() - start, 2),
            "success": False,
            "error": str(e)
        }


# ─────────────────────────────────────────────
# GPT
# ─────────────────────────────────────────────

def convert_with_gpt(
    text: str,
    constraints: Dict[str, Any],
    api_key: str
) -> Dict[str, Any]:
    """Konversi menggunakan OpenAI GPT-4o-mini."""
    start = time.time()

    if OpenAI is None:
        return {
            "model": "GPT (gpt-4o-mini)",
            "model_id": "gpt-4o-mini",
            "output": "",
            "elapsed_seconds": 0,
            "success": False,
            "error": "openai package not installed. Run: pip install openai"
        }

    system_prompt, user_prompt = build_prompt(text, constraints)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        output_text = response.choices[0].message.content.strip()
        elapsed = time.time() - start

        return {
            "model": "GPT (gpt-4o-mini)",
            "model_id": "gpt-4o-mini",
            "output": output_text,
            "elapsed_seconds": round(elapsed, 2),
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "model": "GPT (gpt-4o-mini)",
            "model_id": "gpt-4o-mini",
            "output": "",
            "elapsed_seconds": round(time.time() - start, 2),
            "success": False,
            "error": str(e)
        }


# ─────────────────────────────────────────────
# CLAUDE
# ─────────────────────────────────────────────

def convert_with_claude(
    text: str,
    constraints: Dict[str, Any],
    api_key: str
) -> Dict[str, Any]:
    """Konversi menggunakan Anthropic Claude Haiku."""
    start = time.time()

    if anthropic is None:
        return {
            "model": "Claude (Haiku)",
            "model_id": "claude-haiku-4-5-20251001",
            "output": "",
            "elapsed_seconds": 0,
            "success": False,
            "error": "anthropic package not installed. Run: pip install anthropic"
        }

    system_prompt, user_prompt = build_prompt(text, constraints)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        output_text = response.content[0].text.strip()
        elapsed = time.time() - start

        return {
            "model": "Claude (Haiku)",
            "model_id": "claude-haiku-4-5-20251001",
            "output": output_text,
            "elapsed_seconds": round(elapsed, 2),
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "model": "Claude (Haiku)",
            "model_id": "claude-haiku-4-5-20251001",
            "output": "",
            "elapsed_seconds": round(time.time() - start, 2),
            "success": False,
            "error": str(e)
        }


# ─────────────────────────────────────────────
# RUNNER: Semua model
# ─────────────────────────────────────────────

def run_all_models(
    text: str,
    constraints: Dict[str, Any],
    groq_key: str = "",
    openai_key: str = "",
    anthropic_key: str = ""
) -> Dict[str, Dict[str, Any]]:
    """
    Jalankan ketiga model dan kembalikan hasil masing-masing.
    Model yang tidak punya API key akan di-skip.
    """
    results = {}

    if groq_key:
        results["groq"] = convert_with_groq(text, constraints, groq_key)
    else:
        results["groq"] = {
            "model": "Groq (Llama 3.3 70B)",
            "model_id": "llama-3.3-70b-versatile",
            "output": "",
            "elapsed_seconds": 0,
            "success": False,
            "error": "API key not provided"
        }

    if openai_key:
        results["gpt"] = convert_with_gpt(text, constraints, openai_key)
    else:
        results["gpt"] = {
            "model": "GPT (gpt-4o-mini)",
            "model_id": "gpt-4o-mini",
            "output": "",
            "elapsed_seconds": 0,
            "success": False,
            "error": "API key not provided"
        }

    if anthropic_key:
        results["claude"] = convert_with_claude(text, constraints, anthropic_key)
    else:
        results["claude"] = {
            "model": "Claude (Haiku)",
            "model_id": "claude-haiku-4-5-20251001",
            "output": "",
            "elapsed_seconds": 0,
            "success": False,
            "error": "API key not provided"
        }

    return results
