"""
Fase 2: Rule-Based Pre-Processing
- Text cleaning
- NER (spaCy): persons, orgs, locations, dates, money
- Action verb extraction
- Temporal marker detection
- Constraint generation
"""

import re
import spacy
from typing import Dict, List, Any

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Action verb lexicon (common in news articles)
ACTION_VERBS = {
    "announced", "launched", "released", "developed", "signed", "acquired",
    "merged", "expanded", "opened", "closed", "sold", "bought", "invested",
    "published", "reported", "confirmed", "revealed", "stated", "declared",
    "approved", "rejected", "filed", "submitted", "completed", "started",
    "ended", "began", "finished", "increased", "decreased", "raised", "cut",
    "hired", "fired", "appointed", "resigned", "founded", "established",
    "partnered", "collaborated", "introduced", "implemented", "deployed",
    "upgraded", "updated", "created", "built", "designed", "won", "lost",
    "received", "awarded", "achieved", "reached", "exceeded", "failed",
    "entered", "exited", "joined", "left", "moved", "transferred", "produced"
}

# Temporal markers
TEMPORAL_MARKERS = [
    "first", "then", "next", "after", "before", "following", "subsequently",
    "meanwhile", "later", "finally", "eventually", "initially", "previously",
    "recently", "now", "today", "yesterday", "tomorrow", "when", "while",
    "during", "since", "until", "once", "immediately", "soon", "earlier",
    "afterwards", "consequently", "thereafter"
]


def clean_text(text: str) -> str:
    """Step 1: Bersihkan teks dari karakter non-standar."""
    # Hapus karakter non-ASCII
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\'\"]', ' ', text)
    # Normalisasi spasi berlebihan
    text = re.sub(r'\s+', ' ', text)
    # Perbaiki spasi sebelum tanda baca
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip()


def tokenize_sentences(text: str) -> List[str]:
    """Step 2: Tokenisasi kalimat dengan validasi panjang minimal."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw if len(s.strip()) > 10]
    return sentences


def extract_entities(sentences: List[str]) -> Dict[str, List[str]]:
    """Step 3: NER menggunakan spaCy."""
    entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],      # Geopolitical entities (countries, cities)
        "LOC": [],
        "DATE": [],
        "MONEY": [],
        "MISC": []
    }
    entity_freq = {}

    for sent in sentences:
        doc = nlp(sent)
        for ent in doc.ents:
            label = ent.label_
            text_ent = ent.text.strip()

            # Map label ke kategori kita
            if label == "PERSON":
                cat = "PERSON"
            elif label in ("ORG", "NORP"):
                cat = "ORG"
            elif label == "GPE":
                cat = "GPE"
            elif label == "LOC":
                cat = "LOC"
            elif label in ("DATE", "TIME"):
                cat = "DATE"
            elif label == "MONEY":
                cat = "MONEY"
            else:
                cat = "MISC"

            if text_ent not in entity_freq:
                entity_freq[text_ent] = {"cat": cat, "freq": 0}
            entity_freq[text_ent]["freq"] += 1

    # Sort by frequency descending
    sorted_entities = sorted(entity_freq.items(), key=lambda x: x[1]["freq"], reverse=True)

    for ent_text, info in sorted_entities:
        cat = info["cat"]
        if ent_text not in entities[cat]:
            entities[cat].append(ent_text)

    return entities


def extract_action_verbs(sentences: List[str]) -> List[Dict[str, Any]]:
    """Step 4: Ekstraksi kata kerja aksi."""
    action_sequence = []

    for i, sent in enumerate(sentences):
        doc = nlp(sent)
        for token in doc:
            if token.lemma_.lower() in ACTION_VERBS and token.pos_ == "VERB":
                # Cari subjek dan objek
                subject = ""
                obj = ""
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child.text
                    elif child.dep_ in ("dobj", "attr"):
                        obj = child.text

                action_sequence.append({
                    "verb": token.text,
                    "lemma": token.lemma_,
                    "subject": subject,
                    "object": obj,
                    "sentence_idx": i,
                    "sentence": sent
                })

    return action_sequence


def detect_temporal_markers(sentences: List[str]) -> List[Dict[str, Any]]:
    """Step 5: Deteksi penanda temporal."""
    temporal_info = []

    for i, sent in enumerate(sentences):
        tokens = sent.lower().split()
        for j, word in enumerate(tokens):
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in TEMPORAL_MARKERS:
                temporal_info.append({
                    "marker": clean_word,
                    "sentence_idx": i,
                    "word_position": j,
                    "sentence": sent
                })

    return temporal_info


def generate_constraints(
    entities: Dict[str, List[str]],
    action_sequence: List[Dict[str, Any]],
    temporal_info: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Step 6: Buat constraints untuk memandu LLM."""

    # Mandatory entities: ambil top-5 paling penting
    mandatory = []
    for cat in ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY"]:
        for ent in entities.get(cat, [])[:2]:  # max 2 per kategori
            if len(mandatory) < 8:
                mandatory.append(f"{ent} ({cat})")

    # Event sequence dari action verbs
    event_sequence = []
    for av in action_sequence[:5]:  # max 5 events
        if av["subject"] or av["object"]:
            desc = av["verb"]
            if av["subject"]:
                desc = f"{av['subject']} {desc}"
            if av["object"]:
                desc = f"{desc} {av['object']}"
            event_sequence.append(desc)

    constraints = {
        "mandatory_entities": mandatory,
        "event_sequence": event_sequence,
        "temporal_markers_found": [t["marker"] for t in temporal_info[:5]],
        "format_constraints": [
            "Use numbered list (1., 2., 3., ...)",
            "Start each step with action verb in past tense",
            "Keep each step under 50 words",
            "Use Subject-Verb-Object structure",
            "Maintain chronological order"
        ],
        "linguistic_constraints": [
            "Formal tone",
            "Active voice preferred",
            "Avoid ambiguous pronouns",
            "Include specific names and dates"
        ]
    }

    return constraints


def run_preprocessing(raw_text: str) -> Dict[str, Any]:
    """
    Main function: jalankan seluruh Fase 2.
    Returns dict berisi semua hasil pre-processing.
    """
    # Step 1: Clean
    cleaned = clean_text(raw_text)

    # Step 2: Tokenize
    sentences = tokenize_sentences(cleaned)

    # Step 3: NER
    entities = extract_entities(sentences)

    # Step 4: Action verbs
    action_sequence = extract_action_verbs(sentences)

    # Step 5: Temporal markers
    temporal_info = detect_temporal_markers(sentences)

    # Step 6: Constraints
    constraints = generate_constraints(entities, action_sequence, temporal_info)

    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "sentence_count": len(sentences),
        "entities": entities,
        "action_sequence": action_sequence,
        "temporal_info": temporal_info,
        "constraints": constraints,
        "word_count": len(cleaned.split())
    }
