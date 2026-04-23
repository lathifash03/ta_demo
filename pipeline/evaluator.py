"""
Fase 5: Evaluasi Otomatis
- ROUGE-1, ROUGE-2, ROUGE-L
- BERTScore F1
- Entity Preservation Rate (dari post-processing)

Catatan: BERTScore memerlukan model download pertama kali (~1GB).
Jika tidak tersedia, akan fallback ke estimasi berbasis cosine similarity sederhana.
"""

from typing import Dict, Any, Optional


def compute_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """
    Hitung ROUGE scores menggunakan rouge-score library.
    hypothesis = output LLM (setelah post-processing)
    reference  = teks asli (digunakan sebagai pseudo-reference)
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1_f": round(scores["rouge1"].fmeasure, 4),
            "rouge2_f": round(scores["rouge2"].fmeasure, 4),
            "rougeL_f": round(scores["rougeL"].fmeasure, 4),
            "rouge1_p": round(scores["rouge1"].precision, 4),
            "rouge1_r": round(scores["rouge1"].recall, 4),
        }
    except ImportError:
        return {
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "rouge1_p": 0.0,
            "rouge1_r": 0.0,
            "error": "rouge-score not installed. Run: pip install rouge-score"
        }
    except Exception as e:
        return {
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "rouge1_p": 0.0,
            "rouge1_r": 0.0,
            "error": str(e)
        }


def compute_bertscore(hypothesis: str, reference: str) -> Dict[str, float]:
    """
    Hitung BERTScore menggunakan bert-score library.
    Menggunakan model distilbert untuk kecepatan.
    """
    if not hypothesis or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": "Empty input"}

    try:
        from bert_score import score as bert_score_fn
        import torch

        P, R, F1 = bert_score_fn(
            [hypothesis],
            [reference],
            model_type="distilbert-base-uncased",
            lang="en",
            verbose=False,
            device="cpu"
        )
        return {
            "precision": round(P.mean().item(), 4),
            "recall": round(R.mean().item(), 4),
            "f1": round(F1.mean().item(), 4),
        }
    except ImportError:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": "bert-score not installed. Run: pip install bert-score"
        }
    except Exception as e:
        # Fallback: hitung word overlap sederhana sebagai estimasi
        hyp_words = set(hypothesis.lower().split())
        ref_words = set(reference.lower().split())
        if not hyp_words or not ref_words:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": str(e)}

        overlap = len(hyp_words & ref_words)
        precision = overlap / len(hyp_words) if hyp_words else 0
        recall = overlap / len(ref_words) if ref_words else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "note": f"Estimated (BERTScore unavailable: {str(e)[:50]})"
        }


def run_evaluation(
    validated_text: str,
    original_text: str,
    entity_pass_rate: float
) -> Dict[str, Any]:
    """
    Main function: jalankan seluruh Fase 5.
    
    validated_text  = output setelah post-processing
    original_text   = artikel asli (pseudo-reference)
    entity_pass_rate = dari post-processing (0–100)
    """
    if not validated_text:
        return {
            "rouge": {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0},
            "bertscore": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "entity_preservation_rate": 0.0,
            "overall_quality": 0.0
        }

    rouge = compute_rouge(validated_text, original_text)
    bertscore = compute_bertscore(validated_text, original_text)

    # Overall quality: rata-rata dari semua metrik
    scores = [
        rouge.get("rougeL_f", 0.0),
        bertscore.get("f1", 0.0),
        entity_pass_rate / 100
    ]
    overall = sum(scores) / len(scores)

    return {
        "rouge": rouge,
        "bertscore": bertscore,
        "entity_preservation_rate": entity_pass_rate,
        "overall_quality": round(overall * 100, 1)
    }
