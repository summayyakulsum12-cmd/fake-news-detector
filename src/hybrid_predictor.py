"""
Combines the TF-IDF/LogisticRegression model (fast, pattern-based) with the
LLM analyzer (slower, reads and reasons about the text) into a single
verdict + confidence + human-readable explanation.

Combination logic:
  - If both agree            -> average confidence, high trust.
  - If they disagree         -> weight LLM more heavily (it "understands"
                                 context the TF-IDF model can't see), but
                                 flag the disagreement explicitly for the user.
  - Weights are configurable via ML_WEIGHT / LLM_WEIGHT env vars.
"""
import os

from src import ml_model
from src import llm_analyzer


def _weights():
    ml_w = float(os.environ.get("ML_WEIGHT", 0.35))
    llm_w = float(os.environ.get("LLM_WEIGHT", 0.65))
    total = ml_w + llm_w
    return ml_w / total, llm_w / total


def predict(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    ml_result = ml_model.predict(text) if ml_model.is_ready() else None
    llm_result = llm_analyzer.analyze(text)

    ml_w, llm_w = _weights()

    if ml_result is None:
        final_label = llm_result["verdict"]
        final_confidence = llm_result["confidence"]
        agreement = None
    else:
        agreement = ml_result["label"] == llm_result["verdict"]
        # score each label as weighted vote
        scores = {"FAKE": 0.0, "REAL": 0.0}
        scores[ml_result["label"]] += ml_w * ml_result["confidence"]
        scores[llm_result["verdict"]] += llm_w * llm_result["confidence"]
        final_label = max(scores, key=scores.get)
        final_confidence = round(scores[final_label] / (ml_w + llm_w), 4)

    explanation_parts = [llm_result.get("reasoning", "")]
    if ml_result is not None:
        explanation_parts.append(
            f"Statistical model (TF-IDF + Logistic Regression) leaned "
            f"{ml_result['label']} with {ml_result['confidence']*100:.1f}% confidence."
        )
        if agreement is False:
            explanation_parts.append(
                "Note: the statistical model and the LLM disagreed; the final "
                "verdict weights the LLM's contextual reasoning more heavily."
            )

    return {
        "final_verdict": final_label,
        "final_confidence": round(min(max(final_confidence, 0), 1), 4),
        "explanation": " ".join(p for p in explanation_parts if p),
        "red_flags": llm_result.get("red_flags", []),
        "components": {
            "ml_model": ml_result,
            "llm_analysis": {
                "verdict": llm_result["verdict"],
                "confidence": llm_result["confidence"],
                "reasoning": llm_result.get("reasoning"),
                "provider": llm_result.get("provider"),
            },
            "agreement": agreement,
        },
    }
