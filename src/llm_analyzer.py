"""
LLM-based "understanding" half of the hybrid pipeline.

The LLM is prompted to actually read and reason about the article -- checking
internal consistency, evidence/sourcing, tone, and known misinformation
patterns -- rather than just pattern-matching on word frequencies like the
TF-IDF model does. It returns structured JSON.

Three providers are supported out of the box (pick one via LLM_PROVIDER in .env):
  - "anthropic"  (Claude, via ANTHROPIC_API_KEY)
  - "openai"     (GPT models, via OPENAI_API_KEY)
  - "gemini"     (Google Gemini, via GEMINI_API_KEY)

If no API key is configured, a rule-based fallback analyzer is used instead
so the app still runs end-to-end for local demo/testing purposes. Swap in a
real key for genuine LLM reasoning.
"""
import json
import os
import re

from src.data_preprocessing import sensationalism_score

SYSTEM_PROMPT = """You are a careful, skeptical fact-checking analyst. You will \
be given a news article or claim. Analyze it for:
1. Internal consistency and plausibility of the claims made.
2. Presence of verifiable sources, named experts, dates, or institutions vs. \
vague/unattributed claims ("sources say", "insiders reveal").
3. Emotional/sensational language designed to provoke sharing rather than inform.
4. Whether the claim matches patterns of known misinformation (miracle cures, \
secret conspiracies, impossible claims, urgency/fear tactics).
5. Overall credibility.

Respond ONLY with a JSON object, no other text, in exactly this shape:
{
  "verdict": "FAKE" or "REAL",
  "confidence": <float 0-1>,
  "reasoning": "<2-4 sentence explanation in plain language>",
  "red_flags": ["<short flag>", ...]
}"""


def _call_anthropic(text: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Article:\n\n{text}"}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    return _parse_json(raw)


def _call_openai(text: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Article:\n\n{text}"},
        ],
        temperature=0,
        max_tokens=500,
    )
    raw = resp.choices[0].message.content
    return _parse_json(raw)


def _call_gemini(text: str) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)
    resp = model.generate_content(
        f"Article:\n\n{text}",
        generation_config={"temperature": 0, "response_mime_type": "application/json"},
    )
    return _parse_json(resp.text)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    data["verdict"] = str(data.get("verdict", "REAL")).upper()
    data["confidence"] = float(data.get("confidence", 0.5))
    data["reasoning"] = data.get("reasoning", "")
    data["red_flags"] = data.get("red_flags", [])
    return data


def _fallback_heuristic_analyzer(text: str) -> dict:
    """No API key configured -- deterministic rule-based stand-in so the app
    still runs for demo purposes. NOT a substitute for a real LLM call."""
    score = sensationalism_score(text)
    vague_sourcing = bool(re.search(r"\b(sources say|insider|anonymous tipster|experts baffled)\b", text.lower()))
    if vague_sourcing:
        score += 0.3

    verdict = "FAKE" if score >= 0.3 else "REAL"
    confidence = min(0.55 + score * 0.3, 0.95) if verdict == "FAKE" else min(0.55 + (0.3 - score) * 0.5, 0.9)

    flags = []
    if score > 0:
        flags.append("Sensational or clickbait-style language detected")
    if vague_sourcing:
        flags.append("Vague or unattributed sourcing")
    if not flags:
        flags.append("No strong red flags detected by heuristic scan")

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "reasoning": (
            "No LLM API key configured, so this result comes from a lightweight rule-based "
            "fallback (keyword/tone heuristics), not genuine language understanding. "
            "Configure ANTHROPIC_API_KEY or OPENAI_API_KEY in .env for real LLM analysis."
        ),
        "red_flags": flags,
        "provider": "heuristic_fallback",
    }


def analyze(text: str) -> dict:
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    try:
        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            result = _call_anthropic(text)
            result["provider"] = "anthropic"
            return result
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            result = _call_openai(text)
            result["provider"] = "openai"
            return result
        if provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
            result = _call_gemini(text)
            result["provider"] = "gemini"
            return result
    except Exception as exc:  # noqa: BLE001 - degrade gracefully in a demo app
        fallback = _fallback_heuristic_analyzer(text)
        fallback["reasoning"] = f"LLM call failed ({exc}); using fallback heuristic. " + fallback["reasoning"]
        return fallback

    return _fallback_heuristic_analyzer(text)
