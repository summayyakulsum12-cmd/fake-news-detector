# Fake News Detector — Hybrid LLM + ML Pipeline

Detects whether a piece of news is likely **FAKE** or **REAL** using a hybrid
approach:

1. **Statistical model** — TF-IDF + Logistic Regression, trained on labeled
   examples. Fast, cheap, catches known lexical/stylistic patterns.
2. **LLM understanding layer** — the actual text is sent to an LLM (Claude,
   GPT, or Gemini) with a fact-checking prompt. It reasons about internal consistency,
   sourcing/evidence, sensational tone, and known misinformation patterns,
   and returns a structured verdict + explanation.
3. **Hybrid combiner** — merges both signals (LLM weighted more heavily by
   default, since it actually reads and understands the content) into one
   final verdict, confidence score, and human-readable explanation.

If no LLM API key is configured, the app falls back to a deterministic
rule-based heuristic so it still runs end-to-end for demos — swap in a real
key for genuine LLM reasoning.

## Project structure

```
fake-news-detector/
├── app.py                  # Flask app (web UI + JSON API)
├── requirements.txt
├── .env.example             # copy to .env and fill in
├── Dockerfile
├── docker-compose.yml
├── Procfile                 # Heroku/Render
├── data/
│   └── sample_news.csv      # small demo dataset (replace for production)
├── models/                  # trained model artifacts land here
├── src/
│   ├── data_preprocessing.py
│   ├── train_ml_model.py    # trains the TF-IDF + LogisticRegression model
│   ├── ml_model.py          # loads model, exposes predict()
│   ├── llm_analyzer.py      # LLM reasoning layer (Anthropic/OpenAI/fallback)
│   └── hybrid_predictor.py  # combines both into a final verdict
├── templates/index.html     # web UI
├── static/style.css
└── tests/test_app.py
```

## 1. Local setup

```bash
cd fake-news-detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set LLM_PROVIDER + ANTHROPIC_API_KEY or OPENAI_API_KEY

# Train the baseline ML model (uses the bundled sample dataset by default)
python src/train_ml_model.py --data data/sample_news.csv

# Run the app
python app.py
# -> open http://localhost:5000
```

## 2. Using your own dataset

The bundled `data/sample_news.csv` is a small demo set (~30 rows) so the
project runs immediately. For a real deployment, swap in a larger labeled
dataset (e.g. the Kaggle "Fake and Real News Dataset", ISOT, or LIAR) with the
same two columns: `text,label` (label = `FAKE` or `REAL`), then re-run:

```bash
python src/train_ml_model.py --data data/your_dataset.csv
```

## 3. API usage

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm miracle cure banned by big pharma, insiders say."}'
```

Response:
```json
{
  "success": true,
  "result": {
    "final_verdict": "FAKE",
    "final_confidence": 0.87,
    "explanation": "...",
    "red_flags": ["Sensational or clickbait-style language detected", "..."],
    "components": { "ml_model": {...}, "llm_analysis": {...}, "agreement": true }
  }
}
```

## 4. Running tests

```bash
pytest
```

## 5. Deployment options

### Docker (recommended)
```bash
docker compose up --build
# -> http://localhost:5000
```
The Dockerfile trains the demo model at build time, so the container works
immediately. Mount your own `data/` + re-run training for production data.

### Render / Railway / Heroku
- `Procfile` is included (`web: gunicorn app:app`, plus a `release` step that
  trains the model).
- Set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` and `LLM_PROVIDER` as environment
  variables in the platform's dashboard — do not commit `.env`.

### Any VM / bare server
```bash
pip install -r requirements.txt
python src/train_ml_model.py --data data/sample_news.csv
gunicorn -b 0.0.0.0:5000 -w 2 app:app
```
Put nginx in front for TLS/reverse proxy in production.

## 6. Configuration reference (.env)

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `anthropic`, `openai`, `gemini`, or blank for offline heuristic fallback |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Claude credentials + model name |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI credentials + model name |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Gemini credentials + model name |
| `ML_WEIGHT` / `LLM_WEIGHT` | Relative weight of each signal in the hybrid combiner |
| `PORT` | Port Flask/gunicorn binds to |

## Notes & next steps

- The bundled dataset is intentionally tiny — accuracy numbers from it are
  illustrative only, not a real benchmark. Retrain on a proper dataset before
  relying on this for anything beyond a demo.
- The LLM prompt in `src/llm_analyzer.py` is a starting point — tune it
  (e.g. add few-shot examples, ask it to cite what part of the text triggered
  a flag) to fit your domain.
- Consider adding a caching layer (e.g. Redis) in front of the LLM calls if
  you expect repeated queries on the same text, to control API costs.
