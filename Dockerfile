FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the demo ML model at build time so the container works immediately.
RUN python src/train_ml_model.py --data data/sample_news.csv

ENV PORT=5000
EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "app:app"]
