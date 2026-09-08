# Titanic Survival Assistant -- trains the model at build time and
# serves the Streamlit LLM interface at runtime.
FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost's OpenMP requirement.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch data and train the model into the image so `docker run` works
# immediately without a separate setup step. (For faster iterative
# rebuilds during development, comment these two lines out and instead
# `docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data ...`
# to reuse artifacts trained on the host.)
RUN python scripts/download_data.py \
    && python -m src.train \
    && python -m src.select_best_model

EXPOSE 8501

# NEBIUS_API_KEY / OPENAI_API_KEY are read from the environment at
# runtime (e.g. `docker run -e NEBIUS_API_KEY=... `); without one, the
# app runs on its deterministic rule-based parser.
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
