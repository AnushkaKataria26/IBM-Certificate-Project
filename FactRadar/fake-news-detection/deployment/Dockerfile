# deployment/Dockerfile — Containerized fake news detection API
#
# Uses the locked dependency versions from Phase 0 for exact reproducibility.
# NLTK data is downloaded inside the container (not copied from host) to avoid
# missing-data failures at runtime.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching — deps change less often than code)
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Download NLTK data required by clean_text.py at runtime:
#   - stopwords: for stop word removal
#   - wordnet: for lemmatization
#   - punkt_tab: for word_tokenize
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('punkt_tab', quiet=True)"

# Copy application code and model artifacts
COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
