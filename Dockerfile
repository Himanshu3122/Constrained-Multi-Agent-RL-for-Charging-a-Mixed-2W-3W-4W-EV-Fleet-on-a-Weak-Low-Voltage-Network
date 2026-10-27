# Research API image: FastAPI service for Layer 1 (/v1/intent) and Layer 3 (/v1/dispatch).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch keeps the image small; the API does not need CUDA.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

RUN useradd --create-home --uid 10001 solace
USER solace

EXPOSE 8000
CMD ["uvicorn", "--factory", "solace.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
