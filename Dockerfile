# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final lightweight image
FROM python:3.9-slim AS runner

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . /app

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Default port for FastAPI backend
EXPOSE 8002
# Default port for Streamlit UI
EXPOSE 8503

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8002"]
