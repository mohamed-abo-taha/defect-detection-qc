# CPU image for the classification demo (Streamlit). Detection (ultralytics) is heavier and
# imported lazily, so it simply won't be available in this slim image — that's intentional.
FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
