FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# GROQ_API_KEY, GOOGLE_API_KEY, LANGFUSE_*, REDIS_URL are read from the
# environment at runtime (e.g. via `docker run --env-file .env` or
# docker-compose) — nothing secret is baked into the image.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
