FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rca ./rca
COPY models ./models
ENV RCA_DATA_DIR=/tmp/rca-data PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn rca.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
