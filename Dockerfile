# Dockerfile (multi-stage)

######################
# Stage 1: builder
######################
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

######################
# Stage 2: runtime
######################
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Копіюємо код Flask + модель
COPY app.py model.py ./
COPY templates ./templates
COPY static ./static

# Файл моделі буде покладений workflow у корінь як best_model.pth
COPY best_model.pth ./best_model.pth

ENV MODEL_PATH=/app/best_model.pth

EXPOSE 8000

CMD ["python", "app.py"]

