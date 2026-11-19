FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN useradd --create-home auditor && chown -R auditor /app
USER auditor

# Единая команда проверки. Сеть не нужна: запускайте с --network none.
CMD ["python", "-m", "auditgym", "check"]
