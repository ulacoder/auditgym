FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Разделение привилегий: оценщик и харнесс работают от root внутри контейнера, а каждая
# политика — от отдельного непривилегированного пользователя auditgym-policy. Он не может писать
# в /app, читать /proc процессов оценщика, посылать им сигналы и переживать свой прогон.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin auditgym-policy \
    && chown -R root:root /app && chmod -R u+rwX,go+rX,go-w /app

# Единая команда проверки. Сеть не нужна: запускайте с --network none.
CMD ["python", "-m", "auditgym", "check"]
