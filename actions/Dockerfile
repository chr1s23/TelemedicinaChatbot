FROM python:3.10.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos el requirements.txt y lo instalamos primero para aprovechar cache de docker
COPY requirements.txt .

# Actualizamos pip y setuptools antes de instalar dependencias
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY . /app

#se copia el modelo pre-entrenado en el contenedor
COPY cpu_and_mobile /app/cpu_and_mobile


EXPOSE 5055

CMD ["python", "-m", "rasa_sdk.endpoint", "--actions", "actions"]

