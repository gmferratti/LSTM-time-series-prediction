# Builder stage
FROM --platform=linux/amd64 python:3.12-slim AS builder

WORKDIR /app

# Copiar apenas o necessário para a produção
COPY requirements.txt .

# Criar venv e instalar dependências de produção
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch \
        scikit-learn \
        numpy \
        pandas \
        joblib \
        google-cloud-storage \
        python-multipart \
        pydantic

# Final stage
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# Copiar venv da imagem builder
COPY --from=builder /opt/venv /opt/venv

# Copiar apenas os arquivos necessários
COPY src/api ./src/api
COPY secrets/key.json /app/credentials.json

# Configurar ambiente
ENV PATH="/opt/venv/bin:$PATH"
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
ENV PORT=8080

# Expor porta
EXPOSE ${PORT}

# Iniciar API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]