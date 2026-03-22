# ─────────────────────────────────────────────
# Dockerfile — Compatible Hugging Face Spaces (SDK Docker)
# ─────────────────────────────────────────────
 
# Image de base légère Python (Debian Slim)
FROM python:3.10-slim
 
# Métadonnées
LABEL maintainer="ton_nom"
LABEL description="Faster R-CNN Object Detection API — FastAPI + PyTorch CPU"
 
# ── Variables d'environnement ──────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/weights/model.pth
ENV PORT=7860
 
# ── Répertoire de travail ─────────────────────
WORKDIR /app
 
# ── Dépendances système minimales ─────────────
# libgomp1 : requis par PyTorch pour le multithreading CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
 
# ── Installation des dépendances Python ───────
# Copie d'abord requirements.txt seul pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
 
# ── Copie du code source ───────────────────────
COPY app.py .
COPY model.py .
 
# ── Copie des poids du modèle ─────────────────
COPY weights/ ./weights/
 
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser
 
# ── Exposition du port ────────────────────────
EXPOSE 7860
 
# ── Commande de démarrage ─────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]