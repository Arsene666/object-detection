"""
app.py
API FastAPI pour la détection d'objets avec Faster R-CNN.
Compatible Hugging Face Spaces (SDK Docker) — port 7860.
"""
 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import logging
 
from model import load_model, run_inference
 
# ─────────────────────────────────────────────
# Configuration du logger
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────
# Chemin vers le fichier de poids
# Priorité : variable d'environnement → valeur par défaut locale
# ─────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "weights/fasterrcnn_checkpoint.pth")
 
# Dictionnaire global pour stocker le modèle en mémoire
# (évite de le recharger à chaque requête)
app_state: dict = {}
 
 
# ─────────────────────────────────────────────
# Lifecycle : chargement du modèle au démarrage
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie FastAPI.
    Le modèle est chargé une seule fois avant de traiter la première requête.
    """
    logger.info(f"Chargement du modèle depuis : {MODEL_PATH}")
    try:
        app_state["model"] = load_model(MODEL_PATH)
        logger.info("✅ Modèle chargé avec succès.")
    except Exception as e:
        logger.error(f"❌ Impossible de charger le modèle : {e}")
        raise RuntimeError(f"Erreur de chargement du modèle : {e}")
 
    yield  # L'application tourne ici
 
    # Nettoyage à l'arrêt (optionnel)
    app_state.clear()
    logger.info("Modèle libéré.")
 
 
# ─────────────────────────────────────────────
# Initialisation de l'application FastAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="Object Detection API",
    description="API de détection d'objets basée sur Faster R-CNN (PyTorch). "
                "Déployée sur Hugging Face Spaces via Docker.",
    version="1.0.0",
    lifespan=lifespan,
)
 
 
# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
 
@app.get("/", summary="Health check")
async def root():
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok", "message": "Object Detection API is running 🚀"}
 
 
@app.get("/health", summary="Health check détaillé")
async def health():
    """Retourne l'état de santé de l'API et la disponibilité du modèle."""
    model_loaded = "model" in app_state and app_state["model"] is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }
 
 
@app.post(
    "/predict",
    summary="Détection d'objets",
    response_description="Labels, scores et bounding boxes détectés",
)
async def predict(file: UploadFile = File(..., description="Image à analyser (JPEG, PNG, WebP…)")):
    """
    Reçoit une image et retourne les objets détectés.
 
    - **labels** : noms des classes détectées
    - **scores** : scores de confiance (entre 0 et 1)
    - **boxes**  : coordonnées des bounding boxes [x1, y1, x2, y2]
    """
    # Vérification que le modèle est bien chargé
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore prêt.")
 
    # Vérification basique du type MIME
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(
            status_code=415,
            detail=f"Type de fichier non supporté : {file.content_type}. "
                   "Utilisez JPEG, PNG, WebP ou BMP.",
        )
 
    try:
        image_bytes = await file.read()
        logger.info(f"Image reçue : {file.filename} ({len(image_bytes)} bytes)")
 
        detections = run_inference(app_state["model"], image_bytes)
 
        logger.info(f"{len(detections['labels'])} objet(s) détecté(s).")
        return JSONResponse(content=detections)
 
    except Exception as e:
        logger.error(f"Erreur pendant l'inférence : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {str(e)}")
 
 
# ─────────────────────────────────────────────
# Point d'entrée (développement local)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Sur HF Spaces, le port doit être 7860
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)