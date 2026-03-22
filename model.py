"""
model.py
Chargement du modèle R-CNN et logique d'inférence.
Le modèle est chargé une seule fois au démarrage de l'application.
"""
 
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import io
from typing import Dict, List
 
 
COCO_LABELS = [
    'background',
         'apple',
         'avocado',
         'banana',
         'guavs',
         'kiwi',
         'mango',
         'orange',
         'peach',
         'pineapple'
]
 
# Seuil de confiance minimum pour filtrer les détections
SCORE_THRESHOLD = 0.5
 
 
# ─────────────────────────────────────────────
# 2. Chargement du modèle (appelé une seule fois)
# ─────────────────────────────────────────────
def load_model(model_path: str) -> torch.nn.Module:
    """
    Charge le modèle Faster R-CNN depuis un fichier .pth.
    Tourne exclusivement sur CPU.
 
    Args:
        model_path: Chemin vers le fichier de poids (.pth)
 
    Returns:
        Modèle PyTorch en mode évaluation
    """
    # Instancie l'architecture (adapter num_classes à ton modèle)
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=10)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 10)

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])  
 
    model.eval()  # Désactive dropout / batchnorm en mode inférence
    return model
 
 
# ─────────────────────────────────────────────
# 3. Prétraitement de l'image
# ─────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convertit des bytes bruts en tenseur normalisé attendu par le modèle.
 
    Args:
        image_bytes: Image brute (envoyée via UploadFile)
 
    Returns:
        Tenseur [1, C, H, W] normalisé entre 0 et 1
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
 
    transform = T.Compose([
        T.ToTensor(),  # [H, W, C] uint8 → [C, H, W] float32 in [0, 1]
    ])
 
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Ajoute la dimension batch : [1, C, H, W]
 
 
# ─────────────────────────────────────────────
# 4. Inférence
# ─────────────────────────────────────────────
def run_inference(model: torch.nn.Module, image_bytes: bytes) -> Dict:
    """
    Lance l'inférence sur une image et retourne les détections filtrées.
 
    Args:
        model:       Modèle chargé
        image_bytes: Image brute en bytes
 
    Returns:
        Dictionnaire avec labels, scores et bounding boxes
    """
    tensor = preprocess_image(image_bytes)
 
    with torch.no_grad():  # Désactive le calcul de gradient (économie mémoire)
        outputs = model(tensor)
 
    # outputs[0] correspond au premier (et unique) élément du batch
    result = outputs[0]
 
    boxes  = result["boxes"].cpu().numpy()   # shape: [N, 4] — format [x1, y1, x2, y2]
    labels = result["labels"].cpu().numpy()  # shape: [N]
    scores = result["scores"].cpu().numpy()  # shape: [N]
 
    # Filtre les détections sous le seuil de confiance
    keep = scores >= SCORE_THRESHOLD
 
    detections = {
        "labels": [
            COCO_LABELS[l] if l < len(COCO_LABELS) else f"class_{l}"
            for l in labels[keep].tolist()
        ],
        "scores": [round(float(s), 4) for s in scores[keep].tolist()],
        "boxes":  [
            {
                "x1": round(float(b[0]), 2),
                "y1": round(float(b[1]), 2),
                "x2": round(float(b[2]), 2),
                "y2": round(float(b[3]), 2),
            }
            for b in boxes[keep].tolist()
        ],
    }
 
    return detections
 