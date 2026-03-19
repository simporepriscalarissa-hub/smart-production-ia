from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os

app = FastAPI(title="Smart Production IA - Détection Qualité YOLOv11")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle YOLOv11
model = None
try:
    from ultralytics import YOLO
    model_path = "best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print("✅ Modèle YOLOv11 chargé avec succès !")
    else:
        print("⚠️ best.pt non trouvé — mode simulation activé")
except Exception as e:
    print(f"⚠️ Erreur : {e}")

@app.get("/")
def root():
    return {
        "service": "Smart Production IA",
        "version": "1.0.0",
        "modele": "YOLOv11",
        "modele_charge": model is not None,
        "status": "actif"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modele": "yolov11" if model else "simulation"
    }

@app.post("/analyser")
async def analyser_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        if model is not None:
            results = model(img_array, conf=0.5)
            detections = []

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls]
                    detections.append({
                        "classe": label,
                        "confiance": round(conf * 100, 1),
                        "conforme": False
                    })

            nb_defauts = len(detections)
            conforme = nb_defauts == 0

            return {
                "conforme": conforme,
                "nb_defauts": nb_defauts,
                "nb_conformes": 1 if conforme else 0,
                "detections": detections,
                "confiance_max": max([d["confiance"] for d in detections], default=0),
                "message": "✅ Pièce conforme — Aucun défaut détecté" if conforme else f"❌ {nb_defauts} défaut(s) détecté(s)",
                "mode": "yolov11"
            }

        else:
            import random
            conforme = random.random() > 0.3
            nb_defauts = 0 if conforme else random.randint(1, 3)
            return {
                "conforme": conforme,
                "nb_defauts": nb_defauts,
                "nb_conformes": 1 if conforme else 0,
                "detections": [],
                "confiance_max": 0,
                "message": "✅ Pièce conforme (simulation)" if conforme else f"❌ {nb_defauts} défaut(s) (simulation)",
                "mode": "simulation"
            }

    except Exception as e:
        return {
            "erreur": str(e),
            "conforme": False,
            "mode": "erreur"
        }