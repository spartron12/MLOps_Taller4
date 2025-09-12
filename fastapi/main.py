# fastapi_app_registry.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import logging
import os
import mlflow

# Configuración del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instanciar FastAPI
app = FastAPI(
    title="Penguins Species Prediction API",
    description="API para predecir especies de pingüinos usando modelo en MLflow Registry",
    version="3.0.0"
)

# Variables globales
model = None
species_mapping = {1: "Adelie", 2: "Chinstrap", 3: "Gentoo"}

# Esquema de entrada
class PenguinFeatures(BaseModel):
    bill_length_mm: float = Field(..., example=39.1)
    bill_depth_mm: float = Field(..., example=18.7)
    flipper_length_mm: float = Field(..., example=181.0)
    body_mass_g: float = Field(..., example=3750.0)
    year: int = Field(..., example=2007)
    island_Biscoe: int = Field(0, example=0)
    island_Dream: int = Field(0, example=0)  
    island_Torgersen: int = Field(1, example=1)
    sex_female: int = Field(0, example=0)
    sex_male: int = Field(1, example=1)

# Cargar modelo desde MLflow Registry al iniciar FastAPI
@app.on_event("startup")
async def load_model():
    global model
    try:
        # Configuración de conexión
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "admin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "supersecret")

        # 🔹 Tracking server de MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5003"))

        # 🔹 Cargar modelo desde el Registry
        MODEL_NAME = "lucio_model3"
        MODEL_STAGE = "Production"   # o "Staging", o "1" para versión específica
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

        model = mlflow.pyfunc.load_model(model_uri)

        logger.info(f"✅ Modelo {MODEL_NAME} ({MODEL_STAGE}) cargado exitosamente desde MLflow Registry")
        logger.info(f"Tipo de modelo: {type(model)}")

        # Verificar con predicción dummy
        test_data = pd.DataFrame([{
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "year": 2007,
            "island_Biscoe": 0,
            "island_Dream": 0,
            "island_Torgersen": 1,
            "sex_female": 0,
            "sex_male": 1
        }])
        test_prediction = model.predict(test_data)
        logger.info(f"Predicción de prueba: {test_prediction}")

    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {str(e)}")
        raise e

# Endpoint de predicción
@app.post("/predict")
def predict(features: PenguinFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    try:
        X = pd.DataFrame([features.dict()])
        prediction = model.predict(X)[0]
        return {
            "species_id": int(prediction),
            "species_name": species_mapping.get(int(prediction), "Desconocido"),
            "model_used": "MLflow Registry"
        }
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

# Endpoint de info
@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {"model_type": str(type(model).__name__), "model_loaded": True}