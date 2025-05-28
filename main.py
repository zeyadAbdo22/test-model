import logging
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from utils import load_model_from_azure
# Importing all api routers

from API.brain_xray import router as brain_router
from API.lung_xray import router as lung_xray_router
from API.lung_tissues import router as lung_tissues_router
from API.Kidney import router as kidney_router
from API.knee import router as Knee_router
from API.Diabetic_Retinopathy import router as Diabetic_Retinopathy_router

# Configure environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Disable tensorflow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize FastAPI app
app = FastAPI(title="Medical Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
   
@app.on_event("startup")
async def startup_event():
    """Initialize all models at startup"""
    try:
        logging.info("Loading all models at startup...")

        # Load Brain Tumor model from Azure
        app.state.brain_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/resnet_brain_model.h5"
        )
        logging.info("Brain Tumor model loaded successfully")

        # Load Lung Cancer model from Azure
        app.state.lung_xray_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/chest_xray.py"
        )
        logging.info("Lung X-Ray model loaded successfully")
        
        # Load Lung Cancer model from Azure
        app.state.lung_tissues_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/lung-cancer-resnet-model.h5"
        )
        logging.info("Lung tissues model loaded successfully")

        # Load Kidney model from Azure
        app.state.kidney_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/resnet50_kidney_ct_augmented.h5"
        )
        logging.info("Kidney model loaded successfully")

        # Load Knee model from Azure
        app.state.knee_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/Knee_Osteoporosis.h5"
        )
        logging.info("Knee model loaded successfully")

        # Load Diabetic Retinopathy model from Azure
        app.state.Diabetic_Retinopathy_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/Diabetic-Retinopathy-ResNet50-model.h5"
        )
        logging.info("Diabetic-Retinopathy model loaded successfully")

    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "Medical Scan Detection API is running",
    }

# Include the API routers with specific paths
app.include_router(brain_router, prefix="/brain-XRays", tags=["Brain Detection"])
app.include_router(lung_xray_router, prefix="/Lung-XRays", tags=["Lung XRays Detection"])
app.include_router(lung_tissues_router, prefix="/Lung-tissues", tags=["Lung tissues Detection"])
app.include_router(kidney_router, prefix="/kidney", tags=["kidney Detection"])
app.include_router(Knee_router, prefix="/Knee", tags=[" Knee Detection"])
app.include_router(Diabetic_Retinopathy_router, prefix="/Diabetic-Retinopathy", tags=["Diabetic Retinopathy Detection"])
