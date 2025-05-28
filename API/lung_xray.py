import logging
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image
from io import BytesIO
import json
import numpy as np
from utils import preprocess_image
from similarity import check_similarity
from fastapi.responses import JSONResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define class labels
LUNG_CLASSES = {0: "Covid", 1: "Normal", 2: "Pneumonia", 3: "Tuberculosis"}


@router.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Lung Cancer Detection API is running"}


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        logger.info(f"Receiving prediction request for file: {file.filename}")

        # Get model from app state
        model = request.app.state.lung_tissues_model
        if model is None:
            raise ValueError("Lung tissues model not initialized")

        # Check content type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image bytes
        contents = await file.read()

        # Run similarity check
        similarity_result = check_similarity(contents)
        similarity_data = similarity_result.body.decode()
        similarity_json = json.loads(similarity_data)

        if similarity_json.get("prediction") != "medical":
            raise HTTPException(status_code=400, detail="File is not a valid medical scan")

        logger.info(f"Similarity check passed: {similarity_json}")

        # Preprocess image
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img)

        # Predict
        prediction = model.predict(img_array, verbose=0)
        logger.info(f"Raw model prediction: {prediction}")

        # Decode prediction
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class])
        result_label = LUNG_CLASSES.get(predicted_class, "Unknown")

        logger.info(f"Prediction: {result_label}, Confidence: {confidence:.4f}")

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result_label,
            "confidence": confidence,
        }
        


    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
