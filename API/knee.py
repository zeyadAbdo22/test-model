from fastapi import FastAPI, File, UploadFile, APIRouter, Request, HTTPException
import numpy as np
import json
import tensorflow as tf
from PIL import Image
from io import BytesIO
import logging
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from similarity import check_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Labels for knee model
labels_map = {
    "knee": ["Healthy", "Osteoporosis"]
}

def preprocess_image(img, model_type: str):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if model_type == "knee":
        return mobilenet_preprocess_input(img_array)
    else:
        raise ValueError("Invalid model type for preprocessing")

@router.post("/predict")
async def predict_knee(request: Request, file: UploadFile = File(...)):
    try:
        logger.info(f"Received knee image: {file.filename}")
        contents = await file.read()

        # Parse similarity check response
        similar = check_similarity(contents)
        similarity_data = similar.body.decode()  # Convert bytes to string
        similarity_result = json.loads(similarity_data)  # Parse JSON string

        if similarity_result.get("prediction") == "not-medical":
            raise HTTPException(
                status_code=400,
                detail="File is not a valid medical image"
            )

        # Add logging for debugging
        logger.info(f"Similarity check result: {similarity_data}")

        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img, model_type="knee")

        knee_model = request.app.state.knee_model

        if knee_model is None:
            raise ValueError("Knee model is not loaded")

        prediction = knee_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        return {
            "success": True,
            "filename": file.filename,
            "prediction": labels_map["knee"][predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Knee prediction error: {str(e)}")
        return {"success": False, "error": str(e)}