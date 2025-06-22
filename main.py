import os
import pickle
import pandas as pd
import numpy as np
import insightface
import cv2
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import requests
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI ---
app = FastAPI(
    title="Osun State University Student Recognition API with InsightFace",
    description="An API to recognize students from images and return their profile.",
    version="1.0.0"
)

# --- Configuration ---
MODEL_FILE = "student_model.pkl"

# Load InsightFace model once
face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(224, 224))

# --- Global variable for model data ---
model_data = {}

# --- Helper Functions ---
def load_model():
    """Load the trained model and student data into memory."""
    global model_data
    if not os.path.exists(MODEL_FILE):
        logger.error(f"Model file not found at '{MODEL_FILE}'.")
        raise RuntimeError(f"Model file not found. Please ensure '{MODEL_FILE}' is in the root directory.")
    
    try:
        with open(MODEL_FILE, 'rb') as f:
            model_data = pickle.load(f)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load or parse model file: {e}")
        raise RuntimeError("Model file is corrupt or cannot be read.")

def get_face_embedding_from_bytes(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
    except UnidentifiedImageError:
        return None
    img = np.array(img)
    faces = face_app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding

# --- API Events ---
@app.on_event("startup")
async def startup_event():
    """Load the model when the API starts up."""
    load_model()

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def root():
    """A welcome message to confirm the API is running."""
    return {"message": "Welcome to the Student Recognition API (InsightFace version)."}

@app.post("/predict", tags=["Recognition"])
async def predict(file: UploadFile = File(...), confidence_threshold: float = 0.6):
    """
    Recognize a student from an uploaded image.
    
    - **file**: The image file of the student to recognize.
    - **confidence_threshold**: The minimum confidence required to return a match (0.0 to 1.0).
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    if 'model' not in model_data:
        raise HTTPException(status_code=503, detail="Model is not loaded. The service may be starting up or has encountered an error.")

    # Read image bytes from the uploaded file
    image_bytes = await file.read()
    
    emb = get_face_embedding_from_bytes(image_bytes)

    if emb is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image or no face detected.")

    model = model_data['model']
    probabilities = model.predict_proba([emb])[0]
    confidence = np.max(probabilities)

    if confidence < confidence_threshold:
        return JSONResponse(
            status_code=404,
            content={"detail": "Recognition failed: No student in the database matches the image with sufficient confidence."}
        )

    label_encoder = model_data['encoder']
    student_data_df = model_data['data']
    
    prediction_encoded = model.predict([emb])[0]
    matric_no = label_encoder.inverse_transform([prediction_encoded])[0]
    
    student_profile = student_data_df[student_data_df['MatricNo'] == matric_no].iloc[0].to_dict()

    response_data = {
        "confidence": confidence,
        "student_profile": student_profile
    }
    
    return JSONResponse(status_code=200, content=response_data)

@app.post("/predict-url", tags=["Recognition"])
async def predict_url(image_url: str, confidence_threshold: float = 0.6):
    """
    Recognize a student from an image URL.

    - **image_url**: The public URL of the student image.
    - **confidence_threshold**: Minimum confidence for a match.
    """
    if 'model' not in model_data:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        if not response.headers['Content-Type'].startswith('image/'):
            raise HTTPException(status_code=400, detail="URL does not point to an image.")
        image_bytes = response.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {e}")

    # Re-use the same logic as the file upload endpoint
    emb = get_face_embedding_from_bytes(image_bytes)

    if emb is None:
        raise HTTPException(status_code=400, detail="URL does not point to a valid image or no face detected.")

    model = model_data['model']
    probabilities = model.predict_proba([emb])[0]
    confidence = np.max(probabilities)

    if confidence < confidence_threshold:
        return JSONResponse(
            status_code=404,
            content={"detail": "Recognition failed: No student in the database matches the image with sufficient confidence."}
        )
    
    label_encoder = model_data['encoder']
    student_data_df = model_data['data']
    
    prediction_encoded = model.predict([emb])[0]
    matric_no = label_encoder.inverse_transform([prediction_encoded])[0]
    
    student_profile = student_data_df[student_data_df['MatricNo'] == matric_no].iloc[0].to_dict()

    response_data = {
        "confidence": confidence,
        "student_profile": student_profile
    }
    
    return JSONResponse(status_code=200, content=response_data)


# To run this API locally:
# 1. Make sure student_model.pkl exists. If not, run student_recognition_system.py to train and create it.
# 2. In your terminal, run: uvicorn main:app --reload
# 3. Open your browser to http://127.0.0.1:8000/docs to see the interactive API documentation. 