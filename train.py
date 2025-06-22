import os
import pickle
import pandas as pd
import numpy as np
import insightface
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DATA_FILE = "students.json"
MODEL_FILE = "student_model.pkl"

# Load InsightFace model once
face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(224, 224))

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    faces = face_app.get(img)
    if len(faces) == 0:
        print(f"Warning: No face detected in {image_path}. Skipping.")
        return None
    return faces[0].embedding  # 512-d vector

def train_model():
    """Trains the face recognition model and saves it."""
    print("Loading student data...")
    try:
        student_data = pd.read_json(DATA_FILE, orient='records')
    except Exception as e:
        print(f"Error loading {DATA_FILE}: {e}")
        return

    embeddings = []
    labels = []

    print("Extracting face embeddings...")
    for idx, row in student_data.iterrows():
        image_path = row['ImagePath']
        matric_no = row['MatricNo']
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}. Skipping.")
            continue
        emb = get_face_embedding(image_path)
        if emb is not None:
            embeddings.append(emb)
            labels.append(matric_no)

    if not embeddings:
        print("No embeddings found. Training aborted.")
        return

    print(f"Training model on {len(labels)} faces...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    model = SVC(kernel='rbf', probability=True)
    model.fit(embeddings, encoded_labels)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'encoder': label_encoder, 'data': student_data}, f)

    print(f"Training complete. Model saved to '{MODEL_FILE}'.")

if __name__ == '__main__':
    train_model() 