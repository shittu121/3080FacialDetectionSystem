import cv2
import os
import pickle
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# --- Configuration ---
DATA_FILE = "students.json"
MODEL_FILE = "student_model.pkl"
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
IMAGE_SIZE = (100, 100)

def extract_face(image_path):
    """Detects a face in an image, resizes it, and returns it in grayscale."""
    try:
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('L')
        else:
            img = Image.open(image_path).convert('L') # Open in grayscale
        
        img_np = np.array(img, 'uint8')
        
        faces = FACE_CASCADE.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            print(f"Warning: No face detected in {image_path}. Skipping.")
            return None
        
        # Use the first detected face
        x, y, w, h = faces[0]
        face_roi = img_np[y:y+h, x:x+w]
        
        # Resize and return flattened face
        return cv2.resize(face_roi, IMAGE_SIZE).flatten()
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def train_model():
    """Trains the face recognition model and saves it."""
    print("Loading student data...")
    try:
        # Read data from the JSON file, orient='records' assumes a list of dicts.
        student_data = pd.read_json(DATA_FILE, orient='records')
    except FileNotFoundError:
        print(f"Error: The data file '{DATA_FILE}' was not found.")
        print("Please create it and add your student details in JSON format.")
        return
    except ValueError:
        print(f"Error: Could not parse '{DATA_FILE}'. Please ensure it is a valid JSON file (a list of student records).")
        return

    face_features = []
    labels = []

    print("Extracting faces from images...")
    for index, row in student_data.iterrows():
        image_path = row['ImagePath']
        matric_no = row['MatricNo']
        
        if not os.path.exists(image_path):
            # Also check if it's a URL for remote training data
            if not image_path.startswith(('http://', 'https://')):
                print(f"Warning: Image not found at {image_path}. Skipping.")
                continue
            
        features = extract_face(image_path)
        if features is not None:
            face_features.append(features)
            labels.append(matric_no)

    if not face_features:
        print("\nTraining failed: No faces were detected in any of the provided images.")
        return

    print(f"\nTraining model on {len(labels)} detected faces...")
    
    # Encode MatricNo labels to numbers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Train the SVM classifier
    model = SVC(kernel='rbf', probability=True)
    model.fit(face_features, encoded_labels)

    # Save the trained model and the label encoder
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'encoder': label_encoder, 'data': student_data}, f)

    print(f"Training complete. Model saved to '{MODEL_FILE}'.")


if __name__ == '__main__':
    train_model() 