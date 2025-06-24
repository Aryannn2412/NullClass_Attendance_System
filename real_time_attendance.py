import cv2
import numpy as np
import pandas as pd
import pickle
import os
import time as systime
from datetime import datetime
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.preprocessing.image import img_to_array

# === BASE DIRECTORY ===
BASE_DIR = "/Users/aryanjha/Desktop/NullClass_Internship/Task_2_Attendance_System"

# === Load FaceNet & MTCNN ===
embedder = FaceNet()
detector = MTCNN()

# === Load SVM Classifier & Label Encoder ===
with open(os.path.join(BASE_DIR, 'face_classifier.pkl'), 'rb') as f:
    classifier = pickle.load(f)

with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

# === Load Emotion Detection Model ===
from tensorflow.keras import Model
get_custom_objects().update({'Functional': Model})

with open(os.path.join(BASE_DIR, 'model', 'model_a1.json'), 'r') as f:
    model_json = f.read()

emotion_model = model_from_json(model_json)
emotion_model.load_weights(os.path.join(BASE_DIR, 'model', 'model_weights1.h5'))

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# === Emotion Prediction Function ===
def predict_emotion(face_gray):
    face_resized = cv2.resize(face_gray, (48, 48))
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = img_to_array(face_resized)
    face_resized = np.expand_dims(face_resized, axis=0)
    preds = emotion_model.predict(face_resized)[0]
    return emotion_labels[np.argmax(preds)]

# === CSV Setup ===
csv_path = os.path.join(BASE_DIR, "output", "attendance.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
if not os.path.exists(csv_path):
    pd.DataFrame(columns=["Name", "Emotion", "Timestamp"]).to_csv(csv_path, index=False)

# === Start Webcam with Timeout ===
cap = cv2.VideoCapture(0)
marked = set()
start_time = systime.time()
timeout = 10  # seconds

print("🎥 Attendance system running... (auto-exits in 10s or press ESC)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Auto timeout check
    if systime.time() - start_time > timeout:
        print("⏳ Timeout reached. Exiting...")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_faces(frame_rgb)

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(x, 0), max(y, 0)
        face_rgb = frame_rgb[y:y+h, x:x+w]
        face_gray = frame_gray[y:y+h, x:x+w]

        try:
            face_rgb_resized = cv2.resize(face_rgb, (160, 160))
            emb = embedder.embeddings([face_rgb_resized])[0]
            pred_id = classifier.predict([emb])[0]
            name = label_encoder.inverse_transform([pred_id])[0]
            emotion = predict_emotion(face_gray)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # TEMPORARILY disabled time window for testing
            if name not in marked:
                marked.add(name)
                new_row = pd.DataFrame([[name, emotion, timestamp]],
                                       columns=["Name", "Emotion", "Timestamp"])
                
                # Write and flush to CSV
                with open(csv_path, mode='a', newline='') as f:
                    new_row.to_csv(f, header=False, index=False)
                    f.flush()

                print(f"✅ Logged: {name} | {emotion} | {timestamp}")
                print(new_row)

            # Display on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}, {emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        except Exception as e:
            print(f"⚠️ Error: {e}")

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
