# Task 2 – Attendance System with Emotion Detection

This project is part of the NullClass Internship Program. It implements a real-time attendance system that:
- Detects student faces from a webcam
- Recognizes their identity using FaceNet + SVM
- Detects their current emotion using a pre-trained CNN
- Logs their name, emotion, and timestamp to a CSV file

---

## 🛠️ How to Run

### 1. 📸 Collect Student Face Images

Run the face capture script for each student:

python capture_faces.py
You'll be asked to enter a student name (e.g., Aryan Jha).
The script will capture 20 face images via webcam and save them to:
Task_2_Attendance_System/dataset/Aryan Jha/
Repeat this step for each student in your class.

## 2. 🧠 Train the Face Recognition Model
Inside your training notebook:
Extract FaceNet embeddings using keras-facenet
Train an SVM classifier on those embeddings
Save the model and label encoder as:
face_classifier.pkl
label_encoder.pkl
These are used by the real-time system to recognize faces.

## 3. 😐 Train Emotion Detection Model (or Use Pretrained)
We use the emotion model trained in Task 1:
model_a1.json (architecture)
model_weights1.h5 (weights)
Ensure these are placed inside the model/ folder.

## 4. 🟢 Run the Real-Time Attendance System
python real_time_attendance.py
✅ It will:
Start your webcam
Detect and recognize any known face
Predict emotion
Log name, emotion, and timestamp to output/attendance.csv
⏱️ Webcam will automatically close after 10 seconds.
🧾 Output (attendance.csv)
Example:
Name,Emotion,Timestamp
Aryan Jha,Happy,2025-06-23 09:32:11
Priya Sharma,Sad,2025-06-23 09:34:07
⚙️ Requirements
Install dependencies:
pip install -r requirements.txt

## ✅ Notes
The real_time_attendance.py script is currently set to log attendance even outside 9:30–10:00 AM for testing. You can re-enable time restriction by modifying the condition:

