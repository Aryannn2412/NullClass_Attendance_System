import cv2
import os

def capture_faces(student_name, base_dir="Task_2_Attendance_System", total_images=20):
    dataset_dir = os.path.join(base_dir, "dataset", student_name)
    os.makedirs(dataset_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    print(f"📸 Capturing {total_images} images for student: {student_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            img_path = os.path.join(dataset_dir, f"{count+1}.jpg")
            cv2.imwrite(img_path, face)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{count}/{total_images}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) == 27 or count >= total_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Done! Saved images to: {dataset_dir}")

if __name__ == "__main__":
    student_name = input("Enter student name: ")
    capture_faces(student_name)
