import face_recognition
import cv2
import os

# Directory containing known faces
KNOWN_FACES_DIR = "faces"

known_encodings = []
known_names = []

print("🔍 Loading known faces...")

# Load and encode all known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    path = os.path.join(KNOWN_FACES_DIR, filename)

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Load image
    image = face_recognition.load_image_file(path)

    # Detect face(s)
    face_locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_locations)

    if encodings:
        known_encodings.append(encodings[0])
        name = os.path.splitext(filename)[0]  # filename without extension
        known_names.append(name)
        print(f"✅ Loaded {name}")
    else:
        print(f"⚠️ No face found in {filename}, skipping...")

print("✅ All known faces loaded.")

# Start webcam
cap = cv2.VideoCapture(0)

print("🎥 Starting webcam. Press 'q' to quit.")

process_this_frame = True
face_locations = []
face_names = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for speed (1/4 size)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR -> RGB and make copy to avoid dlib TypeError
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB).copy()

    if process_this_frame:
        # Detect faces & encode them
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)

            name = "Unknown"
            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Draw results (scale back up to original frame size)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
