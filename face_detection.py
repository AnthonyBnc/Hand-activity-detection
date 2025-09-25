import face_recognition
import cv2
import os

# Folder that contains your known faces (jpg/png files)
KNOWN_FACES_DIR = "faces"

known_encodings = []
known_names = []

print("🔍 Loading known faces...")

# Load all images from faces/ folder
for filename in os.listdir(KNOWN_FACES_DIR):
    path = os.path.join(KNOWN_FACES_DIR, filename)

    # Only process image files
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Load image
    image = face_recognition.load_image_file(path)

    # Detect face(s) in the image
    face_locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_locations)

    if encodings:
        known_encodings.append(encodings[0])
        # Use filename (without extension) as the label
        name = os.path.splitext(filename)[0]
        known_names.append(name)
        print(f"✅ Loaded {name}")
    else:
        print(f"⚠️ No face found in {filename}, skipping...")

print("✅ All known faces loaded.")

# Start webcam
cap = cv2.VideoCapture(0)

print("🎥 Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (face_recognition uses RGB, OpenCV uses BGR)
    rgb_frame = frame[:, :, ::-1]

    # Find all faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face with known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)

        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Draw box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Label with name
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Face Recognition", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
