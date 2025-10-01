import face_recognition
import cv2
import os
import time

# ---------------------------
# CONFIG
# ---------------------------
KNOWN_FACES_DIR = "faces"  # folder with known people images
TOLERANCE = 0.45           # lower = stricter, higher = looser
FRAME_RESIZE = 0.25        # scale factor for faster recognition
SHOW_FPS = True            # show FPS on screen
WINDOW_NAME = "Face Recognition"

# ---------------------------
# LOAD KNOWN FACES
# ---------------------------
print("🔍 Loading known faces...")

known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(path)

    face_locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_locations)

    if encodings:
        known_encodings.append(encodings[0])
        name = os.path.splitext(filename)[0]  # filename without extension
        known_names.append(name)
        print(f"✅ Loaded {name}")
    else:
        print(f"⚠️ No face found in {filename}, skipping...")

print(f"✅ Finished loading {len(known_names)} known faces.")

# ---------------------------
# START WEBCAM
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

print("🎥 Starting webcam. Press 'q' to quit.")

process_this_frame = True
face_locations, face_names = [], []
fps = 0
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB).copy()

    if process_this_frame:
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
            name = "Unknown"
            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Draw results (scale back coordinates)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        scale = int(1 / FRAME_RESIZE)
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        # Draw rectangle + label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # FPS counter
    if SHOW_FPS:
        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show video window
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
