import face_recognition
import cv2
import os

known_encodings = []
known_names = []

for file in os.listdir('faces'):
    path = os.path.join("faces", file)
    image = face_recognition.load_image_file(path)
    encs = face_recognition.face_encodings(image)
    if encs: 
        known_encodings.append(encs[0])
        known_names.append(os.path.splitext(file)[0])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()
    if not ret:
        break 

    rgb = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        name = "Unknown"
        if True in matches: 
            idx = matches.index(True)
            name = known_names[idx]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()