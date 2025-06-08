import cv2
import numpy as np
from collections import deque

def count_defects(contour, hull):
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0
    return defects.shape[0]

# Track hand center for waving detection
motion_history = deque(maxlen=10)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip to mirror view
    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]  # Region of interest

    # Convert to grayscale and blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)

    # Threshold image
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(),
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(max_contour) > 3000:
            hull = cv2.convexHull(max_contour, returnPoints=False)
            defects_count = count_defects(max_contour, hull)

            # Draw contour and hull
            cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

            # Get center of contour for motion tracking
            M = cv2.moments(max_contour)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                motion_history.append(cx)

                # Draw center
                cv2.circle(roi, (cx, 200), 5, (255, 0, 0), -1)

                # Check for waving gesture (motion side-to-side)
                if len(motion_history) == motion_history.maxlen:
                    movement = max(motion_history) - min(motion_history)
                    if movement > 50:
                        cv2.putText(frame, "👋 Waving", (50, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Classify gesture
            if defects_count >= 4:
                cv2.putText(frame, "🖐️ Open Hand", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            elif defects_count <= 1:
                cv2.putText(frame, "✊ Fist", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                cv2.putText(frame, "✋ Unknown Gesture", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # Draw region of interest box
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)

    # Show output
    cv2.imshow("Gesture Detection", frame)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
