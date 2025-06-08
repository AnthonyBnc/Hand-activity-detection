import cv2
import numpy as np
from collections import deque

def count_defects(contour, hull):
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0
    return defects.shape[0]

# Motion history to detect waving
motion_history = deque(maxlen=10)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hand_state = "No hand detected"

    # Convert to HSV color space for skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range (adjust as needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create skin mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours on the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area > 3000:
            # Get bounding box for red box
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Detect gesture based on convexity
            hull = cv2.convexHull(max_contour, returnPoints=False)
            defects_count = count_defects(max_contour, hull)

            # Track center point of hand
            M = cv2.moments(max_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                motion_history.append(cx)

                if len(motion_history) == motion_history.maxlen:
                    movement = max(motion_history) - min(motion_history)
                    if movement > 50:
                        hand_state = "👋 Waving"
                    elif defects_count >= 4:
                        hand_state = "🖐️ Open Hand"
                    elif defects_count <= 1:
                        hand_state = "✊ Fist"
                    else:
                        hand_state = "❓ Unknown Gesture"

            # Draw hand contour
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

    # Display the detected hand state
    cv2.putText(frame, hand_state, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the result (just the main view)
    cv2.imshow("Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
