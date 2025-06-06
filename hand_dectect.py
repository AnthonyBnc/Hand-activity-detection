import cv2
import numpy as np

# Define skin color range in HSV
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image to avoid mirror view
    frame = cv2.flip(frame, 1)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask where skin color areas are white
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Noise reduction
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the largest contour (assume it's the hand)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

    # Show results
    cv2.imshow("Hand Detection", frame)
    cv2.imshow("Skin Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
