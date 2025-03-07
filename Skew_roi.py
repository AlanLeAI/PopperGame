import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture(1)  # Use 0 for webcam

# Define the four corner points of the skewed ROI in the original frame
pts_src = np.float32([
    (100, 200),  # Top-left
    (400, 180),  # Top-right
    (450, 350),  # Bottom-right
    (120, 380)   # Bottom-left
])

# Define where these points should be mapped in the output ROI (rectangular)
width, height = 300, 300  # Define desired output size
pts_dst = np.float32([
    (0, 0),
    (width, 0),
    (width, height),
    (0, height)
])

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(pts_src, pts_dst)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the skewed ROI on the original frame
    cv2.polylines(frame, [np.int32(pts_src)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Warp perspective to get a straightened ROI
    warped_roi = cv2.warpPerspective(frame, M, (width, height))

    # Display the original frame with the drawn ROI
    cv2.imshow("Video with Skewed ROI", frame)

    # Display the warped ROI
    cv2.imshow("Warped ROI (Straightened View)", warped_roi)

    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
