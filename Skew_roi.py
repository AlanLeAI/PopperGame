import cv2
import numpy as np

pts_src = []
def click_event(event, x, y, flags, param):
    global pts_src

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts_src) < 4:
            pts_src.append((x, y))

cap = cv2.VideoCapture(0)  
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", click_event)

width, height = 800, 600

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()  

    
    for pt in pts_src:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    if len(pts_src) == 4:
        pts_dst = np.float32([
            (0, 0),
            (width, 0),
            (width, height),
            (0, height)
        ])

        M = cv2.getPerspectiveTransform(np.float32(pts_src), pts_dst)
        warped_roi = cv2.warpPerspective(original_frame, M, (width, height))
        warped_roi = cv2.rotate(warped_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
        warped_roi = cv2.flip(warped_roi, 0)

        cv2.imshow("Warped ROI (Straightened View)", warped_roi)

    cv2.imshow("Select ROI", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        pts_src.clear()
        cv2.destroyWindow("Warped ROI (Straightened View)")

cap.release()
cv2.destroyAllWindows()
