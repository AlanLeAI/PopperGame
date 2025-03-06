from findingBalloons import *

img = cv2.imread("testimage.png")
# cv2.imshow("imgballon",img)
imgballon  = find_balloons(img)

if imgballon.dtype != np.uint8:
    imgballon = imgballon.astype(np.uint8)

# imgballon = cv2.resize(imgballon, None, fx=0.5, fy=0.5)
cv2.imshow("imgballon",imgballon)
cv2.waitKey(0)


