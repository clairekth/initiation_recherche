import cv2
import math
from utils import stackImages
import random
import numpy as np

# Blank image
imgBlank = np.zeros((640, 480), np.uint8)

# Load the picture
img = cv2.imread("database/hand_19.png")
img = cv2.resize(img, (640, 480))

# Colors
col = []
for i in range(0, 255):
    col.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 480)
# cv2.createTrackbar("Gaussian ksiseX", "Parameters", 7, 10, lambda x: x)
# cv2.createTrackbar("Gaussian ksiseY", "Parameters", 5, 10, lambda x: x)
# cv2.createTrackbar("Sigma X", "Parameters", 4, 10, lambda x: x)

cv2.createTrackbar("Bilateral d", "Parameters", 9, 10, lambda x: x)
cv2.createTrackbar("Bilateral sigmaColor", "Parameters", 75, 100, lambda x: x)
cv2.createTrackbar("Bilateral sigmaSpace", "Parameters", 75, 100, lambda x: x)

cv2.createTrackbar("Thresh", "Parameters", 223, 255, lambda x: x)
cv2.createTrackbar("Maxval", "Parameters", 255, 255, lambda x: x)

cv2.createTrackbar("Threshold1", "Parameters", 0, 255, lambda x: x)
cv2.createTrackbar("Threshold2", "Parameters", 0, 255, lambda x: x)

while True:
    # Get the parameters
    # gaussian_ksizeX = cv2.getTrackbarPos("Gaussian ksiseX", "Parameters")
    # gaussian_ksizeY = cv2.getTrackbarPos("Gaussian ksiseY", "Parameters")
    # sigmaX = cv2.getTrackbarPos("Sigma X", "Parameters")

    bilateral_d = cv2.getTrackbarPos("Bilateral d", "Parameters")
    bilateral_sigmaColor = cv2.getTrackbarPos("Bilateral sigmaColor", "Parameters")
    bilateral_sigmaSpace = cv2.getTrackbarPos("Bilateral sigmaSpace", "Parameters")

    thresh = cv2.getTrackbarPos("Thresh", "Parameters")
    maxval = cv2.getTrackbarPos("Maxval", "Parameters")

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    # Check values of parameters
    # if gaussian_ksizeX % 2 == 0:
    #     gaussian_ksizeX += 1
    # if gaussian_ksizeY % 2 == 0:
    #     gaussian_ksizeY += 1

    imgCopy = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # imgGaussian = cv2.GaussianBlur(imgGray, (gaussian_ksizeX, gaussian_ksizeY), sigmaX)
    imgBilateral = cv2.bilateralFilter(imgGray, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)
    _, imgThreshold = cv2.threshold(
        imgBilateral, thresh, maxval, cv2.THRESH_BINARY
    )
  
    imgCanny = cv2.Canny(imgThreshold, threshold1, threshold2)

    # Contour
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(imgCopy, [hull], -1, col[i], 2)
            i+=1
            if i == 255:
                i = 0
         

    imgStack = stackImages(
        0.6, ([img, imgGray, imgBilateral], [imgThreshold, imgCanny, imgCopy])
    )

    cv2.imshow("Result", imgStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
