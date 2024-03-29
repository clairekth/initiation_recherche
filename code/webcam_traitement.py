import cv2
import random
from utils import stackImages
import numpy as np
import time

# Blank image
imgBlank = np.zeros((640, 480), np.uint8)

# Load webcam
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# Colors
col = []
for i in range(0, 255):
    col.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 480)

cv2.createTrackbar("Sigma X", "Parameters", 4, 10, lambda x: x)

cv2.createTrackbar("Thresh", "Parameters", 223, 255, lambda x: x)
cv2.createTrackbar("Maxval", "Parameters", 255, 255, lambda x: x)

cv2.createTrackbar("Threshold1", "Parameters", 0, 255, lambda x: x)
cv2.createTrackbar("Threshold2", "Parameters", 0, 255, lambda x: x)

while True:
    success, img = cap.read()

    imgCopy = img.copy()

    sigmaX = cv2.getTrackbarPos("Sigma X", "Parameters")
    thresh = cv2.getTrackbarPos("Thresh", "Parameters")
    maxval = cv2.getTrackbarPos("Maxval", "Parameters")

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    # Convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Smoothing the image
    blurred = cv2.GaussianBlur(hsv, (5, 5), sigmaX)

    # Binary threshold
    _, threshold = cv2.threshold(blurred, thresh, maxval, cv2.THRESH_BINARY)

    # Canny edge detection
    canny = cv2.Canny(threshold, threshold1, threshold2)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(max_contour)

    cv2.drawContours(imgCopy, [hull], -1, (0, 255, 0), 3)

    # Display the images
    imgStack = stackImages(0.9, ([img, blurred], [threshold, imgCopy]))

    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
