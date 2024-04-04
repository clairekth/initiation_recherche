import cv2
import math
from utils import stackImages
import random
import numpy as np

# Blank image
imgBlank = np.zeros((640, 480), np.uint8)

# Load the picture
img = cv2.imread("database/hands/hand_19.png")
if img is None:
    print("Could not read the image.")
    exit()
img = cv2.resize(img, (640, 480))

# Colors
col = []
for i in range(0, 255):
    col.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 480)
cv2.createTrackbar("Hue Min", "Parameters", 0, 179, lambda x: x)
cv2.createTrackbar("Hue Max", "Parameters", 179, 179, lambda x: x)
cv2.createTrackbar("Sat Min", "Parameters", 70, 255, lambda x: x)
cv2.createTrackbar("Sat Max", "Parameters", 255, 255, lambda x: x)
cv2.createTrackbar("Val Min", "Parameters", 75, 255, lambda x: x)
cv2.createTrackbar("Val Max", "Parameters", 255, 255, lambda x: x)

while True:
    imgCopy = img.copy()

    # Get the parameters
    h_min = cv2.getTrackbarPos("Hue Min", "Parameters")
    h_max = cv2.getTrackbarPos("Hue Max", "Parameters")
    s_min = cv2.getTrackbarPos("Sat Min", "Parameters")
    s_max = cv2.getTrackbarPos("Sat Max", "Parameters")
    v_min = cv2.getTrackbarPos("Val Min", "Parameters")
    v_max = cv2.getTrackbarPos("Val Max", "Parameters")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Convert to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Gaussian blur
    imgGaussian = cv2.GaussianBlur(imgHSV, (7, 7), 1)

    mask = cv2.inRange(imgGaussian, lower, upper)

    # Contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            hull = cv2.convexHull(cnt, returnPoints=False)
            cv2.drawContours(imgCopy, [cnt], -1, col[i], 3)
            i += 1
            if i == 255:
                i = 0

            # Defects
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

                    if angle <= 90:
                        cv2.circle(imgCopy, far, 5, [0, 0, 255], -1)

    imgStack = stackImages(0.6, ([img, imgHSV, imgGaussian], [mask, imgBlank, imgCopy]))

    cv2.imshow("Result", imgStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
