import cv2
import math
from utils import stackImages

# Load the picture
img = cv2.imread("database/hand_19.png")
img = cv2.resize(img, (640, 480))

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 480)
cv2.createTrackbar("Gaussian ksiseX", "Parameters", 1, 10, lambda x: x)
cv2.createTrackbar("Gaussian ksiseY", "Parameters", 1, 10, lambda x: x)
cv2.createTrackbar("Sigma X", "Parameters", 0, 10, lambda x: x)
cv2.createTrackbar("thresh", "Parameters", 0, 255, lambda x: x)
cv2.createTrackbar("maxval", "Parameters", 0, 255, lambda x: x)
cv2.createTrackbar("threshold1", "Parameters", 0, 255, lambda x: x)
cv2.createTrackbar("threshold2", "Parameters", 0, 255, lambda x: x)

while True:
    # Get the parameters
    gaussian_ksizeX = cv2.getTrackbarPos("Gaussian ksiseX", "Parameters")
    gaussian_ksizeY = cv2.getTrackbarPos("Gaussian ksiseY", "Parameters")
    sigmaX = cv2.getTrackbarPos("Sigma X", "Parameters")

    thresh = cv2.getTrackbarPos("Thresh", "Parameters")
    maxval = cv2.getTrackbarPos("Maxval", "Parameters")

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    # Check values of parameters
    if gaussian_ksizeX % 2 == 0:
        gaussian_ksizeX += 1
    if gaussian_ksizeY % 2 == 0:
        gaussian_ksizeY += 1

    imgCopy = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGaussian = cv2.GaussianBlur(imgGray, (gaussian_ksizeX, gaussian_ksizeY), sigmaX)
    imgContour, imgThreshold = cv2.threshold(imgGaussian, thresh, maxval, cv2.THRESH_BINARY)
    imgCanny = cv2.Canny(imgThreshold, threshold1, threshold2)

    # Contour
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(imgCopy, [hull], -1, (0, 255, 0), 2)

    imgStack = stackImages(0.6, ([img, imgGray, imgGaussian], [imgThreshold, imgCanny, imgCopy]))

    cv2.imshow("Result", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
