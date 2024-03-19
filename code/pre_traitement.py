import cv2
import math 

# Load the picture
img = cv2.imread("picture/hand.jpg")
img = cv2.resize(img, (640, 480))

imgCopy = img.copy()

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgGaussien = cv2.GaussianBlur(gray, (5,5), 0)

_,thresh=cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

imgCanny = cv2.Canny(thresh, 100, 200)

# Contour
contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours :
    area = cv2.contourArea(cnt)
    if area > 1000:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(imgCopy, [hull], -1, (0, 255, 0), 2) 

