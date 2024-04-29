import cv2
import numpy as np
from utils import stackImages
import math

def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

handCascade = cv2.CascadeClassifier("code/cascade/cascade1.xml")
if handCascade.empty():
    raise IOError("Unable to load the hand cascade classifier xml file")

img = cv2.imread("database/hands/hand_19.png")
if img is None:
    raise IOError("Unable to load the image file")

# resize en pourcentage
scale_percent = 75
img = cv2.resize(img, (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)))

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hands = handCascade.detectMultiScale(imgGray, scaleFactor=1.2, minNeighbors=1, minSize=(50, 30))
color = (255, 255, 255)
biggest_scare = -np.inf
x_max, y_max, w_max, h_max = 0,0,0,0
for x, y, w, h in hands:
    if w*h > biggest_scare :
        biggest_scare = w*h
        x_max, y_max, w_max, h_max = x, y, w, h

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgGaussian = cv2.GaussianBlur(imgHSV, (7, 7), 1)

roi = imgHSV[y_max:y_max+h_max, x_max:x_max+w_max]

h,s,v = unique_count_app(roi)
epsilon = 45

print(h,s,v)
h_min, s_min, v_min = h-epsilon, s-epsilon, v-epsilon
h_max, s_max, v_max = h+(2*epsilon), 255, 255
lower = np.array([h_min, s_min, v_min])
upper = np.array([h_max, s_max, v_max])

mask = cv2.inRange(imgGaussian, lower, upper)

# Convex hull + defects
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        hull = cv2.convexHull(cnt)

        cv2.drawContours(img, [hull], -1, (0, 255, 0), 2)

        points = hull[:,0]

        print(points)
        # hull = cv2.convexHull(cnt, returnPoints=False)
        # defects = cv2.convexityDefects(cnt, hull)

        # if defects is not None:
        #     for i in range(defects.shape[0]):
        #         s, e, f, d = defects[i, 0]
        #         start = tuple(cnt[s][0])
        #         end = tuple(cnt[e][0])
        #         far = tuple(cnt[f][0])

        #         a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        #         b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        #         c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        #         angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        #         if angle <= 90:
        #             cv2.circle(img, far, 5, [0, 0, 255], -1)


stack = stackImages(0.8, ([img, roi, mask]))

cv2.imwrite("Result.jpg", stack)


cv2.waitKey(0)

