import cv2
import math
from utils import stackImages
import random
import numpy as np

# Blank image
imgBlank = np.zeros((640, 480), np.uint8)

# Load the picture
img = cv2.imread("../database/hands/hand_8.png")
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


# Mask that matches the tips of the fingers (square blended with a circle)
def createMask(scale):
    """
    Create a mask that matches the tips of the fingers (square blended with a circle)
    The mask has a certain size and the shapes are centered
    """
    width = int(50 * scale)
    height = int(50 * scale)
    mask = np.zeros((width, height), np.uint8)

    rec_width = int(width / 2)
    rec_height = int(height / 2)
    rec_x = int(width / 2 - rec_width / 2)
    rec_y = int(height - rec_height)
    mask[rec_y : rec_y + rec_height, rec_x : rec_x + rec_width] = 255
    cv2.circle(mask, (rec_x + int(rec_width / 2), rec_y), int(rec_width / 2), 255, -1)

    # mask = ligne
    # mask = np.zeros((50, 50), np.uint8)

    # mask[23:27, 16:32] = 255
    
    return mask


def detectFingers(imageThres, image):
    """
    Move the mask on the image to detect the fingers

    Parameters:
        - image: the image thresholded
        - mask: the mask to apply to the image to detect the fingers
    """
    imgCopy = imageThres.copy()
    imgRes = image.copy()
    width = imageThres.shape[1]
    height = imageThres.shape[0]

    mask = createMask(1.25)

    mask_width = mask.shape[1]
    mask_height = mask.shape[0]

    cmpt = 0
    imgBlank = np.ones((20, 200), np.uint8)
    rectangles = []
    for i in range(0, width - mask_width, mask_width//2):
        for j in range(0, height - mask_height, mask_height//2 ):
            pif = imgCopy[j : j + mask_height, i : i + mask_width]
            imgTest = imgCopy.copy()
            cv2.rectangle(imgTest, (i, j), (i + mask_width, j + mask_height), 255, cv2.FILLED)
            # stack = stackImages(2, ([pif], [imgBlank], [imgTest]))
            # cv2.imshow("Result", stack)
            # cv2.waitKey(0)
            
            roi = pif.copy()
            res = cv2.bitwise_and(mask, roi)
            if (cv2.countNonZero(res) / (mask_width * mask_height)) > 0.2:
                # cv2.imshow("Result", res)
                # cv2.waitKey(0)
                # cv2.rectangle(imgRes, (i, j), (i + mask_width, j + mask_height), (255,0,0), 2)
                rectangles.append((i, j, mask_width, mask_height))
                break
    # Check les rectangles convexes si connexe on garde celui qui est le plus haut
    # for i in range(len(rectangles)):
    #     x, y, w, h = rectangles[i]
    #     for j in range(i + 1, len(rectangles)):
    #         x2, y2, w2, h2 = rectangles[j]
    #         # check si connexe
    #         # swap si y2 < y
    #         swap = False
    #         if y2 < y:
    #             swap = True
    #             x, y, w, h = rectangles[j]
    #             x2, y2, w2, h2 = rectangles[i]
    #         if x2 - (x + w) < 10 and y2 - (y + h) < 10:
    #             if swap:
    #                 rectangles[i] = (0, 0, 0, 0)
    #             else:
    #                 rectangles[j] = (0, 0, 0, 0)
    #         elif (x2 + w2) - x < 10 and (y2 + h2) - y < 10:
    #             if swap:
    #                 rectangles[i] = (0, 0, 0, 0)
    #             else:   
    #                 rectangles[j] = (0, 0, 0, 0)
            
                
    
    for x, y, w, h in rectangles:
        cv2.rectangle(imgRes, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return imgRes


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
            hull = cv2.convexHull(cnt)
            cv2.drawContours(imgCopy, [hull], -1, col[i], 3)
            i += 1
            if i == 255:
                i = 0

            # # Defects
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
            #         angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

            #         if angle <= 90:
            #             cv2.circle(imgCopy, far, 5, [0, 0, 255], -1)

    imgCopy = img.copy()
    img_finger = detectFingers(mask, imgCopy)
    imgStack = stackImages(
        0.6, ([img, imgHSV, imgGaussian], [mask, createMask(1), img_finger])
    )

    cv2.imshow("Result", imgStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
