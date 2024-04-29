import cv2
import math
from utils import stackImages
import random
import numpy as np

# Blank image
imgBlank = np.zeros((640, 480), np.uint8)

# Load the picture
img = cv2.imread("database/hands/hand_1.png")
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
cv2.createTrackbar("Rate min", "Parameters", 0, 100, lambda x: x)
cv2.createTrackbar("Rate max", "Parameters", 100, 100, lambda x: x)


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

    # cv2.line(mask, (width//3, height//2), (2*width//3, height//2), 255, 4)

    # cv2.circle(mask, (width//2, height//2), int(width/2.5), 255, -1)

    return mask


def countFinger(imageThres, image, scale=1., display=False):
    """
    Count the number of fingers in the image and draw rectangles around them

    Parameters
    ----------
    imageThres : np.ndarray
        Image with the threshold applied
    image : np.ndarray
        Original image
    """
    fingers = []
    mask = createMask(scale)
    finalImage = image.copy()

    mask_height = mask.shape[0]
    mask_width = mask.shape[1]

    image_width = image.shape[1]
    image_height = image.shape[0]

    rate_min = cv2.getTrackbarPos("Rate min", "Parameters") / 100
    rate_max = cv2.getTrackbarPos("Rate max", "Parameters") / 100

    if display:
        cv2.namedWindow("Fingers")

    for x in range(0, image_width - mask_width, mask_width // 2):
        for y in range(0, image_height - mask_height, mask_height // 2):
            imgShow = image.copy()
            roi = imageThres[y : y + mask_height, x : x + mask_width]
            cv2.rectangle(imgShow, (x, y), (x + mask_width, y + mask_height), col[0], 2)
            bitxnor = cv2.bitwise_xor(roi, mask)
            bitxnor = cv2.bitwise_not(bitxnor)
            rate = cv2.countNonZero(bitxnor) / (mask_width * mask_height)
            if rate_min < rate < rate_max:
                cv2.rectangle(
                    finalImage, (x, y), (x + mask_width, y + mask_height), (0,0,255), 2
                )
                fingers.append((x, y))
                break
            print(f"Rate: {rate}")
            if display:
                stack = stackImages(0.5, ([image, roi, bitxnor, imgShow]))
                cv2.imshow("Fingers", stack)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    exit(0)

    return finalImage


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

    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area > 100:
    #         hull = cv2.convexHull(cnt)
    #         cv2.drawContours(imgCopy, [hull], -1, col[0], 3)
    

    imgCopy = img.copy()
    fingers = countFinger(mask, imgCopy)
    imgStack = stackImages(
         0.6, ([img, imgHSV, imgGaussian], [mask, createMask(1), fingers])
    )
    # imgStack = stackImages(
    #     0.6, ([img, imgHSV, imgGaussian], [mask, imgCopy, imgBlank])
    # )
    # imgStack = stackImages(0.9, ([countFinger(mask, imgCopy, 0.6), createMask(1)]))

    cv2.imshow("Result", imgStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
