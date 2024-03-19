import cv2
import math 

# Load the picture
img = cv2.imread("picture/hand.jpg")
img = cv2.resize(img, (640, 480))

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

_,thresh=cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

# Contour
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Hull and defects
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img, [hull], -1, (0, 255, 0), 2)  # Draw the convex contour
        
        hull = cv2.convexHull(cnt, returnPoints=False)
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
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                if angle <= 90:
                    cv2.circle(img, far, 5, [0, 0, 255], -1)



# Display the mask
cv2.imshow("Result", img)

# Wait for the user to press a key
cv2.waitKey(0) 
cv2.destroyAllWindows()


# bdd images
# pré ttt avec tests
# haar cascade explication scientifique 