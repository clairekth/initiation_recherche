import cv2
import numpy as np
from utils import stackImages
import math

def cluster(list):
    """
    From a list of points (x, y), returns a certain number of clusters
    """
    # clusters = np.array(list[0])
    # clusters = {}
    # clusters.append(list[0])
    x, y = list[0][0]
    clusters = []
    clusters.append((x,y))
    epsilon = 80
    for i in range(1, len(list)):
        x, y = list[i][0]

        is_new_cluster = True

        for point in clusters:
            print(point)
            x_tmp, y_tmp = point[0], point[1]
            x_min = x_tmp - epsilon
            x_max = x_tmp + epsilon
            y_min = y_tmp - epsilon
            y_max = y_tmp + epsilon

            if x_min < x < x_max and y_min < y < y_max:
                print(x," - ", y, "ok ok")
                is_new_cluster = False
                break

        
        if is_new_cluster:
            # clusters = np.append(clusters, [[x, y]], axis=0)
            clusters.append((x,y))
       
    y_tot = 0
    y_max = 0
    x_max = 0
    for x, y in clusters:
        y_tot += y
        if y > y_max :
            y_max = y
        if x > x_max :
            x_max = x
    
    y_moy = y_tot / len(clusters)

    # res = [cluster for cluster in clusters if cluster[1] <=  0.5 * y_moy or cluster[0] == x_max]
    res = [cluster for cluster in clusters if cluster[1] <=  0.5 * y_moy]
    filtered = []

    for i in range(0, len(res)) :
        x, y = res[i]
        flag = True
        for j in range(i+1, len(res)):
            x2, y2 = res[j]
            if abs(x - x2) <= 30 : # proche
                print("x: ", x, "x2: ", x2)
                y_coef = y / y_max
                y2_coef = y2 / y_max
                if y_coef > y2_coef and y_coef > 0.8 :
                    print("1")
                    filtered.append((x,y))
                elif y2_coef > y_coef and y2_coef > 0.8:
                    print("2")
                    filtered.append((x2,y2))
                
                flag = False
        if flag:
            filtered.append((x,y))
            flag = True
    # res = [cluster for cluster in res if all(abs(cluster[0] - other_cluster[0]) >= 30 for other_cluster in res if other_cluster != cluster)]

    print("Clusters:")
    print(res)
    
    return filtered

def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

imgBlank = np.zeros((200, 200), np.uint8)

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


handCascade = cv2.CascadeClassifier("code/cascade/cascade1.xml")
if handCascade.empty():
    raise IOError("Unable to load the hand cascade classifier xml file")

# img = cv2.imread("database/hands/hand_19.png")
img = cv2.imread("database/hands/right_hand_3_fingers.png")
if img is None:
    raise IOError("Unable to load the image file")

# resize en pourcentage
scale_percent = 75
img = cv2.resize(img, (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)))

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


while True:
    imgCopy = img.copy()

    # Get the parameters
    h_min = cv2.getTrackbarPos("Hue Min", "Parameters")
    h_max = cv2.getTrackbarPos("Hue Max", "Parameters")
    s_min = cv2.getTrackbarPos("Sat Min", "Parameters")
    s_max = cv2.getTrackbarPos("Sat Max", "Parameters")
    v_min = cv2.getTrackbarPos("Val Min", "Parameters")
    v_max = cv2.getTrackbarPos("Val Max", "Parameters")

    # hands = handCascade.detectMultiScale(imgGray, scaleFactor=1.2, minNeighbors=1, minSize=(50, 30))
    # color = (255, 255, 255)
    # biggest_scare = -np.inf
    # x_max, y_max, w_max, h_max = 0,0,0,0
    # for x, y, w, h in hands:
    #     if w*h > biggest_scare :
    #         biggest_scare = w*h
    #         x_max, y_max, w_max, h_max = x, y, w, h


    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgGaussian = cv2.GaussianBlur(imgHSV, (7, 7), 1)

    # roi = imgHSV[y_max:y_max+h_max, x_max:x_max+w_max]
    # cv2.rectangle(imgRec, (x_max, y_max), (x_max+w_max, y_max+h_max), color, 2)

    # h,s,v = unique_count_app(roi)
    # epsilon = 45

    # Création d'une image couleur unis HSV
    # imgColor = np.zeros((200, 200, 3), np.uint8)
    # imgColor[:] = [h,s,v]
    # imgColor = cv2.cvtColor(imgColor, cv2.COLOR_HSV2BGR)

    # print(h,s,v)
    # h_min, s_min, v_min = h-epsilon, s-epsilon, v-epsilon
    # h_max, s_max, v_max = h+(2*epsilon), 255, 255

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(imgGaussian, lower, upper)

    # Convex hull + defects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            hull = cv2.convexHull(cnt)

            cv2.drawContours(imgCopy, [hull], -1, (0, 255, 0), 2)
            imgPt = imgCopy.copy()


            # print(hull)

            unique_points = cluster(hull)
            print("Unique points:")
            print(unique_points)
            for x, y in unique_points:
                cv2.circle(imgPt, (x,y), 5, (255,0,255), -1)

            # unique_points = cv2.convexHull(cnt, returnPoints=True)
            # for point in unique_points:
            #     x, y = point[0]
            #     cv2.circle(imgPt, (x,y), 5, (255,0,255), -1)


    stack = stackImages(0.8, ([img, imgHSV, mask], [imgCopy, imgPt, imgBlank]))

    cv2.imshow("Result.jpg", stack)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

