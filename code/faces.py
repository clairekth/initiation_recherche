import cv2

faceCascade = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
color = (255,0,255)
frameWidth = 640
frameHeight = 480

img = cv2.imread("../database/faces_detect.jpg")


imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(imgGray, 1.1, 10)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img,"Face",(x,y-5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
    
cv2.imshow("Result", img)
cv2.waitKey(0) 
    