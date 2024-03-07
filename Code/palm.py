import cv2

handCascade = cv2.CascadeClassifier("cascade/lpam_github.xml")
color = (255, 0, 255)
frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hands = handCascade.detectMultiScale(imgGray, 1.1, 10)

    for x, y, w, h in hands:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            img, "Hand", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2
        )

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
