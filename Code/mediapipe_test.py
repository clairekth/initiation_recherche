import mediapipe as mp
import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

mp_hands = mp.solutions.hands.Hands()

while True :
    success, img = cap.read()
    cv2.imshow("Result", img)
    results = mp_hands.process(img)
    if results.multi_hand_landmarks:
        circle = [0]*21
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                circle[id] = (cx, cy)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                cv2.putText(img,str(id),(cx,cy),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,0),1)

            for i in range(0,4):
                cv2.line(img, circle[i], circle[i+1], (255, 0, 255), 3)


            cv2.line(img, circle[0], circle[5], (255, 0, 255), 3)
            for i in range(5,8):
                cv2.line(img, circle[i], circle[i+1], (255, 0, 255), 3)
            
            cv2.line(img, circle[0], circle[9], (255, 0, 255), 3)
            for i in range(9,12):
                cv2.line(img, circle[i], circle[i+1], (255, 0, 255), 3)
            
            cv2.line(img, circle[0], circle[13], (255, 0, 255), 3)
            for i in range(13,16):
                cv2.line(img, circle[i], circle[i+1], (255, 0, 255), 3)

            cv2.line(img, circle[0], circle[17], (255, 0, 255), 3)
            for i in range(17,20):
                cv2.line(img, circle[i], circle[i+1], (255, 0, 255), 3)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


