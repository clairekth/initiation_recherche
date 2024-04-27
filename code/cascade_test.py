import cv2

handCascade = cv2.CascadeClassifier(".\\cascade\\cascade9.xml")
if handCascade.empty():
    raise IOError("Unable to load the hand cascade classifier xml file")
color = (255, 0, 255)
frameWidth = 640
frameHeight = 480

# img = cv2.imread("C:\\Users\\clair\\Documents\\initiation_recherche\\database\\hands\\left_5_fingers.png")
img = cv2.imread("C:\\Users\\clair\\Documents\\initiation_recherche\\database\\hands\\hand_15.png")
if img is None:
    raise IOError("Unable to load the image file")

# resize en pourcentage
scale_percent = 75
img = cv2.resize(img, (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)))


imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hands = handCascade.detectMultiScale(imgGray, scaleFactor=1.2, minNeighbors=1, minSize=(30, 30))
for x, y, w, h in hands:
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        img, "Hand", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2
    )

# x_min = float('inf')
# y_min = float('inf')
# x_max = 0
# y_max = 0

# for x, y, w, h in hands:
#     x_min = min(x_min, x)
#     y_min = min(y_min, y)
#     x_max = max(x_max, x + w)
#     y_max = max(y_max, y + h)

# cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
# cv2.putText(img, "Hand", (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

cv2.imshow("Result", img)
cv2.waitKey(0)

# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10, 150)

# while True:
#     success, img = cap.read()
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     hands = handCascade.detectMultiScale(imgGray, 1.1, 2)
#     print(f"Number of hands found: {hands}")

#     for x, y, w, h in hands:
#         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(
#             img, "Hand", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2
#         )

#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
