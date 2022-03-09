import cv2


detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('pictures/me_rotated.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(img_gray, minNeighbors=3)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv2.imshow('Capture - Face detection', img)
cv2.waitKey(0)
#cv2.imwrite('pictures/res.jpg', img)
