import cv2
import numpy as np
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture('source_video.mp4')# better using the absolute path of file (e.g. C:/Users/Lei/source_video.mp4)
model = load_model('harryTest.h5')# better using the absolute path of file (e.g. C:/Users/Lei/harryTest.h5)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, 150, 200)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    ret, imgBinary = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    imgBinary = cv2.morphologyEx(imgBinary, cv2.MORPH_CLOSE, kernel)


    contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            l = max(w, h)
            img_digit = np.zeros((l, l, 1), np.uint8)
            new_x, new_y = x - (l - w) // 2, y - (l - h) // 2
            img_digit = imgBinary[new_y:new_y + l, new_x:new_x + l]
            kernel = np.ones((5, 5), np.uint8)
            img_digit = cv2.morphologyEx(img_digit, cv2.MORPH_DILATE, kernel)

            img_digit = cv2.resize(img_digit, (28, 28), interpolation=cv2.INTER_AREA)
            img_digit = img_digit / 255.0

            img_input = img_digit.reshape(1, 28, 28,1)
            predictions = model.predict(img_input)
            number = np.argmax(predictions)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, "Number:" + str(number), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
