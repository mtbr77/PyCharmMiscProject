import cv2 as cv
import numpy as np
from multiprocessing import Pool

def is_intersecting(rect1, rect2):
    x_overlap = (rect1[0] < rect2[2]) and (rect1[2] > rect2[0])
    y_overlap = (rect1[1] < rect2[3]) and (rect1[3] > rect2[1])
    return x_overlap and y_overlap

def find_intersections(rectangles):
    intersections = []
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            if is_intersecting(rectangles[i], rectangles[j]):
                intersections.append((rectangles[i], rectangles[j]))
    return intersections

cap = cv.VideoCapture('C:\\Users\\Zhanna\\Downloads\\motion-detection2.mp4')

if not cap.isOpened():
    print("cant get video from camera")
    exit(0)

back_sub = cv.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = back_sub.apply(frame)

    count = np.count_nonzero(mask)

    if count < 500:
        continue

    #mask = cv.threshold(mask, 50, 255, cv.THRESH_BINARY)[1]
    #mask = cv.erode(mask, None, iterations=2)
    #mask = cv.dilate(mask, None, iterations=3)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Motion Detection", frame)
    #cv.imshow("Mask", mask)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()