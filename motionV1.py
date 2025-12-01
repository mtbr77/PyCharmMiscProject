import cv2
cap = cv2.VideoCapture('C:\\Users\\Zhanna\\Downloads\\motion-detection2.mp4')

if not cap.isOpened():
    print("cant get video")
    exit(0)

prev_frame = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("cant get frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray
        continue

    frame_delta = cv2.absdiff(prev_frame, gray)

    #cv2.imshow("delta", frame_delta)

    thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]

    #cv2.imshow("thresh", thresh)

    thresh = cv2.dilate(thresh, None, iterations=2)

    #cv2.imshow("thresh2", thresh)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_rects = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        motion_rects.append((x, y, w, h))

        #largest_contour = max(contours, key=cv2.contourArea)
        #(x, y, w, h) = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Motion", frame)

    prev_frame = gray

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()