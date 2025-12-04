import cv2

# Initialize the KNN background subtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN(history=2, dist2Threshold=700.0, detectShadows=False)

min_detected_area = 5000

def detect_significant_contours_of_motion(frame, threshold_area=400):
    mask = bg_subtractor.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=4)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > threshold_area:
            return True, mask, contours

    return False, mask, contours

if __name__ == '__main__':
    cap = cv2.VideoCapture('pool.mp4')

    default_color = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        motion_detected, mask, contours = detect_significant_contours_of_motion(frame, min_detected_area)

        if motion_detected:
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area >= min_detected_area:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    center = ((x + w / 2), (y + h / 2))
                    color = default_color
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
        cv2.imshow("Original Framw", frame)
        mask = cv2.resize(mask, None, fx=0.4, fy=0.4)
        cv2.imshow("Foreground Mask", mask)

        if cv2.waitKey(300) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
