import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не вдалося отримати доступ до камери")
    exit(0)

# Фонова модель (виявлення руху)
back_sub = cv2.createBackgroundSubtractorMOG2(history=1,
                                              varThreshold=20,
                                              detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Маска руху
    mask = back_sub.apply(frame)

    # Очищення шумів
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)

    # Пошук контурів (рухомих об'єктів)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ігноруємо надто маленькі області
        if area < 2000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Виділяємо рамкою рухомий об'єкт
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

    # Показ результатів
    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Mask", mask)

    # Вихід по "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()