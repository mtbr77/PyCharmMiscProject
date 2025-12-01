import cv2
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Помилка: Не вдалося відкрити камеру.")
    exit(0)

# Ініціалізація змінної для збереження попереднього кадру
prev_frame = None

print("Камера активна. Натисніть 'q' для виходу.")

while True:
    # Захоплення поточного кадру
    ret, frame = cap.read()

    if not ret:
        print("Помилка: Не вдалося отримати кадр.")
        break

    # Конвертація кадру в чорно-білий (для спрощення обробки)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Застосування розмиття (для зменшення шуму)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Ініціалізація попереднього кадру
    if prev_frame is None:
        prev_frame = gray
        continue

    # 3. Обчислення різниці між поточним і попереднім кадром
    # А. Фонова різниця
    frame_delta = cv2.absdiff(prev_frame, gray)

    cv2.imshow("delta", frame_delta)

    # Б. Бінаризація: поріг встановлюємо, щоб виділити значні зміни
    # Області, де різниця перевищує 30, стають білими (255)
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.MORPH_RECT)

    cv2.imshow("thresh", thresh)

    # В. Розширення (Dilate) для заповнення невеликих пробілів у рухомих об'єктах
    thresh = cv2.dilate(thresh, None, iterations=3)

    cv2.imshow("thresh2", thresh)

    # 4. Пошук контурів (областей руху)
    # Знаходимо контури руху на бінаризованому зображенні
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ініціалізація об'єднаної рамки для всіх знайдених рухомих об'єктів
    motion_rects = []

    # Обробка знайдених контурів
    for contour in contours:
        # Якщо площа контуру замала, ігноруємо його (це може бути шум)
        if cv2.contourArea(contour) < 500:  # Мінімальна площа руху
            continue

        # Отримання координат прямокутної рамки для поточного контуру
        (x, y, w, h) = cv2.boundingRect(contour)
        motion_rects.append((x, y, w, h))

    # 5. Виділення рухомого об'єкта однією рамкою

    # Об'єднання всіх рамок руху в одну загальну рамку
    if motion_rects:
        # Обчислення мінімальної та максимальної координати
        # x_min = min(x for x, _, _, _ in motion_rects)
        # y_min = min(y for _, y, _, _ in motion_rects)
        # x_max = max(x + w for x, _, w, _ in motion_rects)
        # y_max = max(y + h for _, y, _, h in motion_rects)

        # Якщо потрібно лише найбільший об'єкт
        # В даному випадку малюємо рамку навколо найбільшого контуру (для прикладу)

        largest_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour)

        # Малювання червоної рамки навколо знайденого об'єкта
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Movement Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Відображення вихідного кадру
    #cv2.imshow("Video Feed with Motion Tracking", frame)

    # Оновлення попереднього кадру для наступної ітерації
    prev_frame = gray

    # Перевірка на натискання клавіші 'q' для виходу
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Звільнення ресурсів
cap.release()
cv2.destroyAllWindows()