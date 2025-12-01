import cv2
from multiprocessing import Pool, Queue, Process

# Function to perform motion detection on a single frame
def detect_motion_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_subtractor = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=False)
    mask = bg_subtractor.apply(frame)

    mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)[1]
    # mask = cv.erode(mask, None, iterations=2)
    # mask = cv.dilate(mask, None, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Motion Detection", frame)

    motion_detected = True
    return motion_detected


def video_capture_process(frame_queue):
    # Capture from webcam (index 0) or video file
    cap = cv2.VideoCapture('C:\\Users\\Zhanna\\Downloads\\motion-detection2.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Put raw frame (numpy array) into the queue
        frame_queue.put(frame)
    cap.release()


def processing_pool_manager(frame_queue, result_queue):
    # Use a Pool to process frames in parallel
    with Pool(processes=4) as pool:  # Adjust number of processes as needed
        while True:
            frame = frame_queue.get()
            if frame is None:  # Signal to stop
                break
            # Apply motion detection function in parallel
            result = pool.apply_async(detect_motion_frame, (frame,))
            result_queue.put(result.get())  # Get result synchronously here or handle asynchronously

def main():
    frame_queue = Queue()
    result_queue = Queue()

    p_capture = Process(target=video_capture_process, args=(frame_queue,))
    p_manager = Process(target=processing_pool_manager, args=(frame_queue, result_queue))

    p_capture.start()
    p_manager.start()

    while True:
        if not result_queue.empty():
            motion = result_queue.get()
            if motion:
                print("Motion Detected!")
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    p_capture.join()
    p_manager.join()

if __name__ == "__main__":
    main()