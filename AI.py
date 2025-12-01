import cv2
import numpy as np
from multiprocessing import Process, Queue

def frame_producer(frame_queue):
    cap = cv2.VideoCapture('C:\\Users\\Zhanna\\Downloads\\motion-detection2.mp4') # Or your video file
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()

def motion_detector_worker(frame_queue, result_queue):
    fgbg = cv2.createBackgroundSubtractorKNN()
    while True:
        frame = frame_queue.get()
        if frame is None: # Sentinel value to stop process
            break

        fgmask = fgbg.apply(frame)
        # Apply morphological operations to clean up the mask
        #fgmask = cv2.erode(fgmask, None, iterations=2)
        #fgmask = cv2.dilate(fgmask, None, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 500: # Filter small contours
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)
                motion_detected_boxes.append((x, y, w, h))

        result_queue.put((frame, motion_detected_boxes))

def result_consumer(result_queue):
    while True:
        data = result_queue.get()
        if data is None: # Sentinel value to stop process
            break
        frame, boxes = data

        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Motion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_queue = Queue(maxsize=10) # Buffer for frames
    result_queue = Queue(maxsize=10) # Buffer for results

    producer_process = Process(target=frame_producer, args=(frame_queue,))
    detector_process1 = Process(target=motion_detector_worker, args=(frame_queue, result_queue))
    detector_process2 = Process(target=motion_detector_worker, args=(frame_queue, result_queue)) # More workers for parallelism
    consumer_process = Process(target=result_consumer, args=(result_queue,))

    producer_process.start()
    detector_process1.start()
    detector_process2.start()
    consumer_process.start()

    # Wait for processes to finish (or handle graceful shutdown)
    producer_process.join()
    frame_queue.put(None) # Signal workers to stop
    frame_queue.put(None) # Signal workers to stop
    detector_process1.join()
    detector_process2.join()
    result_queue.put(None) # Signal consumer to stop
    consumer_process.join()