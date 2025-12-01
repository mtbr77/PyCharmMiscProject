import cv2
from multiprocessing import Pool


def detect_motion_in_frame(frame_data):
    # Unpack frame and previous_frame from the tuple
    current_frame_gray, previous_frame_gray = frame_data

    # Calculate absolute difference between current and previous frame
    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

    # Apply a threshold to highlight significant changes
    _, mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in gaps
    #mask  = cv2.dilate(thresh, None, iterations=2)

    # Find contours of moving objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    bounding_boxes = []

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust this threshold as needed
            motion_detected = True
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

    return motion_detected, bounding_boxes


if __name__ == '__main__':
    cap = cv2.VideoCapture('C:\\Users\\Zhanna\\Downloads\\motion-detection2.mp4')  # Use 0 for webcam, or provide video file path

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    _, previous_frame = cap.read()
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    #previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, (21, 21), 0)

    # Create a pool of worker processes
    num_processes = 4  # Adjust based on yo
    # ur CPU cores
    with Pool(processes=num_processes) as pool:
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            #current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

            # Prepare data for parallel processing
            frame_data_for_pool = (current_frame_gray, previous_frame_gray)

            # Submit the motion detection task to the pool
            result = pool.apply_async(detect_motion_in_frame, (frame_data_for_pool,))

            # Get the result (this will block until the task is complete)
            motion_detected, bounding_boxes = result.get()

            if motion_detected:
                for (x, y, w, h) in bounding_boxes:
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Motion Detection", current_frame)
            previous_frame_gray = current_frame_gray.copy()

            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()