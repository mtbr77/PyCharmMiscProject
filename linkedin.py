import cv2
import numpy as np

video_cap = cv2.VideoCapture('C:\\Users\\Zhanna\\Downloads\\motion-detection2.mp4')
if not video_cap.isOpened():
    print('Unable to open: ')

bg_sub = cv2.createBackgroundSubtractorKNN(history=200)

ksize = (6, 6)  # Kernel size for erosion.
max_contours = 3  # Number of contours to use for rendering a bounding rectangle.
frame_count = 0
frame_start = 5  # Allow this number of frames to bootstrap the generation of a background model.
red = (0, 0, 255)
yellow = (0, 255, 255)

# Process video frames.
while True:
    ret, frame = video_cap.read()
    frame_count += 1
    if frame is None:
        break
    else:
        frame_erode_c = frame.copy()

    # Create a foreground mask for the current frame.
    fg_mask = bg_sub.apply(frame)

    # Wait a few frames for the background model to learn.
    if frame_count > frame_start:

        # Motion area based on foreground mask with erosion.
        fg_mask_erode_c = cv2.erode(fg_mask, np.ones(ksize, np.uint8))
        motion_area_erode = cv2.findNonZero(fg_mask_erode_c)
        if motion_area_erode is not None:
            xe, ye, we, he = cv2.boundingRect(motion_area_erode)
            cv2.rectangle(frame_erode_c, (xe, ye), (xe + we, ye + he), red, thickness=2)

        # Find Contours.
        contours_erode, hierarchy = cv2.findContours(fg_mask_erode_c, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_erode) > 0:

            # Sort contours based on area.
            contours_sorted = sorted(contours_erode, key=cv2.contourArea, reverse=True)

            # Compute bounding rectangle for the top N largest contours.
            for idx in range(min(max_contours, len(contours_sorted))):
                xc, yc, wc, hc = cv2.boundingRect(contours_sorted[idx])
                if idx == 0:
                    x1 = xc
                    y1 = yc
                    x2 = xc + wc
                    y2 = yc + hc
                else:
                    x1 = min(x1, xc)
                    y1 = min(y1, yc)
                    x2 = max(x2, xc + wc)
                    y2 = max(y2, yc + hc)

            # Draw bounding rectangle for top N contours on output frame.
            cv2.rectangle(frame_erode_c, (x1, y1), (x2, y2), yellow, thickness=2)

        # Resize for proper view and display
        cv2.imshow('Motion Detector', frame_erode_c)
        k = cv2.waitKey(20)
        if k == ord('q'):
            break

video_cap.release()
cv2.destroyAllWindows()
