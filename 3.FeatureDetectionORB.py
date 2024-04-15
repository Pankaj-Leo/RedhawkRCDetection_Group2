import cv2
import numpy as np

# Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=650)

# Load the template image and convert it to grayscale
template_image = cv2.imread('Redhawk.png', 0)
if template_image is None:
    raise ValueError("Error: Template image not found. Make sure the file 'Redhawk.png' exists in the current directory.")
# Detect keypoints and descriptors in the template image
kp_template, desc_template = orb.detectAndCompute(template_image, None)

# Define the video Camera
cap = cv2.VideoCapture(0)

# Define the video Resolution
desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Define the brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Counter for the frames
frame_counter = 0
# Specify the interval for frame processing (n)
frame_interval = 10  # Change n to your desired interval

while True:
    # Read the current frame from the video capture object
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Check camera index.")
        break

    # Increment the frame counter
    frame_counter += 1

    # Process the frame when frame_counter reaches the interval, then reset the counter
    if frame_counter == frame_interval:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors in the current frame
        kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

        # Match descriptors between the template image and the current frame
        matches = bf.match(desc_template, desc_frame)
        # Sort the matches based on their distance (the lower the better)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:
            # Extract the coordinates of matched keypoints in the template image and the current frame
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate the homography matrix
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Check if the homography matrix is well-defined
            if M is not None:
                # Get the height and width of the template image
                h, w = template_image.shape
                # Define the points of the template image corners
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                # Project the corners into the current frame using the homography matrix
                dst = cv2.perspectiveTransform(pts, M)

                # Draw a bounding box around the detected object
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                print("Not enough matches or bad homography.")

        # Reset the counter
        frame_counter = 0

    # Display the frame
    cv2.imshow('ESC or Q for Exit', frame)

    # Exit loop if 'q' or 'Q' or 'ESC' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q'), ord('Q')]:  # 27 is the ASCII code for 'ESC'
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
