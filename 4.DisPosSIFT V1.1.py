import numpy as np
import cv2

def left_or_right(frame_width, centroid_x):
    """Determine if the object's centroid is on the left or right side of the screen."""
    return "Right" if centroid_x > frame_width / 2 else "Left"

def distance_to_midline(frame_width, centroid_x):
    """Calculate the horizontal distance of the object's centroid to the midline of the frame."""
    return abs(centroid_x - frame_width / 2)

# Load previously saved calibration data
with np.load('calibration_data.npz') as X:
    camera_matrix, dist_coeffs = [X[i] for i in ('mtx', 'dist')]

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Load the template image and check if it's loaded correctly
template_image_path = 'Redhawk.png'
template_image = cv2.imread(template_image_path, 0)
if template_image is None:
    raise ValueError(f"Error: Template image not found at path '{template_image_path}'.")

# Detect keypoints and descriptors in the template image
kp_template, desc_template = sift.detectAndCompute(template_image, None)

# Set up video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("Error: Could not open video device.")

# Set the resolution
desired_width, desired_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Known dimensions of the object
object_height = 250  # The height of the object in millimeters

# Focal length (from the camera matrix)
focal_length = camera_matrix[1, 1]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = sift.detectAndCompute(gray_frame, None)

    # Feature matching using BFMatcher with default parameters
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc_template, desc_frame)
    matches = sorted(matches, key=lambda x: x.distance)

    # Proceed only if there are enough matches
    if len(matches) > 10:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Calculate Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w = template_image.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            centroid_x = np.mean(dst[:, 0, 0])

            # Calculate the distance and position
            distance = (object_height * focal_length) / cv2.norm(dst[0] - dst[1])
            position = left_or_right(frame_width=frame.shape[1], centroid_x=centroid_x)
            midline_distance = distance_to_midline(frame_width=frame.shape[1], centroid_x=centroid_x)

            print(f"Estimated distance: {distance:.2f} mm; Object is on the: {position}; "
                  f"Horizontal distance to midline: {midline_distance:.2f} pixels")
    else:
        print("Not enough matches found - %d/%d" % (len(matches), 10))

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
