import numpy as np
import cv2

def left_or_right(frame_width, centroid_x):
    """Determine if the object's centroid is on the left or right side of the screen."""
    if centroid_x > frame_width / 2:
        return "Right"
    else:
        return "Left"

def distance_to_midline(frame_width, centroid_x):
    """Calculate the horizontal distance of the object's centroid to the midline of the frame."""
    return abs(centroid_x - frame_width / 2)

# Load previously saved calibration data
calibration_data = np.load('calibration_data.npz')
mtx, dist = calibration_data['mtx'], calibration_data['dist']

# Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=650)

# Load the template image and check if it is loaded correctly
template_image_path = 'Redhawk.png'
template_image = cv2.imread(template_image_path, 0)
if template_image is None:
    raise ValueError(f"Error: Template image not found at path '{template_image_path}'.")

# Detect keypoints and descriptors in the template image
kp_template, desc_template = orb.detectAndCompute(template_image, None)

# Set up video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("Error: Could not open video device.")

# Set the resolution
desired_width, desired_height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Known dimensions of the object (e.g., in mm)
object_height = 250

# Focal length (from the camera matrix)
focal_length = mtx[1, 1]

# Frame processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_template, desc_frame)
    matches = sorted(matches, key=lambda x: x.distance)

    # Proceed if we have enough matches
    if len(matches) > 10:
        # Estimate homography between template and scene
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            # Project the template's corners into the scene
            h, w = template_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Draw the bounding box
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # Calculate centroid of the bounding box
            centroid_x = np.mean(dst[:, 0, 0])

            # Distance and position calculation
            distance = (object_height * focal_length) / cv2.norm(dst[0] - dst[1])
            position = left_or_right(frame_width=frame.shape[1], centroid_x=centroid_x)
            midline_distance = distance_to_midline(frame_width=frame.shape[1], centroid_x=centroid_x)

            print(f"Estimated distance: {distance:.2f} mm; Object is on the: {position}; "
                  f"Horizontal distance to midline: {midline_distance:.2f} pixels")
    else:
        print("Not enough matches found - %d/%d" % (len(matches), 10))

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np
#
# def main():
#     cap = cv2.VideoCapture(0)
#     orb = cv2.ORB_create()
#     template_image = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
#     kp_template, des_template = orb.detectAndCompute(template_image, None)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         kp_frame, des_frame = orb.detectAndCompute(frame, None)
#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#         matches = bf.match(des_template, des_frame)
#         matches = sorted(matches, key=lambda x: x.distance)
#
#         # Assuming the calibration data and object size are known
#         focal_length = 800  # Example focal length
#         known_width = 24.0  # Known width of the object in cm
#         if matches:
#             object_width_in_frame = max([kp_frame[m.trainIdx].pt[0] for m in matches]) - min([kp_frame[m.trainIdx].pt[0] for m in matches])
#             distance = (known_width * focal_length) / object_width_in_frame
#             print(f"Distance to object: {distance} cm")
#
#         cv2.imshow('Frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
