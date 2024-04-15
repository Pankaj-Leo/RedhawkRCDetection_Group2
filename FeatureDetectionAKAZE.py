import cv2
import numpy as np

def load_calibration_data(file_path):
    """Load camera calibration data from a .npz file."""
    data = np.load(file_path)
    return data['mtx'], data['dist']

def initialize_video_capture():
    """Initialize the video capture object with high resolution."""
    cap = cv2.VideoCapture(0)
    # Attempt to set a high resolution for long-distance feature detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def adjust_akaze_parameters():
    """Adjust AKAZE parameters for better long-range detection."""
    # Increase the threshold to be more selective in feature detection
    # Adjust the number of octaves and octave layers for larger range images
    return cv2.AKAZE_create(threshold=0.0001, nOctaves=6, nOctaveLayers=6, diffusivity=cv2.KAZE_DIFF_PM_G2)

def find_features_and_calculate_metrics(frame, akaze, bf, kp_template, desc_template, camera_matrix, object_width_world):
    """Detect features, match them, and calculate distance and angle."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = akaze.detectAndCompute(gray_frame, None)
    matches = bf.match(desc_template, desc_frame)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        object_width_pixels = max(dst_pts[:, 0]) - min(dst_pts[:, 0])
        distance, angle = calculate_distance_and_angle(object_width_pixels, object_width_world, camera_matrix[0, 0])
        return distance, angle, src_pts, dst_pts
    return None, None, None, None

def main():
    mtx, dist = load_calibration_data('calibration_data.npz')
    cap = initialize_video_capture()
    akaze = adjust_akaze_parameters()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    template_image = cv2.imread('Redhawk.png', cv2.IMREAD_GRAYSCALE)
    kp_template, desc_template = akaze.detectAndCompute(template_image, None)

    object_width_world = 200  # mm, known width of the object

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            distance, angle, src_pts, dst_pts = find_features_and_calculate_metrics(frame, akaze, bf, kp_template, desc_template, mtx, object_width_world)
            if distance is not None:
                print(f"Distance to object: {distance:.2f} mm, Angle: {angle:.2f} degrees")

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
