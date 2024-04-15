import numpy as np
import cv2
# Set up video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("Error: Could not open video device.")
def load_calibration_data(file_path):
    """Load camera calibration data from a .npz file."""
    try:
        data = np.load(file_path)
        mtx = data['mtx']
        dist = data['dist']
        print("Calibration data loaded successfully.")
        return mtx, dist
    except IOError:
        print("Error loading calibration data. Ensure the calibration file exists.")
        raise

def initialize_camera(width=1280, height=720):
    """Initialize the camera and set the resolution."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        raise IOError("Camera cannot be opened.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def undistort_frame(frame, mtx, dist):
    """Apply undistortion transformation to a frame using loaded calibration data."""
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    return undistorted

def main():
    mtx, dist = load_calibration_data('calibration_data.npz')  # Load calibration data
    cap = initialize_camera()  # Initialize the camera

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            undistorted = undistort_frame(frame, mtx, dist)  # Undistort the frame
            cv2.imshow('Undistorted Video', undistorted)

            if cv2.waitKey(1) == 27:  # Exit loop when 'ESC' is pressed
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera and windows closed.")

if __name__ == "__main__":
    main()


# When everything done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
