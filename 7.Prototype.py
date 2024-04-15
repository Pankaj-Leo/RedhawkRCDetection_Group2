import numpy as np
import cv2
import serial
import serial.tools.list_ports
import time

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if 'Arduino' in p.description:
            return p.device
    return None

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def left_or_right(frame_width, centroid_x):
    """Determine if the object's centroid is on the left or right side of the screen."""
    return "Right" if centroid_x > frame_width / 2 else "Left"

def distance_to_midline(frame_width, centroid_x):
    """Calculate the horizontal distance of the object's centroid to the midline of the frame."""
    return abs(centroid_x - frame_width / 2)

def process_frame(frame, sift, template_kp, template_desc):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = sift.detectAndCompute(gray_frame, None)
    if desc_frame is not None:
        return kp_frame, desc_frame
    return None, None

def compute_commands(frame, kp_template, desc_template, kp_frame, desc_frame):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_template, desc_frame, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > 10:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h, w = template_image.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            centroid_x = np.mean(dst[:, 0, 0])

            # Calculate position and distance
            position = left_or_right(frame.shape[1], centroid_x)
            midline_distance = distance_to_midline(frame.shape[1], centroid_x)

            return position, midline_distance
    return None, None

def main():
    try:
        cap = initialize_camera()
        sift = cv2.SIFT_create()
        template_image = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
        kp_template, desc_template = sift.detectAndCompute(template_image, None)

        arduino_port = find_arduino_port()
        if not arduino_port:
            raise IOError("Arduino not found")

        with serial.Serial(arduino_port, 9600, timeout=1) as ser:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                kp_frame, desc_frame = process_frame(frame, sift, kp_template, desc_template)
                if kp_frame is not None and desc_frame is not None:
                    position, midline_distance = compute_commands(frame, kp_template, desc_template, kp_frame, desc_frame)
                    if position and midline_distance is not None:
                        # Send commands based on position and midline distance
                        # Example: send_command_to_arduino(ser, f"Position: {position}, Distance: {midline_distance}")
                        print(f"Position: {position}, Distance to midline: {midline_distance}")

                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
