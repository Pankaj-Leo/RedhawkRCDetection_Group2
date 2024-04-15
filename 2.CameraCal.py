import numpy as np
import cv2
import glob

# Termination criteria for the corner refinement process
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Square size in millimeters (change this to your actual square size)
square_size = 23.0

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,9,0)
objp = np.zeros((7*10, 3), np.float32)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# List of images for calibration
images = glob.glob('./*.png')

# Ensure we have some images to process
if not images:
    print("No images found. Check your file path and try again.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (10,7), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)

        # Refine corner location
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (10,7), corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
    else:
        print(f"Chessboard not found in image: {fname}")

cv2.destroyAllWindows()

# Camera calibration
if objpoints and imgpoints:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the calibration data
    np.savez('calibration_data.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    # Output the calibration data to the console
    print("Camera calibrated successfully.")
    print(f"Camera matrix: \n{mtx}")
    print(f"Distortion coefficients: \n{dist}")

    # Calculate and output the total reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"Total reprojection error: {mean_error / len(objpoints)}")
else:
    print("Not enough images with detected chessboard corners for calibration.")
