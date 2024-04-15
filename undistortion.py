import cv2
import numpy as np
import glob
import os

def load_calibration_data(file_path):
    """ Load camera calibration data from a .npz file. """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Calibration data file not found.")
    with np.load(file_path) as X:
        mtx, dist = X['mtx'], X['dist']
    return mtx, dist

def find_images(directory, exclude='redhawk.png'):
    """ Find all .png images in the specified directory, excluding specific filenames. """
    pattern = f"{directory}/*.png"
    all_images = glob.glob(pattern)
    images = [img for img in all_images if exclude not in img]
    if not images:
        raise FileNotFoundError("No suitable .png images found in the directory.")
    return images

def undistort_images(images, mtx, dist):
    """ Undistort images using the provided calibration matrix and distortion coefficients. """
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Failed to load image {fname}")
            continue
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(f"./undistorted_{i}.png", dst)
        print(f"Undistorted image saved as ./undistorted_{i}.png")

def main():
    try:
        mtx, dist = load_calibration_data('calibration_data.npz')
        images = find_images('.')
        undistort_images(images, mtx, dist)
        print("Undistortion process completed successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
