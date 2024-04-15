import cv2
from imageio.plugins import opencv

# Print OpenCV version
print(cv2.__version__)


#%%

#%%

#%%
# Specific Scripts Considerations
# 0.CheckOpenCV.py: This is likely a script to check if OpenCV is installed and working correctly. Ensure that it checks the specific version you're using for the project.
# 1.CameraCap.py: A script for capturing video from the camera. Here, you might want to ensure it handles multiple cameras or reconnection attempts if the camera disconnects.
# 2.CameraCal.py: The camera calibration script is crucial. Ensure the calibration is accurate and maybe automate the calibration process with a user-friendly interface.
# 3.FeatureDetectionORB.py: The feature detection script using ORB. You may want to ensure that it is optimized for speed since this will run in real-time on a remote control car.
# 4.DisPos.py and 4.DisPosSIFT V1.1.py: These scripts are likely for calculating the distance and position of the detected image. Consider integrating angle detection logic here and fine-tuning the algorithms for different distances and lighting conditions.
# 5.DisPosSIFTFLANN.py: Using SIFT with FLANN for feature matching may provide better results but at the cost of higher computational load. You'll need to balance the accuracy with the speed for real-time processing.
# 6.RCDriveTest.py: This could be a test script for driving the RC car. Here, you might want to add safety checks and remote emergency stop capabilities.
# 7.Prototype.py: Possibly an integration of all the previous components into a prototype. This should be tested extensively and have clear instructions on how to operate the car and troubleshoot common issues.