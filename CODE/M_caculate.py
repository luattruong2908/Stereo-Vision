import numpy as np
import cv2
import matplotlib.pyplot as plt

# Check for left and right camera IDs
# These values can change depending on the system
CamL_id = 2  # Camera ID for left camera
CamR_id = 0  # Camera ID for right camera

CamL = cv2.VideoCapture(CamL_id)
CamL.set(3,320)
CamL.set(4,240)
CamR = cv2.VideoCapture(CamR_id)
CamR.set(3,320)
CamR.set(4,240)

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("D:\VSCODE\Stereo_vision\data\param\params_best1.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

# These parameters can vary according to the setup
# Keeping the target object at max_dist we store disparity values
# after every sample_delta distance.
max_dist = 110  # max distance to keep the target object (in cm)
min_dist = 50  # Minimum distance the stereo setup can measure (in cm)
sample_delta = 10  # Distance between two sampling points (in cm)

Z = max_dist
Value_pairs = []

disp_map = np.zeros((600, 600, 3))

# Reading the stored the StereoBM parameters
cv_file = cv2.FileStorage("D:\VSCODE\Stereo_vision\data\param\depth_estmation_params_py.xml", cv2.FILE_STORAGE_READ)
numDisparities = int(cv_file.getNode("numDisparities").real())
blockSize = int(cv_file.getNode("blockSize").real())
preFilterType = int(cv_file.getNode("preFilterType").real())
preFilterSize = int(cv_file.getNode("preFilterSize").real())
preFilterCap = int(cv_file.getNode("preFilterCap").real())
textureThreshold = int(cv_file.getNode("textureThreshold").real())
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
speckleRange = int(cv_file.getNode("speckleRange").real())
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
minDisparity = int(cv_file.getNode("minDisparity").real())
cv_file.release()


# Defining callback functions for mouse events
def mouse_click(event, x, y, flags, param):
    global Z
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if disparity[y, x] > 0:
            Value_pairs.append([Z, disparity[y, x]])
            print("Distance: %r cm  | Disparity: %r" % (Z, disparity[y, x]))
            Z -= sample_delta


cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 640, 480)
cv2.namedWindow('right image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('right image', 640, 480)
cv2.setMouseCallback('disp', mouse_click)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()

while True:

    # Capturing and storing left and right camera images
    retR, imgR = CamR.read()
    retL, imgL = CamL.read()

    # Proceed only if the frames have been captured
    if retL and retR:
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        # Applying stereo image rectification on the left image
        Left_nice = cv2.remap(imgL_gray,
                              Left_Stereo_Map_x,
                              Left_Stereo_Map_y,
                              cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT,
                              0)

        # Applying stereo image rectification on the right image
        Right_nice = cv2.remap(imgR_gray,
                               Right_Stereo_Map_x,
                               Right_Stereo_Map_y,
                               cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT,
                               0)

        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice, Right_nice)
        # NOTE: compute returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_16S and scale it down 16 times.

        # Converting to float32
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity = (disparity / 16.0 - minDisparity) / numDisparities

        # Displaying the disparity map
        cv2.imshow("disp", disparity)
        cv2.imshow("right image", imgR)

        if cv2.waitKey(1) == 27:
            break

        if Z < min_dist:
            break

    else:
        CamL = cv2.VideoCapture(CamL_id)
        CamR = cv2.VideoCapture(CamR_id)

# solving for M in the following equation
# ||    depth = M * (1/disparity)   ||
# for N data points coeff is Nx2 matrix with values
# 1/disparity, 1
# and depth is Nx1 matrix with depth values

value_pairs = np.array(Value_pairs)
z = value_pairs[:, 0]
disp = value_pairs[:, 1]
z = np.array([z]).T
disp = np.array([disp]).T
ones = np.ones((len(disp),1))
disp = np.concatenate((ones,disp), axis = 1)
disp_inv = 1 / disp

#Plotting the relation depth and corresponding disparity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(disp, z, 'o-')
ax1.set(xlabel='Normalized disparity value', ylabel='Depth from camera (cm)',
        title='Relation between depth \n and corresponding disparity')
ax1.grid()
ax2.plot(disp_inv, z, 'o-')
ax2.set(xlabel='Inverse disparity value (1/disp) ', ylabel='Depth from camera (cm)',
        title='Relation between depth \n and corresponding inverse disparity')
ax2.grid()
plt.show()

# Solving for M using Linear regression
A = np.dot(disp_inv.T,disp_inv)
B = np.dot(disp_inv.T,z)
M = np.dot(np.linalg.pinv(A),B)
print('M = ', M)
C = float(M[0])
M = float(M[1])




# Storing the updated value of M along with the stereo parameters
cv_file = cv2.FileStorage("D:\VSCODE\Stereo_vision\data\param\depth_estmation_params_py.xml", cv2.FILE_STORAGE_WRITE)

cv_file.write("numDisparities", numDisparities)
cv_file.write("blockSize", blockSize)
cv_file.write("preFilterType", preFilterType)
cv_file.write("preFilterSize", preFilterSize)
cv_file.write("preFilterCap", preFilterCap)
cv_file.write("textureThreshold", textureThreshold)
cv_file.write("uniquenessRatio", uniquenessRatio)
cv_file.write("speckleRange", speckleRange)
cv_file.write("speckleWindowSize", speckleWindowSize)
cv_file.write("disp12MaxDiff", disp12MaxDiff)
cv_file.write("minDisparity", minDisparity)
cv_file.write("M", M)
cv_file.write("C", C)
cv_file.release()