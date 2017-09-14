import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle

class Undistort:
    def __init__(self):
        try:
            data = pickle.load(open('calibration.p', 'rb'))
        except:
            data = self.compute_undistortion_coeffs()
        self.mtx = data['mtx']
        self.dist = data['dist']
        
    def compute_undistortion_coeffs(self):
        # prepare object points
        nx = 9#TODO: enter the number of inside corners in x
        ny = 6#TODO: enter the number of inside corners in y
        
        images = glob.glob('camera_cal/calibration*.jpg')
        matched_objpoints = []
        imgpoints = []
        
        for fname in images:
            # Make a list of calibration images
            #fname = 'calibration_test.png'
            img = cv2.imread(fname)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            shape = gray.shape
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
            objp = np.zeros((nx * ny, 3), np.float32)
            objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
            
            # If found, draw corners
            if ret == True:
                imgpoints.append(corners)
                matched_objpoints.append(objp)
                # Draw and display the corners
                #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                #plt.imshow(img)
                
        cv2.calibrateCamera(matched_objpoints, imgpoints, shape[::-1], None, None)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(matched_objpoints, imgpoints, shape, None, None)
        data = {'mtx':mtx, 'dist':dist}
        with open('calibration.p', 'wb') as calibration:
            pickle.dump(data, calibration)
        return data

    def __call__(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


if __name__ == '__main__':
    undistorter = Undistort()
    images = glob.glob('camera_cal/calibration*.jpg')
    for fname in images:
        test_img = cv2.imread(fname)
        undist_img = undistorter(test_img)
        cv2.imwrite(fname.replace('camera_cal', 'undistorted'), undist_img)
        #plt.imshow(test_img)
        #plt.imshow(undist_img)