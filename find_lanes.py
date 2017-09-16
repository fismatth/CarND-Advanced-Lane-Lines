import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import collections
from perspective_transform import PerspectiveWarper
from constants import src, dst, scale_y_warped
from numpy import sign


def get_binary(channel, color_thresh, grad_thresh):
    # Sobel x
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= grad_thresh[0]) & (scaled_sobel <= grad_thresh[1])] = 1
    # Threshold color channel
    color_binary = np.zeros_like(channel)
    color_binary[(channel >= color_thresh[0]) & (channel <= color_thresh[1])] = 1
    return color_binary, grad_binary
    
def get_potential_lane_pixels(img, s_thresh=(170, 255), sx_thresh=(30, 100), l_thresh=(200, 255), lx_thresh=(200, 255)):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    l_binary, lx_binary = get_binary(l_channel, l_thresh, lx_thresh)
    s_binary, sx_binary = get_binary(s_channel, s_thresh, sx_thresh)
    #return color_binary
    result = np.zeros_like(s_channel)
    result[(s_binary == 1) | (sx_binary == 1) | (l_binary == 1) | (lx_binary == 1)] = 1
    #result[(l_binary == 1) | (lx_binary == 1)] = 1
    return result

class SlidingWindowSearch:
    def __init__(self):
        # current frame
        self.binary_warped = None
        # polynomial fit of lanes
        self.left_fit = None
        self.right_fit = None
        self.left_fit_history = collections.deque(maxlen=10)
        self.right_fit_history = collections.deque(maxlen=10)
        self.nonzeroy = None
        self.nonzerox = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.sliding_windows = None
        # Set the width of the windows +/- margin
        self.margin = 75
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        self.state = 'no_fit'
        
    def init_nonzero(self):
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        
    def almost_parallel(self, poly_fit1, poly_fit2):
        sign_check = np.sign(poly_fit1[0]) == np.sign(poly_fit2[0]) or (abs(poly_fit1[0]) < 1e-3 and abs(poly_fit2[0] < 1e-3)) 
        return sign_check and abs(poly_fit1[0] - poly_fit2[0]) < 1e-3 and abs(poly_fit1[1] - poly_fit2[1]) < 30.0
    
    def almost_equal(self, poly_fit1, poly_fit2):
        return self.almost_parallel(poly_fit1, poly_fit2) and (abs(poly_fit1[2] - poly_fit2[2])) < 50.0
        
    def fit_polynomial(self):
        # Extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]
        
        try:
            # Fit a second order polynomial to each
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
        except TypeError:
            return False
        return True
        
    def append_history(self):
        self.left_fit_history.append(self.left_fit)
        self.right_fit_history.append(self.right_fit)
        
    def initial_sliding_window_search(self):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[int(self.binary_warped.shape[0]/2):,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(self.binary_warped.shape[0]/ self.nwindows)
        
        self.init_nonzero()

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.sliding_windows = []
        
        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
            win_y_high = self.binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            self.sliding_windows.append([win_xleft_low,win_y_low, win_xleft_high,win_y_high])
            self.sliding_windows.append([win_xright_low,win_y_low, win_xright_high, win_y_high])
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & 
            (self.nonzerox >= win_xleft_low) &  (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & 
            (self.nonzerox >= win_xright_low) &  (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)
        
        if self.fit_polynomial() and self.almost_parallel(self.left_fit, self.right_fit):
            a = 0.5 * (self.left_fit[0] + self.right_fit[0])
            b = 0.5 * (self.left_fit[1] + self.right_fit[1])
            self.left_fit[0] = 0.5 * (self.left_fit[0] + a)
            self.left_fit[1] = 0.5 * (self.left_fit[1] + b)
            self.right_fit[0] = 0.5 * (self.right_fit[0] + a)
            self.right_fit[1] = 0.5 * (self.right_fit[1] + b)
            self.append_history()
            self.state = 'initial_fit'

    def sliding_window_search(self):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        self.init_nonzero()
        self.left_lane_inds = ((self.nonzerox > (self.left_fit[0]*(self.nonzeroy**2) + self.left_fit[1]*self.nonzeroy + self.left_fit[2] - self.margin))
                          & (self.nonzerox < (self.left_fit[0]*(self.nonzeroy**2) + self.left_fit[1]*self.nonzeroy + self.left_fit[2] + self.margin))) 
        
        self.right_lane_inds = ((self.nonzerox > (self.right_fit[0]*(self.nonzeroy**2) + self.right_fit[1]*self.nonzeroy + self.right_fit[2] - self.margin))
                           & (self.nonzerox < (self.right_fit[0]*(self.nonzeroy**2) + self.right_fit[1]*self.nonzeroy + self.right_fit[2] + self.margin)))  
        
        last_left_fit = self.left_fit
        last_right_fit = self.right_fit
        
        if self.fit_polynomial() and self.almost_parallel(self.left_fit, self.right_fit) and self.almost_parallel(last_left_fit, self.left_fit) and self.almost_parallel(last_right_fit, self.right_fit):
            self.append_history()
            self.state = 'successive_fit'
        else:
            self.left_fit_history.popleft()
            self.right_fit_history.popleft()
            if len(self.left_fit_history) == 0:
                self.state = 'no_fit'
    
    def get_fitted_pixels(self):
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        return ploty, left_fitx, right_fitx
    
    def visualize_fit(self, draw_lines=True):
        if self.state == 'no_fit':
            return

        ploty, left_fitx, right_fitx = self.get_fitted_pixels()
        
        out_img = np.array(cv2.merge((self.binary_warped, self.binary_warped, self.binary_warped)),np.uint8) * 255
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
        if draw_lines:
            if self.state == 'initial_fit':
                for window in self.sliding_windows:
                    cv2.rectangle(out_img,(window[0], window[1]), (window[2], window[3]), (0,255,0), 2)
            else:
                # Generate a polygon to illustrate the search window area
                # And recast the x and y points into usable format for cv2.fillPoly()
                left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
                left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
                left_line_pts = np.hstack((left_line_window1, left_line_window2))
                right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
                right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
                right_line_pts = np.hstack((right_line_window1, right_line_window2))
                
                # Draw the lane onto the warped blank image
                window_img = np.zeros_like(out_img)
                cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
                cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
                out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        return out_img
        
    def get_lanes(self):
        if self.state == 'no_fit':
            return None
        ploty, left_fitx, right_fitx = self.get_fitted_pixels()
        left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_pts = np.hstack((left_line, right_line))
        
        lane_pixels = self.visualize_fit(False)
        lane_area = np.zeros_like(lane_pixels)
        cv2.fillPoly(lane_area, np.int_([lane_pts]), (0, 255, 0))
        lane_overlay = cv2.addWeighted(lane_pixels, 0.5, lane_area, 0.5, 0)
        
        left_curvature, right_curvature, offset = self.get_curvature_offset(ploty, left_fitx, right_fitx)
        
        return lane_overlay, left_curvature, right_curvature, offset
    
    def eval_poly(self, poly, y):
        return poly[0] * y**2 + poly[1] * y + poly[2]
    
    def get_curvature_offset(self, ploty, leftx, rightx):
        y_eval = np.max(self.binary_warped.shape[0])
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/ (720 * scale_y_warped) # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curv_m = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curve_m = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        x_center = self.binary_warped.shape[1] / 2
        x_car = 0.5 * (self.eval_poly(self.left_fit, y_eval) + self.eval_poly(self.right_fit, y_eval))
        x_offset_m = (x_center - x_car) * xm_per_pix
        
        return left_curv_m, right_curve_m, x_offset_m
    
    def filter_fit(self):
        np_left_hist = np.array(self.left_fit_history)
        np_right_hist = np.array(self.right_fit_history)
        self.left_fit = np.mean(np_left_hist, axis=0)
        self.right_fit = np.mean(np_right_hist, axis=0)
    
    def __call__(self, binary_warped):
        self.binary_warped = binary_warped
        if self.state == 'no_fit':
            self.initial_sliding_window_search()
        else:
            self.sliding_window_search()
        if self.state != 'no_fit':
            self.filter_fit()
        return self.get_lanes()

if __name__ == '__main__':
    # Read in a thresholded image
    img = cv2.imread('test_images/test4.jpg')
    pw = PerspectiveWarper(src, dst)
    binary = get_potential_lane_pixels(img)
    #plt.imshow(binary, cmap='gray')
    #plt.show()
    warped = pw.transform(binary)
    #plt.imshow(warped, cmap='gray')
    #plt.show()
    
    lane_searcher = SlidingWindowSearch()
    lane_searcher(warped)
    out_img = lane_searcher.visualize_fit()
    plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()