import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from perspective_transform import PerspectiveWarper
from constants import src, dst


# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720

def get_potential_lane_pixels(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    #return color_binary
    result = np.zeros_like(s_channel)
    result[sxbinary == 1] = 1
    result[s_binary == 1] = 1
    return result

class SlidingWindowSearch:
    def __init__(self):
        # current frame
        self.binary_warped = None
        # polynomial fit of lanes
        self.left_fit = None
        self.right_fit = None
        self.nonzeroy = None
        self.nonzerox = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.sliding_windows = None
        # Set the width of the windows +/- margin
        self.margin = 100
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
        
    def fit_polynomial(self):
        # Extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds] 
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds] 
        
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
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
        
        self.fit_polynomial()
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
        
        self.fit_polynomial()
        self.state = 'successive_fit'
    
    def get_fitted_pixels(self):
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        return ploty, left_fitx, right_fitx
    
    def visualize_fit(self):
        if self.state == 'no_fit':
            return

        ploty, left_fitx, right_fitx = self.get_fitted_pixels()
        
        out_img = np.array(cv2.merge((self.binary_warped, self.binary_warped, self.binary_warped)),np.uint8) * 255
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
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
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        
    def get_lane_overlay_img(self):
        if self.state == 'no_fit':
            return None
        zeros = np.zeros_like(self.binary_warped)
        lane_overlay = np.array(cv2.merge((zeros, zeros, zeros)),np.uint8) 
        ploty, left_fitx, right_fitx = self.get_fitted_pixels()
        thickness = 10.0
        thickness_half = 0.5 * thickness 
        left_line_left = np.array([np.transpose(np.vstack([left_fitx-thickness_half, ploty]))])
        left_line_right = np.array([np.flipud(np.transpose(np.vstack([left_fitx+thickness_half, ploty])))])
        left_line_pts = np.hstack((left_line_left, left_line_right))
        right_line_left = np.array([np.transpose(np.vstack([right_fitx-thickness_half, ploty]))])
        right_line_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx+thickness_half, ploty])))])
        right_line_pts = np.hstack((right_line_left, right_line_right))
        lane_pts = np.hstack((left_line_right, right_line_left))
        
        cv2.fillPoly(lane_overlay, np.int_([left_line_pts]), (255, 0, 0))
        cv2.fillPoly(lane_overlay, np.int_([right_line_pts]), (0, 0, 255))
        cv2.fillPoly(lane_overlay, np.int_([lane_pts]), (0, 255, 0))
        
        return lane_overlay
    
    def __call__(self, binary_warped):
        self.binary_warped = binary_warped
        if self.state == 'no_fit':
            self.initial_sliding_window_search()
        else:
            self.sliding_window_search()
        return self.get_lane_overlay_img()

if __name__ == '__main__':
    # Read in a thresholded image
    img = cv2.imread('test_images/test2.jpg')
    pw = PerspectiveWarper(src, dst)
    binary = get_potential_lane_pixels(img)
    #plt.imshow(binary, cmap='gray')
    #plt.show()
    warped = pw.transform(binary)
    #plt.imshow(warped, cmap='gray')
    #plt.show()
    
    lane_searcher = SlidingWindowSearch()
    lane_searcher(warped)
    lane_searcher.visualize_fit()
    lane_searcher(warped)
    lane_searcher.visualize_fit()