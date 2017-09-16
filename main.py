from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from find_lanes import get_potential_lane_pixels, SlidingWindowSearch
from perspective_transform import PerspectiveWarper
from constants import src, dst
from camera_calibration import Undistort

undistorter = Undistort()
lane_searcher = SlidingWindowSearch()
perspective_warper = PerspectiveWarper(src, dst)


def process_image(img):
    undistorted = undistorter(img)
    binary_img = get_potential_lane_pixels(img)
    #return np.array(cv2.merge((binary_img, binary_img, binary_img)),np.uint8) * 255
    #return cv2.addWeighted(undistorted, 1, , 0.3, 0)
    binary_warped = perspective_warper.transform(binary_img)
    #return np.array(cv2.merge((binary_warped, binary_warped, binary_warped)),np.uint8) * 255
    lanes_result = lane_searcher(binary_warped)
    if lanes_result is None:
        #return np.zeros_like(undistorted)
        return undistorted
    lane_overlay_warped, left_curvature, right_curvature, offset = lanes_result
    #return lane_overlay_warped
    #fit_img_warped = lane_searcher.visualize_fit()
    #fit_img = perspective_warper.inv_transform(fit_img_warped)
    #return fit_img
    lane_overlay = perspective_warper.inv_transform(lane_overlay_warped)
    cv2.putText(lane_overlay, 'curvature: {}m'.format( 0.5* (left_curvature + right_curvature)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
    cv2.putText(lane_overlay, 'offset from center: {}m'.format(offset), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))

    return cv2.addWeighted(undistorted, 1, lane_overlay, 0.5, 0)


if __name__ == '__main__':
    video_fname = 'project_video.mp4'
    clip1 = VideoFileClip(video_fname)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(video_fname.replace('.mp4', '_annotated.mp4'), audio=False)