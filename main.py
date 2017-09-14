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
    # TODO put get_potential_lane_pixels here
    undistorted = undistorter(img)
    binary_img = get_potential_lane_pixels(img)
    binary_warped = perspective_warper.transform(binary_img)
    lane_overlay_warped = lane_searcher(binary_warped)
    lane_overlay = perspective_warper.inv_transform(lane_overlay_warped)
    return cv2.addWeighted(undistorted, 1, lane_overlay, 0.3, 0)


if __name__ == '__main__':
    video_fname = 'project_video.mp4'
    clip1 = VideoFileClip(video_fname)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(video_fname.replace('.mp4', '_annotated.mp4'), audio=False)