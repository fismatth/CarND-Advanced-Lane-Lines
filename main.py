from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from find_lanes import get_potential_lane_pixels, LaneSearcher
from perspective_transform import PerspectiveWarper
from constants import warp_src, warp_dst
from camera_calibration import Undistort

undistorter = Undistort()
lane_searcher = LaneSearcher()
perspective_warper = PerspectiveWarper(warp_src, warp_dst)

def convert_binary(binary):
    return np.array(cv2.merge((binary, binary, binary)),np.uint8) * 255

def process_image(img):
    undistorted = undistorter(img)
    #cv2.imwrite('./output_images/undistorted.jpg', undistorted)
    binary_img = get_potential_lane_pixels(undistorted)
    #cv2.imwrite('./output_images/binary_img.jpg', convert_binary(binary_img))
    binary_warped = perspective_warper.transform(binary_img)
    #cv2.imwrite('./output_images/binary_warped.jpg', convert_binary(binary_warped))
    lanes_result = lane_searcher(binary_warped)
    if lanes_result is None:
        return undistorted
    lane_overlay_warped, left_curvature, right_curvature, offset = lanes_result
    #cv2.imwrite('output_images/lane_overlay_warped.jpg', lane_overlay_warped)
    lane_overlay = perspective_warper.inv_transform(lane_overlay_warped)
    #cv2.imwrite('output_images/lane_overlay.jpg', cv2.addWeighted(undistorted, 1, lane_overlay, 0.5, 0))
    cv2.putText(lane_overlay, 'curvature: {}m'.format( 0.5* (left_curvature + right_curvature)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
    cv2.putText(lane_overlay, 'offset from center: {}m'.format(offset), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
    
    return cv2.addWeighted(undistorted, 1, lane_overlay, 0.5, 0)


if __name__ == '__main__':
    #fname = 'test_images/test1.jpg'
    #img = cv2.imread(fname)
    #process_image(img)
    
    video_fname = 'project_video.mp4'
    clip1 = VideoFileClip(video_fname)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(video_fname.replace('.mp4', '_annotated.mp4'), audio=False)