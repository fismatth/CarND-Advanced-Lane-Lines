import cv2
import glob
from constants import scale_y_warped, warp_src, warp_dst

class PerspectiveWarper:
    def __init__(self, warp_src, warp_dst):
        self.M = cv2.getPerspectiveTransform(warp_src, warp_dst)
        self.Minv = cv2.getPerspectiveTransform(warp_dst, warp_src)
        
    def transform(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0] * scale_y_warped))
    
    def inv_transform(self, img):
        return cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0] // scale_y_warped))


if __name__ == '__main__':
    persp_warper= PerspectiveWarper(warp_src, warp_dst)
    images = glob.glob('test_images/straight_lines*.jpg')
    
    for fname in images:
        img = cv2.imread(fname)
        warped_img = persp_warper.transform(img)
        cv2.line(img, tuple(warp_src[0]), tuple(warp_src[1]), (0,0,255), 2)
        cv2.line(img, tuple(warp_src[1]), tuple(warp_src[3]), (0,0,255), 2)
        cv2.line(img, tuple(warp_src[3]), tuple(warp_src[2]), (0,0,255), 2)
        cv2.line(img, tuple(warp_src[2]), tuple(warp_src[0]), (0,0,255), 2)
        cv2.imwrite(fname.replace('test_images', 'output_images')[:-4] + '_srcpoints.jpg', img)
        cv2.line(warped_img, tuple(warp_dst[0]), tuple(warp_dst[1]), (0,0,255), 2)
        cv2.line(warped_img, tuple(warp_dst[1]), tuple(warp_dst[3]), (0,0,255), 2)
        cv2.line(warped_img, tuple(warp_dst[3]), tuple(warp_dst[2]), (0,0,255), 2)
        cv2.line(warped_img, tuple(warp_dst[2]), tuple(warp_dst[0]), (0,0,255), 2)
        cv2.imwrite(fname.replace('test_images', 'output_images')[:-4] + '_warped.jpg', warped_img)