import cv2
import numpy as np
import glob
from constants import scale_y_warped, src, dst

#src = np.float32([[253, 685], [1050, 685], [610, 440], [667, 440]])
#dst = np.float32([[253, 685], [1050, 685], [253, 0], [1050, 0]])

class PerspectiveWarper:
    def __init__(self, src, dst):
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        
    def transform(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0] * scale_y_warped))
    
    def inv_transform(self, img):
        return cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0] // scale_y_warped))


if __name__ == '__main__':
    persp_warper= PerspectiveWarper(src, dst)
    images = glob.glob('test_images/straight_lines*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        warped_img = persp_warper.transform(img)
        cv2.imwrite(fname.replace('test_images', 'output_images')[:-4] + '_warped.jpg', warped_img)