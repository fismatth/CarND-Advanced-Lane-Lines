import numpy as np

# scaling factor for #pixels in y direction
scale_y_warped = 2

# source and destination points for warping perspective
warp_src = np.float32([[253, 685], [1050, 685], [592, 452], [687, 452]])
warp_dst = np.float32([[253, 720 * scale_y_warped], [1050, 720 * scale_y_warped], [253, 0], [1050, 0]])