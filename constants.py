import numpy as np

src = np.float32([[253, 685], [1050, 685], [592, 452], [687, 452]])
scale_y_warped = 2
dst = np.float32([[253, 685 * scale_y_warped], [1050, 685 * scale_y_warped], [253, 0], [1050, 0]])