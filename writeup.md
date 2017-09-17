## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted"
[image2]: ./output_images/undistorted/calibration1.jpg "Undistorted"
[image3]: ./test_images/test1.jpg "Straight lines input image"
[image4]: ./output_images/undistorted.jpg "Undistorted"
[image5]: ./output_images/lane_overlay.jpg "Undistorted image with lane overlay"
[image9]: ./output_images/straight_lines1_srcpoints.jpg "Image with source points"
[image10]: ./output_images/straight_lines1_warped.jpg "Warped image with source points"
[image11]: ./output_images/lane_fit.png "Lane pixels with sliding windows and polynomial fit"
[video1]: ./project_video_annotated.mp4 "Potential lane pixels"
[video2]: ./project_video_annotated.mp4 "Potential lane pixels warped"
[video3]: ./project_video_annotated.mp4 "Warped lane area and matched lane pixels"
[video4]: ./project_video_annotated.mp4 "Project Video with lane area and matched lane Pixels drawn"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code to compute the camera calibration is contained in "./camera_calibration.py". 
The Undistort class initially computes the undistortion coefficients (function compute_undistortion_coeffs) and saves them to "./calibration.p" - if this file is already there, the class only loads the stored coefficients.
To compute the undistortion coefficients, I do the following steps:

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

Undistortion of an image can then be applied using the __call__ function of the Undistort class (which uses cv2.undistort()).
For the following chessboard image

![alt text][image1]

this is the undistorted result:

![alt text][image2]

### Pipeline (single images)

The pipeline steps are demonstrated using the following input image:

![alt text][image3]

#### 1. Example of a distortion-corrected image.

Applying the distortion correction as describe before, the resulting image looks like this:
![alt text][image3]

#### 2. Creating binary image using color and gradient thresholds

In "./find_lane_lines.py", function get_potential_lane_pixels(), I'm first converting the image to HLS color space. Then I apply color and gradient thresholds to the L and S channel and combine the resulting four binary images.
For the test image, this is the result:

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image4]

#### 3. Perspective transform

To perform the perspective transform of an image, I use the PerspectiveWarper class in "./perspective_transform.py". 
The source and destination points are defined in "./constants.py":

```python
# scaling factor for #pixels in y direction
scale_y_warped = 2

# source and destination points for warping perspective
warp_src = np.float32([[253, 685], [1050, 685], [592, 452], [687, 452]])
warp_dst = np.float32([[253, 720 * scale_y_warped], [1050, 720 * scale_y_warped], [253, 0], [1050, 0]])
```

I'm scaling the warped image in y-direction by a factor of 2 (variable scale_y_warped) to make the ratio of x and y more realistic and thus make it easier to identify the lane pixels later on.
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 253, 685      | 253, 1440        | 
| 1050, 685      | 1050, 1440      |
| 592, 452     | 253, 0      |
| 687, 452      | 1050, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image9]
![alt text][image10]

#### 4. Identifying lane pixels and fit polynomial

The previously compute binary image is warped as described before. To search for lane pixels and fit a polynomial, the LaneSearcher class in "./find_lanes.py" is used.
To get an initial fit, a sliding window search is used as in the lecture. To fit a polynomial to the left and right lane pixels, np.polyfit is used.
When we already have a fit, the area around the polynomials is used to search for lane pixels, which are then again used to fit a new polynomial.
As a consistency check for the identified lanes, I test if the polynomials are almost parallel.
To align the identified left/right lanes (make them more parallel), I compute the mean of the coefficients (excluding the constant offset) and then updating the left/right lane with averaging itself with these mean coefficients.
To smooth the identified lanes I'm averaging over the last (at most) 10 frames.

---

The identified lane pixels with the sliding windows and fitted polynomials are visualized in the following picture:

![alt text][image11]

#### 5. Calculating radius of curvature and offset to center

The radius of curvature and offset to the center is computed in "./find_lanes.py", function get_curvature_offset(). 
For the radius of curvature, I map the fitted polynomials into world space and evaluate the curvature at the bottom of the image using the formula provided in the lecture.
To compute the offset from the lane center, I'm computing the midpoint of the lane by taking the mean of the fitted left/right polynomial evaluated again at the bottom of the image.
Then, assuming the center of the image in x-direction is the center of the car, the difference of the center and previously computed car position scaled by approximated meters per pixel (3.7/700) gives the required offset in meters.

#### 6. Map lanes back to original image

A (warped) lane overlay image is computed in "./find_lanes.py", function get_lane_overlay(), which visualizes the lane area and identified left and right lane pixels. 
To map this image back to the original perspective, the inv_transform() function of the PerspectiveWarper class is used (in "./perspective_transform.py").
For the example image, the result looks like this:

![alt text][image5]

---

### Pipeline (video)

#### 1. Final video output

I've visualized the intermediate results of the pipeline with some videos (always using the project video as input):

---

[Potential lane pixels](./project_video_binary_annotated.mp4) after thresholding.

---

[Warped potential lane pixels](./project_video_binary_warped_annotated.mp4).

---

[Warped lane area and identified lane pixels](./project_video_lane_overlay_warped_annotated.mp4).

---

Here's the [final video result](./project_video_annotated.mp4)

---

### Discussion

#### 1. Problems and possible improvements

An issue that is clearly visible in the [binary video](./project_video_binary_annotated.mp4), is that sometimes a lot of non-lane pixels on or directly next to the road are identified as potential lane pixels (e.g. at 0:41 in the video). Some possible improvements could be:
* Tune thresholding parameters
* Don't use channels/ corresponding binary images that have too much potential lane pixels
* Filter gradients by direction

Another issue that can be seen in the [warped binary video ](./project_video_binary_warped_annotated.mp4) is that for curves not all previously identified lane pixels are still visible. We could simply adapt the destination points to avoid this (and potentially scale the warped image in x-direction).

---

Another issue visible in the [warped lane overlay video](./project_video_lane_overlay_warped_annotated.mp4) is that the polynomials are not always nearly parallel/ the width is not everywhere the same. 
To improve this, we could try the following steps:
* Estimate lane center line using left/right lane polynomial
* Estimate width (averaging)
* Generate lane pixels around lane center line, move them left and right according to estimated width (add them to the binary image)
* Re-fit left/right lane polynomials as before