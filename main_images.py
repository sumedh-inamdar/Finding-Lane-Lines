import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import helpers

import os
image_filesList = os.listdir("test_images/")

imgList = ['solidYellowCurve.jpg','solidYellowCurve2.jpg','solidYellowLeft.jpg','solidWhiteCurve.jpg','solidWhiteRight.jpg','whiteCarLaneSwitch.jpg']
image_files = [img for img in image_filesList if any(x in img for x in imgList)]

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

# Read in and grayscale the image
index = 1
fig = plt.figure(figsize=(12,8))
for image in image_files:

    os.chdir("/Users/sumedhinamdar/Documents/GitHub/CarND-LaneLines-P1/test_images")
    currentImage = mpimg.imread(image)
    hsvMasked = helpers.hsvMaskConv(currentImage)
    gray = helpers.grayscale(hsvMasked)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = helpers.kernel_size
    blur_gray = helpers.gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = helpers.low_threshold
    high_threshold = helpers.high_threshold
    edges = helpers.canny(blur_gray, low_threshold, high_threshold)

    # Next we'll isolate the region of interest to apply the Hough transform upon
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    (imHeight, imWidth, __) = currentImage.shape
    vertices = np.array([[(.10*imWidth,imHeight),(0.45*imWidth, 0.55*imHeight), (0.55*imWidth, 0.55*imHeight), (imWidth,imHeight)]], dtype=np.int32)
    masked_edges = helpers.region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = helpers.rho # distance resolution in pixels of the Hough grid
    theta = helpers.theta # angular resolution in radians of the Hough grid
    threshold = helpers.threshold   # minimum number of votes (intersections in Hough grid cell)
    min_line_length = helpers.min_line_length #minimum number of pixels making up a line
    max_line_gap = helpers.max_line_gap    # maximum gap in pixels between connectable line segments
    line_image = np.copy(currentImage)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_img, lines = helpers.hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = helpers.weighted_img(line_img, color_edges, α=0.8, β=1., γ=0.)
    os.chdir("/Users/sumedhinamdar/Documents/GitHub/CarND-LaneLines-P1/test_images_output")
    mpimg.imsave(image.replace('.jpg','_houghT.jpg'),lines_edges[index])
    fig.add_subplot(2,3,index)
    plt.imshow(lines_edges)
    index += 1

plt.show()
