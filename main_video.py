from moviepy.editor import VideoFileClip
#from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import helpers
import numpy as np
import cv2
import os

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    hsvMasked = helpers.hsvMaskConv(image)
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
    (imHeight, imWidth, __) = image.shape
    vertices = np.array([[(.10*imWidth,imHeight),(0.45*imWidth, 0.60*imHeight), (0.55*imWidth, 0.60*imHeight), (0.9*imWidth,imHeight)]], dtype=np.int32)
    masked_edges = helpers.region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = helpers.rho # distance resolution in pixels of the Hough grid
    theta = helpers.theta # angular resolution in radians of the Hough grid
    threshold = helpers.threshold   # minimum number of votes (intersections in Hough grid cell)
    min_line_length = helpers.min_line_length #minimum number of pixels making up a line
    max_line_gap = helpers.max_line_gap    # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_img, lines = helpers.hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    result = helpers.weighted_img(line_img, image, α=0.8, β=1., γ=0.)
    return result

video_filesList = os.listdir('/Users/sumedhinamdar/Documents/GitHub/CarND-LaneLines-P1/test_videos/')
video_files = [vid for vid in video_filesList if '.mp4' in vid]

for video in video_files:

    helpers.left_m_ave = [] #define list to be used as global variable for EMA filter
    helpers.right_m_ave = [] #define list to be used as global variable for EMA filter
    helpers.left_b_ave = [] #define list to be used as global variable for EMA filter
    helpers.right_b_ave = [] #define list to be used as global variable for EMA filter

    os.chdir('/Users/sumedhinamdar/Documents/GitHub/CarND-LaneLines-P1/test_videos')
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip(video)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    os.chdir('/Users/sumedhinamdar/Documents/GitHub/CarND-LaneLines-P1/test_videos_output')
    white_output = video.replace('.mp4', '_lines.mp4')
    white_clip.write_videofile(white_output, audio=False)
    plt.show()
