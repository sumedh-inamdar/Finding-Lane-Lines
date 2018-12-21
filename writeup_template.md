@@ -1,4 +1,4 @@
# **Finding Lane Lines on the Road**
# **Finding Lane Lines on the Road**

## Writeup Template

@ -23,81 +23,25 @@ The goals / steps of this project are the following:

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.


My software pipeline consisted of 10 steps:
[image1]:[./test_images/solidYellowCurve.jpg] "Original Solid Yellow Curve"
[image2]: [./test_images/solidYellowCurve2.jpg] "Original Solid Yellow Curve 2"
[image3]: [./test_images/solidYellowLeft.jpg] "Original Solid Yellow Left"

1. HSV conversion + Mask to expose yellow and white colors (lanes)
Convert original image to HSV space and create yellow and white binary masks. Masks are then combined and applied on input image to expose lane lines.

[image4]:[./test_images/solidYellowCurve_hsvMasked.jpg] "Solid Yellow Curve"
[image5]: [./test_images/solidYellowCurve2_hsvMasked.jpg] "Solid Yellow Curve 2"
[image6]: [./test_images/solidYellowLeft_hsvMasked.jpg] "Solid Yellow Left"

2. Apply Grayscale conversion to increase contrast between road and edges of lane
[image7]:[./test_images/solidYellowCurve_gray.jpg] "Solid Yellow Curve"
[image8]: [./test_images/solidYellowCurve2_gray.jpg] "Solid Yellow Curve 2"
[image9]: [./test_images/solidYellowLeft_gray.jpg] "Solid Yellow Left"

3. Apply gaussian blur filter to smoothen edges and remove gaussian noise
[image8]:[./test_images/solidYellowCurve_gaussianBlur.jpg] "Solid Yellow Curve"
[image9]: [./test_images/solidYellowCurve2_gaussianBlur.jpg] "Solid Yellow Curve 2"
[image10]: [./test_images/solidYellowLeft_gaussianBlur.jpg] "Solid Yellow Left"

4. Apply canny edge filter to identify edges through the intensity gradient of the image
[image11]:[./test_images/solidYellowCurve_canny.jpg] "Solid Yellow Curve"
[image12]: [./test_images/solidYellowCurve2_canny.jpg] "Solid Yellow Curve 2"
[image13]: [./test_images/solidYellowLeft_canny.jpg] "Solid Yellow Left"

5. Apply image mask to only retain the portion of the image defined by a polygon with given vertices
[image14]:[./test_images/solidYellowCurve_regionMask.jpg] "Solid Yellow Curve"
[image15]: [./test_images/solidYellowCurve2_regionMask.jpg] "Solid Yellow Curve 2"
[image16]: [./test_images/solidYellowLeft_regionMask.jpg] "Solid Yellow Left"

6. Draw hough lines on cropped canny image through a hough transformation.
[image17]:[./test_images/solidYellowCurve_houghLinesRaw.jpg] "Solid Yellow Curve"
[image18]: [./test_images/solidYellowCurve2_houghLinesRaw.jpg] "Solid Yellow Curve 2"
[image19]: [./test_images/solidYellowLeft_houghLinesRaw.jpg] "Solid Yellow Left"

For 4-6: For a given image, you can use canny edge detection to find points associated with edges. These points then become lines in hough space. The intersection of these lines then determine where we have identified a line.

7. Reject lines that do not meet slope and location requirements of left and right lanes (e.g., left lane: 11 deg < slope < 58 deg and resides on left side of image) and color code lane lines as left(red) and right(blue).
[image20]:[./test_images/solidYellowCurve_houghT.jpg] "Solid Yellow Curve"
[image21]: [./test_images/solidYellowCurve2_houghT.jpg] "Solid Yellow Curve 2"
[image22]: [./test_images/solidYellowLeft_houghT.jpg] "Solid Yellow Left"

8. Fit a first degree polynomial to the (x,y) points for the left and right lanes separately.

"        left = np.polyfit(leftLaneX, leftLaneY, 1) #get slope and y-intercept of polyfit line
        right = np.polyfit(rightLaneX, rightLaneY, 1)
"

9. Feed the polynomial coefficients (slope and y-intercept) into an exponential moving average filter. This is to help smoothen the "jumpy" lines that appear moving frame to frame during the video. The filter applies an exponential smoothing filter on the previous 10 lines while applying the highest weighting to the newest lines.
[image23]: [./test_images/EMAweights.png] "EMA weights"

9. Extrapolate lines to end of lane using slope and y-intercept values from EMA filter.
[image24]:[./test_images/solidYellowCurve_extrapolate.jpg] "Solid Yellow Curve"
[image25]: [./test_images/solidYellowCurve2_extrapolate.jpg] "Solid Yellow Curve 2"
[image26]: [./test_images/solidYellowLeft_extrapolate.jpg] "Solid Yellow Left"
### 2. Identify potential shortcomings with your current pipeline

10. Draw lines on input image for every frame
[image27]:[./test_images/solidYellowCurve_finalImage.jpg] "Solid Yellow Curve"
[image28]: [./test_images/solidYellowCurve2_finalImage.jpg] "Solid Yellow Curve 2"
[image29]: [./test_images/solidYellowLeft_finalImage.jpg] "Solid Yellow Left"

### 2. Identify potential shortcomings with your current pipeline
One potential shortcoming would be what would happen when ...

Shortcomings include:
Another shortcoming could be ...

1. Choosing lines from a car if it were to cross into the region of interest.
2. Can only detect lines under well lit conditions with sufficient contrast between lanes and road.
3. Sharp curves
4. Identifying missing, choppy, or noisy lane lines

### 3. Suggest possible improvements to your pipeline

Improvements include:
A possible improvement would be to ...

1. Placing higher weight on longer lane lines to reduce effect of noise
2. Using higher power polynomial to draw curves instead of lines
3. Automating the tuning of gaussian blur filter, canny edge, and hough transform through a parameter sweep
4. Use AI to train and auto detect lines
Another potential improvement could be to ...
