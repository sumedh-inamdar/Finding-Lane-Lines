import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Define variables in helpers function to be referenced in other files (main_images.py, main_video.py)
kernel_size = 3
low_threshold = 30
high_threshold = 100

rho = 1.5 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 1   # minimum number of votes (intersections in Hough grid cell)
min_line_length = 12 #minimum number of pixels making up a line
max_line_gap = 10    # maximum gap in pixels between connectable line segments

left_m_ave = [] #define list to be used as global variable for EMA filter
right_m_ave = [] #define list to be used as global variable for EMA filter
left_b_ave = [] #define list to be used as global variable for EMA filter
right_b_ave = [] #define list to be used as global variable for EMA filter

def hsvMaskConv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #white color mask
    lower_w = np.array([0,0,200])
    upper_w = np.array([255,30,255])
    hsv_mask_white = cv2.inRange(hsv, lower_w, upper_w)

    #yellow color mask
    lower_y = np.array([10,100,100])
    upper_y = np.array([40,255,255])
    hsv_mask_yellow = cv2.inRange(hsv, lower_y, upper_y)

    #Combine both binary masks and apply on original image
    combinedMask = cv2.bitwise_or(hsv_mask_white, hsv_mask_yellow)
    masked_image = cv2.bitwise_and(img, img, mask = combinedMask)
    return masked_image

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=20):

    leftLaneX = []
    leftLaneY = []
    rightLaneX = []
    rightLaneY = []
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y2-y1)/(x2-x1)
                if (slope > .2 and slope <1.6 and x1>img.shape[1]/2 and x2>img.shape[1]/2):
                    rightLaneX += [x1, x2]
                    rightLaneY += [y1, y2]
                elif (slope < -0.2 and slope > -1.6 and x1<img.shape[1]/2 and x2<img.shape[1]/2):
                    leftLaneX += [x1, x2]
                    leftLaneY += [y1, y2]
    except TypeError or ZeroDivisionError:
        pass
    #function based off 1st order linear regression model fitting x,y points
    try:
        left = np.polyfit(leftLaneX, leftLaneY, 1) #get slope and y-intercept of polyfit line
        right = np.polyfit(rightLaneX, rightLaneY, 1)

        newLeft, newRight = getLaneValuesEMA(left, right) #get weighted moving average of slope, y-intercept

        leftFunction = np.poly1d(newLeft) #turn polyfit line into function that outputs y-value based on x-input
        rightFunction = np.poly1d(newRight)

        leftX1 = (img.shape[0] - newLeft[1])/newLeft[0] #extrapolate line to the y-intercept
        leftY1 = leftFunction(leftX1)
        leftX2 = 0.48*img.shape[1] #extrapolate line to end of lane
        leftY2 = leftFunction(leftX2)

        rightX1 = .52*img.shape[1] #same extrapolation to right lane markers
        rightY1 = rightFunction(rightX1)
        rightX2 = (img.shape[0] - newRight[1])/newRight[0]
        rightY2 = rightFunction(rightX2)

        cv2.line(img, (int(leftX1), int(leftY1)), (int(leftX2), int(leftY2)), color, thickness) #red line for left
        cv2.line(img, (int(rightX1), int(rightY1)), (int(rightX2), int(rightY2)), [0,0,255], thickness) #blue line for right
    except TypeError or ValueError:
        pass

def getLaneValuesEMA(left, right):
    global left_m_ave
    global left_b_ave
    global right_m_ave
    global right_b_ave

    left_m = left[0]
    left_b = left[1]
    right_m = right[0]
    right_b = right[1]

    left_m_ave += [left_m] #append slope of left lane to list
    right_m_ave += [right_m] #append slope of right lane to list
    left_b_ave += [left_b] #append y-int of left lane to list
    right_b_ave += [right_b] #append y-int of right lane to list

    #exponential moving average of series
    newLeftM = EMAcalc(left_m_ave, min(10,len(left_m_ave)), 0.1)
    newRightM = EMAcalc(right_m_ave, min(10,len(right_m_ave)),0.1)
    newLeftB = EMAcalc(left_b_ave, min(10,len(left_b_ave)),0.1)
    newRightB = EMAcalc(right_b_ave, min(10,len(right_b_ave)),0.1)
    newLeft = [newLeftM, newLeftB]
    newRight = [newRightM, newRightB]

    return newLeft, newRight

# Apply exponential moving average smoothing filter to previous 10 lines to estimate next line
# Reference: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
def EMAcalc(series, window, alpha):
    sLen = len(series)-1
    num = 0
    den = 0
    alpha = 2/(window + 1)
    for idx in range(window):
        num += series[(sLen-idx)]*(1-alpha)**idx
        den += (1-alpha)**idx
    return num/den

# DEBUG function
# def EMAcalc1(series, window, alpha):
#     sLen = len(series)-1
#     num = []
#     den = 0
#     alpha = 2/(window + 1)
#     for idx in range(window):
#         num += [series[(sLen-idx)]*(1-alpha)**idx]
#         den += (1-alpha)**idx
#     return [x/den for x in num]

#helper function to debug raw lines
def draw_rawLines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            try:
                slope = (y2-y1)/(x2-x1)
                if (slope > .2 and slope <1.6 and x1>img.shape[1]/2 and x2>img.shape[1]/2):
                    # rightLaneX += [x1, x2]
                    # rightLaneY += [y1, y2]
                    cv2.line(img, (x1,y1),(x2,y2), color, thickness)
                elif (slope < -0.2 and slope > -1.6 and x1<img.shape[1]/2 and x2<img.shape[1]/2):
                    # leftLaneX += [x1, x2]
                    # leftLaneY += [y1, y2]
                    cv2.line(img, (x1,y1),(x2,y2), [0,0,255], thickness)
            except ZeroDivisionError:
                pass

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    # Debug code below - ignore
    #draw_rawLines(line_img, lines)
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(line_img, (x1,y1),(x2,y2), [0,255,0], 5)
    return line_img, lines

def hough_rawLines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_rawLines(line_img, lines)

    # Debug code below - ignore
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(line_img, (x1,y1),(x2,y2), [0,255,0], 5)
    return line_img, lines

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):

    return cv2.addWeighted(initial_img, α, img, β, γ)
