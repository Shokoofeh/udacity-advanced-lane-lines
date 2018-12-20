import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from itertools import groupby, islice,  cycle
import os.path
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# %matplotlib qt


### helper functions for visualization and saving results ###

def save_image(img, fname):
    cv2.imwrite(fname ,img)

def save_matrix(M, Minv, name):
    np.save(name + '_warp_matrix' , M)
    np.save(name +'_unwarp_matrix' , Minv)

### Camera Calibration ### 
#### Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. ####

def get_camera_calibration(img_size):
    if(os.path.isfile(name + 'calibrateCamera_mtx.npy') & os.path.isfile(name + 'calibrateCamera_dist.npy')):
        mtx = np.load(name + "calibrateCamera_mtx.npy")
        dist = np.load(name + "calibrateCamera_dist.npy")
        print('Loading camera calibration matrix ...')
        return mtx, dist

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('./camera_cal/calibration*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    np.save(name + "calibrateCamera_mtx", mtx)
    np.save(name + "calibrateCamera_dist", dist)
    print('Done calculating camera calibration ...')

    return mtx, dist

#### Apply a distortion correction to raw images. ####

def get_undistorted_image(img):
    img_size = (img.shape[1], img.shape[0])
    mtx, dist = get_camera_calibration(img_size)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    print('Done undistorting the image ...')
    return dst

###############################################################################

### Apply a perspective transform to rectify binary image ("birds-eye view"). ### 

def measure_warp(img, name):
    if(os.path.isfile(name + '_warp_matrix.npy') & os.path.isfile(name + '_unwarp_matrix.npy')):
        M = np.load(name + '_warp_matrix.npy')
        Minv = np.load(name + '_unwarp_matrix.npy')
        print('Loading saved perspective transformation matrix for ' + name)
        return M, Minv
    top = 0
    bottom = img.shape[0]
    def handler(e):
        if len(src)<4:
            plt.axhline(int(e.ydata), linewidth=2, color='r')
            plt.axvline(int(e.xdata), linewidth=2, color='r')
            src.append((int(e.xdata),int(e.ydata)))
        if len(src)==4:
            dst.extend([(100,bottom),(100,top),(1180,top),(1180,bottom)])
    was_interactive = matplotlib.is_interactive()
    if not matplotlib.is_interactive():
        plt.ion()
    fig = plt.figure()
    plt.imshow(img)
    global src                                                            
    global dst                                                            
    src = []
    dst = []
    cid1 = fig.canvas.mpl_connect('button_press_event', handler)
    cid2 = fig.canvas.mpl_connect('close_event', lambda e: e.canvas.stop_event_loop())
    fig.canvas.start_event_loop(timeout=-1)
    M = cv2.getPerspectiveTransform((np.asfarray(src, np.float32)), (np.asfarray(dst, np.float32)))
    Minv = cv2.getPerspectiveTransform(np.asfarray(dst, np.float32), np.asfarray(src, np.float32))
    matplotlib.interactive(was_interactive)
    np.save(name + '_warp_matrix', M)
    np.save(name + '_unwarp_matrix' , Minv)
    return M, Minv



def get_warped_image(img, name):

    img_size = (img.shape[1], img.shape[0])
    M, Minv = measure_warp(img, name)
    warped = cv2.warpPerspective(img, M, img_size)
    unwarp = cv2.warpPerspective(warped, Minv, img_size)
    print('Done Perspective Transform ...')
    return warped, unwarp, M, Minv
    

###############################################################################

### Use color transforms, gradients, etc., to create a thresholded binary image. ###

def binary_threshold(img, thresh):
    binary = np.zeros_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 1
    return binary

def abs_sobel_func(img, orient, sobel_kernel = 3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient == 'y', ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    return abs_sobel

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20,100)):
    abs_sobel = abs_sobel_func(img, orient, sobel_kernel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = binary_threshold(scaled_sobel, thresh)
    return sxbinary

def mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100)):
    abs_sobelx = abs_sobel_func(img, 'x', sobel_kernel)
    abs_sobely = abs_sobel_func(img, 'y', sobel_kernel)
    abs_sobel = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    mask = binary_threshold(scaled_sobel, mag_thresh)
    return mask

def dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3)):
    abs_sobelx = abs_sobel_func(img, 'x', sobel_kernel)
    abs_sobely = abs_sobel_func(img, 'y', sobel_kernel)
    dir_gradient = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = binary_threshold(dir_gradient, thresh)
    return binary_output

def apply_gradian_threshold(image):
    ksize = 3
    gradx = abs_sobel_thresh(image, orient='x',  sobel_kernel=3, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y',  sobel_kernel=3, thresh=(20, 100))   
    mag_binary = mag_thresh(image,  sobel_kernel=3, mag_thresh=(70, 100))    
    dir_binary = dir_threshold(image,  sobel_kernel=3, thresh=(0.7, 0.9))
    grad_binary = np.zeros_like(dir_binary)
#     grad_binary[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    grad_binary[(gradx == 1)] = 1
    return grad_binary

def apply_color_threshold(img):
    r_channel = img[:,:,0]
    thresh = (150, 255)
    r_binary = binary_threshold(r_channel, thresh)
    
    g_channel = img[:,:,1]
    thresh = (200, 255)
    g_binary = binary_threshold(g_channel, thresh)
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    thresh = (170, 255)
    s_binary = binary_threshold(s_channel, thresh)
    
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    u_channel = yuv[:,:,1]
    thresh = (0, 0)
    u_binary = binary_threshold(u_channel, thresh)
    
    color_binary = np.zeros_like(s_binary)
#     color_binary[(s_binary == 1) | (u_binary == 1) | ((r_binary == 1) & (g_binary == 1))] = 1
    color_binary[(s_binary == 1) ] = 1
    return s_binary

def get_binary_image(img):
    gradient_binary = apply_gradian_threshold(img)
    color_binary = apply_color_threshold(img)

    combined_binary = np.zeros_like(gradient_binary)
    combined_binary[(gradient_binary == 1) | (color_binary == 1)] = 1
    print('Done binary transform ...')
    return combined_binary
        


###############################################################################
### Detect lane pixels and fit to find the lane boundary. ### 

def detect_lines_sliding_window(warped_binary):
    histogram = np.sum(warped_binary[warped_binary.shape[0]/2:,:], axis=0)
    out_img = np.dstack((warped_binary, warped_binary, warped_binary))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped_binary.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_binary.shape[0] - (window+1)*window_height
        win_y_high = warped_binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, 719, 720)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    print('Done finding lines ...')
    return left_fit, right_fit,out_img
  
    
def draw_lane(undistorted, warped_binary,left_fit, right_fit, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    nonzero = warped_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])
    left_fitx = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]
    right_fitx = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]
    
    margin = 50
    left_lane_inds = ((left_fitx - margin < nonzerox) & (nonzerox < left_fitx + margin))
    right_lane_inds = ((right_fitx - margin < nonzerox) & (nonzerox < right_fitx + margin))
    
        ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped_binary, warped_binary, warped_binary))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, nonzeroy]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              nonzeroy])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, nonzeroy]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              nonzeroy])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result_line = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Draw the lane onto the warped_binary blank image
    pts = np.hstack((np.array([np.flipud(np.transpose(np.vstack([left_fitx, 
                              nonzeroy])))]), np.array([np.transpose(np.vstack([right_fitx, nonzeroy]))])))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0])) 
    result_lane = cv2.addWeighted(undistorted, 1, newwarp, 0.6, 0)
    y_eval = warped_binary.shape[0]
    ym_per_pix = 30.0/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(nonzeroy*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(nonzeroy*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    cv2.putText(result_lane, "L. Curvature: %.2f km" % (left_curverad/1000), (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    cv2.putText(result_lane, "R. Curvature: %.2f km" % (right_curverad/1000), (50,80), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    # Annotate image with position estimate
    cv2.putText(result_lane, "C. Position: %.2f m" % ((np.average((left_fitx + right_fitx)/2) - warped_binary.shape[1]//2)*3.7/700), (50,110), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    
    print('Done drawing lane ...')
    return result_lane, result_line


def process_image(test_img, save = False):
    undistorted_img = get_undistorted_image(test_img)
    binary_image = get_binary_image(undistorted_img)
    binary_warped_image, unwarped_image, M, Minv = get_warped_image(binary_image, name)
    l_fit, r_fit, window_line_image = detect_lines_sliding_window(binary_warped_image)
    lane_image, line_image = draw_lane(undistorted_img, binary_warped_image, l_fit, r_fit, Minv)

    if(save):
        save_image(window_line_image, name + '_window_line_image.jpg')
        save_image(unwarped_image, name + '_unwarped_image.jpg')
        save_image(binary_warped_image, name + '_binary_warped_image.jpg')
        save_image(binary_image, name + '_binary_image.jpg')
        save_image(undistorted_img, name + '_undistorted_img.jpg')
        save_image(lane_image, name + '_lane_image.jpg')
        save_image(line_image, name + '_line_image.jpg')
    return lane_image

##### For Single image
name = './output_images/single_images/test_images/test3'
test_img = cv2.imread(name + '.jpg')
process_image(test_img, True)

##### For video
# name = 'harder_challenge_video'
# video_output = name+'_output.mp4'
# clip1 = VideoFileClip('project_video.mp4')
# white_clip = clip1.fl_image(process_image) 
# white_clip.write_videofile(video_output, audio=False)