from tkinter import Frame
import cv2
import numpy as np
import math

"""*****************************参数设置********************************"""
startx = 307
starty = 700
length_pers = 680
width_pers = 665
aP = [0.0,0.0]
dstps = np.float32([[(startx, starty), (startx, starty - length_pers), (startx + width_pers, starty - length_pers),
                    (startx + width_pers, starty)]])
srcps = np.array([[238., 681.], [546., 497.], [762., 483.], [1069., 678.]], dtype="float32")
M = cv2.getPerspectiveTransform(srcps, dstps)
Minv = cv2.getPerspectiveTransform(dstps,srcps)
    
"""视频读取"""
cap = cv2.VideoCapture("f:/lane/lane_3.mp4")
ret = cap.isOpened()
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D') #MP4格式
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

out = cv2.VideoWriter('f:/lane/lane_out_withbackg.mp4', fourcc, 5, (int(w), int(h)))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""*****************************找线与画线********************************"""
def find_line(binary_warped):
    # Take a histogram of the bottom half of the image
    #histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histogram = np.sum(binary_warped[:,:], axis=0)
    
    img1 = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR) 
    num = 0
    for num in range(len(histogram)-1):
        cv2.line(img1, (num,int(720-histogram[num]/200)),(num+1,int(720-histogram[num+1]/200)),(255,255,0), 5)
       
       
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 25
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 25
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            cv2.rectangle(img1,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,0,255),3)
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            cv2.rectangle(img1,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(255,0,255),3)

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
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    num = 0
    for num in range(len(ploty)-1):
        cv2.line(img1, (int(left_fitx[num]),int(ploty[num])),(int(left_fitx[num+1]),int(ploty[num+1])),(0,0,255), 20)
        cv2.line(img1, (int(right_fitx[num]),int(ploty[num])),(int(right_fitx[num+1]),int(ploty[num+1])),(0,0,255), 20)
    
    
    vertices = np.array([[(int(left_fitx[0]),int(ploty[0])),(int(left_fitx[num-1]),int(ploty[num-1])),
                          (int(right_fitx[num-1]),int(ploty[num-1])),(int(right_fitx[0]),int(ploty[0]))]])
    cv2.fillPoly(img1, vertices,(0,255, 0))
    return img1
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""*****************************视频帧处理********************************"""

while(ret):
    
    ret, frame=cap.read()
    if ret == True:
        img = frame
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        binary = np.zeros_like(gray)
        binary[(gray >= 127)] = 255 
        kernel = np.ones((3,3),np.uint8)
        binary_Blur = cv2.morphologyEx(binary, cv2.MORPH_OPEN,kernel) #开运算

        binary_warped = cv2.warpPerspective(binary_Blur, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        img1 = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR) 
        
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)     

        lane_base = np.argmax(histogram)
        nwindows = 25
        window_height = int(binary_warped.shape[0]/nwindows)
        nonzero = binary_warped.nonzero()
        #print(nonzero)        
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        lane_current = lane_base
                
        margin = 25
        minpix = 25
                
        lane_inds = []
            
        img1 = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR) 
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height            
            win_y_high = binary_warped.shape[0] - window*window_height             
            win_x_low = lane_current - margin             
            win_x_high = lane_current + margin 
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            #cv2.rectangle(img1,(win_x_low,win_y_low),(win_x_high,win_y_high),(0,0,255),3)
                        
            lane_inds.append(good_inds)
                    
            if len(good_inds) > minpix:
                lane_current = int(np.mean(nonzerox[good_inds])) 

        #print(lane_inds)
        lane_inds = np.concatenate(lane_inds) 
        pixelX=nonzerox[lane_inds]
        pixelY=nonzeroy[lane_inds]
        fit = np.polyfit(pixelY, pixelX, 2)
        #p1 = np.poly1d(fit)
        #print(p1)
        #pixlY_max = max(pixelY)
        ploty =np.array(list(set(pixelY)))
        #print(ploty)
        plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        num = 0
        for num in range(len(ploty)-1):
            cv2.line(img1, (int(plotx[num]),int(ploty[num])),(int(plotx[num+1]),
                                                            int(ploty[num+1])),(0,255,0), 10)
            
        aP = [0.0,0.0]

        if(lane_base >=620):
            LorR = -1.0  #Right
        else:
            LorR = 1.0  #Left


        aimLaneP = [int(plotx[len(ploty)//2]),int(ploty[len(ploty)//2])] 
        img1 = cv2.circle(img1, (aimLaneP[0],aimLaneP[1]), 25, (255,0,0), -1)

        lanePk =(1/(2*fit[0]))*aimLaneP[0] - fit[1]/(2*fit[0])
        k_ver = -1/lanePk
        theta = math.atan(k_ver)
        aP[0] = aimLaneP[0] + math.cos(theta)*(LorR)*width_pers/2
        aP[1] = aimLaneP[1] + math.sin(theta)*(LorR)*width_pers/2
        
        img1 = cv2.circle(img1, (int(aP[0]),int(aP[1])), 25, (0,0,255), -1)

        
        
        ''''''
        origin_thr = np.zeros_like(gray)  #尺寸一致的黑色图
        origin_thr[(gray >= 165)] = 255  
        kernel = np.ones((3,3),np.uint8)
        gray_Blur = cv2.morphologyEx(origin_thr, cv2.MORPH_OPEN,kernel) #开运算
    
        binary_warped = cv2.warpPerspective(gray_Blur, M, frame.shape[1::-1], flags=cv2.INTER_LINEAR) #透视变换
    
        color_warp=find_line(binary_warped)
        ''''''
    
        """图片叠加"""
        newwarp = cv2.warpPerspective(color_warp, Minv, img1.shape[1::-1]) 
        line = cv2.warpPerspective(img1, Minv, img1.shape[1::-1])       
        result = cv2.addWeighted(newwarp, 1, img, 0.3, 0)
        result = cv2.addWeighted(line, 1, result, 0.3, 0) #output with backg
        
        """
        darkback = np.zeros_like(img)
        result = cv2.addWeighted(newwarp, 1, darkback, 0.3, 0)
        result = cv2.addWeighted(line, 1, result, 0.3, 0)
        
        #output without backg
        """
            
        cv2.imshow("result",result) 
        out.write(result)
        k = cv2.waitKey(75)
        if( k   == ord('q')):
            break   
        
cv2.waitKey(0)
cap.release()
out.release()
cv2.destroyAllWindows()