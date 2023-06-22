# reading Video
import cv2
import numpy as np
from collections import deque

away_h = 1
away_w = 200
rewap = 14
H = 720
W = 1280

color_Low1 = np.array([100,240,240])
color_High = np.array([220,255,255])

kp = 1.5
ki = 0.0
kd = 2
control = 0.0
error_1 = error_2 = error_now = 0.0

alpha = 0.8


dstps = np.float32( [[away_w, H-away_h], [away_w+rewap, away_h], 
                     [W-away_w-rewap, away_h],    [W-away_w, H-away_h]])

srcps = np.array([[4, 583], [205,563], [1065,563], [1266,583]], dtype="float32")



M = cv2.getPerspectiveTransform(srcps, dstps)
Minv = cv2.getPerspectiveTransform(dstps,srcps)

last_left = 200
last_right = 1080
i_right = 0
i_left = 0
lane_dist =1000
left = deque(maxlen=5)
right = deque(maxlen=5)


def get_center(histogram,last_left,last_right): 
    global lane_dist
    global i_left ,i_right

    leftbase = np.array(np.where(histogram[:W//2]>20)).mean()
    
    if np.isnan(leftbase):
        rightbase = np.array(np.where(histogram>20)).mean()
        if np.isnan(rightbase):
            rightbase = 1180
            leftbase = 100
        else :
            leftbase = rightbase - lane_dist
    else:
        rightbase = np.array(np.where(histogram[W//2:]>20)).mean()+W//2
        print(rightbase)
        if abs(rightbase - leftbase)<800:
            leftbase = rightbase - lane_dist
        if np.isnan(rightbase):
            rightbase = leftbase + lane_dist
        else :
            lane_dist = rightbase - leftbase
    ## 防止左右颠倒
    if abs(leftbase - last_right) < 300:
        rightbase = leftbase
        leftbase = rightbase - lane_dist
    if abs(rightbase - last_left) < 300 :
        leftbase = rightbase
        rightbase = leftbase + lane_dist
    ##############
    #防止突变和不变
    if abs(rightbase - last_right) > 50 and i_right < 3 :
        rightbase = last_right
    if abs(leftbase - last_left) > 50 and i_right < 3:
        leftbase = last_left
    if rightbase == last_right or leftbase == last_left:
        i_right += 1
    else : i_right = 0
    return leftbase,rightbase

def get_best(data,alpha):
    if(len(data)) == 1:
        return data[0]
    # get moving average value
    else:
        data_array = np.array(data)
        a = data_array[:-1].mean(axis=0)
        b = data_array[-1]
        return ((1. - alpha)*a + alpha*b)



cap = cv2.VideoCapture('F:/test.mp4')

#cv2.namedWindow("A1")
ret = cap.isOpened()
while(ret):
    ret, frame = cap.read()
    if ret == True:
        warped = cv2.warpPerspective(frame, M, frame.shape[1::-1], 
                                     flags=cv2.INTER_LINEAR)
        
        blur = cv2.inRange(warped,color_Low1,color_High)
        
        histogram1 = np.sum(blur,axis=0)##(160,20)well
        leftbase,rightbase = get_center(histogram1,last_left,last_right)
        left.append(leftbase)
        right.append(rightbase)
        best_left = get_best(left,alpha)
        best_right = get_best(right,alpha)
        last_left = leftbase
        last_right = rightbase
        best_center = (best_left + best_right)/2
        
       
        img1 = cv2.circle(warped, (int(best_center),50), 20, (0,0,255), -1)   
        cv2.imshow("result",img1)
        newwarp = cv2.warpPerspective(img1, Minv, frame.shape[1::-1]) 
        # Combine the result with the original image
    
        result = cv2.addWeighted(frame, 1, newwarp, 0.8, 0)
    
    
        
        
        cv2.imshow("result",result)
        
        error_1 = error_now
        error_now = (640-best_center)*0.1
        control = kp * error_now + ki*(error_now - error_1)
        control = int(max(min(58,control),-38))
        direction = int(50 - control)
        cv2.putText(result, str(direction), (1000,100),     cv2.FONT_HERSHEY_COMPLEX,      3,       (0, 255, 255),     5)
        cv2.imshow("warped",result)
        
        k = cv2.waitKey(50)
        if( k & 0xff == ord('q')):
              break
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()