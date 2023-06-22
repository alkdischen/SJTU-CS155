#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import cv2
import os
import sys
import glob
import numpy as np
import math

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from collections import deque
from lane_test import get_best, get_center
from std_msgs.msg import Int32


#距离映射
x_cmPerPixel = 90/665.00
y_cmPerPixel = 81/680.00
roadWidth = 665

y_offset = 50.0 #cm

#轴间距
I = 58.0
#摄像头坐标系与车中心间距
D = 18.0
#计算cmdSteer的系数
k = -19





away_h = 1
away_w = 200
rewap = 14
H = 720
W = 1280



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


direc = 50





class camera:
    def __init__(self):

        self.camMat = []
        self.camDistortion = []

        self.cap = cv2.VideoCapture('/dev/video10')
        #self.cap = cv2.VideoCapture('video2.avi')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.imagePub = rospy.Publisher('images', Image, queue_size=1)
        self.cmdPub = rospy.Publisher('lane_vel', Int32, queue_size=10)
        self.cam_cmd = Twist()
        self.cvb = CvBridge()
        
        src_points = np.array([[3,570], [387,460], [906,452], [1041,485]], dtype="float32")
        dst_points = np.array([[266., 686.], [266., 19.], [931., 20.], [931., 701.]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        self.aP = [0.0, 0.0]
        self.lastP = [0.0, 0.0]
        self.Timer = 0
    
    def __del__(self):
        self.cap.release()
    
 
    
       

    def spin(self):
        global last_left
        global last_right
        global error_now
        global direc
        ret, frame = self.cap.read()

        if ret == True:
            warped = cv2.warpPerspective(frame, M, frame.shape[1::-1], 
                                        flags=cv2.INTER_LINEAR)
            
            color_Low1 = np.array([100,240,240])
            color_High = np.array([220,255,255])
            
            blur = cv2.inRange(warped,color_Low1,color_High)
            
            histogram1 = np.sum(blur,axis=0)##(160,20)well
            leftbase,rightbase = get_center(histogram1, last_left , last_right)
            left.append(leftbase)
            right.append(rightbase)
            best_left = get_best(left,alpha)
            best_right = get_best(right,alpha)
            last_left = leftbase
            last_right = rightbase
            best_center = (best_left + best_right)/2
            
        
            img1 = cv2.circle(warped, (int(best_center),50), 20, (0,0,255), -1)   
            #cv2.imshow("result",img1)
            newwarp = cv2.warpPerspective(img1, Minv, frame.shape[1::-1]) 
            # Combine the result with the original image
        
            result = cv2.addWeighted(frame, 1, newwarp, 0.8, 0)
             #cv2.imshow("result",result)
        
            error_1 = error_now
            error_now = (640-best_center)*0.1
            control = kp * error_now + ki*(error_now - error_1)
            control = int(max(min(58,control),-38))
            direction = int(50 - control)
            #print(direction)
            direc = direction
            cv2.putText(result, str(direction), (1000,100),     cv2.FONT_HERSHEY_COMPLEX,      3,       (0, 255, 255),     5)
            #cv2.imshow("warped",result)





            self.cmdPub.publish(direc)
            # self.imagePub.publish(result)  # binary_warped
            cv2.imshow('binary_warped', result)
            cv2.waitKey(1)




def getdire(direc):
    findire = direc
    return findire



if __name__ == '__main__':
    rospy.init_node('lane_vel', anonymous=True)
    rate = rospy.Rate(10)
    
    
    try:
        cam = camera()
        print(rospy.is_shutdown())  # FALSE
        while not rospy.is_shutdown():
            cam.spin()
            print(direc)
            print('betweeen == cam.spin ==')
            rate.sleep()
    except rospy.ROSInterruptException:
        print(rospy.ROSInterruptException)
        pass


