#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import cv2
import os
import sys
import glob
import numpy as np
import math
import time
#from sensor_msgs.msg import Image
#from geometry_msgs.msg import Twist
#from cv_bridge import CvBridge
from std_msgs.msg import Int32,Int32MultiArray,Float32MultiArray
import threading
from collections import deque

perspective_src = np.float32(
[
    [147,459],
    [431,364],
    [835,370],
    [1108,475]])
'''
147,459
431,364
835,370
1108,475

148,460
476,348
792,352
1109,476
'''
away_h = 1
away_w = 200
rewap = 14
H = 720
W = 1280
perspective_des = np.float32(
    [[away_w, H-away_h],
    [away_w+rewap, away_h],
    [W-away_w-rewap, away_h],
    [W-away_w, H-away_h]]
)
M = cv2.getPerspectiveTransform(perspective_src, perspective_des)
def get_pers_transform(img_):
        img_size = (img_.shape[1], img_.shape[0])
        return cv2.warpPerspective(img_, M, img_size, flags=cv2.INTER_LINEAR)
class camera:
    def __init__(self):

        self.camMat = []
        self.camDistortion = []

        self.cap = cv2.VideoCapture('/dev/video10')
        #print(self.cap.isOpened())
        #self.cap = cv2.VideoCapture(0)
        #self.cap = cv2.VideoCapture('video2.avi')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        #self.imagePub = rospy.Publisher('images', Image, queue_size=1)
        #self.cmdPub = rospy.Publisher('lane_vel', Twist, queue_size=1)

        #self.cam_cmd = Twist()

        #self.cvb = CvBridge()

        self.aP = [0.0, 0.0]
        self.lastP = [0.0, 0.0]
        self.Timer = 0
        self.angularScale=6
        self.abc=0

    def __del__(self):
        self.cap.release()

def white_select(img):
    #yellow BGR 0,255,255
    #black BGR 0,0,0
    color_Low1 = np.array([240,240,240])
    color_High = np.array([255,255,255])
    return cv2.inRange(img,color_Low1,color_High)

def yellow_select(img):
    #yellow BGR 0,255,255
    #black BGR 0,0,0
    color_Low1 = np.array([100,240,240])
    color_High = np.array([255,255,255])
    return cv2.inRange(img,color_Low1,color_High)
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
last_left = 200
last_right = 1080
i_right = 0
i_left = 0
def thread_job():        
    rospy.spin()
def obs_callback(msg):
    global has_obs
    global has_back_obs
    #print(msg.data)
    has_obs = msg.data[0]
    has_back_obs = msg.data[1]
def park_callback(msg):
    global park_time
    park_time = msg.data
def laserlane_callback(msg):
    global leftdist
    global rightdist
    leftdist = msg.data[0]
    rightdist = msg.data[1]
leftdist = 10
rightdist = 10
has_obs = 0
has_back_obs = 0
park_time = 0
def thetaX_callback(msg):
    global thetaX
    #print(msg.data)
    #thetaX.append(msg.data)
    #print(thetaX)
    thetaX = msg.data
thetaX = -239

def hilens_callback(msg):
    global label
    global label_1
    label_1 = label
    label = msg.data
    if label_1 == 3:
        label = 3
    #print(class_names[label])
label = 0
label_1 = 0
class_names = ['green_go', 'pedestrian_crossing', 'red_stop', 'speed_high', 'speed_limited', 'speed_unlimited', 'yellow_back'] 
#             [0              1                      2              3             4                  5               6       ]
def get_best(data,alpha):
    if(len(data)) == 1:
        return data[0]
    # get moving average value
    else:
        data_array = np.array(data)
        a = data_array[:-1].mean(axis=0)
        b = data_array[-1]
        return ((1. - alpha)*a + alpha*b)


left = deque(maxlen=5)
right = deque(maxlen=5)
#thetaX = deque(maxlen = 10)
lane_dist = 1000
if __name__ == "__main__":
    downstart = time.time()
    cam = camera()
    #cv2.namedWindow('testcamera')
    #cv2.namedWindow('img_yellow')
    pub1 = rospy.Publisher('/auto_driver/send/speed', Int32, queue_size=10)
    pub2 = rospy.Publisher('/auto_driver/send/gear', Int32, queue_size=10)
    pub3 = rospy.Publisher('/auto_driver/send/direction', Int32 , queue_size=10)
    rospy.init_node('car_cmd', anonymous=True)
    add_thread = threading.Thread(target = thread_job)     # 阻塞循环接收节点
    add_thread.start()
    rate = rospy.Rate(30)
    rospy.Subscriber('/auto_driver/send/obstacle', Int32MultiArray, obs_callback)
    rospy.Subscriber('/auto_driver/send/park', Int32, park_callback)
    rospy.Subscriber('/auto_driver/send/lanedist', Float32MultiArray, laserlane_callback)
    rospy.Subscriber('/vcu/thetaX', Int32, thetaX_callback)
    rospy.Subscriber('/hilens/label', Int32, hilens_callback)
    #yellow BGR 0,255,255
    #black BGR 0,0,0
    color_Low1 = np.array([120,240,240])
    color_High = np.array([220,255,255])
    kp = 1.5
    ki = 0.0
    kd = 2
    control = 0.0
    error_1= error_now = 0.0
    error_laser_1 = laser_error = 0.0
    gear = 1
    stage = 0
    i = 0
    alpha = 0.8
    crossing_time = 0
    part_time_1 = part_time_now = 100
    while not rospy.is_shutdown():
        start = time.time()
        if leftdist < 1.2 and rightdist <1.2 and label == 1:# and label != 3: ##false close laser
            #rightdist = 0.7
            laser_error_1 = laser_error
            laser_error  = -(rightdist - leftdist)*150
            control = 0.7 * laser_error + 2 * (laser_error - laser_error_1)
            gear = 1
            speed = 30
            #if thetaX <-250 and thetaX > -320:
                #downstart = time.time()
            if thetaX < -800 and thetaX >-1800:
                print("low -1200")
                if time.time()- downstart > 0.4:
                    gear = 2
                    speed = 10
                    downstart = time.time()
                    print("shake")
            print("laser_control")
            rate.sleep()
        else:
            print("cv_control")
            ret, raw = cam.cap.read()
            print(ret)
            blur = raw[450:470,0:1180,:]
            blur = yellow_select(blur)
            histogram1 = np.sum(blur,axis=0)##(160,20)well
            leftbase,rightbase = get_center(histogram1,last_left,last_right)
            left.append(leftbase)
            right.append(rightbase)
            best_left = get_best(left,alpha)
            best_right = get_best(right,alpha)
            last_left = leftbase
            last_right = rightbase
            best_center = (best_left + best_right)/2
            #error_2 = error_1
            error_1 = error_now
            error_now = (640-best_center)*0.1
            control = kp * error_now + ki*(error_now - error_1)
            control = int(max(min(58,control),-38))
            gear = 1
            speed = 35
            
            if label == 4 :
                gear = 1
                speed = 10
            elif label == 1:
                #crossing_time += 1
                print("crossing")
                speed = 50
            elif label == 6:
                speed = 50
            else:
                speed = 50
            #if control > 30:
                #speed = 35



        direction = int(60 - control)
        if (has_obs == 1 and label == 3) or label == 2:
            gear = 2
        '''    
        if crossing_time >=1 and  crossing_time <= 2:
            gear = 2
            crossing_time += 10
            for k in range(10):
                pub2.publish(gear)
            time.sleep(5)
        '''
        
        if label == 1 and crossing_time == 0:
            speed = 35
            se = raw[600:700,:100,:]
            se = white_select(se)
            if se.mean()>1:
                gear = 2
                print("cross")
                crossing_time += 1
                for k in range(20):
                    pub2.publish(gear)
                time.sleep(1)
        
        part_time_1 = part_time_now
        part_time_now = park_time
        
        ############################# daoche
        if part_time_now - part_time_1 ==1  and label ==3 :
            speed = 20
            gear = 1
            direction = 65
            for k in range(10):
                pub1.publish(speed)
                pub2.publish(gear)
                pub3.publish(direction)
            time.sleep(2.1)
            #################### zhi
            for k in range(10):
                pub1.publish(speed)
                pub2.publish(gear)
                pub3.publish(5)
            time.sleep(2.7)
            ##################   zuo 
            for k in range(10):
                pub1.publish(speed)
                pub2.publish(gear)
                pub3.publish(direction)
            time.sleep(1.6)

            ################   zhi
            for k in range(10):
                pub1.publish(speed)
                pub2.publish(4)
                pub3.publish(20)
            time.sleep(3)
            ##################  zuo hou

            for k in range(10):
                pub1.publish(speed)
                pub2.publish(4)
                pub3.publish(92)
            time.sleep(4.5)
            ###################  you  hou
            for k in range(10):
                pub1.publish(0)
                pub2.publish(1)
                pub3.publish(60)
            sys.exit()
      
        pub1.publish(speed)
        pub2.publish(gear)
        pub3.publish(direction)
        ####################show img
        '''
        cv2.circle(raw, (int(best_right),460 ), 100, 255, -1)
        cv2.circle(raw, (int(best_left),460 ), 100, 10, -1)
        show = cv2.resize(raw, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        cv2.imshow("img_yellow",show)
        #cv2.imshow("testcamera", frame)
        #print("hehe")
        
        if cv2.waitKey(1) & 0xff == ord('q'):
           break
        '''
        #############################
        end = time.time()
        #print(se.mean())
        #rospy.loginfo("e:"+str(best_center)+ " "+"d:"+str(direction)+" "+"fps:"+ str(1/(end - start))+" o:"+str(has_obs))
        #print(park_time)
        #rospy.loginfo("fps:"+ str(1/(end - start)))
        print(class_names[label],gear,crossing_time)
        

    cam.cap.release()
    cv2.destroyAllWindows()