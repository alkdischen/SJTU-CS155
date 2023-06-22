#!/usr/bin/env python
#coding=utf-8
import rospy
#倒入自定义的数据类型
import time
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np
import threading


def thread_job():
    
    rospy.spin()

def kinematicCtrl():
    
    #Publisher 函数第一个参数是话题名称，第二个参数 数据类型，现在就是我们定义的msg 最后一个是缓冲区的大小
    #queue_size: None（不建议）  #这将设置为阻塞式同步收发模式！
    #queue_size: 0（不建议）#这将设置为无限缓冲区模式，很危险！
    #queue_size: 10 or more  #一般情况下，设为10 。queue_size太大了会导致数据延迟不同步。
    
    pub1 = rospy.Publisher('/bluetooth/received/manul', Int32 , queue_size=10)
    pub2 = rospy.Publisher('/auto_driver/send/direction', Int32 , queue_size=10)
    pub3 = rospy.Publisher('/auto_driver/send/speed', Int32 , queue_size=10)
    pub4 = rospy.Publisher('/auto_driver/send/gear', Int32 , queue_size=10)
    
    manul=0       # 0 - Automatic(自动); 1 - Manual (手动操控)
    speed=10      # SPEED (0～100之间的值)
    direction=30  # 0-LEFT-50-RIGHT-100 (0-49:左转，50:直行，51～100:右转)
    gear=1        # 1 - DRIVE, 2 - NEUTRAL, 3 - PARK, 4 - REVERSE
                  # 1:前进挡 2:空挡 3:停车挡 4:倒挡
    
    rospy.init_node('kinematicCtrl', anonymous=True)
    
    add_thread = threading.Thread(target = thread_job)
    
    add_thread.start()
    
    rate = rospy.Rate(8) # 8Hz
        
    #更新频率是1hz
    #rospy.loginfo(rospy.is_shutdown())
    
    pub1.publish(manul)
    while not rospy.is_shutdown():
        pub2.publish(direction)
        pub3.publish(speed)
        pub4.publish(gear)
        rate.sleep()

if __name__ == '__main__':
    kinematicCtrl()

