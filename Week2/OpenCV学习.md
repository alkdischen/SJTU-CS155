---
title: OpenCV学习笔记
cover: https://raw.githubusercontent.com/AlkdisChen/PicGo/main/DarkSouls/DarkSouls%20(8).jpg
top_img: https://raw.githubusercontent.com/AlkdisChen/PicGo/main/DarkSouls/DarkSouls%20(8).jpg
abbrlink: 94d7160a
date: 2022-09-21 20:19:00
update: 2022-09-21 20:19:00
categories:
- CV
tags:
- 课程笔记
- SJTU-CS155
- OpenCV
---
# OpenCV学习笔记

# 安装

```python
pip install opencv-python
```

# <br>图片操作

## 读图

~~~python
import cv2
img = cv2.imread('url')
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows
~~~

<br>

## 图像特性

~~~python
img.shape
#(行，列，通道）
~~~

<br>

## 读取像素点

~~~python
img[100,200]
#输出[100,200]处[B,G,R]和dtype
#[行，列]  [y,x]
~~~

~~~python
print(img)
#以第一行开始，输出[B,G,R]
~~~

<br>

## 新建图片

~~~python
import cv2
import numpy as np
img_new = np.zeros((540,648,3),np.uint8)
#((高，宽，通道),dytype)
#((img.shape[0],img.shape[1],3),dtype)
cv2.imshow('img',img_new)
cv2.waitKey()
cv2.destroyAllWindows()
~~~

~~~python
#可以简化为
img_new = np.zeros_like(img,np.unit8)
~~~

<br>

## 绘画

~~~py
#画直线
cv2.line(img_new,(80,50),(800,500),(0,0,255),3)
#(image,起点，终点，颜色，粗细)
#line中点的坐标为[列，行] [x,y]，与img.shape中正好相反
~~~

~~~python
#画园
cv2.circle(img_new,(200,250),200,(255,0,0),5)
#（image,圆心，半径，颜色，粗细)
#如果粗细为负值，则为实心圆
~~~

~~~python
#画填充区域
vertices = np.array([[(0,540),(460,325),(520,325),(960,540)]])
#填充区域的四个顶点
cv2.fillPoly(img_new,vertices, (255,255,255) )
~~~

<br>

## 保存图片

~~~python
cv2.imwrite('url',img_new)
#return true
~~~

<br>

# 图像转换

## 灰值化

~~~python
#方法1

import cv2
import numpy as np
img = cv2.imread('url')
gray = np.zeros_like(img, np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        gray[i,j] = 0.11 * img[i,j][0] + 0.59 * img[i,j][1] + 0.3 * img[i,j][2]
        #对每一个像素点的GBR进行处理
cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()、
#自定义，但是更慢

#值得一提的是，此gray仍然是三通道
gray.shape()
~~~

~~~PYTHON
#方法2

import cv2
img = cv2.imread('url')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()
#运用函数更加便捷，更快速

#此gray为单通道
gray.shape()
~~~

~~~py
#方法3

import cv2
gray = cv2.imread('url', 0)
cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()
#直接应用灰值处理

#此gray为单通道
gray.shape()
~~~

<br>

## 二值化处理

二值化处理也即黑白化（黑-0，白-255）

~~~py
#方法1

reta, binary_a = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#100,255为阈值，前者越小，图片越明亮

reta, binary_b = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
#反转黑白

cv2.imshow('binary_a', binary_a)
cv2.waitKey()
cv2.destroyAllWindows()
~~~

~~~py
#方法2

binary = np.zeros_like(gray, np.uint8)
binary[ (gray >= 127) ] = 255 
#灰度大于127变为白色

cv2.imshow('binary', binary)
cv2.waitKey()
cv2.destroyAllWindows()

#binary为单通道
binary.shape
~~~

<br>

# 视频处理

## 读取视频

~~~py
import cv2
import numpy as np

cap = cv2.VideoCapture('url') #读文件

#'url'也可以替换为摄像头
#cap = cv2.VideoCapture(0)
#默认摄像头为0
ret = cap.isOpened()

while(ret):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('frame', frame)
        k = cv2.waitKey(100) 
        #wiatKet内为毫秒
        if (k == ord('q') ):
            break
            #输入q退出
cv2.waitKey()
cap.release()
cv2.destroyAllWindows
~~~

<br>

## 存储视频

~~~py
#存储视频文件
import cv2
import numpy as np

cap = cv2.VideoCapture('url') #读文件
ret = cap.isOpened()

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D') #MP4格式
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

out = cv2.VideoWriter('url', fourcc, 50, (int(w), int(h)))
#50为帧数

while(ret):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        
        #单通道无法写入，改成三通道
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out.write(gray)
        
        k = cv2.waitKey(100) 
        #wiatKet内为毫秒
        if (k == ord('q')):
            break
            #输入q退出
cv2.waitKey()
cap.release()
out.release()
cv2.destroyAllWindows
~~~

<br>



