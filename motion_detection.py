# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 08:00:18 2019

@author: Smita
"""

import cv2
import numpy as np
import math

import time

k = 6 
size = 1
thre = 5

def arrowdraw (img, x1, y1, x2, y2) :
    radians = math.atan2(x1 - x2, y2 - y1)
    x11 = 0
    y11 = 0
    x12 = -2
    y12 = -2
    u11 = 0
    v11 = 0
    u12 = 2
    v12 = -2
    x11_ = x11 * math.cos(radians) - y11 * math.sin(radians) + x2
    y11_ = x11 * math.sin(radians) + y11 * math.cos(radians) + y2
    x12_ = x12 * math.cos(radians) - y12 * math.sin(radians) + x2
    y12_ = x12 * math.sin(radians) + y12 * math.cos(radians) + y2
    u11_ = u11 * math.cos(radians) - v11 * math.sin(radians) + x2
    v11_ = u11 * math.sin(radians) + v11 * math.cos(radians) + y2
    u12_ = u12 * math.cos(radians) - v12 * math.sin(radians) + x2
    v12_ = u12 * math.sin(radians) + v12 * math.cos(radians) + y2
    img = cv2.line(img, (x1, y1) ,(x2, y2) ,(255, 255, 255 ), 1)
    img = cv2.line(img, (int(x11_) ,int(y11_)), (int(x12_) ,int(y12_)), (255, 255, 255), 1)
    img = cv2.line(img, (int(u11_), int(v11_)), (int(u12_), int(v12_)) , (255, 255, 255), 1)
    
    return img

def visualizeBoundary(img, frame):
    
    kernel = np.ones((10,10),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    (_, cnts, _) = cv2.findContours(erosion.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        rect = cv2.boundingRect(c)
        if rect[2] <= 2 * k + 1 or rect[3] < 2 * k + 1: continue
        
        x,y,w,h = rect
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    return frame

def detectMotionVector(pre, cur) : 
    
    start = time.time()
    vecs = []
    
    h = pre.shape[1]
    w = pre.shape[0]
    
    for x in range(k, w - k - 1, 2 * k + 1):
        for y in range(k, h - k - 1, 2 * k + 1):
            
            x0 = -1
            y0 = -1
                        
            minVal = 255 * 255 * k * k
            orinVal = 0
            for x_ in range(x - size, x + size):
                for y_ in range(y - size, y + size):
                    
                    if (x_ < k or x_ >= w - k or y_ < k or y_ >= h - k):
                        continue
                    
                    sum2 = 0
                    
                    for i in range(-k, k):
                        for j in range(-k, k):
                            r = cur[x + i, y + j, 0]
                            g = cur[x + i, y + j, 1]
                            b = cur[x + i, y + j, 2]
                            r_ = pre[x_ + i, y_ + j, 0]
                            g_ = pre[x_ + i, y_ + j, 1]
                            b_ = pre[x_ + i, y_ + j, 2]
                            
                            sum2 = sum2 + (r - r_) * (r - r_) + (g - g_) * (g - g_) + (b - b_) * (b - b_)
                    
                    if sum2 < minVal:
                        minVal = sum2
                        x0 = x_
                        y0 = y_
                    
                    if (x == x_ and y == y_):
                        orinVal = sum2
            
            if minVal == orinVal or minVal < thre * thre * 3 * k * k:
                x0 = x
                y0 = y
                            
            vec = [x, y, x0, y0]
            vecs.append(vec)
    
    end = time.time()
    print(end - start)
    return vecs

cap = cv2.VideoCapture('cityscapes.mp4')

if (cap.isOpened() == False):
    print("Error opening video stream or file")
    
prevFrame = []

isPrevDefined = False

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        
        if isPrevDefined == False:
            prevFrame = frame
            isPrevDefined = True
            continue
            
        vectors = detectMotionVector(prevFrame, frame)
        rlt = frame.copy()
        rlt_bianry = np.zeros((frame.shape[0], frame.shape[1], 1), dtype = "uint8")
        
        for vec in vectors:
            if (vec[1] == vec[3] and vec[0] == vec[2]):
                continue
            rlt = arrowdraw(rlt, vec[1], vec[0], vec[3], vec[2])
            cv2.rectangle(rlt_bianry, (vec[3] - k, vec[2] - k), (vec[3] + k, vec[2] + k), (255, 255, 255), -1)
        
        prevFrame = frame
        
        rlt2 = visualizeBoundary(rlt_bianry, rlt)
        cv2.imshow('Mod', rlt)
        
        prevFrame = frame
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else:
        break

cap.release()

cv2.destroyAllWindows()
'''
first = cv2.imread("assets/11.jpg")
second = cv2.imread("assets/12.jpg")
vectors = detectMotionVector(first, second)
rlt = second
for vec in vectors:
    if vec[1] == vec[3] and vec[0] == vec[2]:
        continue
    rlt = arrowdraw(rlt, vec[1], vec[0], vec[3], vec[2])
cv2.imshow('first', first)    
cv2.imshow('second', rlt)
cv2.waitKey(0)
'''
