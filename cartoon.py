# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:08:08 2020

@author: SM51998
"""

import cv2
import numpy as np

# no of downsampling steps
num_down = 2
# no of bilateral filtering steps
num_bilateral = 7

img_rgb = cv2.imread('1.jpg')
#print(img_rgb.shape) 

#resizing s as to get optimal results after unsampling is done
img_rgb=cv2.resize(img_rgb,(800,800))

# downsample image using gaussian pyramid
img_color = img_rgb
for _ in range(num_down):
    img_color = cv2.pyrDown(img_color)
    
# repeatedly apply small bilateral filter instead of applying one large filter
for _ in range(num_bilateral):
    img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    
    
# upscale image to original size
for _ in range(num_down):
    img_color = cv2.pyrUp(img_color)
    
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)

img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY,
                                blockSize=9,
                                C=2)

# convert back to color, bit-AND with color image
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
img_cartoon = cv2.bitwise_and(img_color, img_edge)

#display
# cv2.imshow("cartoon",img_cartoon)
stack = np.hstack([img_rgb,img_cartoon])
cv2.imshow('Stacked Images',stack)