import numpy as np
import scipy.ndimage as ndimage
import cv2
img = cv2.imread('1.jpg')
# Note the 0 sigma for the last axis, we don't wan't to blurr the color planes together!
img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
cv2.imwrite("out.jpg",img)
