import cv2
import os
import numpy as np
from os import listdir
from os.path import join,isfile

path="/home/ipcvg/tensorflow_enet/main/"
path1="/home/ipcvg/tensorflow_enet/segmentation/motion12/"
path2="/home/ipcvg/tensorflow_enet/segmentation/concatinated_output/"
pathout="/home/ipcvg/tensorflow_enet/segmentation/result.avi"
onlyfiles = [ f for f in listdir(path) if isfile(join(path,f)) ]
onlyfiles.sort()
arr1=np.empty(len(onlyfiles),dtype="object")
arr2=np.empty(len(onlyfiles),dtype="object")
count=0
#######################################################################
for i in range(len(onlyfiles)):
    arr1[i]=cv2.imread(join(path,onlyfiles[i]))
onlyfiles = [ f for f in listdir(path1) if isfile(join(path1,f)) ]
onlyfiles.sort()
for i in range(len(onlyfiles)):
    arr2[i]=cv2.imread(join(path1,onlyfiles[i]))
m,n,l=arr2[0].shape
size=(2*m,n)
#######################################################################
fps=30
out=cv2.VideoWriter(pathout,cv2.VideoWriter_fourcc(*'DIVX'),fps,size)
for i in range(len(arr1)):
    x=arr1[i]
    y=arr2[i]
    arr3=np.concatenate((x,y),axis=1)
    cv2.imwrite(os.path.join(path2,"frame%08d.jpg"%count),arr3)
    count+=1
cv2.destroyAllWindows()    
    
