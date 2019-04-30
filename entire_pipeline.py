import cv2
import os
from os import listdir
from os.path import isfile,join
import time as t
import numpy as np
import imutils
import pickle as cPickle
import tensorflow as tf

from utilities import label_img_to_color

from model import ENet_model

project_dir = "/home/ipcvg/tensorflow_enet/segmentation/"

data_dir = project_dir + "data/"

model_id = "sequence_run"

label=[]
batch_size = 1
img_height =512
img_width = 1024
count=0
x=0
z=0
y=0
alpha=0.1
beta = (1.0 - alpha)

model = ENet_model(model_id, img_height=img_height, img_width=img_width, batch_size=batch_size)

no_of_classes = model.no_of_classes

results_dir = model.project_dir + "motion14/"
#results_dir1=model.project_dir+"motion15/"
path=results_dir 

# load the mean color channels of the train imgs:
train_mean_channels = cPickle.load(open("data/mean_channels.pkl"))
################################FEATURE EXTRACTION#########################################
def feature(frame):
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges=cv2.Canny(frame,100,200) 
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 10  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50 # minimum number of pixels making up a line
        max_line_gap = 20 # maximum gap in pixels between connectable line segments
        line_image = np.copy(frame) * 0  # creating a blank to draw lines on
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)
        lines_edges = cv2.addWeighted(frame, 1, line_image, 1, 0)
        ret,th = cv2.threshold(gray,127,255, 0) # First obtain the threshold using the greyscale image

        _,contours,hierarchy = cv2.findContours(th,2,1) #--- Find all the contours in the binary image ---
        cnt = contours
        big_contour =[]
        max =0
        for i in cnt:
            area = cv2.contourArea(i) #--- find the contour having biggest area ---
            if(area > max):
                max = area
                big_contour = i 

        final = cv2.drawContours(lines_edges, big_contour, -1, (0,0,255), 2)
	return  final	
#####################################SEGMENTATION FUNCTION####################################################
def segmentation(img):
    no_of_frames =1
    no_of_batches = int(no_of_frames/batch_size)
    saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)       
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, project_dir + "training_logs/model_4/checkpoints/model_4_epoch_50.ckpt")
      #  print("weights loaded")
        step=0
        batch_pointer = 0
        for j in range(batch_size):
	    batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
	    img = cv2.resize(img, (img_width, img_height))
	    img = img - train_mean_channels
            img=np.expand_dims(img,axis=0)
 	    batch_imgs = img
	    batch_feed_dict = model.create_feed_dict(imgs_batch=batch_imgs,
			early_drop_prob=0.0, late_drop_prob=0.0)
	    logits = sess.run(model.logits, feed_dict=batch_feed_dict)
	    predictions = np.argmax(logits, axis=3)
            pred_img = predictions[j]
	    pred_img_color = label_img_to_color(pred_img)
            img = batch_imgs[j] + train_mean_channels
            #overlayed_img = 0.2*img + 0.8*pred_img_color
            overlayed_img=1*pred_img_color
    return[overlayed_img,pred_img_color]  
#######################################IMAGE WARPING ######################################################
def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res 
################################################LABEL TRACKING#####################################################
def label_tracking(img,previous,seg):
    frame1=np.copy(previous)
    frame2=np.copy(img)
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    next1 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    res=warp_flow(seg,flow) 
    return res
#####################################TEMPORAL CONSISTENCY###############################################################
start=t.time()
cap = cv2.VideoCapture('input_videos/cut4.mp4')
allframes=[]
lines=[]
angles=[]
ret,frame=cap.read()
first=1
count=0
srikar=0
images=[]
while(ret):
    if frame is None:
	break
    else:
        allframes.append(frame)
    ret,frame=cap.read()
for i in range(len(allframes)):
    frame=allframes[i]
    if(i==0):
	std=feature(frame)
	list=segmentation(frame)   #####call segmentation funcrion
	segimg=list[0] 
	pred_img_color=list[1]
	cv2.imwrite(os.path.join(path , "frame%08d.jpg" % count),segimg)
	track=pred_img_color
	count+=1
        previous=np.copy(frame)
    else:
	frame2=feature(frame)
	mean1=np.std(std)
	mean2=np.std(frame2)
	if(abs(mean2-mean1)>1.2):
           srikar+=1
           list=segmentation(frame)   #####call segmentation funcrion
           segimg=list[0] 
           pred_img_color=list[1]
           cv2.imwrite(os.path.join(path , "frame%08d.jpg" % count),segimg)
           track=pred_img_color
           count+=1
           std=frame2  
           previous=np.copy(frame)
	else:
		frame=cv2.resize(frame,(img_width,img_height),interpolation=cv2.INTER_AREA)
		previous=cv2.resize(previous,(img_width,img_height),interpolation=cv2.INTER_AREA)
		track_img=label_tracking(frame,previous,track)
		#track_img1=0.2*frame+0.8*track_img
		track_img1=1*track_img
                #cv2.imwrite(os.path.join(path1 , "frame%08d.jpg" % count),track_img1)
                out = cv2.addWeighted(segimg,alpha,track_img1, beta, 0.0)
		cv2.imwrite(os.path.join(path , "frame%08d.jpg" % count),out)
		count+=1
		track=track_img
                segimg=out
                previous=np.copy(frame)
		
    
end=t.time()
print(srikar)
print('time consumed: {:.0f}m {:.0f}s '.format((end-start)//60,(end-start)%60))      
