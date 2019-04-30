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


batch_size = 1
img_height = 512
img_width = 1024
count=0
x=0
z=0
y=0
model = ENet_model(model_id, img_height=img_height, img_width=img_width, batch_size=batch_size)

no_of_classes = model.no_of_classes

results_dir = model.project_dir + "motion_estimation_result/"
path=results_dir 

# load the mean color channels of the train imgs:
train_mean_channels = cPickle.load(open("data/mean_channels.pkl"))



#####################################SEGMENTATION FUNCTION####################################################
def segmentation(img):
    no_of_frames =1
    print("number of frames")
    print(no_of_frames)
    no_of_batches = int(no_of_frames/batch_size)
    print("number of batches")
    print(no_of_batches)


    
    saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)       
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, project_dir + "training_logs/model_4/checkpoints/model_4_epoch_50.ckpt")
        print("weights loaded")
        step=0
        batch_pointer = 0
        start=t.time()
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
            overlayed_img = 0.0*img + 1.0*pred_img_color
    return(overlayed_img)  
################################################LABEL TRACKING#####################################################
def label_tracking(img,previous,seg):
    frame1=np.copy(previous)
    frame2=np.copy(img)
    prvs = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame2)
    hsv[...,1] = 255
    next1 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    fram3=np.array(seg)
    m,n=prvs.shape
    i=0
    j=0
    p=0
    q=0
    for i in range(m):
        for j in range(n):
            p=int(round(i+(flow[...,0][i][j])))
            q=int(round(j+(flow[...,1][i][j])))
            if(p>=m or p<0):
                break
            if(q>=n or q<0):
                break
            fram3[i,j]=seg[p,q]
           
    return fram3
#####################################VIDEO READ###############################################################
count=0
cap=cv2.VideoCapture('/home/ipcvg/tensorflow_enet/test1.mp4')
start=t.time()
#all_frames=np.empty(dtype=object)
all_frames=[]
while(cap.isOpened):
	ret,frame=cap.read()
        if frame is None:
                break
        else:
                all_frames.append(frame)
###############################TEMPORAL CONSISTENCY##########################################################
dim=(600,600)
first=0
count2=0
count=0
previous=np.empty(frame)
fgbg=cv2.bgsegm.createBackgroundSubtractorMOG()
for i in range(len(all_frames)):
    frame=all_frames[i]
    if (first==0):
	segimg=segmentation(frame)####call segmentation function here
        first=first+1
	cv2.imwrite(os.path.join(path , "frame%08d.jpg" % count),segimg)
	count=count+1
	previous=np.copy(frame)
        track=segimg
    rframe=cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
    frame1=fgbg.apply(rframe)
    if frame1 is None:
        print('null')
    else:

        kernel = np.ones((5,5),np.uint8)
        frame1=cv2.erode(frame1,kernel,iterations=1) #Marphological operation to decrese the whitenoise in the frame
        kernel = np.ones((3,3),np.uint8)
        frame1=cv2.dilate(frame1,kernel,iterations=1) #Marphological operation to restore the shap of the objects present in the frame
        #Function which finds contours present in the frame
        cnts = cv2.findContours(frame1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]  
        detection_count=0 #Counter to count number of objects prsent in the given frame
        for c in cnts:
            if cv2.contourArea(c)<4000: #Sensitivity measure  to ignore small objects.Increaing this value decreases the number of objects identified 
                continue
            else :
                detection_count=detection_count+cv2.contourArea(c)
            (x,y,w,h) = cv2.boundingRect(c)
                    
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)


            cv2.drawContours(frame1,[box],0,(255,0,0),2 )#drawing contours on the background subtracted frame,after all the above operations were done


        if detection_count>800: #Retaining the frame if the number of objects identified is greater than the specified number i'e is  0 in this case
            segimg=segmentation(frame)   #####call segmentation funcrion
            print('sccc',count2)
            previous=np.copy(frame)
            cv2.imwrite(os.path.join(path , "frame%08d.jpg" % count),segimg)
            track=segimg
            count+=1

        else:
            frame=cv2.resize(frame,(img_width,img_height),interpolation=cv2.INTER_AREA)
           # previous=cv2.resize(previous,(img_width,img_height),interpolation=cv2.INTER_AREA)
            track_img=label_tracking(frame,previous,track)
            cv2.imwrite(os.path.join(path , "frame%08d.jpg" % count),track_img)
            count+=1
           # track=track_img
        count2=count2+1
      
end=t.time()
print('time consumed: {:.0f}m {:.0f}s '.format((end-start)//60,(end-start)%60))      
