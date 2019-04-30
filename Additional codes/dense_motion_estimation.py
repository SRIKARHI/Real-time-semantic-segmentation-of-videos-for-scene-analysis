import cv2
import numpy as np

frame1=cv2.imread("frame00000031.jpg")
frame2=cv2.imread("frame00000032,jpg")

x=[]
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
next1 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(prvs,next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
fram3=np.array(prvs)
m,n=prvs.shape
i=0
j=0
p=0
q=0
prev_p=0
prev_q=0
for i in range(m):
    #p=int((i*np.cos(hsv[...,0][i][j]))-(j*np.sin(hsv[...,0][i][j])))
    for j in range(n):
        p=i+round(flow[...,0][i][j])
        #p=round(i+(np.cos(ang[i][j])))
        p=int(p)
        if(p>=m):
            break
        if(p<0):
            break
        q=int(j+(round(flow[...,1][i][j])))
        #q=int(round(j+(mag[i][j]*np.sin(ang[i][j]))))
       # q=int((i*np.sin(ang[i][j]))-(j*np.cos(ang[i][j])))
        if(q>=n):
            break
        if(q<0):
            break
        if(prev_p!=p and prev_q!=q):
            fram3[p,q]=prvs[i,j]
        #fram3[i,j]=prvs[i,j]
            prev_p=p
            prev_q=q
      
        #print(i,j,p,q)

#bicubic_img = cv2.resize(fram3,None, fx =5, fy = 5, interpolation = cv2.INTER_CUBIC)
        
        
'''m,n=prvs.shape
j=0
for i in range(m):
    x=(ang[i][j])*180/np.pi
    p=int(i+(mag[i][j]*ang[i][j]))
    if(p>=m):
            break
    if(p<0):
        continue
    for j in range(n):
        x=(ang[i][j])*180/np.pi
        q=int(j+(mag[i][j]*ang[i][j]))
        if(q>=n):
            break
        if(q<0):
            continue
        fram3[p,q]=prvs[i,j]
        
        print(i,j,p,q)
    j=0
'''
#cv2.imshow("Interpolation",bicubic_img)
cv2.imshow("Frame3",fram3)
cv2.imshow("frame1",frame1)
cv2.imshow("frame2",frame2)
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()