import cv2

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


img=cv2.imread("1.jpg")
ret=feature(frame)
std=ret
