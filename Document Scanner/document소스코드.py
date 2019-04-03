import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('receipt.jpg') 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, cv2.CV_8UC1) # gray scale image

blurred = cv2.GaussianBlur(gray, (11,11) ,0) # bluerring

edges = cv2.Canny(blurred,100,200) # edge detection 

blurred = cv2.GaussianBlur(edges, (11,11) ,0)

titles = ['gray'] # show image

images = [gray]

for i in range(1):
    plt.subplot(1,1,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show() # show image

titles = ['edges'] # show image

images = [blurred]

for i in range(1):
    plt.subplot(1,1,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show() # show image

(_,cnts,_) = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # findconturs

#approx_cnts = [] #conturs Approximation
#for cnt in cnts:
#    epsilon = 0.1*cv2.arcLength(cnt,True)
#    approx = cv2.approxPolyDP(cnt,epsilon,True)
#    approx_cnts.append(approx)

c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

#leftmost = tuple(cnt[cnt[:,:,0].argmin()][0]) # method1: find 4 point 
#rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
#topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
#bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

leftmost = tuple(box[0]) #method2: find 4 point
rightmost = tuple(box[1])
topmost = tuple(box[2])
bottommost = tuple(box[3])


cv2.circle(img, leftmost,5,(0,0,255),2) # 4 point draw
cv2.circle(img, rightmost,5,(0,0,255),-1)
cv2.circle(img, topmost,5,(0,0,255),-1)
cv2.circle(img, bottommost,5,(0,0,255),-1)

#cv2.drawContours(img, approx_cnts, -1, (0,255,0), 10) # contours draw
cv2.drawContours(img, [box], -1, (0,255,0), 10) # contours draw

rows,cols,_ = img.shape # perspective transform
#pts1 = np.float32([leftmost, topmost, bottommost, rightmost]) # case: methon 1 
pts1 = np.float32([box[1], box[2], box[0], box[3]]) # case: methon 2
pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(gray, M,(500, 500))


thresh = cv2.adaptiveThreshold(dst, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,9) # adaptive thresholding 

titles = ['img','perspective transform','adaptive thresholding'] # show image

images = [img,dst,thresh]

for i in range(3):
    plt.subplot(2,2,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show() # show image