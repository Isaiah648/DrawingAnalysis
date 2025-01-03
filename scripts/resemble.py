import cv2
import imutils
from imutils import contours
import numpy as np
from skimage.metrics import structural_similarity
import skimage.segmentation

img = cv2.imread('pattern1.jpg')
img2 = cv2.imread('pattern2(1).jpg')
img3 = cv2.imread('y.jpg')

#img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

#img2 = cv2.detailEnhance(img2, sigma_s=10, sigma_r=0.15)

#opp = cv2.absdiff(before_gray, after_gray)
#cv2.imshow(opp)

edges = cv2.Canny(image=img2, threshold1=100, threshold2=200)
edges2 = cv2.Canny(image=img, threshold1=100, threshold2=200)

#w = 200
#h = 200
#up = (h,w)
#resized = cv2.resize(img, up, interpolation = cv2.INTER_LINEAR)

#w1 = 200
#h1 = 200
#up1 = (h1,w1)
#resized1 = cv2.resize(img2, up1, interpolation = cv2.INTER_LINEAR)

#if height > height1 and width > width1:
#    resized = cv2.resize(img, width1, height1, interpolation = cv2.INTER_AREA)
#else:
#    resized = cv2.resize(img2, width, height, interpolation = cv2.INTER_AREA)

before_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

opp = cv2.absdiff(before_gray, after_gray)
cv2.imshow("absolute", opp)

gray0 = cv2.GaussianBlur(before_gray, (7,7), 0)
gray1 = cv2.GaussianBlur(after_gray, (7,7), 0)
edges00 = cv2.Canny(gray0, 50, 100)
edges01 = cv2.Canny(gray1, 50, 100)
edges00 = cv2.dilate(edges00, None, iterations=1)
edges01 = cv2.dilate(edges01, None, iterations=1)
edges00 = cv2.erode(edges00, None, iterations=1)
edges01 = cv2.erode(edges01, None, iterations=1)

contours00 = cv2.findContours(edges00.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours00 = imutils.grab_contours(contours00)

contours01 = cv2.findContours(edges01.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours01 = imutils.grab_contours(contours01)

(score, diff) = structural_similarity(edges00, edges01, full=True)
print("Edges Sim: {:.4f}%".format(score*100))
diff = (diff*255).astype("uint8")
diff_box = cv2.merge([diff,diff,diff])

(score1, diff1) = structural_similarity(before_gray, after_gray, full=True)
print("Gray-Scale Sim: {:.4f}%".format(score1*100))
diff1 = (diff1*255).astype("uint8")
diff1_box = cv2.merge([diff1,diff1,diff1])

mask = np.zeros(img.shape, dtype='uint8')
maskimg = img2.copy()

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2. THRESH_OTSU)[1]
contours0 = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours0 = imutils.grab_contours(contours0)
for i in contours0:
    (a,b,c,d) = cv2.boundingRect(i)
    cv2.rectangle(before_gray, (a,b), (a+c, b+d), (0,0,255),2)
    cv2.rectangle(after_gray, (a,b), (a+c, b+d), (0,0,255),2)
    cv2.drawContours(mask, [i], 0, (255,255,255),-1)
    cv2.drawContours(maskimg, [i], 0, (0,255,0), -1)

cv2.imshow("threshy", thresh)
#contours0 = contours0[0] if len(contours0) == 2 else contours0[1]
#for i in contours0:
#    area = cv2.contourArea(i)

cv2.imshow

(score2, diff2) = structural_similarity(gray0, gray1, full=True)
print("Gray Sim: {:.4f}%".format(score2*100))
diff2 = (diff2*255).astype("uint8")
diff2_box = cv2.merge([diff2,diff2,diff2])

rect = 5
rect1 = 40

cv2.imshow('image 1', img)
cv2.imshow('image 2', img2)
cv2.imshow('gray-scale image 1', before_gray)
cv2.imshow('gray-scale', after_gray)
cv2.imshow('difference', diff)
cv2.imshow('difference1', diff1)
cv2.imshow('difference2', diff2)
cv2.imshow('region of image', img2[rect:rect+25,rect1:rect1+25])
#cv2.rectangle(img,(284,0)(300,128))
cv2.imshow('edges', edges)
cv2.imshow('edges2', edges2)
cv2.imshow('edges00', edges00)
cv2.imshow('edges01', edges01)

ret, thresh0 = cv2.threshold(img2, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('binary', thresh0)
ret, thresh1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('binary1', thresh1)

#(score3, diff3) = structural_similarity(thresh, thresh1, full=True)
#print("Sim: {:.4f}%".format(score3*100))
#diff3 = (diff3*255).astype("uint8")
#diff3_box = cv2.merge([diff3, diff3, diff3])
#cv2.imshow('difference3', diff3)

contours, hierarchy = cv2.findContours(image=diff, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
image2_copy = img.copy()
cv2.drawContours(image=image2_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)



cv2.imshow("contours", image2_copy)
cv2.waitKey(0)
cv2.imwrite('contours_img', image2_copy)

cv2.waitKey()


