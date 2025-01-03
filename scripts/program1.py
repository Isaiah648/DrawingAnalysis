"""
Better attempt at perfection
Isaiah Kenny
7/29/22

"""
import argparse
import numpy as np
import cv2
from skimage.metrics import structural_similarity
from skimage import measure
from skimage import feature
from skimage import transform
from skimage.filters import threshold_otsu
import skimage.segmentation
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as img
import imutils
from imutils import contours

# pip install numpy opencv-python scikit-image matplotlib
# --user for root

gold = cv2.imread('pattern1.jpg', 0)

width = 706
height = gold.shape[0]
dimension = (width, height)

students = []
scores = []

directoryForResults = 'Documents/Results'
directory = 'Documents/Check'
files = Path(directory).glob('*')

for file in files:
    i = -1
    print(file)
    name = file
    student = cv2.imread(str(file),0)
    student = cv2.resize(student, dimension, interpolation=cv2.INTER_AREA)
    students.append(student)
    cv2.imshow('student drawing', students[i]) #students[i]
    cv2.imshow('gold standard / ground truth', gold)

    edges0 = cv2.Canny(image=gold, threshold1=100, threshold2=200)
    edges1 = cv2.Canny(image=students[i], threshold1=100, threshold2=200)

    #kernel0 = np.ones((3,3), np.uint8)
    #edges0 = cv2.morphologyEx(edges0, cv2.MORPH_CLOSE, kernel0, iterations=2)
    #edges1 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel0, iterations=2)
    #kernel00 = np.ones((3,3), np.uint8)
    #edges0 = cv2.dilate(edges0, kernel00, iterations=2)
    #edges1 = cv2.dilate(edges1, kernel00, iterations=2)

    (score, diff) = structural_similarity(gold, students[i], full=True) #students[i]
    print("Sim: {:.4f}%".format(score*100))
    diff = (diff*255).astype("uint8")
    diff_box = cv2.merge([diff,diff,diff])
    (score1, diff1) = structural_similarity(edges0, edges1, full=True)
    print("edges Sim: {:.4f}%".format(score1*100))
    diff1 = (diff1*255).astype("uint8")
    diff1_box = cv2.merge([diff1,diff1,diff1])

    scores.append(name.stem + str(score1))
 
    error = cv2.absdiff(gold, students[i])
    error0 = cv2.multiply(error, 2)

    threshold = 0.40
    low = np.quantile(students[i], threshold)
    ret, error1 = cv2.threshold(students[i], low, 255, cv2.THRESH_BINARY)
    ret, error11 = cv2.threshold(gold, low, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    error1 = cv2.morphologyEx(error1, cv2.MORPH_CLOSE, kernel, iterations=3) #cv2.MORPH_OPEN iterations=3
    error11 = cv2.morphologyEx(error1, cv2.MORPH_CLOSE, kernel, iterations=3)
    kernel_ = np.ones((5,5),np.uint8)
    thresh = cv2.dilate(error1, kernel_, iterations=2)
    cv2.imshow("thresh",thresh)

    contours1, hierarchy=cv2.findContours(error1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours0, hierarchy=cv2.findContours(error11,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #imager = diff1.copy()
    #for i in range(len(contours1)):
    #    color=(0,255,0)
    #    cv2.drawContours(imager, contours1, i, color, 1, 8, hierarchy)
    #yea = error.copy()
    #cv2.imshow("1", imager)
    #cv2.drawContours(yea, contours1, -1,(0,255,0),3)
    #yea = cv2.absdiff(contours0, contours1)
    #cv2.imshow("real", yea)

    #imutilsedge = imutils.auto_canny(diff1)
    #cv2.imshow("ImutilsEdge", imutilsedge)
    
    cv2.imshow("Comparison", error)
    cv2.imshow("diff", diff)
    cv2.imshow("gain", error0)
    #cv2.imshow("threshy", error1)
    cv2.imshow("Edge diff", diff1)

    imaging = diff1.copy()
    contours, hierarchy = cv2.findContours(diff1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(diff1, contours, -1, (0,255,0))
    #contour = contours[0]
    #shape = 0.1*cv2.arcLength(contour, True)
    #improvedShape = cv2.approxPolyDP(contour, shape, True)
    #contours2, hierarchy = cv2.findContours(diff1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours2 = imutils.grab_contours(contours2)
    #c = max(contours, key=cv2.contourArea)
    #imaging = diff1.copy()
    cv2.drawContours(imaging, contours, -1, (0,255,0), 1)
    #(x,y,w,h) = cv2.boundingRect(contours2[0])
    cv2.imshow("Genius", imaging)

   #plot = img.imread(error)
   #plt.imshow(plot)
   #cv2.waitKey(0)
   
    cv2.imwrite(str(Path(directoryForResults)) + "/" + name.stem + "_image.jpg", students[i])
    cv2.imwrite(str(Path(directoryForResults)) + "/" + name.stem + "_groundTruth.jpg", gold)
    cv2.imwrite(str(Path(directoryForResults)) + "/"  + name.stem + '_comparison.jpg', diff1)
    cv2.imwrite(str(Path(directoryForResults))+ "/" +  name.stem + "_absdiff.jpg", error0)

    threshsci = threshold_otsu(gold)
    threshsci1 = threshold_otsu(student[i])
    cv2.imshow("wau", threshsci)
    cv2.imshow("waathea", threshsci1)
    #diffy = cv2.absdiff(cv2.imread(str(threshsci)), cv2.imread(str(threshsci1)))
    #yes = plt.imread(threshsci)
    #yes2 = plt.imread(threshsci1)
    

    """
    student1 = transform.resize(student[i], output_shape=(706,1024))
    gold1 = transform.resize(gold, output_shape=(706,1024))
    otsu_threshold, image_result = cv2.threshold(student1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )
    otsu1_threshold, image_result1 = cv2.threshold(gold1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )

    cv2.imshow("stud", image_result)
    cv2.imshow("gol", image_result1)
    yay = cv2.absdiff(image_result, image_result1)
    cv2.imshow("comp", yay)
    """
    """ 
    pcituresss = feature.canny(diff1, sigma=3)
    fig, ax = plt.subplots()
    ax.imshow(pcituresss)
    plt.show()
    """
    contoursgray = measure.find_contours(diff1, 0.8)
    fig, ax = plt.subplots()
    ax.imshow(diff1, cmap=plt.cm.gray)
    for contour in contoursgray:
        ax.plot(contour[:,1], contour[:,0], linewidth=2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
    #plot = img.imread(diff1)
    #plt.rcParams['axes.facecolor'] = 'silver'
   # plotted = plt.imshow(diff1)
    #plt.rcParams['axes.facecolor'] = 'xkcd:silver'
    #plt.plot(diff1)
   # plt.show()
    #cv2.imread(str(diff1))
    #cv2.imwrite(str(Path(directoryForResults)) + "/" + name.stem + "_plot.jpg", plotted)

    student = ""
    i+=1
    cv2.waitKey(0)
    cv2.destroyAllWindows

cv2.imwrite(str(Path(directoryForResults)) + "Scores.png", scores)
cv2.waitKey()
cv2.destroyAllWindows

