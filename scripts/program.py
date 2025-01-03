"""
Attempt at perfection
Isaiah Kenny
7/25/22

"""
import argparse
import numpy as np
import cv2
from skimage.metrics import structural_similarity
from pathlib import Path

gold = cv2.imread('pattern1.jpg',0)

students = []

directoryForResults = 'Documents/Results'
directory = 'Documents/Check'
files = Path(directory).glob('*')

for file in files:
    i = 0
    print(file)
    student = cv2.imread(str(file),0)
    #students.append(student)
    cv2.imshow('student drawing', student) #students[i]
    cv2.imshow('gold standard / ground truth', gold)
    
    edges0 = cv2.Canny(image=gold, threshold1=100, threshold2=200)
    edges1 = cv2.Canny(image=student, threshold1=100, threshold2=200) #students[i]

    (score, diff) = structural_similarity(gold, student, full=True) #students[i]
    print("Sim: {:.4f}%".format(score*100))
    diff = (diff*255).astype("uint8")
    diff_box = cv2.merge([diff,diff,diff])
    (score1, diff1) = structural_similarity(edges0, edges1, full=True)
    print("Edges Sim: {:.4f}%".format(score1*100))
    diff1 = (diff1*255).astype("uint8")
    diff1_box = cv2.merge([diff1,diff1,diff1])
    
    error = cv2.absdiff(gold, student) #students[i]
    error0 = cv2.multiply(error, 2)
    error1 = cv2.threshold(error, 50, 100, cv2.THRESH_BINARY)[1]

    cv2.imshow("Comparison", error)
    cv2.imshow("diff", diff)
    cv2.imshow("gain", error0)
    cv2.imshow("threshy", error1)
    cv2.imshow("Edge diff", diff1)

    #result = cv2.imwrite(str(Path(directoryForResults)) + student + '_reviewed.jpg', error)
    #student = ""
    #i+=1
    cv2.waitKey()

cv2.waitKey()
cv2.destroyAllWindows

