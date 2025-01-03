import numpy as np
import cv2
import skimage.segmentation
from skimage.metrics import structural_similarity
from matplotlib import pyplot as plt
import matplotlib.image as img

img = plt.imread("pattern1.jpg")
res = skimage.segmentation.felzenszwalb(img, scale=1000)
res1 = skimage.segmentation.felzenszwalb(img, scale=500)

graph = plt.imread(res)
plt.imshow("graph", graph)

cv2.imshow("graph", res)
cv2.imshow("other graph", res1)

fig = plt.figure(figsize=(12,5))
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
ax0.imshow(res); ax0.set_xlabel("10")
ax1.imshow(res1); ax1.set_xlabel("5")
fig.suptitle("graph")
plt.tight_layout()       

cv2.imshow("10",ax0)
cv2.imshow("5",ax1)

cv2.waitKey(0)
