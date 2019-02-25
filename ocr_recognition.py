#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:15:07 2019
Computer Vision assignment 1
OCR recognition
@author: aj611
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickel



# reading an image
img = io.imread('a.bmp');

print(img.shape)

# visualizing an image


io.imshow(img)
plt.title('Original Image')
io.show()


# image histogram
hist = exposure(img)

plt.bar(hist[1]. hist[0])
plt.title('Histogram')
plt.show()


# binarization by thresholding
th = 200
img_binary = (img < th).astype(np.double)

# display binary image
io.imshow(img_binary)
plt.title('Binayr Image')
io.show()


# connected component analysis
img_label = label(img_binary, background = 0)
io.imshow(img_label)
plt.title('Labeled Image')
io.show()

# print how many components are ther by max label value
print(np.amax(img_label))

# displaying component bounding boxes
i
regions = regionprops(img_label)
io.imshow()
ax = plt.gca()

for props in regions:
    minr, minc, maxr, maxc = props.bbox
    ax.add_patch(Rectange((minc, minr), maxc - minc, maxr - minr, fill = False, edgecolor='red', linewidth = 1))
   
ax.title('Bounding Boxes')
io.show()

# Computing Hu moments and removing small components
roi = img_binary[minr:maxr, minc:maxc]
m = moments(roi)
cr = m[0, 1]/m[0, 0]
cc = m[1, 0]/m[0, 0]
mu = moments_central(roi, cs, cc)
nu = moments_normalized(mu)
hu = moments_hu(nu)

# storing features
Features = []i
Features.append(hu)

# recognition of training data
D = cdist(Features, Features)
io.imshow()
plt.title('Distance matrix')
io.show()


# sort the columns of D along each row and find the index of the seconds smallest distance in the row
D_index = np.argsort(dist, axis = 1)

# confusion matrix(Ytrue, Ypred)

io.imshow(confM)
plt.title('confusion Matrix')
io.show()

### testing

# reading image and binarization
# extracting characters and their features
# enhancements


# load ground truth locations and classes
pkl_file = open('test2_gt.pkl', 'rb')
mydict = pickle.load(pkl_file)
pkl_file.close()
classes = mydict['classes']
locations = mydict['locations']

# classes contain ground truth and locations contain their center coordinates