#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:15:07 2019
Computer Vision assignment 1
OCR recognition
@author: aj611
"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from PIL import Image
import os, os.path, glob


# Compute the features for all the characters present in all the files
files = glob.glob("?.bmp")
print(files)
Features = []
# stores the actual labels - a, d, f, ..z
Labels = []
file = open("maxc_minc_maxr_minr_values.txt", "w+")
# stores the means and standard deviations of the entire column vectors
Mean = []
SD = []
for j in files:
    # reading an image
    print(j)
    img = io.imread(j);
    
    print(img.shape)
    
    # visualizing an image
    
    '''
    io.imshow(img)
    plt.title('Original Image')
    io.show()
    '''
    
    # image histogram
    hist = exposure.histogram(img)
    '''
    plt.bar(hist[1], hist[0])
    plt.title('Histogram')
    plt.show()
    '''
    
    # binarization by thresholding
    th = 200
    img_binary = (img < th).astype(np.double)
    
    '''
    # display binary image
    io.imshow(img_binary)
    plt.title('Binary Image')
    io.show()
    '''
    
    '''
    hist1 = exposure.histogram(img_binary)
    plt.bar(hist1[1], hist1[0])
    plt.title('Histogram of the binary image')
    plt.show()
    '''
    
    
    # connected component analysis
    img_label = label(img_binary, background = 0)
    '''
    io.imshow(img_label)
    plt.title('Labeled Image')
    io.show()
    '''
    
    # print how many components are ther by max label value
    print(np.amax(img_label))
    
    # displaying component bounding boxes
    
    regions = regionprops(img_label)
    '''
    io.imshow(img_binary)
    ax = plt.gca()
    '''
    i = 0;
    
    
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        # removing small components
        #file.write("maxc - minc: ")
        #file.write(maxc - minc)
        #file.write("maxr - minr: ")
        #file.write(maxr - minr)
        if (maxc - minc) < 15 and (maxr - minr) < 15:
            i = i + 1
            #print("maxc - minc: ")
            #print(maxc - minc)
            #print("maxr - minr: ")
            #print(maxr - minr)
        else:
            #ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill = False, edgecolor='red', linewidth = 1))
            # Computing Hu moments
            roi = img_binary[minr:maxr, minc:maxc]
            m = moments(roi)
            cr = m[0, 1]/m[0, 0]
            cc = m[1, 0]/m[0, 0]
            mu = moments_central(roi, cr, cc)
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            Features.append(hu)
            Labels.append(j[:1])
    
    
    print("number of noise components: ")   
    print(i)
    #file.write("number of noise components: ")
    #file.write(i)

    Features_scaled = preprocessing.scale(Features)
    Mean = Features_scaled.mean(axis = 0)
    SD = Features_scaled.std(axis = 0)
    #np.transpose(Features)

    
    '''
    #ax.title('Bounding Boxes')
    ax.set_title('Bounding Boxes')
    io.show()
    '''



# storing features


# recognition of training data
D = cdist(Features, Features)
'''
io.imshow(D)
plt.title('Distance matrix')
io.show()
'''


# sort the columns of D along each row and find the index of the seconds smallest distance in the row
D_index = np.argsort(D, axis = 1)
D_index_transpose = np.transpose(D_index)

Ypred = []
Ytrue = []

for i in D_index_transpose[2]:
    Ypred.append(Labels[i])
    
Ytrue = Labels

confM = confusion_matrix(Ytrue, Ypred)

io.imshow(confM)
plt.title('confusion Matrix')
io.show()

### testing

# reading image and binarization
# extracting characters and their features
# enhancements

'''
# load ground truth locations and classes
pkl_file = open('test2_gt.pkl', 'rb')
mydict = pickle.load(pkl_file)
pkl_file.close()
classes = mydict['classes']
locations = mydict['locations']

# classes contain ground truth and locations contain their center coordinates
'''