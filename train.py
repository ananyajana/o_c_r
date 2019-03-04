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
import codecs


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

def train():
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
        
        io.imshow(img_binary)
        ax = plt.gca()
       
        i = 0;
        
        
        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            # removing small components
            #file.write("maxc - minc: ")
            #file.write(maxc - minc)
            #file.write("maxr - minr: ")
            #file.write(maxr - minr)
            if (maxc - minc) < 7 and (maxr - minr) < 7:
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

train()

Features_scaled = preprocessing.scale(Features)
Features_mod = np.asarray(Features)
Mean = Features_mod.mean(axis = 0)
SD = Features_mod.std(axis = 0)
print(Mean)
print(SD)
    #np.transpose(Features)

    
'''
#ax.title('Bounding Boxes')
ax.set_title('Bounding Boxes')
io.show()
'''



# storing features


# recognition of training data
D = cdist(Features_scaled, Features_scaled)

io.imshow(D)
plt.title('Distance matrix')
io.show()



# sort the columns of D along each row and find the index of the seconds smallest distance in the row
D_index = np.argsort(D, axis = 1)
D_index_transpose = np.transpose(D_index)
Ypred = []
Ytrue = []

for i in D_index_transpose[2]:
    Ypred.append(Labels[i])
    
Ytrue = Labels
confM = confusion_matrix(Ytrue, Ypred, labels=['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z'])
'''
io.imshow(confM)
plt.title('confusion Matrix')
io.show()
'''
### testing

# reading image and binarization
# extracting characters and their features
# enhancements



test_files = glob.glob("test1.bmp")
print(test_files)
test_features = []
# stores the actual labels - a, d, f, ..z
test_labels = []

def test():
        for j in test_files:
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
                if (maxc - minc) < 1 and (maxr - minr) < 1:
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
                    test_features.append(hu)
                    test_labels.append(j[:1])
            
            
            print("number of noise components: ")   
            print(i)
        #file.write("number of noise components: ")
        #file.write(i)
test()

# subtract mean and divide with standard deviation to make sure 0 mean and unit variance
test_features_mod = np.asarray(test_features)
test_features_transpose = np.transpose(test_features_mod)
test_features_transpose_new = [] 
'''
#ax.title('Bounding Boxes')
ax.set_title('Bounding Boxes')
io.show()
'''

for i in range(0, 7):
    #print("Before modification\n")
    #print(test_features_transpose[i])
    print("hello")
    test_features_transpose_new.append((test_features_transpose[i] - Mean[i])/SD[i])
    #print("After modification\n")
    #print(test_features_transpose[i])
# storing features
test_features_transpose_mod = np.asarray(test_features_transpose_new)
test_features_scaled = np.transpose(test_features_transpose_mod)

# recognition of test data
D1 = cdist(test_features_scaled, Features_scaled)
'''
io.imshow(D1)
plt.title('Distance matrix')
io.show()
'''

# sort the columns of D1 along each row and find the index of the smallest distance in the row
D1_index = np.argsort(D1, axis = 1)
D1_index_transpose = np.transpose(D1_index)

Ypred_test = []
Ytrue_test1 = []
Ytrue_test2 = []

for i in D1_index_transpose[1]:
    Ypred_test.append(Labels[i])
 
'''
# load ground truth locations and classes
pkl_file1 = open('test2_gt.pkl', 'rb')
mydict = pickle.load(pkl_file1, encoding="utf-8")
pkl_file1.close()
classes = mydict['classes']
locations = mydict['locations']
print(classes)
print(locations)
# classes contain ground truth and locations contain their center coordinates
'''
Ytrue_test2 = ['f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
               'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h',
               'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k',
               's', 's', 's', 's', 's', 's', 's', 's',
               'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
               'z', 'z', 'z', 'z', 'z', 'z', 'z', 'z',
               'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
               'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o',
               'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w',
               'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm']

Ytrue_test1 = ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
               'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd',
               'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm',
               'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n',
               'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o',
               'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p',
               'q', 'q', 'q', 'q', 'q', 'q', 'q', 'q',
               'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
               'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u',
               'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w']


#Ytrue_test = Ytrue_test1
print("Ytrue_test is")
print(Ytrue_test1)
print("Ypred_test is")
print(Ypred_test)
             
i = 0
for j in range(0, 80):
    if Ypred_test[j] == Ytrue_test1[j]:
        i = i + 1

print("count is")
print(i)

confM1 = confusion_matrix(Ytrue_test1, Ypred_test, labels=['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z'])

io.imshow(confM1)
plt.title('confusion Matrix')
io.show()
