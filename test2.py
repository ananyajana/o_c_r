#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 10:41:25 2019

@author: reshistory
"""

import numpy as np

features = numpy.array([[5, 5, 5], [6, 6, 6]])
print(features.shape)
print(features[0])

Mean = numpy.array([1, 2])
print(Mean.shape)

for i in range(0, 2):
    features[i] = features[i] - Mean[i]
    
print(features)