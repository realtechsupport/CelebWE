#!/usr/bin/env python
from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import scipy
import csv
import scipy.misc as sm
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#get attributes
attrvalues = {}
X = []
with open('list_attr_celeba.txt') as f:
    numimages = int(f.readline())
    attrnames = f.readline().split()
    for line in f:
        tokens = line.strip().split()
        attrvalues[tokens[0]] = np.array([int(a) for a in tokens[1:]])

# pick a random image and get its features
img_features = []
sampleid = np.random.permutation(numimages)[0]
attrs = attrvalues[str(sampleid).zfill(6)+'.jpg']
print()
print('random image attributes are:')
for a in np.where(attrs == 1)[0]:
    img_features.append(attrnames[a])
print(img_features)

#pick a random rule
numfeatures = np.random.permutation(5)[0] + 1
selfeatures = np.random.permutation(40)[0:numfeatures]
print()
print("Here is a randomly selected rule:")
print([attrnames[i] for i in selfeatures])

#place the randomly selected features in the 40 feature vector
selfeaturesvec = -np.ones([40,])
for i in selfeatures:
    selfeaturesvec[i] = 1
selfeaturesvec = selfeaturesvec.astype(int)
print()
print('randomly selected features (1) in the 40 feature vector of (-1)')
print(selfeaturesvec)

# create labels
labelsall = []
X = []
for k in attrvalues.keys():
    if all(attrvalues[k][selfeatures] == 1):
        labelsall.append(1)
    else:
        labelsall.append(-1)

#show the images
location = 'home/test/'
fname = location+str(sampleid).zfill(6)+'.jpg'
img = sm.imread(fname)
plt.imshow(img)
plt.show()
