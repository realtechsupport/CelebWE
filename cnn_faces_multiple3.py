#!/usr/bin/env python
#-------------------------------------------------------------------------------
# cnn_faces_multiple3.py
# logfile naming convention:
# cnn: convoluttional neural network
# 32pix: resized image_size
# 10k: number of images used (training and testing together)
# 2convpass: number of times the image is run through the convolution and pooling block
# include the training size
# > change the name of the log file to refect the chosen settings.... <
#-------------------------------------------------------------------------------
from cnn_faces_util3 import *
launcher = "cnn_faces_launch.py"
logfile = '_CNN_32pix_100k_75per_train_1convpass.txt'

path = os.getcwd()
doit = path + '/' + launcher
allfeatures = 40
#-------------------------------------------------------------------------------
#settings
n_epoch = 50                    #50, 100
n_images = 100000               #25k, 50k, 100k
image_size = 32                 #32, 64, 128
conv_pass = 1                   #1, 2
training = 75                   #percentage of training data (test = 100-training)

FIX = False
if (FIX):
    fix = [20, 30]              #array of features not processed the first time... so do again..
#-------------------------------------------------------------------------------

if(FIX):
    for i in range (0, len(fix)):
        os.system(doit + " " + str(fix[i]) + " " + str(n_epoch)+ " " + str(n_images)+ " " + str(image_size) + " " + str(conv_pass) + " " + str(training) + " " + logfile)
else:
    for i in range (0, allfeatures):
        os.system(doit + " " + str(i) + " " + str(n_epoch)+ " " + str(n_images)+ " " + str(image_size) + " " + str(conv_pass) + " " + str(training) + " " + logfile)
#clean up
for afile in glob.glob("face-classifier.*"):
    os.remove(afile)
for afile in glob.glob("face_dataset*.*"):
    os.remove(afile)
