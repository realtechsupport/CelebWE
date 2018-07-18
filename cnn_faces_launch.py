#!/usr/bin/env python
#-------------------------------------------------------------------------------
# cnn_faces_main.py
#-------------------------------------------------------------------------------

from cnn_faces_util3 import *

value = int(sys.argv[1])
n_epoch = int(sys.argv[2])
n_images = int(sys.argv[3])
image_size = int(sys.argv[4])
conv_pass = sys.argv[5]
training = sys.argv[6]
logfile = sys.argv[7]

test_attribute(value, n_epoch, n_images, image_size, conv_pass, training, logfile)
