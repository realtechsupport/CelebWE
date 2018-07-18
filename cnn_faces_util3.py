#!/usr/bin/env python
#-------------------------------------------------------------------------------
# cnn_faces_util3.py
#-------------------------------------------------------------------------------
from __future__ import division, print_function, absolute_import
import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
from scipy.io import loadmat
import math
import pickle
import glob
#-------------------------------------------------------------------------------
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
#-------------------------------------------------------------------------------

def test_attribute(featurenum, n_epoch, n_images, image_size, conv_pass, training, logfile):
    print()
    print('iteration: ', featurenum)
    print('n_epochs: ', n_epoch)
    print('number of images: ', n_images)
    print('selected image size: ', image_size)
    print('number of convolution block passes: ', conv_pass)
    print('training set (%): ', training)
    print()

    attrvalues = {}

    with open('list_attr_celeba.txt') as f:
        numimages = int(f.readline())
        attrnames = f.readline().split()
        for line in f:
            tokens = line.strip().split()
            attrvalues[tokens[0]] = np.array([int(a) for a in tokens[1:]])

    numfeatures = 1
    selfeatures = [featurenum]
    selected = ([attrnames[i] for i in selfeatures])

    print('current rule is: ', selected)
    selfeaturesvec = -np.ones([40,])
    for i in selfeatures:
        selfeaturesvec[i] = 1
    selfeaturesvec = selfeaturesvec.astype(int)

    # create labels
    labelsall = []
    X = []
    for k in attrvalues.keys():
        if all(attrvalues[k][selfeatures] == 1):
            labelsall.append(1)
        else:
            labelsall.append(0)

    print('resizing images')
    num = int(n_images)
    sz = int(image_size)
    rids = np.random.permutation(numimages)[0:num]
    data = np.zeros([num,sz,sz,3])
    labels = np.zeros([num,2])

    for i in range(num):
        sampleid = rids[i]                  #error fix 0 image
        fname = './img_align_celeba/'+str(sampleid+1).zfill(6)+'.jpg'
        #print(fname)
        img = Image.open(fname)
        attrs = attrvalues[str(sampleid+1).zfill(6)+'.jpg']
        img1 = img.resize([sz,sz])
        imgarr = np.array(img1.getdata()).reshape(img1.size[0], img1.size[1], 3)
        data[i,:,:,:] = imgarr
        labels[i,labelsall[sampleid]] = 1           # possible eror here

    print('splitting into training and test data')
    #split data into train and test (cast to int)
    '''
    spl = int(math.floor(num/2))
    X = data[0:spl,:,:,:]
    X_test = data[spl:,:,:,:]
    Y = labels[0:spl,:]
    Y_test = labels[spl:,:]
    '''
    tr = int(math.floor(num*(int(training)/100)))
    te = int(num-tr)
    print("training and testing numbers: ", tr,te)
    tot = tr+te
    print("checking for equality ", tot, num)

    X = data[0:tr,:,:,:]
    X_test = data[tr:,:,:,:]
    Y = labels[0:te,:]
    Y_test = labels[te:,:]

    print('pickling data')
    # save data
    pickle.dump(X,open('face_dataset_X.pkl','wb'))
    pickle.dump(Y,open('face_dataset_Y.pkl','wb'))
    pickle.dump(X_test,open('face_dataset_X_test.pkl','wb'))
    pickle.dump(Y_test,open('face_dataset_Y_test.pkl','wb'))

    print('loading data')
    X = pickle.load(open("face_dataset_X.pkl","rb"))
    X_test = pickle.load(open("face_dataset_X_test.pkl","rb"))
    Y = pickle.load(open("face_dataset_Y.pkl","rb"))
    Y_test = pickle.load(open("face_dataset_Y_test.pkl","rb"))

    print('image preparations')
    print("not shuffling when training not equal to test size")
    #X, Y = shuffle(X, Y)
    
    # Make sure the data is normalized
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    # Create extra synthetic training data by flipping, rotating and blurring the images
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)

    print('defining network architecture')
    # Define network architecture:Input is a 32x32 image with 3 color channels
    #network = input_data(shape=[None, 32, 32, 3],data_preprocessing=img_prep,data_augmentation=img_aug)
    network = input_data(shape=[None, sz, sz, 3],data_preprocessing=img_prep,data_augmentation=img_aug)
    #2 passes through convolution and pooling

    for j in range (1, (int(conv_pass)+1)):
        print('convolution and pooling sequence: ', str(j))
        # Step 1: Convolution
        network = conv_2d(network, 32, 3, activation='relu')
        # Step 2: Max pooling
        network = max_pool_2d(network, 2)
        # Step 3: Convolution again
        network = conv_2d(network, 64, 3, activation='relu')
        # Step 4: Convolution yet again
        network = conv_2d(network, 64, 3, activation='relu')
        # Step 5: Max pooling again
        network = max_pool_2d(network, 2)

    print('finished convolution and pooling' )

    # Step 6: Fully-connected 512 node neural network
    network = fully_connected(network, 512, activation='relu')
    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    network = dropout(network, 0.5)
    # Step 8: Fully-connected neural network with two outputs (0=isn't a face, 1=is a face) to make the final prediction
    network = fully_connected(network, 2, activation='softmax')
    # Tell tflearn how we want to train the network
    network = regression(network, optimizer='adam', loss='categorical_crossentropy',learning_rate=0.001)
    print('wraping network in a model')
    # Wrap the network in a model object
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='face-classifier.tfl.ckpt')
    # Train it! We'll do 100 training passes and monitor it as it goes.
    print('training with ' + str(n_epoch) + ' passes...') # batch_size was 96 for 32x32 image;  n*image size
    bs = 3*sz
    model.fit(X, Y, int(n_epoch), validation_set=(X_test, Y_test),show_metric=True, batch_size=bs)

    accuracy_score = model.evaluate(X_test, Y_test)
    accuracy = '... accuracy: %0.4f%%' % (accuracy_score[0] * 100)
    print(accuracy)
    print('saving results')
    message = 'feature #' + str(featurenum) + ': ' +selected[0] + ' ' + str(accuracy) +  ' with ' + str(n_epoch) + ' passes'
    logstatus(os.getcwd(), logfile, message, 'a')

#------------------------------------------------------------------------------
def logstatus(path, logfile, message, method):
	datafilepath = path+logfile
	tempfile = open(datafilepath, method)
	result = time.strftime("%d %b %Y %H:%M:%S") + " " + message + "\n"
	tempfile.write(result)
	tempfile.close()
