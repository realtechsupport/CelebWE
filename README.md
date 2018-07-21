# celebWE

general:

Code experiments to explore the classification and generation of beauty as represented in neural networks (a convolutional neural network {CNN} and a generative adversarial neural network {GAN}). 
Test database for the CNN is the CelebA dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and for the GAN a video collected in a supermarket. 

CNN ------------------------------------------------

A vanilla convolutional neural network architecture (see image) under tflearn. Set number of epochs, training data percentage and features to select from the set of 40 possible features. Network architecture: see CNN_architecture.png

CelebA dataset:
202k+ images of celebrities with 40 binary attributes:
5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair 	Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face 	Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair 	Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young

requirements:
python 2.7x, tflearn, pil, pickle, numpy, scipy, csv, cuda

usage:
python cnn_faces_multiple3.py



GAN ------------------------------------------------
A pytorch implementation of a the dcgan network (https://github.com/pytorch/examples/tree/master/dcgan), altered to create 128 x 128 pixel output images.

usage:

usage: GAN_p128.py --dataset DATASETname --dataroot DATAROOTlocation --imageSize 128 --cuda


requirements:
python 2.7x, torch, cuda


sample output CNN:

07 Feb 2018 01:41:20 feature #3: Bags_Under_Eyes ... accuracy: 79.8520% with 50 passes
07 Feb 2018 03:15:05 feature #4: Bald ... accuracy: 97.7280% with 50 passes
07 Feb 2018 04:48:46 feature #5: Bangs ... accuracy: 85.1500% with 50 passes
07 Feb 2018 06:22:22 feature #6: Big_Lips ... accuracy: 75.6820% with 50 passes
07 Feb 2018 07:56:10 feature #7: Big_Nose ... accuracy: 76.6140% with 50 passes
07 Feb 2018 09:29:59 feature #8: Black_Hair ... accuracy: 76.2240% with 50 passes


sample output GAN (trained on avideo of a playground):

fake_playground.png


tested on:
Lambda Labs 'Lambda Single' computer (1 GPU) 
https://lambdal.com/products/single 

publication on CNN experiments:
https://arxiv.org/abs/1711.08801v1

license for all code:
GNU General Public Licence v3.0
