# celebWE

general:
A vanilla convolutional neural network architecture (see image) under tflearn to train a network on the CelebA dataset (220k+images). Set number of epochs, training data percentage and features to select from the set of 40 possible features.

5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair 	Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face 	Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair 	Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young


(see diagram)


license:
GNU General Public Licence v3.0

requirements:
python 2.7x, tflearn, pil, pickle, numpy, scipy, csv

usage:
python cnn_faces_multiple3.py

dataset:
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

tested:
..on a lambda Labs Lambda Single computer (1 GPU) https://lambdal.com/products/single 


sample comments in verbose mode:
iteration:  0
n_epochs:  50
number of images:  100000
selected image size:  32
number of convolution block passes:  1
training set (%):  50
current rule is:  ['5_o_Clock_Shadow']
resizing images
splitting into training and test data
training and testing numbers:  50000 50000
checking for equality  100000 100000
pickling data
loading data
image preparations
not shuffling when training not equal to test size
defining network architecture
convolution and pooling sequence:  1
finished convolution and pooling
wraping network in a model
training with 50 passes...


sample output:
07 Feb 2018 01:41:20 feature #3: Bags_Under_Eyes ... accuracy: 79.8520% with 50 passes
07 Feb 2018 03:15:05 feature #4: Bald ... accuracy: 97.7280% with 50 passes
07 Feb 2018 04:48:46 feature #5: Bangs ... accuracy: 85.1500% with 50 passes
07 Feb 2018 06:22:22 feature #6: Big_Lips ... accuracy: 75.6820% with 50 passes
07 Feb 2018 07:56:10 feature #7: Big_Nose ... accuracy: 76.6140% with 50 passes
07 Feb 2018 09:29:59 feature #8: Black_Hair ... accuracy: 76.2240% with 50 passes


