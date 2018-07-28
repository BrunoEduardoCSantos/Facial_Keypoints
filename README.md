[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

## Overview

In this project, computer vision techniques and deep learning architectures will be used to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. The model look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.
![Facial Keypoint Detection][image1]

# Project Files
The project will be broken up into a few main parts in four Python notebooks:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses



## Setup Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/udacity/P1_Facial_Keypoints.git
cd P1_Facial_Keypoints
```

2. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__: 
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```
	
	At this point your command line should look something like: `(cv-nd) <User>:P1_Facial_Keypoints <user>$`. The `(cv-nd)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```


### Data

All of the data to train a neural network is in the subdirectory `data`. 


## Model Architecture and Training Strategy
### Data preparation
1. Rescaling and/or cropping the data, such that you are left with a square image (the suggested size is 224x224px)
2. Normalizing the images and keypoints; turning each RGB image into a grayscale image with a color range of [0, 1] and transforming the given keypoints into a range of [-1, 1]
3.Turning these images and keypoints into Tensors


### Model design
The model designed for predicting the facial points in this project is given by:
![model](https://github.com/BrunoEduardoCSantos/Facial_Keypoints/blob/master/images/FacialKeysModel.png)

1. Adam was chosen for optimization and SmoothL1 for loss function. SmoothL1 was used as loss function because is less prone to outliers compared to MSE for larger values. In addition, it can include a error function like MSE for smaller value. Regarding the optimization Adam was the chosen one because it suitable for regression with deep learning architecture

2. The initial model architecture was based on inception model, where it was setup layers 5X5 , 3X3 and 1X1. Afterward, it progressed by adding more convolution layers together with dropout layers to reduce the loss as well as avoid overfitting.

3. The number of epochs was chosen by the loss function evaluation as well this hold monotonic descent. Regarding the batch_size it started from 10 and increased to 40 so that  loss function decreases error smoothly and continuosly.


## Further improvements

* The next level : try transfert learning from pretrained network like [vgg and  resnet] (https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). The goal here is to used already working architecture and add a model for a specific task.









