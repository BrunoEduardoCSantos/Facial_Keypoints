## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 64 output channels/feature maps, 3x3 square convolution kernel
        #(W-F)/S +1 = (224-3)/1 +1 =222 
        self.conv1 = nn.Conv2d(1, 32, 3)
        #111X111X32
        self.pool1= nn.MaxPool2d(2,2)
        #(W-F)/S +1 = (111-2)/1 +1 = 110
        self.conv2 = nn.Conv2d(32,  64, 2)
        #55X55X64
        self.pool2= nn.MaxPool2d(2,2)
        
        self.fc1= nn.Linear(55*55*64,136)
        self.dropout = nn.Dropout(p=0.3)
     
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## Define the feedforward behavior of this model
       
        x= self.pool1(F.relu(self.conv1(x)))
        x= self.dropout(x)
        x= self.pool2(F.relu(self.conv2(x)))
        x= self.dropout(x)
        x = x.view(x.size(0), -1)
        x= F.relu(self.fc1(x))
        x= self.dropout(x)
       
        # a modified x, having gone through all the layers of your model, should be returned
        return x
