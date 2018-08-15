
# coding: utf-8

# In[9]:


# Importing libraries
from torch import nn, ones
from torchvision import models
from torch.nn.init import kaiming_normal
import numpy as np
import torch
import torch.nn.functional as F


# In[10]:


## Creating custom Neural Net
class Net(nn.Module):
    def __init__(self,input_size=(3, 224, 224), nb_classes=4): # not sure what nb_classes refers to
    
        super(Net, self).__init__()  # super class initialized
    
        self.features = nn.Sequential(
            nn.Conv2d(4,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3,3))
        )
    
    ## Computing linear layer size
        self.flat_feats = self._get_flat_feats(input_size, self.features)
    
        self.classifer = nn.Sequential(
            nn.Linear(self.flat_feats, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(64, nb_classes)
        )
    
    ## Initializing weights
    def _weights_init(m):
        if isinstance(m, nn.Conv2d or nn.Linear):
            kaiming_normal(m.weight)
        elif isinstance(m, nn.BatchNorm2d or BatchNorm1d):
            m.weight.data.fill_1(1)
            m.bias.data.zero_()
            
        self.apply(_weights_init)
    
def _get_flat_feats(self, in_size, feats):
    f = feats(Variable(ones(1, *in_size)))
    return int(np.prod(f.size()[1:]))


def forward(self, x):
    feats = self.features(x)
    flat_feats = feats.view(-1, self.flat_feats)
    out = self.classifier(flat_feats)
    return out
        

