
# coding: utf-8

# In[11]:


# Need Custom imports of dataload, training, validation, logger, prediction, data augmentation, custom loss, sampler for validation
# dataloader
from src.neuro import Net # Baseline model


# Importing Utilities
import random
import logging
import time
from timeit import default_timer as timer
import os

# Libraries
import numpy as np
import math

# Torch
import torch.optim as optim
import torch.nn.functional as F # Transform params
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
########################################################
mode = Net(4).cuda()

epochs = 100
batch_size = 352

# Normalization param
# normalize = transforms.Normalize(mean=[x,x,x],
                                #     std=[x,x,x])
    
# Optimizer param
optimizer = optim.SGD(model.parameters(),lr=1e-3, momentum=0.9,) # Fine tune these
criterion = CrossEntropyLoss()
# Need to figure this parameter out
# .cuda() at end

# These are also binary labels, no weight
classes = [1, 2, 3, 4]

save_dir = './snapshots'

#########################################################
## Loading the dataset

## Augmentation + Normalization for full training
ds_transform_augmented = transforms.Compose([
                 transforms.RandomSizedCrop(224),
                 PowerPIL(),
                 transforms.ToTensor(),
                 # normalize
])

## Normalization only for validation and test though
ds_transform_raw = transforms.Compose([
                 transforms.Scale(224),
                 transforms.ToTensor(),
                 # normalize
                 ])
##########################################################
## Train and Validation
X_train = DiabRetinopathyDataset('./Datasets/trainLabels.csv','./data/train/','.jpg',
                                 ds_transform_augmented
                                 )
X_val = DiabRetinopathyDataset('./Datasets/trainLabels.csv','.data/train/','.jpg',
                               ds_transform_raw
                               )
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetSampler(valid_idx)

##########################################################
## Both dataloader loads from the same dataset but with different indices
train_loader = DataLoader(X_train,
                          batch_size=batch_size,
                          sampler=train_sampler,
                          num_workers=4,
                          pin_memory=True)

valid_loader = DataLoader(X_val,
                          batch_size=batch_size,
                          sampler=valid_sampler,
                          num_workers=4,
                          pin_memory=True)

##########################################################
## Start training
best_score = 0
for epoch in range(epochs):
    epoch_timer = timer()
    
    # Train and validate
    train(epoch, train_loader, model, criterion, optimizer)
    score, loss, threshold = validate(epoch, valid_loader, model, criterion, X_train.getLabelEncoder())
    # Save scores
    is_best = score > best_score
    best_score = max(score, best_score)
    snapshot(save_dir, run_name, is_best,{
        'epoch': epoch + 1,
        'state_dict': best_score,
        'optimizer': optimizer.state_dict(),
        'threshold': threshold,
        'val_loss': loss
    })
    
    end_epoch_timer = timer()
    logger.info("#### End epoch{}, elapsed time: {}".format(epoch, end_epoch_timer - epoch_timer))

###########################################################
## Prediction
X_test = DiabRetinopathyDataset('./Dataset/sampleSubmission.csv','./data/test/','.jpg',
                                 ds_transform_raw
                                )
test_loader = DataLoader(X_test,
                         batch_size=batch_size,
                         num_workers=4,
                         pin_memory=True)

# Load model from best iteration
# logger.info('----Loading best model for prediction')
checkpoint = torch.load(os.path.join(save_dir,
                                     run_name + '-model_best.pth'
                                     )
                       )
model.load_state_dict(checkpoint['state_dict'])

# Predict
predictions = predict(test_loader, model) # TODO Set up module

output(predictions,
       checkpoint['threshold'],
       X_test,
       X_train.getLabelEncoder(),
       './out',
       run_name,
       checkpoint['best_score'])

###########################################################

end_global_timer = timer()
logger.info("------------------Success-----------------")
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

