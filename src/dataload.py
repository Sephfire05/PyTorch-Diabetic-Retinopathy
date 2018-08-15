
# coding: utf-8

# In[10]:


# import libraries
import torch
import torchvision
import torchvision.transforms as transform
from torch.utils.data.dataset import Dataset
import pandas as pd
import os
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer


# In[11]:


class DiabRetinopathy(Dataset):
    """Dataset wrapping images and target labels
    
    Arguments:
        A CSV file patch
        Path to image folder
        Extension of images
    """
    
    def __init__(self, csv_path, img_path, img_ext, transform=transform):
        
        self.df = pd.read_csv(csv_path)
        assert self.df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), "Some images references in the CSV file were not found"

        # Ordering classification
        self.mlb = MultiLabelBinarizer(
                classes = ['1', '2', '3', '4']
                                      )
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform
    
        self.x = self.df['image_name']
        self.y = self.mlb.fit_transform(self.df['label'])

    def X(self):
        return self.x

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.x[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = from_numpy(self.y[index])
        return img, label

    def __len__(self):
        return len(self.df.index)

    def getLabelEncoder(self):
        return self.mlb

    def getDF(self):
        return self.df


# In[4]:




