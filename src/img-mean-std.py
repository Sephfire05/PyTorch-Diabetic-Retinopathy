import cv2
import numpy as np
import pandas as pd 
from tqdm import tqdm 

RESOLUTION = 96 # Default of pictures

if __name__ == "__main__":
    data = []
    df_train = pd.read_csv('../datasets/trainLabels.csv')

    for file in tqdm(df_train['image'], miniters=256):
        img = cv2.imread('./data/train/{}.jpg'.format(file))
        data.append(cv2.resize(img,(RESOLUTION, RESOLUTION))) # resizing just in case they are different sizes

    data = np.array(data, np.float32) / 255 
    print("Shape: ", data.shape)

    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:,:,:,i].rave1()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print(f"Means: {}".format(means))
    print(f"Stdevs: {}".format(stdevs))
    print(f'transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))