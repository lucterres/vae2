import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import math
from os.path import exists

TRAIN_CSV = './data/train.csv'
DEPTH_CSV = 'depths.csv'

TRAIN_IMAGE_DIR = 'D:/_phd/datasets/tgsSalt/train/images/'
TRAIN_MASK_DIR  = 'D:/_phd/datasets/tgsSalt/train/masks/'

TEST_IMAGE_DIR  = 'D:/_phd/datasets/tgsSalt/test/images/'

df_train = pd.read_csv(TRAIN_CSV)
df = df_train
df['salt'] = df['rle_mask'].notnull().replace([False, True], [0,1]) #0 = no_salt #1 = salt
salt = df[df['rle_mask'].notnull()]

def limites(msk):
        number_of_white_pix = np.sum(msk == 255)
        if (number_of_white_pix>(10201*0.1)) & (number_of_white_pix<(10201*0.9)):
                result = True
        else:
                result = False
    
        return result

def coverage(label):
        path = TRAIN_MASK_DIR + f'{label}.png'
        white_pix = -1.0
        if exists(path):
                mask = cv2.imread(path)
                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                white_pix = np.sum(gray == 255)
        return white_pix/10201

def class_cover(val):
    r = math.trunc(10*val)
    return r

# a coluna coverage em todas as linhas do df recebe o resultado da função coverage(id)
df['coverage']=df['id'].map(coverage)

df['class']=df.coverage.map(class_cover)

salt=df[df['class']>0]
salt=salt[salt['class']<9]