import urllib
# from IPython.display import Image, display, clear_output
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns 
# get_ipython().run_line_magic('matplotlib', 'inline')

import json
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import argparse

sns.set_style('whitegrid')


# %%
import os
gpu_id = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import h5py
import numpy as np
import pandas as pd

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint, History
from keras.models import Model

def severity_assessment(image_path, model):
    print ("Determining severity of damage...")
    img = load_img(image_path, target_size=( img_width, img_height)) # this is a PIL image 
    x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
    x = x.reshape((1,) + x.shape)/255 # this is a Numpy array with shape (1, 3, 256, 256)

    pred = model.predict(x)
    pred_label = np.argmax(pred, axis=1)
    d = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
    for key in d:
        if pred_label[0] == key:
            print ("Assessment: {} damage to vehicle".format(d[key]))
    print ("Severity assessment complete.")

img_width, img_height = 256, 256
ft_model = load_model('ft_model.h5')
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--image_path", help="Image to be classified", required=True)
args = parser.parse_args()

### severity result
severity_assessment(args.image_path, ft_model)