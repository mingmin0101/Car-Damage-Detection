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

def car_categories_gate(image_path, model):
    # urllib.urlretrieve(image_path, 'save.jpg') # or other way to upload image
    img = load_img(image_path, target_size=( img_width, img_height)) # this is a PIL image 
    x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
    x = x.reshape((1,) + x.shape)/255 # this is a Numpy array with shape (1, 3, 256, 256)
    # x = preprocess_input(x)
    pred = model.predict(x)
    print( "Validating that damage exists...")
    print( pred)
    if pred[0][0] >=.5:
        print( "Validation complete - proceed to location and severity determination")
    else:
        print( "Are you sure that your car is damaged? Please submit another picture of the damage.")
        print( "Hint: Try zooming in/out, using a different angle or different lighting")

img_width, img_height = 256, 256
ft_model = load_model('ft_model.h5')
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--image_path", help="Image to be classified", required=True)
args = parser.parse_args()

### classification result
car_categories_gate(args.image_path, ft_model)