# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render
from django import template
from django.template.loader import get_template

from .forms import ImageForm
from .models import Image

from django.conf import settings
#from django.conf.settings import MEDIA_ROOT

#########################################################
# model import
#########################################################
import warnings  
warnings.filterwarnings("ignore",category=FutureWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from collections import Counter, defaultdict
import json
import pickle as pk
import numpy as np
import os

from keras.utils.data_utils import get_file
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras import backend as K

import h5py



def image_upload_view(request):

    return render(request, 'upload.html', locals())


def result_view(request):
    """Process images uploaded by users"""
    
    if request.method == 'POST':
        K.clear_session()
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            
            new_img = Image(image = request.FILES['image'])
            new_img.save()
           
            img_id = new_img.id

            img_path = str(settings.BASE_DIR) + new_img.image.url  # os.path.join 找路徑會有錯誤
            model1_result = model1(img_path)
            K.clear_session()

            if model1_result == None:
              model1_response = 0

              return render(request, 'result.html', {'form': form, 'img_obj': Image.objects.get(pk=img_id), 'model1_response':model1_response})   

            else:
              ft_model2 = load_model(os.path.join(settings.STATICFILES_DIRS[0], 'trained_model/model2.h5'))
              ft_model3 = load_model(os.path.join(settings.STATICFILES_DIRS[0], 'trained_model/model3.h5'))
              ft_model4 = load_model(os.path.join(settings.STATICFILES_DIRS[0], 'trained_model/model4.h5'))

              img_preprocessed = preprocess_img(img_path)
              model1_response = 1
              model1_type = model1_result
              model2_response = model2(img_preprocessed, ft_model2)

              if model2_response == 0:
                return render(request, 'result.html', {'form': form, 'img_obj': Image.objects.get(pk=img_id), 'model1_response':model1_response, 'model2_response':model2_response})

              model3_result = model3(img_preprocessed, ft_model3)
              if model3_result == None:
                model3_response = 0
              else:
                model3_response = model3_result

              model4_result = model4(img_preprocessed, ft_model4)
              if model4_result == None:
                model4_response = 0
              else:
                model4_response = model4_result

        return render(request, 'result.html', {'form': form, 'img_obj': Image.objects.get(pk=img_id), 'model1_response':model1_response, 'model2_response':model2_response, 'model3_response':model3_response, 'model4_response':model4_response})   
    else:
        form = ImageForm()

    return render(request, 'result.html', {'form': form}) 



#########################################################
# model1
#########################################################
CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def car_categories_gate(image_path, cat_list):
    # urllib.urlretrieve(image_path, 'save.jpg') # or other way to upload image
    img = prepare_image(image_path)
    vgg16 = VGG16(weights='imagenet')
    out = vgg16.predict(img)
    top = get_predictions(out, top=5)
    print (" Validating that this is a picture of your car...")
    for j in top[0]:
        if j[0:2] in cat_list:
            print(j[1]) #(j[0:2])
            return j[1]#"Validation complete - proceed to damage evaluation"
    
    print("Not a car!")
    return None #"Are you sure this is a picture of your car? Please take another picture (try a different angle or lighting) and try again."

def model1(image_path):
	with open(os.path.join(settings.STATICFILES_DIRS[0], 'trained_model/cat_counter.pk'), 'rb') as f:
	    cat_counter = pk.load(f)
	cat_list = [k for k, v in cat_counter.most_common()[:50]]

	return car_categories_gate(image_path, cat_list)


#########################################################
# model2
#########################################################
gpu_id = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def preprocess_img(img_path):
    img = load_img(img_path, target_size=(256, 256)) # this is a PIL image 
    x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
    x = x.reshape((1,) + x.shape)/255 # this is a Numpy array with shape (1, 3, 256, 256)
    return x

def car_damage_gate(x, model):
    # img = load_img(image_path, target_size=(256, 256)) # this is a PIL image 
    # x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
    # x = x.reshape((1,) + x.shape)/255 # this is a Numpy array with shape (1, 3, 256, 256)
    pred = model.predict(x)
    print( "Validating that damage exists...")
    print( pred)
    if pred[0][0] >=.5:
        print( "Validation complete - proceed to location and severity determination")
        return 1
    else:
        print( "Are you sure that your car is damaged? Please submit another picture of the damage.")
        print( "Hint: Try zooming in/out, using a different angle or different lighting")
        return 0


def model2(image_path, ft_model):
    #ft_model = load_model(os.path.join(settings.STATICFILES_DIRS[0], 'trained_model/model2.h5'))
    return car_damage_gate(image_path, ft_model)

#########################################################
# model3
#########################################################
def location_assessment(x, model):
    print ("Determining location of damage...")
    # img = load_img(image_path, target_size=(256, 256)) # this is a PIL image 
    # x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
    # x = x.reshape((1,) + x.shape)/255 # this is a Numpy array with shape (1, 3, 256, 256)
    pred = model.predict(x)
    pred_label = np.argmax(pred, axis=1)
    d = {0: 'Front', 1: 'Rear', 2: 'Side'}
    for key in d:
        if pred_label[0] == key:
            print ("Assessment: {} damage to vehicle".format(d[key]))
            return d[key]

    print ("Location assessment complete. Coud not find the position.")
    return None

def model3(image_path, ft_model):
    #ft_model = load_model(os.path.join(settings.STATICFILES_DIRS[0], 'trained_model/model3.h5'))

    return location_assessment(image_path, ft_model)

#########################################################
# model4
#########################################################
def severity_assessment(x, model):
    print ("Determining severity of damage...")
    # img = load_img(image_path, target_size=(256, 256)) # this is a PIL image 
    # x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
    # x = x.reshape((1,) + x.shape)/255 # this is a Numpy array with shape (1, 3, 256, 256)
    pred = model.predict(x)
    pred_label = np.argmax(pred, axis=1)
    d = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
    for key in d:
        if pred_label[0] == key:
            print ("Assessment: {} damage to vehicle".format(d[key]))
            return d[key]
    print ("Severity assessment complete.")
    
    return None

def model4(image_path, ft_model):
    #ft_model = load_model(os.path.join(settings.STATICFILES_DIRS[0], 'trained_model/model4.h5'))

    return severity_assessment(image_path, ft_model)