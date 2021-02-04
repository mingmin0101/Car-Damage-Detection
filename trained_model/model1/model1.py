# import urllib
from collections import Counter, defaultdict
import json
import argparse
import pickle as pk
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from keras.utils.data_utils import get_file
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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
    out = vgg16.predict(img)
    top = get_predictions(out, top=5)
    print ("Validating that this is a picture of your car...")
    for j in top[0]:
        if j[0:2] in cat_list:
            print (j[0:2])
            return "Validation complete - proceed to damage evaluation"
    return "Are you sure this is a picture of your car? Please take another picture (try a different angle or lighting) and try again."

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--image_path", help="Image to be classified", required=True)
args = parser.parse_args()

vgg16 = VGG16(weights='imagenet')

with open('cat_counter.pk', 'rb') as f:
    cat_counter = pk.load(f)
cat_list = [k for k, v in cat_counter.most_common()[:50]]

### classification result
print(car_categories_gate(args.image_path, cat_list))