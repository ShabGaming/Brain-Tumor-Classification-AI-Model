import tensorflow as tf  # pip install tensorflow==2.12.0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('model.h5', compile=False)
model = load_model('model.h5')

# Define the class labels
class_labels = ['Glioma Tumor', 'Meningioma Tumor',
                'No Tumor', 'Pituitary Tumor']

# Get the path to the input image from the user
img_path = input("Enter the path to the input image: ")

# Load and preprocess the input image
img = image.load_img(img_path, target_size=(1250, 1250))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.

# Classify the input image
preds = model.predict(x)
class_idx = np.argmax(preds)
class_label = class_labels[class_idx]
print("The input image is classified as: ", class_label)