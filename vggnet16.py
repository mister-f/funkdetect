# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:49:56 2020

@author: funka
"""

from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

''' 
Load the image for object recognition.
Change the file path to whatever source image you are using.
'''
image_path = "images/testing/animal3.jpg"
image = load_img(image_path)

# Resize image to conform to VGGNet model (224x224)
image = image.resize((224, 224))

# Convert image to an array
image_array = img_to_array(image)

# Convert image array from 3 dimensions (height, width, channels)
# to 4 dimensions (batch size, height, width, channels) for proper processing
image_array = np.expand_dims(image_array, axis = 0)

# Preprocess the image array
image_array = imagenet_utils.preprocess_input(image_array)

# Load the pretrained model (large file!)
pretrained_model = VGG16(weights = "imagenet")

# Predict the image using predict()
prediction = pretrained_model.predict(image_array)

# Decode the prediction
decoded_prediction = imagenet_utils.decode_predictions(prediction)

print(decoded_prediction)

# Display the image and overlay the prediciton
display = cv2.imread(image_path)
overlay_text = (decoded_prediction[0][0][1] + " - " + str(round((decoded_prediction[0][0][2] * 100), 2)) + "%")

# Add description text over image
cv2.putText(display, overlay_text, (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (105, 105, 105))

# Show image window
cv2.imshow("Prediction", display)
cv2.waitKey(0)
cv2.destroyWindow("Prediction")
