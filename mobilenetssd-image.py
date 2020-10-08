# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:39:00 2020

@author: funka
"""

import numpy as np
import cv2

''' 
Load the image for object recognition.
Change the file path to whatever source image you are using.
'''
input_image = cv2.imread("images/testing/scene7.jpg")

# Get image width/height
height = input_image.shape[0]
width = input_image.shape[1]

# Resize image
resized_image = cv2.resize(input_image, (300, 300))

# Convert image to blob (Binary Large OBject) for further processing.
# 127.5 is a mean intensity value we are subtracting from each RGB channel and 
# 0.007843 is the scale factor, in this case equal to 1/127.5. The mean value
# is subtracted from each pixel and then normalized between 0 and 1 when
# multiplied by the scale factor.
image_blob = cv2.dnn.blobFromImage(resized_image, 0.007843, (300, 300), 127.5)

# Set class labels (background + 20 more alphabetized classes)
class_labels = ["background", "airplane", "bicycle", "bird", "boat", 
                "bottle", "bus", "car", "cat", "chair", "cow", 
                "dining table", "dog", "horse", "motorcycle", "person", 
                "potted plant", "sheep", "sofa", "train", "tv/monitor"]

# Load the pretrained model from prototext and caffemodel files
mobilenetssd = cv2.dnn.readNetFromCaffe("datasets/mobilenetssd.prototext", "datasets/mobilenetssd.caffemodel")

# Pass preprocessed blob into model
mobilenetssd.setInput(image_blob)

# Use forward() method to obtain object detections
# Output of forward() is a 4D array with shape (1, 1, N, 7) where N is the 
# total number of detections. Each detection has a vector of 7 values
# associated with it (batchID, classID, confidence, left, right, top, bottom).
# So, for example, dectections[0, 0, i, 1] has the class ID for dectection at 
# index i, and dectections[0, 0, i, 2] has the confidence of that detection.
detections = mobilenetssd.forward()

number_detections = detections.shape[2]

# Loop through all the detected objects
for index in np.arange(0, number_detections):
    confidence = detections[0, 0, index, 2]
    '''
    Only consider detected objects above a certain confidence value
    This value can be changed higher or lower depending on the desired accuracy.
    '''
    if confidence > 0.4:
        # Get the label of the prediction (from class_labels)
        predicted_label = class_labels[int(detections[0, 0, index, 1])]
        # Get the bounding box coordinates of the prediction.
        # We need to multiply the boundaries by our original height and width
        # because we resized to 300x300 for preocessing and need to convert back
        bounding_box = detections[0, 0, index, 3:7] * np.array([width, height, width, height])
        # Bounding box coords are float, convert to int
        (start_x, start_y, end_x, end_y) = bounding_box.astype("int")
        
        # Print detected object in console
        print("Predicted object: {} - {:.2f}%".format(predicted_label, confidence * 100))
        
        # Draw bounding box around image
        cv2.rectangle(input_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(input_image, predicted_label + ": " + str(round(confidence * 100, 2)) + "%", (start_x, start_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)

# Display image
cv2.imshow("Detected Objects", input_image)
cv2.waitKey(0)
cv2.destroyWindow("Detected Objects")