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
input_image = cv2.imread("images/testing/scene3.jpg")

# Get image width/height
height = input_image.shape[0]
width = input_image.shape[1]

# Convert image to blob (Binary Large OBject) for further processing.
# RGB must be swapped to BGR for accurate processing, do not crop image.
image_blob = cv2.dnn.blobFromImage(input_image, swapRB = True, crop = False)

# Set class labels
class_labels = ["person", "bicycle", "car", "motorbike", "airplane", "bus",
                "train", "truck", "boat", "traffic light", "fire hydrant",
                "street sign", "stop sign", "parking meter", "bench", "bird",
                "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
                "eyeglasses", "handbag", "tie", "suitcase", "frisbee", "skis",
                "snowboard", "sports ball", "kite", "baseball bat", 
                "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "plate", "wine glass", "cup", "fork", "knife", 
                "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                "chair", "sofa", "potted plant", "bed", "mirror", "dining table",
                "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven",
                "toaster", "sink", "refrigerator", "blender", "book", "clock",
                "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"]

# Load the pretrained model from model weights and config files
maskrcnn = cv2.dnn.readNetFromTensorflow("datasets/maskrcnn_buffermodel.pb", "datasets/maskrcnn_bufferconfig.txt")

# Pass preprocessed blob into model
maskrcnn.setInput(image_blob)

# Use forward() method to obtain object detections
# Detection_boxes output of forward() is a 4D array with shape (1, 1, N, 7) 
# where N is the total number of detections. Each detection has a vector of 7 values
# associated with it (batchID, classID, confidence, left, right, top, bottom).
# So, for example, dectections[0, 0, i, 1] has the class ID for dectection at 
# index i, and dectections[0, 0, i, 2] has the confidence of that detection.
# detection_out_final and detection_masks are output layers from the blob
# being passed into the network
(detection_boxes, detection_masks) = maskrcnn.forward(["detection_out_final", "detection_masks"])

number_detections = detection_boxes.shape[2]

# Loop through all the detected objects
for index in np.arange(0, number_detections):
    confidence = detection_boxes[0, 0, index, 2]
    '''
    Only consider detected objects above a certain confidence value
    This value can be changed higher or lower depending on the desired accuracy.
    '''
    if confidence > 0.4:
        # Get the label of the prediction (from class_labels)
        predicted_label = class_labels[int(detection_boxes[0, 0, index, 1])]
        # Get the bounding box coordinates of the prediction.
        # We need to multiply the boundaries by our original height and width
        # because we resized to 300x300 for preocessing and need to convert back
        bounding_box = detection_boxes[0, 0, index, 3:7] * np.array([width, height, width, height])
        # Bounding box coords are float, convert to int
        (start_x, start_y, end_x, end_y) = bounding_box.astype("int")
        
        # Print detected object in console
        print("Predicted object: {} - {:.2f}%".format(predicted_label, confidence * 100))
        
        # Draw bounding box around image
        cv2.rectangle(input_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(input_image, predicted_label + ": " + str(round(confidence * 100, 2)) + "%", 
                    (start_x, start_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)

# Display image
cv2.imshow("Detected Objects", input_image)
cv2.waitKey(0)
cv2.destroyWindow("Detected Objects")