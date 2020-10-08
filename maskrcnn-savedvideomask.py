# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:39:00 2020

@author: funka
"""

import numpy as np
import cv2

''' 
Load the video for object recognition.
Change the file path to whatever source video you are using.
'''
video_stream = cv2.VideoCapture('images/testing/video_sample.mp4')

# Infinite while loop to continuously capture video images
while True:
    ret, current_frame = video_stream.read()

    # Load image to detect
    input_image = current_frame
    
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
    
    # Create a list of colors for the masks (green, blue, red, cyan, yellow, purple)
    # Convert color list into integer arrays, splitting at commas
    # Colors are tiled to uniquely represent a single class label
    mask_colors = ["0, 255, 0", "0, 0, 255", "255, 0, 0", "255, 255, 0", "0, 255, 255", "255, 0, 255"]
    mask_colors = [np.array(color.split(",")).astype("int") for color in mask_colors]
    mask_colors = np.array(mask_colors)
    mask_colors = np.tile(mask_colors, (15, 1))
    
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
        if confidence > 0.7:
            # Get the label of the prediction (from class_labels)
            predicted_label = class_labels[int(detection_boxes[0, 0, index, 1])]
            # Get the bounding box coordinates of the prediction.
            bounding_box = detection_boxes[0, 0, index, 3:7] * np.array([width, height, width, height])
            # Bounding box coords are float, convert to int
            (start_x, start_y, end_x, end_y) = bounding_box.astype("int")
            
            # Get bounding box width and height
            bounding_box_width = end_x - start_x
            bounding_box_height = end_y - start_y
            
            # Get coordinates for object mask
            mask = detection_masks[index, int(detection_boxes[0, 0, index, 1])]
            
            # Resize the mask to fit the bounding box
            mask = cv2.resize(mask, (bounding_box_width, bounding_box_height))
            
            # Mask object is a array of floats. Convert to a binary array.
            mask = (mask > 0.3)
            
            # Slice image to are of bounding box
            region_of_interest = input_image[start_y:end_y, start_x:end_x]
            # Slice bounding box to masked region
            region_of_interest = region_of_interest[mask]
            
            # Assign a color to mask based on label index
            mask_color = mask_colors[int(detection_boxes[0, 0, index, 1])]
            
            # Create a transparent cover for masked region
            transparent_cover = ((0.3 * mask_color) + (0.5 * region_of_interest).astype("uint8"))
            # Place the transparent cover into the original image
            input_image[start_y:end_y, start_x:end_x][mask] = transparent_cover
            
            # Convert the mask color to a list to be able to color the bounding box
            mask_color = [int(i) for i in mask_color]
            
            # Print detected object in console
            print("Predicted object: {} - {:.2f}%".format(predicted_label, confidence * 100))
            
            # Draw bounding box around image
            #cv2.rectangle(input_image, (start_x, start_y), (end_x, end_y), mask_color, 2)
            cv2.putText(input_image, predicted_label + ": " + str(round(confidence * 100, 2)) + "%", 
                        (start_x, start_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, mask_color, 1)
    
    # Display image
    cv2.imshow("Detected Objects", input_image)
    
    # Break out of while loop if 'q' is pressed
    # waitKey return value is 32 bits while ord() returns 8 bit value
    # 0xFF is to mask the larger value so the comparison is valid
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream and close open image window   
video_stream.release()
cv2.destroyAllWindows()