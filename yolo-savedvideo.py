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
    # 127.5 is a mean intensity value we are subtracting from each RGB channel and 
    # 0.003922 is the scale factor, in this case equal to 1/255. Acceptable blob
    # sizes for YOLO are 320x320, 416x416, and 609x609. Larger size gives better 
    # accuaracy but is processed slower. RGB is swapped to BGR.
    image_blob = cv2.dnn.blobFromImage(input_image, 0.003922, (416, 416), swapRB = True, crop = False)
    
    # Set class labels
    class_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
                    "train", "truck", "boat", "traffic light", "fire hydrant",
                    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                    "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                    "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                    "potted plant", "bed", "dining table", "toilet", "tv/monitor",
                    "laptop", "mouse", "remote", "keyboard", "cell phone",
                    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                    "clock", "vase", "scissors", "teddy bear", "hair dryer",
                    "toothbrush"]
    
    # Create a list of colors for the masks (green, blue, red, yellow, purple)
    # Convert color list into integer arrays, splitting at commas
    # Colors are tiled to uniquely represent a single class label
    box_colors = ["0, 255, 0", "0, 0, 255", "255, 0, 0", "0, 255, 255", "255, 0, 255"]
    box_colors = [np.array(color.split(",")).astype("int") for color in box_colors]
    box_colors = np.array(box_colors)
    box_colors = np.tile(box_colors, (16, 1))
    
    '''
    Load the pretrained model from cfg and weights files
    Any of the YOLO model versions can be entered here. YOLO v4 does require 
    OpenCV version 4.4.0 or higher.
    '''
    yolo = cv2.dnn.readNetFromDarknet("datasets/yolov4.cfg", "datasets/yolov4.weights")
    
    # Processing is done with an input layer, hidden layers, and an output layer.
    # We must get the layers and loop to the last one (the output layer)
    # for proper detection
    yolo_layers = yolo.getLayerNames()
    output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo.getUnconnectedOutLayers()] 
    
    # Pass preprocessed blob into model
    yolo.setInput(image_blob)
    
    # Use forward() method to obtain output layers
    detection_layers = yolo.forward(output_layer)
    
    # Initialization for non-max suppression (NMS)
    # NMS is used because the YOLO method can identify many overlapping detections 
    # of the same object. NMS will only show the highest confidence detection.
    # Declare lists for class ID, bounding boxes, and confidences
    class_ids_list = []
    boxes_list = []
    confidences_list = []
    
    # Loop through all output layers
    for layer in detection_layers:
        # Loop through all the detected objects in each layer
        for detection in layer:
            # detection[0 through 4] contains the two center points of box along 
            # with box width and height
            # detection[5+] conatins confidence score for all objects detected in box
            scores = detection[5:]
            predicted_class = np.argmax(scores)
            confidence = scores[predicted_class]
            
            '''
            Only consider detected objects above a certain confidence value
            This value can be changed higher or lower depending on the desired accuracy.
            '''
            if confidence > 0.5:
                # Get the bounding box info of the prediction.
                # We need to multiply the coordinates by our original height and width
                # because we resized the blob for processing and need to convert back
                bounding_box = detection[0:4] * np.array([width, height, width, height])
                # Bounding box coords are float, convert to int
                (center_x, center_y, box_width, box_height) = bounding_box.astype("int")
                # Determine x & y boundaries based on center coords and height/width
                start_x = int(center_x - (box_width / 2))
                start_y = int(center_y - (box_height / 2))
                
                # Save class ID, bounding box info,  and confidences in a list for 
                # NMS processing.
                # Make sure to pass confidence as float and width and height as integers
                class_ids_list.append(predicted_class)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                
    # Applying the NMS will return only the selected max value ids while suppressing 
    # the non maximum (weak) overlapping bounding boxes      
    # Non-Maxima Suppression confidence set as 0.5 & max_suppression 
    # threhold for NMS as 0.4 (can be adjusted)
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
    
    # Loop through the final set of NMS max detections remaining after NMS and draw 
    # the bounding boxes and ID text
    for max_value_id in max_value_ids:
        max_class_id = max_value_id[0]
        box = boxes_list[max_class_id]
        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]
        
        # Get the predicted class id, label, & confidence
        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        confidence = confidences_list[max_class_id]          
                
        end_x = start_x + box_width
        end_y = start_y + box_height
        
        # Assign a color to mask based on label index
        box_color = box_colors[predicted_class_id]
        
        # Convert the mask color to a list to be able to color the bounding box
        box_color = [int(i) for i in box_color]
        
        # Print detected object in console
        print("Predicted object: {} - {:.2f}%".format(predicted_class_label, confidence * 100))
        
        # Draw bounding box around image
        cv2.rectangle(input_image, (start_x, start_y), (end_x, end_y), box_color, 1)
        cv2.putText(input_image, predicted_class_label + ": " + str(round(confidence * 100, 2)) + "%", (start_x, start_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, box_color, 1)
    
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