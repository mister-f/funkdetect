# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:55:47 2020

@author: funka
"""

# Import libraries
from PIL import Image
import cv2
import pytesseract

''' 
Declare .exe path for Tesseract
This is the default install path on Windows, but may need to be changed if 
installed in a custom location.
'''
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

''' 
Read in image for OCR.
Change the file path to whatever source image you are using.
'''
ocr_image = cv2.imread("images/testing/sonnet18.jpg")

# Image Preprocessing
# Convert color image to grayscale
pre_image = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image using Otsu's Method to create black/white binary image
pre_image = cv2.threshold(pre_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Use median blur to smooth image
pre_image = cv2.medianBlur(pre_image, 3)

# Save the preprocessed image
cv2.imwrite('temp_image.jpg', pre_image)

# Read in temporary image file in PIL format
pil_image = Image.open('temp_image.jpg')

# Use Tesseract for read pre-processed PIL image
ocr_text = pytesseract.image_to_string(pil_image)
ocr_text = ocr_text[:-1]

# Display OCR text and original image
print(ocr_text)
cv2.imshow("Original Image", ocr_image)