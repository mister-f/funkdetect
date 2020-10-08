# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:20:19 2020

@author: funka
"""

# Import libraries
import cv2
import pytesseract
import pkg_resources

# Declare .exe path for tesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Print Tesseract version
print(pkg_resources.working_set.by_key['pytesseract'].version)

# Print OpenCV version
print(cv2.__version__)