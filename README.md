# funkdetect
Funkdetect is a small collection of python scripts that can perform OCR (optical character recognition) and object recognition in still images, saved video, streaming video.

## Tesseract OCR
The tesseract_ocr.py script can be used to extract text from image files. Tesseract works best with structured text, so handwritten images are less likely to yield accurate results. The captured text is output to the console. 
  #### Dependencies/Libraries
  * Tesseract executable - [Installer](https://github.com/UB-Mannheim/tesseract/wiki) or [Source](https://github.com/tesseract-ocr/tesseract/releases)
  * Tesseract library - 'pip install tesseract'
  * Python Tesseract library - 'pip install pytesseract'
  * OpenCV (Open Computer Vision) library - 'pip install opencv-python'
  #### Usage
  1. Install all necessary libraries.
  1. Change the file path for the 'ocr_image' variable to whatever image file is being analyzed.
      1. Supported file formats: .bmp, .dib, .exr, .hdr, .jpeg, .jpg, .jpe, .jp2, .png, .webp, .pbm, .pic, .pgm, .ppm .pxm, .pnm, .pfm, .ras, .sr, .tiff, .tif
  1. Run the script, output will be printed to the console.
  #### Example Output
  
## Single Object Recognition
There are several scripts (inceptionv3.py, resnet.py, vggnet16.py, vggnet19.py, xception.py) that use various models to identify the primary object in an image. Each script uses a pretrained model for the identification. The highest confidence result is added to the displayed original image and the five highest confidence results are output to the console. These methods will download the pretrained models from the internet when used for the first time. These models can be large (~500MB) but once the model has been downloaded once it will not need to be re-downloaded.
  #### Dependencies/Libraries
  * Tensorflow - 'pip install tensorflow'
  * Keras - 'pip install keras'
  * OpenCV (Open Computer Vision) library - 'pip install opencv-python'
  * NumPy - 'pip install numpy'
  #### Usage
  1. Install all necessary libraries.
  1. Usage is the same for: inceptionv3.py, resnet.py, vggnet16.py, vggnet19.py, and xception.py
  1. Change the file path for the 'image_path' variable to whatever image file is being analyzed.
      1. Supported file formats: .bmp, .gif, .jpeg, .jpg, .png
  1. Run the script, top 5 confidence detections will be printed to the console and the image will be displayed with the top result added.
      1. If running the first time, the model will automatically download for the given method. This download only hapens once.
  #### Example Output
 
## Multi Object Recognition (Image and Video)
There are several models represented for determining multiple objects in a scene. MobileNet-SSD is the fastest method, but also has lower accuracy and only detects 20 classes of objects. Mask R-CNN can be used to obtain masking information for detected objects and detects 90 object classes. It is much slower however and mroe difficult to use with video. The YOLO models have high accuracy and can detect 80 object classes. The Tiny-YOLO model can also be used with has faster detection but sacrifices accuracy. Output for each script displays picture or video with bounding boxes surrounding objects (or object masks for Mask R-CNN) along with listed confidences. Detected objects and confidences are also listed as console output.
  #### Dependencies/Libraries
  * OpenCV (Open Computer Vision) library - 'pip install opencv-python'
  * NumPy - 'pip install numpy'
  * Model weights/configurations
    * MobileNet-SSD - Weights/config are in 'datasets' folder
    * Mask R-CNN - [Weights](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API)
    * YOLO - [Weights](https://github.com/AlexeyAB/darknet)
  #### Usage
  1. Install all necessary libraries and download any necessary weight/configuration files into 'datasets' folder.
  1. Usage is the same for maskrcnn, mobilenetssd, and yolo methods
  1. For image analysis, change the file path for the 'input_image' variable to whatever image file is being analyzed.
      1. Supported file formats: .bmp, .dib, .exr, .hdr, .jpeg, .jpg, .jpe, .jp2, .png, .webp, .pbm, .pic, .pgm, .ppm .pxm, .pnm, .pfm, .ras, .sr, .tiff, .tif
  1. For analysis of saved video, change the file path for the 'video_stream' variable to whatever video file is being analyzed.
      1.  Supported file formats: accepted file formats can vary depending on what codecs are installed.
  1. Make sure that the correct file names/directories are used for the model wights and configurations in the 'maskrcnn', 'mobilenetssd', or 'yolo' variables
  1. Run the script, detections will be printed to the console and the image or video will be displayed with bounding boxes or masks. Press 'q' to exit out of any video windows.
  #### Example Output
  
