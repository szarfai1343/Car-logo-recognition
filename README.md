This project uses YOLOv5 and OpenCV in C++ to detect:
-Car brand logos
-License plates

Project structure:
ProjektDetekcje.cpp - Main C++ source file
config_files/ - Configuration files and model storage
car_logo_model.zip - Zipped YOLOv5 model for logo detection
license_model.zip - Zipped YOLOv5 model for license plate detection

Unzip both `car_logo_model.zip` and `license_model.zip` into the `config_files/` directory.
Make sure you have installed OpenCV.

You can use the system on:
Single images
Video files
Live camera feed (with minor modifications)
