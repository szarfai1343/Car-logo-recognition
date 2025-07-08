# Car Logo and License Plate Detection

This project uses **YOLOv5** and **OpenCV** in **C++** to detect:
- Car brand logos
- License plates

## 📁 Project Structure
ProjektDetekcje.cpp - Main C++ source file
config_files/ - Configuration files and model storage
car_logo_model.zip - Zipped YOLOv5 model for logo detection
license_model.zip - Zipped YOLOv5 model for license plate detectio

## ⚙️ How to Run

### 1. Unpack the Models

Unzip both `car_logo_model.zip` and `license_model.zip` into the `config_files/` directory.

After extraction, the folder should contain model files such as `.pt`, `.onnx`, or others used in your code.

### 2. Build the Project

Make sure you have installed:
- OpenCV
- (Optionally) libtorch, if you're using PyTorch C++ API
- A C++ compiler like `g++` or use `CMake`

Example build command:
```bash
g++ ProjektDetekcje.cpp -o detector `pkg-config --cflags --libs opencv4`

