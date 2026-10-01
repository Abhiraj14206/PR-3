# Computer Vision & Real-Time Object Detection

## 📌 Project Overview

This project explores different **Computer Vision and Deep Learning techniques** using Python, OpenCV, YuNet, and YOLOv8.

The project starts with basic image processing techniques such as grayscale conversion, thresholding, morphological operations, bitwise operations, histograms, and brightness/contrast adjustment. It then moves toward deep-learning-based **face detection using YuNet** and **object detection using YOLOv8n**.

Finally, YuNet and YOLOv8 are combined into a real-time webcam detection system, followed by an **FPS performance comparison**.

Project Explanation Video : https://drive.google.com/drive/folders/1Qb-lVKs8lF5754JAESuW11-ihtzuAcyj
---

## 🎯 Objectives

The main objectives of this project are:

* Understand basic image preprocessing techniques.
* Perform morphological image operations.
* Understand bitwise operations on images.
* Analyze grayscale and color histograms.
* Perform brightness and contrast adjustment.
* Detect faces using the YuNet face detector.
* Detect multiple objects using YOLOv8n.
* Perform real-time detection using a webcam.
* Combine face and object detection in a single application.
* Compare the processing speed of YuNet, YOLOv8, and the combined system.

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **Ultralytics YOLO**
* **YuNet Face Detector**
* **Jupyter Notebook**

---

## 📂 Project Structure

```text
A-PR-3/
│
├── A-PR-3.ipynb
│
├── images/
│   ├── Face images
│   ├── Object detection images
│   └── Other test images
│
├── Models/
│   └── face_detection_yunet_2023mar.onnx
│
└── README.md
```

---

# 🔹 Task 1 – Image Processing & Morphological Operations

The first part of the project focuses on basic image processing.

### Operations Performed

* Image loading
* Grayscale conversion
* Binary thresholding
* Erosion
* Dilation
* Opening
* Closing
* Different kernel shapes and sizes

### Morphological Operations

**Erosion**

Shrinks white regions and can remove small bright noise.

**Dilation**

Expands white regions and can fill small gaps.

**Opening**

```text
Opening = Erosion → Dilation
```

It can be used to remove small noise while preserving the main object.

**Closing**

```text
Closing = Dilation → Erosion
```

It can be used to fill small holes and gaps.

Different structuring elements were also tested:

* Rectangular
* Elliptical
* Cross

with different kernel sizes such as:

```text
3 × 3
5 × 5
9 × 9
```

---

# 🔹 Task 2 – Bitwise Operations & Histograms

The project demonstrates image bitwise operations using masks.

### Bitwise Operations

* AND
* OR
* XOR
* NOT

These operations were demonstrated using a circular mask and rectangular mask.

A circular mask was also applied to an input image to extract a selected region.

---

## 📊 Histogram Analysis

The project calculates:

* Grayscale histogram
* RGB color histograms

Histograms help understand the distribution of pixel intensities.

The project also demonstrates brightness and contrast adjustment using:

```text
new_pixel = alpha × pixel + beta
```

Where:

* **Alpha** controls contrast.
* **Beta** controls brightness.

The project compares:

* Original image
* Bright/high-contrast image
* Dark/low-contrast image

and displays their corresponding histograms.

---

# 🔹 Task 3 – YuNet Face Detection

The project uses **YuNet**, a lightweight deep-learning-based face detector.

The YuNet model used in the project is:

```text
face_detection_yunet_2023mar.onnx
```

### Features

The detector provides:

* Face bounding boxes
* Confidence scores
* Five facial landmarks

The project performs face detection on multiple images and also implements **real-time webcam face detection**.

### Confidence Threshold

Different confidence thresholds can affect detection results.

A lower threshold may detect more faces but can increase false detections.

A higher threshold can provide more reliable detections but may miss difficult faces.

---

# 🔹 Task 4 – YOLOv8 Object Detection

The project uses the lightweight:

```text
YOLOv8n
```

model through the Ultralytics library.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

YOLOv8n is used for detecting multiple object classes in images.

### Operations Performed

* Image object detection
* Detection confidence analysis
* Multiple-image detection
* Real-time webcam object detection
* Confidence threshold comparison
* IoU threshold comparison
* Per-class detection counting
* Detection summary visualization

---

## 🎚️ Confidence Threshold

The project tests different confidence values:

```text
0.25
0.50
0.75
```

A higher confidence threshold generally results in fewer but more confident detections.

---

## 📐 IoU Threshold

Different IoU values are also tested:

```text
0.30
0.50
0.70
```

IoU is used during Non-Maximum Suppression to determine how overlapping bounding boxes are handled.

---

# 🔹 Task 5 – Combined Face & Object Detection

The final task combines:

```text
YuNet + YOLOv8n
```

YuNet is responsible for detecting faces, while YOLOv8n detects general objects.

The combined system works with a webcam and displays both detection results simultaneously.

### Pipeline

```text
Webcam Frame
      │
      ├───────────────┐
      ↓               ↓
   YuNet            YOLOv8n
      │               │
 Face Detection   Object Detection
      │               │
      └───────┬───────┘
              ↓
       Combined Output
```

---

# ⚙️ Image Preprocessing Experiment

Morphological opening is also applied to webcam frames before detection.

The project compares:

```text
Original Frame → Detection
```

with:

```text
Morphological Opening → Detection
```

This demonstrates how preprocessing can affect computer vision detection.

Light preprocessing can help with noise, but excessive preprocessing may remove useful image details.

---

# 🚀 FPS Performance Benchmark

The project benchmarks three approaches:

1. YuNet
2. YOLOv8n
3. YuNet + YOLOv8n

The FPS is calculated using a set of captured webcam frames.

```text
FPS = Number of Frames / Processing Time
```

The results are displayed in both console output and a bar chart.

### Benchmark

| Method   | Purpose                  |
| -------- | ------------------------ |
| YuNet    | Face detection           |
| YOLOv8n  | General object detection |
| Combined | Face + object detection  |

The actual FPS values depend on the computer hardware and environment where the notebook is executed.

---

# 📊 Final Comparison

| Technique                | Type                      | Main Purpose                       |
| ------------------------ | ------------------------- | ---------------------------------- |
| Thresholding             | Classical Computer Vision | Basic image segmentation           |
| Histograms               | Classical Computer Vision | Pixel intensity analysis           |
| Morphological Operations | Classical Computer Vision | Noise removal and shape processing |
| YuNet                    | Deep Learning             | Face detection                     |
| YOLOv8n                  | Deep Learning             | General object detection           |
| YuNet + YOLOv8n          | Deep Learning             | Combined face and object detection |

---

# 💡 Key Learnings

Through this project, I learned:

* How images are represented using pixels.
* How grayscale and binary images are created.
* How morphological operations affect image regions.
* How masks can be manipulated using bitwise operations.
* How histograms can be used to analyze images.
* How brightness and contrast affect image distributions.
* How YuNet can be used for face detection.
* How YOLOv8 can perform general object detection.
* How confidence and IoU thresholds affect detection.
* How to implement real-time webcam detection.
* How to combine multiple computer vision models.
* How to benchmark computer vision models using FPS.

---

# 🔧 Installation

Install the required Python libraries:

```bash
pip install opencv-python opencv-contrib-python ultralytics matplotlib numpy
```

---

# ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the notebook

```text
A-PR-3.ipynb
```

Open it using:

* Jupyter Notebook
* JupyterLab
* VS Code

### 3. Install dependencies

```bash
pip install opencv-python opencv-contrib-python ultralytics matplotlib numpy
```

### 4. Update file paths

The notebook currently contains local Windows paths such as:

```text
D:\Deep_Learning\PR-3\
```

Update these paths according to your own project folder.

### 5. Run the notebook

Run the cells sequentially to reproduce the image-processing, face-detection, object-detection, and FPS experiments.

---

# ⚠️ Requirements

For the complete project, you need:

* Python 3.x
* Webcam for real-time detection
* YuNet ONNX model
* YOLOv8n model
* Required Python libraries
* Test images

YOLOv8n can be downloaded automatically by Ultralytics when the model is loaded.

---

# 🌍 Applications

The techniques demonstrated in this project can be applied to:

* Security camera systems
* Face detection systems
* Object monitoring
* Smart surveillance
* Robotics
* Industrial inspection
* Real-time computer vision
* Automated monitoring systems
* Image preprocessing pipelines

---

# 🔮 Future Improvements

Possible improvements include:

* Add face recognition.
* Track detected objects across video frames.
* Store detection results in a database.
* Add FPS and detection statistics to the live interface.
* Improve preprocessing automatically based on image quality.
* Test larger YOLO models for improved detection accuracy.
* Build a graphical user interface.
* Deploy the application as a standalone computer vision application.
* Optimize the combined model for low-power devices.


---

## ⭐ Project Summary

This project demonstrates a progression from **classical computer vision techniques to modern deep-learning-based detection**.

It combines OpenCV image processing, YuNet face detection, and YOLOv8 object detection to create a real-time computer vision pipeline and evaluates the processing performance using FPS benchmarking.
