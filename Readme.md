# **CS5330‑Proj1 — By Yuyang Tian**

### **Project Overview**
This repository explores the fundamentals of capturing, manipulating, and writing images and videos in real time. It showcases several filters and transformations—grayscale, sepia, blur, Sobel operators, face detection, a meme generator, and even a 3D point cloud generator.

| **Filename**         | **Description**                                                                                                          |
|----------------------|--------------------------------------------------------------------------------------------------------------------------|
| `imgDisplay.cpp`     | Basic still‑image operations (open, display, and save).                                                                  |
| `vidDisplay.cpp`     | Live‑stream video application, enabling multiple filters via keyboard inputs.                                            |
| `filterDisplay.cpp`  | Allows real‑time filters on images (keyboard‑driven) with grayscale, sepia, blur.                                        |
| `filter.cpp`         | Core filter implementations (grayscale, sepia, blur, Sobel X/Y, gradient magnitude, face detection, etc.).              |
| `da-video.cpp`       | Generates a 3D point cloud from depth data.                                                                              |
| `pcl3d.h / pcl3d.cpp`| Supports Task 11: 3D Point Cloud generation/visualization logic.                                                          |
| `memeGenerator.cpp`  | Meme generator extension (adds customizable top/bottom text).                                                            |

---

## **1. Project Description**

This project is split between two main functionalities:

- **Still images** (handled in `imgDisplay.cpp`).
- **Live video** (handled in `vidDisplay.cpp`), applying real-time filters triggered by keyboard input.

The filters include:
- Grayscale (traditional & custom‑weighted)
- Sepia tone
- Blur (with two performance approaches)
- Sobel X, Sobel Y, and gradient magnitude
- Face detection
- 3D point cloud generation (depth‑based)
- Meme generator extension

Below are details for each major task.

---

## **2. Tasks Breakdown**

### **Task 1: Display Image**
- **File:** `imgDisplay.cpp`
- **Description:**
   - Opens and displays an image using OpenCV.
   - Press `s` to save the image, or any other key to close and exit.

### **Task 2: Display Live Video**
- **File:** `vidDisplay.cpp`
- **Description:**
   - Uses `VideoCapture` to open the camera and read frames.
   - Displays each frame live.
   - Press `s` to save a frame, or `q` to quit.

### **Task 3: Grayscale Filter**
- Triggered by pressing `g`.
- Converts each frame to grayscale using:  
  \[
  Y = 0.299R + 0.587G + 0.114B
  \]
- Pressing another key reverts to BGR color.

### **Task 4: Alternative Grayscale**
- Uses custom weights, e.g.  
  \[
  Y = 0.005R + 0.8G + 0.195B
  \]
- Produces a darker effect on shadowed areas.

### **Task 5: Sepia Tone & Vignetting**
- **Sepia**: Manual matrix transform on each pixel, then clipping values to `[0, 255]`.
- **Vignette**: Darkens edges by computing a distance from the image center.  
  \[
  \text{vignetteFactor} = 1 - \frac{\text{distanceFromCenter}}{\text{maxDistance}}
  \]

### **Task 6: Blur Filter**
- **Version 1** (5x5 kernel) vs. **Version 2** (separable filter).
- *Version 2* is faster since it breaks a 5×5 convolution into a pair of 1×5 and 5×1 passes, reducing multiplications from 25 to 10 per pixel.

### **Task 7: Sobel X and Sobel Y**
- Sobel filters approximate derivatives in the X and Y directions using separable filters:
   - Sobel X uses `[1, 0, -1]` (horizontal) × `[1, 2, 1]` (vertical).
   - Sobel Y uses `[1, 0, -1]` (vertical) × `[1, 2, 1]` (horizontal).
- Gaussian smoothing is applied first; results are stored in `CV_16S3C`, then converted to 8‑bit for display.

### **Task 8: Gradient Magnitude**
- Combines Sobel X & Y via Euclidean norm:
  \[
  \sqrt{(S_x)^2 + (S_y)^2}
  \]
- Visualizes the strength of intensity changes in both directions.

### **Task 9: Color Quantization**
- Reduces the color space to fewer “levels.”
- Creates “steps” or “buckets” to clamp each pixel’s color to discrete bins.
- Higher levels = more colors; lower levels = more posterized effect.

### **Task 10: Face Detection**
- Utilizes `faceDetection` to detect and draw rectangles around faces in a frame.

### **Task 11: 3D Point Cloud Generation (DA2)**
- **Files:** `da-video.cpp`, `pcl3d.h`, `pcl3d.cpp`
- Uses camera intrinsics to project each pixel into 3D (x, y, z + color) from a depth map.
- Saves to PCD format and visualizes with the PCL library.

### **Task 12: Additional Filters**
1. **Blur Background Filter**
   - Uses face detection bounding boxes to exclude faces from blur. A mask is created: faces remain sharp, everything else is blurred.

2. **Fog Filter**
   - Computes a *fogFactor* from depth:  
     \[
     \text{fogFactor} = 1.0 - e^{(-\text{depth}^2 \cdot \text{fogDensity})}
     \]
   - Blends pixels with white to mimic fog.

3. **Selective Color**
   - Keeps faces in color but renders the rest as grayscale using masks.

---

## **Meme Generator (Extension)**
- **File:** `memeGenerator.cpp`
- **Description:**
   1. Opens an image.
   2. Allows adding top/bottom text with an outlined styling for readability.
   3. Live preview via OpenCV.
   4. Keyboard shortcuts:
      - `t`: Edit top text
      - `b`: Edit bottom text
      - `c`: Clear all text
      - `s`: Save and exit
      - `q`: Quit without saving

**Video Showcase:** [Google Drive Link](https://drive.google.com/file/d/1_aYZx_wTCwwtKma4UQ_nRCyc7jp9FglT/view?usp=drive_link)

---

## **Reflection**

> *“My biggest takeaway from CS5330 is how vast and fascinating Computer Vision can be. Implementing these filters—from simple grayscale to a 3D point cloud—made me realize how theory translates into practice. Even though I’d seen these concepts in lectures, actually coding them forced me to internalize the details. I was also able to explore more advanced features like turning 2D images into 3D models using depth maps. I had lots of ambitious ideas (like gesture-based triggers), which I learned were more complex to implement than they initially seemed, but it was still a fun, enlightening journey!”*

---

## **Acknowledgments & References**

- [Baeldung: Focal Length & Intrinsic Parameters](https://www.baeldung.com/cs/focal-length-intrinsic-camera-parameters)
- [PCL Tutorials: Reading & Visualizing PCD Files](https://pcl.readthedocs.io/projects/tutorials/en/master/reading_pcd.html)
- [PyImageSearch Face Detection Examples](https://pyimagesearch.com/2021/01/04/opencv-augmented-reality-ar/)
- [OpenCV Docs 4.5.1](https://docs.opencv.org/4.5.1/)
- …and various StackOverflow/YouTube resources listed in the original README text.

---

## **Build & Run**

### **Prerequisites**
- [OpenCV](https://opencv.org/)
- [PCL](https://pointclouds.org/) (for the 3D point cloud task)
- CMake toolchain for building C++ projects.

### **Steps**
1. **Clone the Repository**
   ```bash
   git clone https://github.com/anpu9/CS5330-proj1.git
