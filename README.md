# Seven Segment Digit Recognizer
This Python script uses **PaddleOCR** and **OpenCV** to detect and extract numerical digits from images of seven-segment displays or other text-containing regions.
## Features
- Automatically detects text regions using PaddleOCR
- Preprocesses the region of interest (ROI) using CLAHE and thresholding
- Extracts digits with cleaning heuristics (e.g., replacing similar letters like 'O' → '0')
- Designed to work with `.png`, `.jpg`, or `.jpeg` image files

---
##  Requirements
Make sure you have **Python 3.7+** installed.
Install the required packages with pip:
```
pip install paddleocr opencv-python numpy matplotlib
```
PaddleOCR also depends on paddlepaddle. Install it based on your system and whether you use GPU or CPU.
For most users (CPU only), you can run:
```
pip install paddlepaddle
```
Or for GPU:
```
pip install paddlepaddle-gpu
```
## How to Run

From your terminal, navigate to the directory where the script is located, and run:
```
python digitFinder.py path/to/image.jpg
```
