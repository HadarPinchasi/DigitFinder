import sys
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt  

def preprocess_roi(img, x_min, y_min, x_max, y_max):
    """
    Preprocesses the region of interest (ROI) containing the potential digits.
    """
    if y_min >= y_max or x_min >= x_max:
        print(f"Warning: Invalid ROI coordinates ({x_min},{y_min}) to ({x_max},{y_max}). Skipping preprocessing.")
        return np.zeros((10, 10), dtype=np.uint8)  # Return small black image

    roi = img[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        print(
            f"Warning: Empty ROI extracted for coordinates ({x_min},{y_min}) to ({x_max},{y_max}). Skipping preprocessing.")
        return np.zeros((10, 10), dtype=np.uint8)  # Return small black image

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    photo = b

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(photo)

    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh


def extract_number_from_image(img_path, ocr_instance):
    """
    Reads an image, detects text boxes,
    preprocesses the ROI, and extracts digits.
    """
    #reading the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return "read_error"

    try:
        blurred = cv2.GaussianBlur(img, (15, 15), 4.5) #blurring the image
        result = ocr_instance.ocr(blurred, cls=False, det=True, rec=False)#detecting the arees with text
    except Exception as e:
        print(f"Error during initial OCR detection on {img_path}: {e}")
        return "ocr_detect_error"
    if not result or not result[0]:
        print(f"Warning: No text boxes detected by initial OCR in {os.path.basename(img_path)}")
        x_min, y_min, x_max, y_max = 0, 0, img.shape[1], img.shape[0]
        processed_roi = preprocess_roi(img, x_min, y_min, x_max, y_max)
        print(f"-> Processing full image for {os.path.basename(img_path)}")
    else:
        boxes = []
        for box_info in result[0]:
            coords = box_info
            for x, y in coords:
                boxes.append((int(x), int(y)))

        if not boxes:
            print(f"Warning: No coordinates extracted from detection results in {os.path.basename(img_path)}")
            return "no_box_coords"
        #calculating the borders of the ROI
        x_coords, y_coords = zip(*boxes)
        padding = 0
        x_min = max(min(x_coords) - padding, 0)
        y_min = max(min(y_coords) - padding, 0)
        x_max = min(max(x_coords) + padding, img.shape[1])
        y_max = min(max(y_coords) + padding, img.shape[0])

        img_with_bbox = img.copy()
        cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        processed_roi = preprocess_roi(img, x_min, y_min, x_max, y_max)

    if processed_roi is None or processed_roi.size == 0 or processed_roi.shape[0] < 5 or processed_roi.shape[1] < 5:
        print(f"Warning: Preprocessed ROI is invalid or too small for {os.path.basename(img_path)}. Skipping OCR.")
        return "roi_invalid"

    try:
        result_roi = ocr_instance.ocr(processed_roi, cls=False, det=True, rec=True)
    except Exception as e:
        print(f"Error during OCR on ROI for {img_path}: {e}")
        return "ocr_roi_error"

    digits = ""
    if result_roi and result_roi[0]:
        try:
            sorted_results = sorted(result_roi[0], key=lambda item: item[0][0][0])
        except (IndexError, TypeError):
            print(f"Warning: Could not sort results for {os.path.basename(img_path)}, using original order.")
            sorted_results = result_roi[0] if result_roi[0] else []

        for line in sorted_results:
            try:
                text_info = line[1]
                text = text_info[0]
                confidence = text_info[1]

                if confidence > 0.5:
                    text = text.replace('O', '0').replace('o', '0')
                    text = text.replace('I', '1').replace('l', '1')
                    text = text.replace('S', '5').replace('s', '5')
                    text = text.replace('Z', '2').replace('z', '2')
                    text = text.replace('B', '8')
                    text = text.replace('G', '6')

                    cleaned_text = ''.join([c for c in text if c in '0123456789.'])
                    digits += cleaned_text
                    if digits.endswith("00"):
                        digits = digits[:-2]

            except (IndexError, TypeError):
                print(f"Warning: Malformed result line encountered in ROI OCR for {os.path.basename(img_path)}")
                continue

    return digits if digits else "none"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]

    if not os.path.isfile(img_path):
        print(f"Error: File '{img_path}' not found.")
        sys.exit(1)
    
    # Validate image format
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("Error: Only PNG and JPEG image formats are supported.")
        sys.exit(1)

    #Initialize PaddleOCR Once
    print("Initializing PaddleOCR...")
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False, det_db_box_thresh=0.7)
        print("PaddleOCR initialized.")
    except Exception as e:
        print(f"Error initializing PaddleOCR: {e}")
        print("Please ensure PaddleOCR and its dependencies are installed correctly.")
        sys.exit(1)

    print(f"\n--- Processing: {os.path.basename(img_path)} ---")
    try:
        extracted_number = extract_number_from_image(img_path, ocr)
        print(f"Result for {os.path.basename(img_path)}: {extracted_number}")
    except Exception as e:
        print(f"!! Critical error processing {os.path.basename(img_path)}: {e}")
