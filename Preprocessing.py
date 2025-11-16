import numpy as np
import cv2
import os
import csv

# -----------------------------
# IMAGE PREPROCESSING FUNCTION
# -----------------------------
minValue = 70

def preprocess_image(path):    
    frame = cv2.imread(path)
    if frame is None:
        print(f"Error: Cannot read image {path}")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res


# -----------------------------
# DATA PREPARATION PIPELINE
# -----------------------------

# Output directories
output_root = "data2"
train_dir = os.path.join(output_root, "train")
test_dir = os.path.join(output_root, "test")

# Create directories if missing
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Input dataset path
input_root = "data/train"

# CSV headers for ML models (not writing CSV now but kept)
pixels_header = ["label"] + [f"pixel{i}" for i in range(64*64)]

label = 0
total_count = 0
train_count = 0
test_count = 0

# Loop through each class folder
for root, dirs, files in os.walk(input_root):
    for class_name in dirs:

        class_input_path = os.path.join(input_root, class_name)
        class_train_output = os.path.join(train_dir, class_name)
        class_test_output = os.path.join(test_dir, class_name)

        os.makedirs(class_train_output, exist_ok=True)
        os.makedirs(class_test_output, exist_ok=True)

        print(f"Processing class: {class_name}")

        file_list = os.listdir(class_input_path)

        # Change this to 0.75 for real train/test split
        split_point = 100000000000000000  # your very large number keeps everything in train

        i = 0
        for file in file_list:
            total_count += 1

            img_path = os.path.join(class_input_path, file)
            processed = preprocess_image(img_path)

            if processed is None:
                continue  # Skip unreadable images

            # Decide where to save
            if i < split_point:
                train_count += 1
                save_path = os.path.join(class_train_output, file)
            else:
                test_count += 1
                save_path = os.path.join(class_test_output, file)

            cv2.imwrite(save_path, processed)
            i += 1

        label += 1

# Print statistics
print("Total images:", total_count)
print("Train images:", train_count)
print("Test images :", test_count)
