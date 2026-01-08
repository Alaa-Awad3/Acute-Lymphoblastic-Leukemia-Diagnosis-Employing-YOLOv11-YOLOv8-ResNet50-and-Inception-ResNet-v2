import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Define the folder paths for input images and output images
input_folder = "The path containing original images"
output_folder = "The path where the segmented images will be saved"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Define HSV color thresholds for purple (dominant color of blast cells)
lower_purple = np.array([120, 50, 50])  # Lower boundary for purple in HSV
upper_purple = np.array([160, 255, 255])  # Upper boundary for purple in HSV

# 3. Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".tif") or filename.endswith(".jpg") or filename.endswith(
            ".jpeg"):  # Add other formats if necessary
        # Construct the full file path
        img_path = os.path.join(input_folder, filename)

        # 4. Load the image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display purposes

        # 5. Convert the image from RGB to HSV color space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 6. Create a binary mask where the purple areas are white (255) and other areas are black (0)
        mask = cv2.inRange(image_hsv, lower_purple, upper_purple)

        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 7. Apply the mask to the original image to segment the purple cells
        segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_cleaned)

        # 8. Save the resulting segmented image
        output_path = os.path.join(output_folder, f"segmented_{filename}")
        segmented_bgr = cv2.cvtColor(segmented_image,
                                     cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR for saving with OpenCV
        cv2.imwrite(output_path, segmented_bgr)

        print(f"Processed and saved: {filename}")

print("Processing complete!")
