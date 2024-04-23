
# import os
# from skimage.io import imread
# from skimage import color, util
# from skimage.filters import threshold_otsu, gaussian
# import numpy as np
# from PIL import Image

# # Function to preprocess and save leads
# def preprocess_and_save_leads(image_folder, output_folder):
#     print("Preprocessing images...")
#     for root, files in os.walk(image_folder):
#         for filename in files:
#             if filename.endswith('.jpg'):  # Adjust file format here
#                 class_folder = os.path.basename(root)
#                 class_output_folder = os.path.join(output_folder, class_folder)
#                 os.makedirs(class_output_folder, exist_ok=True)
#                 image_path = os.path.join(root, filename)
#                 preprocess_and_save_single(image_path, class_output_folder)
#     print("Preprocessing completed.")

# # Function to preprocess a single image and save leads
# def preprocess_and_save_single(image_path, output_folder):
#     # Load the ECG image
#     image = imread(image_path)

#     # Convert to grayscale if image is RGB
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         grayscale_image = color.rgb2gray(image)
#     else:
#         grayscale_image = image

#     # Divide the ECG image into 12 leads
#     leads = divide_leads(grayscale_image)

#     # Create a folder for the ECG image
#     image_name = os.path.splitext(os.path.basename(image_path))[0]
#     image_folder = os.path.join(output_folder, image_name)
#     os.makedirs(image_folder, exist_ok=True)

#     # Preprocess each lead and save
#     for i, lead in enumerate(leads, start=1):
#         preprocessed_lead = preprocess_lead(lead)
#         lead_output_path = os.path.join(image_folder, f'{image_name}_lead_{i}.png')  # Adjust output format here
#         preprocessed_lead.save(lead_output_path)

# # Function to divide ECG image into leads
# def divide_leads(image):
#     """
#     This function divides the ECG image into 13 leads including long lead.
#     Returns: List containing all 13 leads divided
#     """
#     # Define the coordinates for each lead
#     leads_coordinates = [
#         (300, 600, 150, 643),   # Lead 1
#         (300, 600, 646, 1135),  # Lead aVR
#         (300, 600, 1140, 1625), # Lead V1
#         (300, 600, 1630, 2125), # Lead V4
#         (600, 900, 150, 643),   # Lead 2
#         (600, 900, 646, 1135),  # Lead aVL
#         (600, 900, 1140, 1625), # Lead V2
#         (600, 900, 1630, 2125), # Lead V5
#         (900, 1200, 150, 643),  # Lead 3
#         (900, 1200, 646, 1135), # Lead aVF
#         (900, 1200, 1140, 1625),# Lead V3
#         (900, 1200, 1630, 2125),# Lead V6
#         (1250, 1480, 150, 2125) # Long Lead
#     ]

#     leads = []
#     for top, bottom, left, right in leads_coordinates:
#         leads.append(image[top:bottom, left:right])

#     return leads

# # Function to preprocess a lead (remove background, etc.)
# def preprocess_lead(lead):
#     """
#     This function performs preprocessing on the extracted leads.
#     """
#     # Smoothing image
#     blurred_image = gaussian(lead, sigma=1)

#     # Thresholding to distinguish foreground and background
#     global_thresh = threshold_otsu(blurred_image)
#     binary_global = blurred_image < global_thresh

#     # Convert boolean image to unsigned byte image
#     preprocessed_lead = util.img_as_ubyte(binary_global)

#     # Convert image mode to 'L' (8-bit pixels, black and white)
#     preprocessed_lead = preprocessed_lead.astype(np.uint8)

#     # Convert to PIL image
#     preprocessed_pil_image = Image.fromarray(preprocessed_lead)

#     return preprocessed_pil_image

# # Example usage:
# if __name__ == "__main__":
#     input_folder_train = './NEW_DATA/train'
#     input_folder_validation = './NEW_DATA/validation'
#     output_folder_train = './PreprocessedData/train'
#     output_folder_validation = './PreprocessedData/validation'

#     # Preprocess train images
#     preprocess_and_save_leads(input_folder_train, output_folder_train)

#     # Preprocess test images
#     preprocess_and_save_leads(input_folder_validation, output_folder_validation)

# # the dimensions of the entire image should be at least 1480 pixels in height and 2125 pixels
# # in width to accommodate all the specified regions of interest.

import os
from skimage.io import imread
from skimage import color, util
from skimage.filters import threshold_otsu, gaussian
import numpy as np
from PIL import Image

# Function to preprocess and save leads
def preprocess_and_save_leads(image_folder, output_folder):
    if not os.path.exists(image_folder):
        print(f"The folder '{image_folder}' does not exist.")
        return
    
    print("Preprocessing images...")
    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith('.jpg'):  # Adjust file format here
                class_folder = os.path.basename(root)
                class_output_folder = os.path.join(output_folder, class_folder)
                os.makedirs(class_output_folder, exist_ok=True)
                image_path = os.path.join(root, filename)
                preprocess_and_save_single(image_path, class_output_folder)
    print("Preprocessing completed.")

# Function to preprocess a single image and save leads
def preprocess_and_save_single(image_path, output_folder):
    # Load the ECG image
    image = imread(image_path)

    # Convert to grayscale if image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        grayscale_image = color.rgb2gray(image)
    else:
        grayscale_image = image

    # Divide the ECG image into 12 leads
    leads = divide_leads(grayscale_image)

    # Create a folder for the ECG image
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_folder = os.path.join(output_folder, image_name)
    os.makedirs(image_folder, exist_ok=True)

    # Preprocess each lead and save
    for i, lead in enumerate(leads, start=1):
        preprocessed_lead = preprocess_lead(lead)
        lead_output_path = os.path.join(image_folder, f'{image_name}_lead_{i}.png')  # Adjust output format here
        preprocessed_lead.save(lead_output_path)

# Function to divide ECG image into leads
def divide_leads(image):
    """
    This function divides the ECG image into 13 leads including long lead.
    Returns: List containing all 13 leads divided
    """
    # Define the coordinates for each lead
    leads_coordinates = [
        (300, 600, 150, 643),   # Lead 1
        (300, 600, 646, 1135),  # Lead aVR
        (300, 600, 1140, 1625), # Lead V1
        (300, 600, 1630, 2125), # Lead V4
        (600, 900, 150, 643),   # Lead 2
        (600, 900, 646, 1135),  # Lead aVL
        (600, 900, 1140, 1625), # Lead V2
        (600, 900, 1630, 2125), # Lead V5
        (900, 1200, 150, 643),  # Lead 3
        (900, 1200, 646, 1135), # Lead aVF
        (900, 1200, 1140, 1625),# Lead V3
        (900, 1200, 1630, 2125),# Lead V6
        (1250, 1480, 150, 2125) # Long Lead
    ]

    leads = []
    for top, bottom, left, right in leads_coordinates:
        leads.append(image[top:bottom, left:right])

    return leads

# Function to preprocess a lead (remove background, etc.)
def preprocess_lead(lead):
    """
    This function performs preprocessing on the extracted leads.
    """
    # Smoothing image
    blurred_image = gaussian(lead, sigma=1)

    # Thresholding to distinguish foreground and background
    global_thresh = threshold_otsu(blurred_image)
    binary_global = blurred_image < global_thresh

    # Convert boolean image to unsigned byte image
    preprocessed_lead = util.img_as_ubyte(binary_global)

    # Convert image mode to 'L' (8-bit pixels, black and white)
    preprocessed_lead = preprocessed_lead.astype(np.uint8)

    # Convert to PIL image
    preprocessed_pil_image = Image.fromarray(preprocessed_lead)

    return preprocessed_pil_image

# Example usage:
if __name__ == "__main__":
    input_folder_train = './NEW_DATA/train'
    input_folder_validation = './NEW_DATA/validation'
    output_folder_train = './PreprocessedData/train'
    output_folder_validation = './PreprocessedData/validation'

    # Preprocess train images
    preprocess_and_save_leads(input_folder_train, output_folder_train)

    # Preprocess test images
    preprocess_and_save_leads(input_folder_validation, output_folder_validation)
