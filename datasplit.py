import os
import shutil
import random

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None):
    # Create destination directories if they don't exist
    os.makedirs(dest_dir, exist_ok=True)
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'validation')
    test_dir = os.path.join(dest_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Get the list of class directories
    classes = os.listdir(source_dir)
    
    # Iterate over each class directory
    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        # Get list of image files for the current class
        images = os.listdir(class_dir)
        random.shuffle(images)
        num_images = len(images)
        
        # Calculate number of images for each split
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        num_test = num_images - num_train - num_val
        
        # Split images into train, validation, and test sets
        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]
        
        # Copy images to destination directories
        for image in train_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(train_dir, class_name)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)
        
        for image in val_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(val_dir, class_name)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)
        
        for image in test_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(test_dir, class_name)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)

# Example usage
source_dir = './IMAGE_DATASET'
dest_dir = './NEW_DATA'
split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
