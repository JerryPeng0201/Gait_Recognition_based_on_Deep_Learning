import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.feature import hog


def img_process(roi, img_address, transform=None):
    # Define transforms
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    # Load gait image
    img = cv2.imread(img_address, cv2.IMREAD_GRAYSCALE)

    # Extract ROI
    x, y, w, h = roi
    roi_img = img[y:y+h, x:x+w]

    # Normalize image
    mean, std = roi_img.mean(), roi_img.std()
    norm_img = (roi_img - mean) / std

    # Threshold image
    _, thresh = cv2.threshold(norm_img, 0.5, 1, cv2.THRESH_BINARY)

    # Apply morphological operations
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)

    # Convert image to tensor
    img_tensor = transform(opening)

    return opening, img_tensor


def generate_gem(img_np):
    # Transfer image to Gray
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Define HOG parameters
    cell_size = (8, 8)
    block_size = (2, 2)
    nbins = 9
    
    # Compute HOG descriptors
    hog_desc, hog_img = hog(img_np, orientations=nbins, pixels_per_cell=cell_size,
                            cells_per_block=block_size, visualize=True, block_norm='L2-Hys')
    
    # Normalize HOG descriptors
    hog_desc = hog_desc / np.linalg.norm(hog_desc)
    
    # Compute GEM
    gem = np.zeros_like(hog_img)
    gem[hog_img > np.max(hog_img) * 0.5] = 1
    
    return gem