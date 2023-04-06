import os
import cv2
import numpy as np
import scipy.signal as signal
from torch.utils.data import Dataset
from skimage.feature import hog

class GaitDataset(Dataset):
    def __init__(self, img_folder, transform=None) -> None:
        self.img_folder = img_folder
        self.transform = transform
        self.gait_cycles = self.extract_gait_cycles()
    
    def extract_gait_cycles(self):
        gait_cycles = []
        for root, _, files in os.walk(self.img_folder):
            for file in files:
                # Load the image and generate the GEM
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                gem = self.generate_gem(img)
                
                # Extract gait cycles from the GEM
                cycles = self.extract_cycles(gem)
                
                # Append the cycles to the list
                gait_cycles.append(cycles)
        
        return gait_cycles
    
    def generate_gem(self, img):
        # Define HOG parameters
        cell_size = (8, 8)
        block_size = (2, 2)
        nbins = 9
        
        # Compute HOG descriptors
        hog_desc, hog_img = hog(img, orientations=nbins, pixels_per_cell=cell_size,
                                cells_per_block=block_size, visualize=True, block_norm='L2-Hys')
        
        # Normalize HOG descriptors
        hog_desc = hog_desc / np.linalg.norm(hog_desc)
        
        # Compute GEM
        gem = np.zeros_like(hog_img)
        gem[hog_img > np.max(hog_img) * 0.5] = 1
        
        return gem
    
    def extract_cycles(self, gem):
        cycles = []

        # Use the peaks and valleys in the GEM to extract cycles
        # and return them as a list of numpy arrays
        peaks, _ = signal.find_peaks(np.sum(gem, axis=1), prominence=20)
        valleys, _ = signal.find_peaks(-np.sum(gem, axis=1), prominence=20)
        peaks = np.sort(peaks)
        valleys = np.sort(valleys)

        # Determine the number of cycles
        num_cycles = min(len(peaks), len(valleys))

        for i in range(num_cycles):
            cycle = gem[valleys[i]:peaks[i], :]
            cycles.append(cycle)

        return cycles
    
    def __len__(self):
        return len(self.gait_cycles)

    def __getitem__(self, idx):
        cycle = self.gait_cycles[idx]
        if self.transform:
            cycle = self.transform(cycle)
        return cycle