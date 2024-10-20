import os
import numpy as np
import cv2
from descriptors import Descriptor
import glob

class cvpr_compute_descriptors:
    def __init__(self, DATASET_FOLDER, OUT_FOLDER):
        self.DATASET_FOLDER = DATASET_FOLDER
        self.OUT_FOLDER = OUT_FOLDER

    def compute_descriptors(self, descriptor_name):
        # Ensure the output directory exists
        
        os.makedirs(os.path.join(self.OUT_FOLDER, descriptor_name), exist_ok=True)
        image_paths = glob.glob(self.DATASET_FOLDER + "*.bmp", recursive=True)
        descriptors_path = self.OUT_FOLDER + descriptor_name + '/'

        # Iterate through all BMP files in the dataset folder
        descriptor = Descriptor(descriptor_name)
        for image_path in image_paths:
            print(f"Processing file {image_path}")
            image = cv2.imread(image_path)
            feature = descriptor.extract_descriptors(image, 8)
            file_name = os.path.basename(image_path).split('.')[0]

            save_path = descriptors_path + file_name + '.npy' # extract filename add .npy extension
            print(save_path)
            np.save(save_path, feature)


# compute = cvpr_computedescriptors('MSRC_ObjCategImageDatabase_v2/Images/', 'python/descriptors/')
# compute.compute_descriptors('color_hist')
# DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
# OUT_FOLDER = 'descriptors'
# OUT_SUBFOLDER = 'globalRGBhisto'

# # Ensure the output directory exists
# os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)

# # Iterate through all BMP files in the dataset folder
# for filename in os.listdir(os.path.join(DATASET_FOLDER, 'Images')):
#     if filename.endswith(".bmp"):
#         print(f"Processing file {filename}")
#         img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
#         img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
#         fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))
        
#         # Call extractRandom (or another feature extraction function) to get the descriptor
#         F = extractRandom(img)
        
#         # Save the descriptor to a .mat file
#         sio.savemat(fout, {'F': F})

