import os
import numpy as np
import cv2
from descriptors import Descriptor
import glob
from tqdm import tqdm
from sklearn.cluster import KMeans
class cvpr_compute_descriptors:
    def __init__(self, DATASET_FOLDER, OUT_FOLDER):
        self.DATASET_FOLDER = DATASET_FOLDER
        self.OUT_FOLDER = OUT_FOLDER

    def compute_descriptors(self, descriptor_name, bins=8):
        # Ensure the output directory exists
        
        os.makedirs(os.path.join(self.OUT_FOLDER, descriptor_name), exist_ok=True)
        image_paths = glob.glob(self.DATASET_FOLDER + "*.bmp", recursive=True)
        descriptors_path = self.OUT_FOLDER + descriptor_name + '/'

        # Iterate through all BMP files in the dataset folder
        descriptor = Descriptor(descriptor_name)
        print(f'Computing {descriptor_name} descriptors...')
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if descriptor_name == 'sift':
                feature = descriptor.extract_descriptors(image, bins)
            else:
                feature = descriptor.extract_descriptors(image, bins).flatten()
            file_name = os.path.basename(image_path).split('.')[0]

            save_path = descriptors_path + file_name + '.npy' # extract filename add .npy extension
            np.save(save_path, feature)

    def build_visual_vocab(self, descriptors_folder, dictionary_size=128):   
        descriptor_paths = glob.glob(descriptors_folder + "*.npy", recursive=True)
        print(f'Bulding BoVW with {dictionary_size} clusters...')
        all_descriptors = []
        for descriptor_path in tqdm(descriptor_paths):
            image_descriptors = np.load(descriptor_path)
            for desc in image_descriptors:
                all_descriptors.append(desc)
        all_descriptors = np.vstack(all_descriptors)
        print('Clustering...')
        kmeans = KMeans(n_clusters=dictionary_size, random_state=42).fit(all_descriptors)
        visual_vocabulary = kmeans.cluster_centers_
        return visual_vocabulary, kmeans
    
    def compute_bovw(self, descriptors_folder, kmeans, dictionary_size=128):
        os.makedirs(os.path.join(self.OUT_FOLDER, 'sift_bovw'), exist_ok=True)
        descriptor_paths = glob.glob(descriptors_folder + "*.npy", recursive=True)
        print(f'Computing BoVW histograms...')
        for descriptor_path in tqdm(descriptor_paths):
            descriptor_basename = os.path.basename(descriptor_path).split('.')[0]
            descriptor = np.load(descriptor_path)
            visual_words = kmeans.predict(descriptor)
            histogram, _ = np.histogram(visual_words, bins=np.arange(dictionary_size+1), density=True)
            np.save(self.OUT_FOLDER + 'sift_bovw/' + descriptor_basename + '.npy', histogram)


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

