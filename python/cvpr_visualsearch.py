import os
import numpy as np
import cv2
import random
from cvpr_compare import CompareDistance
import glob

DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'


class VisualSearch:
    def __init__(self, descriptor_folder, image_folder, performance_folder):
        self.DESCRIPTOR_FOLDER = descriptor_folder
        self.IMAGE_FOLDER = image_folder
        self.PERFORMANCE_FOLDER = performance_folder
        # Load all descriptors
        self.descriptors = {}
        self.ALLFEAT = []
        self.ALLFILES = []
        self.test = ''
        self.distance_calculator = CompareDistance()
    # Load image descriptors based on the descriptor name
    def load_descriptors(self, descriptor_name):
        self.descriptor_name = descriptor_name
        descriptor_folder = os.path.join(self.DESCRIPTOR_FOLDER, (descriptor_name + '/'))
        self.test = descriptor_folder
        descriptor_paths = glob.glob(descriptor_folder + "*.npy", recursive=True)
        self.descriptors_dict = {}
        for feature_path in descriptor_paths:
            name = os.path.basename(feature_path).split('.')[0]
            F = np.load(feature_path)
            self.descriptors_dict[name] = F
        return self.descriptors_dict
    def visual_search(self, query_image_path=None, num_results=15, 
                      distance_metric='euclidean', pca=False):
        print(f"Performing visual search using {self.descriptor_name} descriptors")
        if query_image_path == None:
            query_image_path = random.choice(glob.glob(self.IMAGE_FOLDER + "*.bmp", recursive=True))
        query_image = cv2.imread(query_image_path)
        query_basename = os.path.basename(query_image_path).split('.')[0]
        if pca:
            self.descriptors_dict = self.distance_calculator.apply_pca(self.descriptors_dict)
        query_feature = self.descriptors_dict[query_basename]
        image_paths = glob.glob(self.IMAGE_FOLDER + "*.bmp", recursive=True)
        similarity_list = []
        if distance_metric == 'mahalanobis_distance':
            # Calculate mean and inverse covariance matrix once for all descriptors
            mean, cov_matrix, inv_cov_matrix = self.distance_calculator.calculate_eigenmodel(list(self.descriptors_dict.values()))
            
            for image_path in image_paths:
                image_basename = os.path.basename(image_path).split('.')[0]
                image_feature = self.descriptors_dict[image_basename]
                dist = self.distance_calculator.mahalanobis_distance(query_feature, image_feature, inv_cov_matrix)
                if np.isnan(dist):
                    continue
                similarity_list.append({'image_name': image_basename, 'dist': dist})
        
        else:
            for image_path in image_paths:
                image_basename = os.path.basename(image_path).split('.')[0]
                image_feature = self.descriptors_dict[image_basename]
                dist = self.distance_calculator.calculate_distance(query_feature, image_feature)
                similarity_list.append({'image_name': image_basename, 'dist': dist})
        sorted_list = sorted(similarity_list, key=lambda dist: dist['dist'])
        return query_basename, sorted_list
    def visual_search_custom_descriptors(self, query_image_path=None, 
                      distance_metric='euclidean', pca=False, descriptors_dicts=[]):
        
        print(f"Performing visual search using multiple descriptors")
        combined_descriptors = {}
        if len(descriptors_dicts) > 1:
            for key in descriptors_dicts[0].keys():
                combined_descriptors[key] = np.concatenate([descriptor[key] for descriptor in descriptors_dicts])
            self.descriptors_dict = combined_descriptors
        if query_image_path == None:
            query_image_path = random.choice(glob.glob(self.IMAGE_FOLDER + "*.bmp", recursive=True))
        query_image = cv2.imread(query_image_path)
        query_basename = os.path.basename(query_image_path).split('.')[0]
        if pca:
            self.descriptors_dict = self.distance_calculator.apply_pca(self.descriptors_dict)
        query_feature = self.descriptors_dict[query_basename]
        image_paths = glob.glob(self.IMAGE_FOLDER + "*.bmp", recursive=True)
        similarity_list = []
        if distance_metric == 'mahalanobis_distance':
            # Calculate mean and inverse covariance matrix once for all descriptors
            mean, cov_matrix, inv_cov_matrix = self.distance_calculator.calculate_eigenmodel(list(self.descriptors_dict.values()))
            
            for image_path in image_paths:
                image_basename = os.path.basename(image_path).split('.')[0]
                image_feature = self.descriptors_dict[image_basename]
                dist = self.distance_calculator.mahalanobis_distance(query_feature, image_feature, inv_cov_matrix)
                if np.isnan(dist):
                    continue
                similarity_list.append({'image_name': image_basename, 'dist': dist})
        else:
            for image_path in image_paths:
                image_basename = os.path.basename(image_path).split('.')[0]
                image_feature = self.descriptors_dict[image_basename]
                dist = self.distance_calculator.calculate_distance(query_feature, image_feature)
                similarity_list.append({'image_name': image_basename, 'dist': dist})
        sorted_list = sorted(similarity_list, key=lambda dist: dist['dist'])
        return query_basename, sorted_list
    
    def calculate_precision_recall(self, query, retreived_images, top_n=15):
        query_label = query.split('_')[0]
        TP = 0
        FP = 0
        FN = 0
        for image in retreived_images[:top_n]:
            image_label = image['image_name'].split('_')[0]
            if image_label == query_label:
                TP += 1
            else:
                FP += 1
        for image in retreived_images[top_n:]:
            image_label = image['image_name'].split('_')[0]
            if image_label == query_label:
                FN += 1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        return precision, recall
    
    def calculate_average_precision(self, query_basename, retrieved_images_sorted):
        # Calculate AP as described in the slide
        relevant_images = [image for image in retrieved_images_sorted if image['image_name'].split('_')[0] == query_basename.split('_')[0]]
        num_relevant = len(relevant_images)

        if num_relevant == 0:
            return 0

        precision_sum = 0
        relevant_count = 0

        for n, image in enumerate(retrieved_images_sorted, 1):
            image_label = image['image_name'].split('_')[0]
            if image_label == query_basename.split('_')[0]:
                relevant_count += 1
                precision_at_n = relevant_count / n
                precision_sum += precision_at_n

        return precision_sum / num_relevant
# sorted_list = sorted(similarity_list, key=lambda dist: dist['dist'])

# DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images/'  # Modify this path as needed
# OUT_FOLDER = './python/descriptors/better_descriptors/'  # Modify this path as needed
# if not os.path.exists(OUT_FOLDER):
#     os.makedirs(OUT_FOLDER)
# image_paths = glob.glob(DATASET_FOLDER + "*.bmp", recursive=True)

# for image_path in image_paths:
#     image = cv2.imread(image_path)
#     #feature = extract_color_descriptor(image)
#     feature = extract_color_hist_descriptor(image, 8)

#     save_path = image_path.split('/')[-1].split('.')[0] + '.npy' # extract filename add .npy extension
#     np.save(OUT_FOLDER + save_path, feature)
# # Convert ALLFEAT to a numpy array
# ALLFEAT = np.array(ALLFEAT)

# # Pick a random image as the query
# NIMG = ALLFEAT.shape[0]
# queryimg = randint(0, NIMG - 1)

# # Compute the distance between the query and all other descriptors
# dst = []
# query = ALLFEAT[queryimg]
# for i in range(NIMG):
#     candidate = ALLFEAT[i]
#     distance = cvpr_compare(query, candidate)
#     dst.append((distance, i))

# # Sort the distances
# dst.sort(key=lambda x: x[0])

# # Show the top 15 results
# SHOW = 15
# for i in range(SHOW):
#     img = cv2.imread(ALLFILES[dst[i][1]])
#     img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
#     cv2.imshow(f"Result {i+1}", img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

