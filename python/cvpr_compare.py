
import numpy as np
import cv2
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler


class CompareDistance:
    def __init__(self):
        pass
    def random_distance(self, F1, F2):
        # This function should compare F1 to F2 - i.e. compute the distance
        # between the two descriptors
        # For now it just returns a random number
        dst = np.random.rand()
        return dst
    def calculate_distance(self, F1, F2):
        dist = np.linalg.norm(F1 - F2)
        return dist
    def calculate_sift_dustabce(self, F1, F2):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(F1, F2)

    def apply_pca(self, descriptors, n_components=0.95):
        
        descriptors_list = list(descriptors.values())
        scaler = StandardScaler()
        descriptors_normalized = scaler.fit_transform(descriptors_list)
        pca = PCA()
        reduced_descriptors = pca.fit_transform(descriptors_normalized, n_components)
        descriptors = dict(zip(descriptors.keys(), reduced_descriptors))
        return descriptors
    # def mahalanobis_distance(self, query_descriptor, mean, inv_cov_matrix):
    #     # Calculate Mahalanobis distance
    #     dist = distance.mahalanobis(query_descriptor, mean, inv_cov_matrix)
    #     return dist
    def mahalanobis_distance(self, query_descriptor, image_descriptor, inv_cov_matrix):
        # Calculate Mahalanobis distance between the query descriptor and another descriptor
        delta = query_descriptor - image_descriptor
        dist = np.sqrt(np.dot(np.dot(delta.T, inv_cov_matrix), delta))
        if np.any(np.isnan(delta)):
            return np.nan
        
        # Calculate Mahalanobis distance
        dist = np.sqrt(np.dot(np.dot(delta.T, inv_cov_matrix), delta))
        
        # Check if the result is NaN
        if np.isnan(dist):
            return np.nan
        return dist
    def calculate_eigenmodel(self, descriptors):
        mean = np.mean(descriptors, axis=0)
        cov_matrix = np.cov(descriptors, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        return mean, cov_matrix, inv_cov_matrix