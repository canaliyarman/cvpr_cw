
import numpy as np

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

