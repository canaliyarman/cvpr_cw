import numpy as np
import cv2
class Descriptor:

    def __init__(self, name=None):
        self.name = name

    def extract_descriptors(self, img, bin_number):
        if self.name == 'joint_color_hist':
            return self.joint_color_histogram(img, bin_number)
        elif self.name == 'color_hist':
            return self.extract_color_hist_descriptor(img, bin_number)
        elif self.name == 'color':
            return self.extract_color_descriptor(img)
        elif self.name == 'random':
            return self.extractRandom(img)
        elif self.name == 'sift':
            return self.sift_descriptor(img)
        elif self.name == 'color_grid_descriptor':
            return self.color_grid_descriptor(img, bin_number)
        elif self.name == 'edge_descriptor':
            return self.edge_descriptor(img)
        elif self.name == 'edge_grid_descriptor':
            return self.edge_grid_descriptor(img, bin_number)
        else:
            return self.extractRandom(img)
    def extractRandom(self, img):
        # Generate a random row vector with 30 elements
        F = np.random.rand(1, 30)
        # Returns a row [rand rand .... rand] representing an image descriptor
        # computed from image 'img'
        # Note img is expected to be a normalized RGB image (colors range [0,1] not [0,255])
        return F
    def extract_color_descriptor(self, img):
        # Compute the average red, green, and blue values as a basic color descriptor
        R = np.mean(img[:, :, 2])  # Note: OpenCV uses BGR format
        G = np.mean(img[:, :, 1])
        B = np.mean(img[:, :, 0])
        return np.array([R, G, B])

    def extract_color_hist_descriptor(self, img, bin_number=16):
        red_hist, bins_red = np.histogram(img[:,:,2], bins=bin_number, range=(0,255))
        green_hist, bins_green = np.histogram(img[:,:,1], bins=bin_number, range=(0,255))
        blue_hist, bins_blue = np.histogram(img[:,:,0], bins=bin_number, range=(0,255))
        # plt.figure(figsize=(10, 5))

        # plt.subplot(3, 1, 1)
        # plt.bar(bins_red[:-1], red_hist, width=np.diff(bins_red), color='red', edgecolor='black', align='edge')


        # plt.subplot(3, 1, 2)
        # plt.bar(bins_green[:-1], green_hist, width=np.diff(bins_green), color='green', edgecolor='black', align='edge')


        # plt.subplot(3, 1, 3)
        # plt.bar(bins_blue[:-1], blue_hist, width=np.diff(bins_blue), color='blue', edgecolor='black', align='edge')

        # plt.tight_layout()
        # plt.show()
        return np.array((red_hist, green_hist, blue_hist))
        # return np.array([red_hist,green_hist,blue_hist])

    def joint_color_histogram(self, img, Q):
        # Normalize the image values to be between 0 and (Q-1)
        qimg = np.floor((img.astype(float) / 256.0) * Q).astype(int)
        
        # Create a single integer value for each pixel that represents the RGB combination
        bin_index = qimg[:,:,0] * Q**2 + qimg[:,:,1] * Q**1 + qimg[:,:,2]
        
        # Reshape the 2D matrix into a 1D vector of values (flatten the image)
        vals = bin_index.flatten()
        
        # Create a histogram with Q^3 bins
        hist, _ = np.histogram(vals, bins=Q**3, range=(0, Q**3))
        
        # Normalize the histogram so that the sum of all bin values equals 1
        hist = hist / np.sum(hist)
        
        return hist

    def sift_descriptor(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        return des


    def color_grid_descriptor(self, img, grid_size=16):
        height, width, _ = img.shape
        grid_h, grid_w = height // grid_size, width // grid_size


        histograms = []
        for row in range(grid_size):
            for col in range(grid_size):
                grid_img = img[row * grid_h: (row + 1) * grid_h, col * grid_w: (col + 1) * grid_w]
                F = self.extract_color_hist_descriptor(grid_img, 16)
                histograms.append(F)
        
        return np.concatenate(histograms)
    
    def edge_descriptor(self, img, block_size=2, ksize=3, k=0.04):
        
        img = np.float32(img)

        # Harris corner detection
        dst = cv2.cornerHarris(img, block_size, ksize, k)

        # Dilate corner points for marking them
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.01 * dst.max()] = [0, 0, 255]

        # Count the number of corners detected (you can also return other statistics)
        avg_response = np.mean(dst[dst > 0.01 * dst.max()])  # Average response of strong corners


        # You can return the number of corners or corner locations as part of the descriptor
        return avg_response

    def edge_grid_descriptor(self, img, grid_size=16):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape
        grid_h, grid_w = height // grid_size, width // grid_size


        histograms = []
        for row in range(grid_size):
            for col in range(grid_size):
                grid_img = img[row * grid_h: (row + 1) * grid_h, col * grid_w: (col + 1) * grid_w]
                F = self.edge_descriptor(grid_img)
                histograms.append(F)
        
        return np.concatenate(histograms)
    