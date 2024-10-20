
import numpy as np

def extractRandom(img):
    # Generate a random row vector with 30 elements
    F = np.random.rand(1, 30)
    # Returns a row [rand rand .... rand] representing an image descriptor
    # computed from image 'img'
    # Note img is expected to be a normalized RGB image (colors range [0,1] not [0,255])
    return F
def extract_color_descriptor(img):
    # Compute the average red, green, and blue values as a basic color descriptor
    R = np.mean(img[:, :, 2])  # Note: OpenCV uses BGR format
    G = np.mean(img[:, :, 1])
    B = np.mean(img[:, :, 0])
    return np.array([R, G, B])

def extract_color_hist_descriptor(img, bin_number):
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

def joint_color_histogram(img, Q):
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
