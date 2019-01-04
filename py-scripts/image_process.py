import cv2
import numpy as np
#import matplotlib.pyplot as plt
import os
 
path1 = '../images'
path2 = 'processed_images'
 
listing = os.listdir(path1) 
num_samples=len(listing)
print(num_samples)
 
for file in listing:
    print(file)
    image = cv2.imread(path1 + '/' + file, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(im_bw,kernel,iterations = 1)
    kernel = np.ones((1,1),np.uint8)
    erosion = cv2.erode(erosion,kernel,iterations = 2)
    cv2.imwrite(path2 + '/' + file, erosion)
