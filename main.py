import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

import skimage

# Set the directory where the images are stored and define the categories
DIRECTORY = r'C:\Users\sabulikailik\Desktop\Dataset\train'
CATEGORIES = ['cats', 'dogs']

# Set the size of the images
img_size = 100

# Create an empty list to store image data
data = []

# Loop through each category and read in the images
for category in CATEGORIES:
    # Get the full path to the category folder
    folder = os.path.join(DIRECTORY, category)
    # Assign a label to the category
    labels = CATEGORIES.index(category)
    # Loop through each image in the category folder
    for img in os.listdir(folder):
        # Get the full path to the image
        img_path = os.path.join(folder, img)
        # Read in the image and resize it to the desired size
        img = cv2.imread(img_path)
        img_gray= skimage.color.rgb2gray(img)
        img_array = cv2.resize(img_gray, (img_size, img_size))
        # Add the image and its label to the data list
        data.append([img_array, labels])

# Shuffle the data randomly
random.shuffle(data)

# Create empty lists to store features (x) and labels (y)
x = []
y = []

# Loop through each item in the data list and add the feature and label to the x and y lists
for features, labels in data:
    x.append(features)
    y.append(labels)

# Convert x and y to numpy arrays
x = np.array(x)
y = np.array(y)


pickle.dump(x,open('x.pkl','wb'))
pickle.dump(y,open('y.pkl','wb'))


