from glob import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import BinaryCrossentropy
import rasterio as rio
from PIL import Image
import os

# Function to load satellite images
def load_images(data_path):
    files = glob(data_path + "/*B?*.tiff")
    files.sort()

    images = []
    for file in files:
        with rio.open(file, 'r') as f:
            images.append(f.read(1))
    return np.stack(images)

# Function to load flood masks
def load_flood_mask(data_path, file_format='.png', mask_threshold=0.6):
    # Initialize an empty list to store the loaded flood mask images
    mask_images = []

    # Iterate over each file in the data directory
    for filename in os.listdir(data_path):
        # Check if the file is of the specified format
        if filename.endswith(file_format):
            # Construct the full path to the image file
            file_path = os.path.join(data_path, filename)
            
            # Open the image file using PIL (Python Imaging Library)
            image = Image.open(file_path)

            # Convert the image to a numpy array
            mask_array = np.array(image)

            # Apply threshold to generate binary labels
            mask_binary = (mask_array > mask_threshold).astype(int)

            # Append the numpy array to the list of mask images
            mask_images.append(mask_binary)

    # Stack the mask images along the first axis to create a single numpy array
    if mask_images:
        mask_data = np.stack(mask_images, axis=0)
    else:
        mask_data = np.empty((0, 0, 0))

    return mask_data

# Load images
before_floods_data_path = r"C:\Users\HP\Desktop\Final EL\Satellite_Imagery_Analysis-main\Data\Madagascar_data\Madagascar_18_01_2017"
during_floods_data_path = r"C:\Users\HP\Desktop\Final EL\Satellite_Imagery_Analysis-main\Data\Madagascar_data\Madagascar_27_01_2020"

arr_bef = load_images(before_floods_data_path)
arr_dur = load_images(during_floods_data_path)

# Reshape the data to add channel dimension
arr_bef = arr_bef[..., np.newaxis]
arr_dur = arr_dur[..., np.newaxis]

# Load flood masks
flood_mask_bef = load_flood_mask(before_floods_data_path)
flood_mask_dur = load_flood_mask(during_floods_data_path)

# Generate binary labels (0 for non-flooded pixels, 1 for flooded pixels)
labels_bef = (flood_mask_bef > 0.6).astype(int)
labels_dur = (flood_mask_dur > 0.6).astype(int)

# Define CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=arr_bef.shape[1:]),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

# Train the model
print("Shape of arr_bef:", arr_bef.shape)
print("Shape of labels_bef:", labels_bef.shape)
model.fit(arr_bef, labels_bef, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(arr_bef, labels_bef)
print("Loss:", loss)
print("Accuracy:", accuracy)
