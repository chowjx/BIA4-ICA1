import pandas as pd
import numpy as np
import os
from skimage.io import imread, imsave
import cv2
from skimage.exposure import equalize_hist
from skimage import img_as_ubyte
from skimage.filters import median, gaussian



def convert_to_RGB_image(image):
    #Convert all the grayscale images as RGB, since some images are not identical in all three RGB channels
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    else:
        image
    return image

def cut_white_edge(image,thres1=250,thres2=255):
    # Convert to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    a, thres = cv2.threshold(gray, thres1, thres2, cv2.THRESH_BINARY_INV)
    # Find the contour of this image
    contour, b = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Cut the white edge
    max_area=0
    for c in contour:
        if cv2.contourArea(c) > max_area:
            max_area = cv2.contourArea(c)
            m, n, p, q = cv2.boundingRect(c)
    return image[n:n+q, m:m+p]
        
def resize_and_rescale(image, size=(320, 320)):
    # Resize the image
    image_resized = cv2.resize(image, size)
    # Rescale pixel values
    image_rescaled = image_resized / 255.0
    return image_rescaled

def enhance_contrast(image):
    # Enhance contrast using histogram equalization
    # Perform this step in both b, g, r channels
    b, g, r = cv2.split(image)
    be = cv2.equalizeHist(b)
    ge = cv2.equalizeHist(g)
    re = cv2.equalizeHist(r)
    equalized_image = cv2.merge((be, ge, re))
    return equalized_image

def reduce_noise(image):
    # Perform noise reduction through Gaussian blur
    denoised_image = gaussian(image, sigma=1,channel_axis=-1)
    return denoised_image

def preprocess_image(image_path, size=(320, 320)):
    # read the images
    image = imread(image_path)
    #Perform above steps
    image = convert_to_RGB_image(image)
    image = cut_white_edge(image)
    image = enhance_contrast(image)
    image = resize_and_rescale(image, size=size)
    image = reduce_noise(image)
    return image

            
def process_input(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Deal with single image file
    if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        processed_image = preprocess_image(input_path, size=(320, 320))
        fn = os.path.basename(input_path)
        imsave(os.path.join(output_folder, fn), img_as_ubyte(processed_image))
    # Deal with a folder
    elif os.path.isdir(input_path):
        for fn in os.listdir(input_path):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(input_path, fn)
                processed_image = preprocess_image(file_path, size=(320, 320))
                imsave(os.path.join(output_folder, fn), img_as_ubyte(processed_image))
    else:
        print("The provided path is neither a file nor a directory. Please check your input path.")


process_input('../input/tb-chest-radiography/TB_Chest_Radiography_Database/Normal', 'processed_normal')
process_input('../input/tb-chest-radiography/TB_Chest_Radiography_Database/Tuberculosis', 'processed_tb')
print("Data preprocessing completed")


###### Choose an example for visualisation ######

xray = imread("../input/tb-chest-radiography/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-154.png")
print(xray.shape)
xray = np.stack((xray,) * 3, axis=-1)
print(xray.shape)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(xray)
ax[1].hist(xray[:,:,0].ravel(), bins=256, range=(0, 256), color='red')
ax[2].hist(xray[:,:,1].ravel(), bins=256, range=(0, 256), color='green')
ax[3].hist(xray[:,:,2].ravel(), bins=256, range=(0, 256), color='blue')

(b, g, r) = cv2.split(xray)
be = cv2.equalizeHist(b)
ge = cv2.equalizeHist(g)
re = cv2.equalizeHist(r)
xray_equalised = cv2.merge((be, ge, re))
fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].imshow(xray)
ax[0].set_title("Original",fontsize=20)
ax[1].imshow(xray_equalised)
ax[1].set_title("Equalised",fontsize=20)

plt.show()

xray = cv2.resize(xray, (320,320))
xray=xray/255.0

xray_filter = gaussian(xray_equalised, sigma=1,channel_axis=-1)
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(xray_equalised[160:320,160:320], cmap="gray")
ax[0].set_title("Equalised",fontsize=20)
ax[1].imshow(xray_filter[160:320,160:320], cmap="gray")
ax[1].set_title("median filter", fontsize=20)
