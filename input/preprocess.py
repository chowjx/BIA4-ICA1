import pandas as pd
import numpy as np
import os
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import cv2
from skimage.exposure import equalize_hist
from skimage import img_as_ubyte
from skimage.filters import median, gaussian



normal = pd.read_excel('../input/tb-chest-radiography/TB_Chest_Radiography_Database/Normal.metadata.xlsx')
tb = pd.read_excel('../input/tb-chest-radiography/TB_Chest_Radiography_Database/Tuberculosis.metadata.xlsx')
count_normal=len(normal)
count_tb=len(tb)
print("There are "+ str(count_normal)+" images of Normal and "+str(count_tb)+" images of Tuberculosis.")
print("......Image preprocessing......")

def convert_to_RGB_image(image):
    #Save all the grayscale images as RGB
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    else:
        image
    return image

def cut_white_edge(image,threshold1=254,threshold2=255):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, threshold1, threshold2, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y+h, x:x+w]
        
def resize_and_rescale(image, size=(320, 320)):
    image_resized = cv2.resize(image, size) # Resize image
    image_rescaled = image_resized / 255.0    # Rescale pixel values
    return image_rescaled

def enhance_contrast(image):
    # Enhance contrast using histogram equalization
    (b, g, r) = cv2.split(image)
    be = cv2.equalizeHist(b)
    ge = cv2.equalizeHist(g)
    re = cv2.equalizeHist(r)
    equalised_image = cv2.merge((be, ge, re))
    return equalised_image

def reduce_noise(image):
    # Perform noise reduction through Gaussian blur
    filter_image = gaussian(image, sigma=1,channel_axis=-1)
    return filter_image

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


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            processed_image = preprocess_image(file_path,size=(320, 320))
            imsave(os.path.join(output_folder, filename), img_as_ubyte(processed_image))

process_folder('../input/tb-chest-radiography/TB_Chest_Radiography_Database/Normal', 'processed_normal')
process_folder('../input/tb-chest-radiography/TB_Chest_Radiography_Database/Tuberculosis', 'processed_tb')
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
