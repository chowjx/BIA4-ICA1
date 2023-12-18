import pandas as pd
import numpy as np
import os
from skimage.io import imread, imsave
import cv2
from skimage.exposure import equalize_hist
from skimage.color import rgb2hsv, hsv2rgb
from skimage import img_as_ubyte
from skimage.filters import median, gaussian
import argparse

def parse_size(size):
    try:
        width, height = map(int, size.split(','))
        return width, height
    except ValueError:
        raise argparse.ArgumentTypeError("Size must be width,height")

def parse_parameters():
    """
    pass the parameters from terminal

    :return: a list of parameters
    """
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument("-input_path","-i", type=str, default=None, help="The input could be a single image file or a folder containing several images.")
    parser.add_argument("-output_path","-o", type=str, default="./preprocessed", help="The output files will be saved in the specified path.")
    parser.add_argument("-size","-s",type=parse_size, default="320,320",
                        help="Set the size of the output images"
                             "Default: 320*320")
    return parser.parse_args()


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
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    equalized_image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
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

            
def Preprocess(input_path, output_path, size=(320, 320)):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Deal with single image file
    if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        processed_image = preprocess_image(input_path, size=size)
        fn = os.path.basename(input_path)
        imsave(os.path.join(output_path, fn), img_as_ubyte(processed_image))
    # Deal with a folder
    elif os.path.isdir(input_path):
        for fn in os.listdir(input_path):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(input_path, fn)
                processed_image = preprocess_image(file_path, size=size)
                imsave(os.path.join(output_path, fn), img_as_ubyte(processed_image))
    else:
        print("The provided path is neither a file nor a directory. Please check your input path.")

parameters = parse_parameters()
Preprocess(parameters.input_path,parameters.output_path,parameters.size)
