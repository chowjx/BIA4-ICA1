# Preprocessing Image Data

This guide provides instructions for using the preprocessing script, which is designed to process images by converting them to RGB format, trimming white edges, resizing, rescaling, enhancing contrast, and reducing noise. This preprocessing pipeline is essential for preparing image data to be more effectively analyzed and processed by deep learning models, improving the performance and reliability of the resulting insights or predictions.
## Requirements

Before running the script, ensure you have the following installed:
- Python 3.10.12
- `numpy` version 1.25.2
- `pandas` version 2.1.4
- `opencv-python` (`cv2`) version 4.8.1.78
- `scikit-image` version 0.22.0

## Usage

### Function Descriptions

- `convert_to_RGB_image(image)`: Converts grayscale images to RGB.
- `cut_white_edge(image, thres1=250, thres2=255)`: Removes white edges from images.
- `resize_and_rescale(image, size=(320, 320))`: Resizes and rescales images to the specified size.
- `enhance_contrast(image)`: Enhances image contrast using histogram equalization.
- `reduce_noise(image)`: Applies Gaussian blur to reduce noise.
- `preprocess_image(image_path, size=(320, 320))`: Main function to preprocess images, calling all above functions sequentially.
- `Preprocess(input_path, output_path, size=(320, 320))`: Processes either a single image file or all images in a folder and saves the processed images in the specified output directory.

### Steps for Preprocessing

1. **Prepare Your Data**: Organize your images. You can preprocess single images or multiple images in a folder.

2. **Set Input and Output Paths**:
   - `-input_path [-i]`:  Path to the image or folder containing images.     Default: None.
   - `-output_path [-o]`: Path where preprocessed images will be saved.      Default: "./preprocessed".
   - `-size [-s]`:        Size of output images.                             Format: width,height. Default: 320,320.

4. **Run the Script**:
   - Use `python preprocess.py -i path/to/your/image/or/folder -o path/to/save/processed/images -s width,height` to start preprocessing.
   - The script will automatically handle different file formats (`.png`, `.jpg`, `.jpeg`).

5. **Check Output**: Processed images will be saved in the specified output directory.



