# Tuberculosis Detector - Chest X-Ray Analysis Tool

## Overview
This software, compatible with Python 3.7 or later, is designed to assist clinical researchers in accurately segmenting the thoracic region and classifying tuberculosis from patients' chest X-ray images.

## System Requirements
- **High Memory Requirement**: Image classification and segmentation may require considerable running memory. Use on an adequately powerful device, with **Kaggle** and **Google Colab** recommended for optimal performance.

## Installation
Before using this software, ensure the following Python packages are installed in your environment:
- **PyQt5** version 5.15.10
- **TensorFlow** version 2.15.0
- **opencv-python** version 4.8.1.78
- **NumPy** version 1.26.2

## Contents of the Software Package
The software zip file contains:
- Three Python script files.
- Three `.h5` model files for the AI models.
- Three test cases for demonstration and testing purposes.

## Usage
- **Input Image Format**: The software accepts chest X-ray images in `.png`, `.jpg`, or `.jpeg` formats.
- **Single Image Processing**: Capable of segmenting and classifying a single X-ray image at a time.
- **Main Script**: Run `GUImain_final.py` to start the Tuberculosis Detector software.
- **Graphical User Interface**: `GUImain.py` generates the software's user interface.

## Additional Resources
- **Documentation**: Follow the instructions provided in the software documentation for guidance.
- **User Guide Video**: Watch the video tutorial ["Tuberculosis_Detector_user_guide.mp4"](https://github.com/chowjx/BIA4-ICA1/blob/main/Tuberculosis_Detector_user_guide.mp4) for detailed instructions on running the software on your server.

## Note
- Regularly update your Python environment and packages to ensure compatibility and optimal performance of the software.

