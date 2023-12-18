"""
Tuberculosis Detector: a deep-learning based software to help with diagnosing tuberculosis by displaying segmentation and classification on chest X-ray images.
It supports U-net model for chest X-ray segmentation.
Also, it supports DenseNet 201 model for classification.
"""


# Import libraries
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5 import QtGui
from keras.models import load_model
import sys, cv2
import numpy as np

# From GUIwindows_final.py package import mainWindow_ui
from GUIwindows_final import mainWindow_ui
# From classification_implement.py package input_and_judge function to achieve classification
from classification_implement import input_and_judge


class Main(QMainWindow, mainWindow_ui):

    # Initialization of the main window
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)

    def openfile(self):
        """Open the image file path

        :return: Save the file path as the variable self.input_path
                Call the show_input_img function.
        """

        # Open a file dialog to select an image file path
        openfile = QFileDialog.getOpenFileName()
        if openfile[0] != '':
            # Get the image path of the input image
            self.input_path = openfile[0]
            # Call the function to show the input image
            self.show_input_img(self.input_path)

    def show_input_img(self, file_path):
        """Display the input image

        :param file_path: File path of the input image

        :return: None - Display the image in the import_img QLabel
        """

        input_img = QtGui.QPixmap(file_path)
        self.IMG_import.setPixmap(input_img)

    def segmentation(self):
        """Start of the segmentation function.

        :param file_path: File path of the input image

        :return: Display the processed image in the export_img QLabel
        """

        def dice_coef(y_true, y_pred, smooth=1):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        if hasattr(self, 'input_path'):
            # Load your model with custom metrics
            model = load_model('./segmentation_model.h5', custom_objects={'dice_coef': dice_coef})

            # Read the input image
            input_image = cv2.imread(self.input_path, cv2.IMREAD_GRAYSCALE)

            # Preprocess the image (resize and normalize)
            input_image_resized = cv2.resize(input_image, (256, 256))
            input_image_normalized = input_image_resized / 255.0
            input_image_reshaped = input_image_normalized.reshape(1, 256, 256, 1)

            # Predict the mask
            predicted_mask = model.predict(input_image_reshaped)[0, :, :, 0]

            # Threshold the predicted mask to binary
            predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)

            # Create an overlay image
            original_image_colored = cv2.cvtColor(input_image_resized, cv2.COLOR_GRAY2BGR)
            mask_colored = cv2.applyColorMap(predicted_mask_binary * 255, cv2.COLORMAP_HOT)
            overlay_image = cv2.addWeighted(original_image_colored, 0.7, mask_colored, 0.3, 0)

            # Convert to QPixmap for display in QLabel
            height, width, channel = overlay_image.shape
            bytesPerLine = 3 * width
            qImg = QtGui.QImage(overlay_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)

            # Display the composite image in the IMG_export QLabel
            self.IMG_export.setPixmap(pixmap)
        else:
            QMessageBox.warning(self, 'Warning', 'Please import an image first!', QMessageBox.Close)

    def saveImage(self):

        """Save the segmented image

        :return: None - Save the segmented image to a selected path
        """

        try:
            # Save the pixmap of the image in the IMG_export QLabel
            img = self.IMG_export.pixmap().toImage()
            fpath, ftype = QFileDialog.getSaveFileName(self.centralwidget, "Save the segmented picture",
                                                       "Segmented Picture",
                                                       "*.jpg;;*.png;;All Files(*)")
            img.save(fpath)
        except: #if there is no image input or don't have the segmentation action, Tuberculosis Detector will give a warning
            QMessageBox.warning(self, 'Warning',
                                'Tuberculosis Detector Warning \n\nSave Function: Please Check If An Image Has Been Imported!',
                                QMessageBox.Close)

    def classification(self):

        """Start of classifying the input image by densenet 201 model.

        :param file_path: File path of the input image

        :return: Call the input_and_judge function and show the classification result in the label_4 text browser"""

        try:
            model_type = "DenseNet 201"
            image_path = str(self.input_path)
            model = load_model("./3_fold_densenet201_chest_xray_model.h5")
            densenet201_classification = input_and_judge(model_type, model, image_path)
            self.textBrowser_result.setText(
                f"The DenseNet 201 learning model suggests that the classification of this chest X-ray is  {densenet201_classification}")
            self.textBrowser_result.repaint()
        except: #if there is no image input, tuberculosis detector will give a warning
            QMessageBox.warning(self, 'Warning',
                                'Tuberculosis Detector Warning \n\nDenseNet 201 Model: Please Check If An Image Has Been Imported!',
                                QMessageBox.Close)


if __name__ == '__main__':
    # Create GUI object, create main window ui class object and show the main window
    app = QApplication(sys.argv)
    main = Main()
    main.show()

    # Connect the buttons to the defined functions
    main.PushButton_filebrowsing.clicked.connect(main.openfile)
    main.pushButton_segmentationprocess.clicked.connect(main.segmentation)
    main.pushButton_saving.clicked.connect(main.saveImage)
    main.pushButton_run.clicked.connect(main.classification)

    # The program will run unless the exit closes the window
    sys.exit(app.exec_())