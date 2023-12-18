"""
The PyQt5 UI code generator 5.15.10 automatically created this GUIwindow.py after the Qt designer generated the BIA.ui file.
We made some changes to this file to better connect our UI to the functions.
"""
# import libraries
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog


# define a mainWindiow_ui class
class mainWindow_ui(object):
    def setupUi(self, mainWindow):
        # set the mainwindow in 821*721 size
        mainWindow.setObjectName("Main Window")
        mainWindow.resize(821, 740)
        # Ensure the MainWindow size is stable
        mainWindow.setMinimumSize(QtCore.QSize(821, 740))
        mainWindow.setMaximumSize(QtCore.QSize(821, 740))
        screen = QDesktopWidget().screenGeometry()
        size = mainWindow.geometry()

        # Ensure the MainWindow can be displayed on the center
        mainWindow.move((int)((screen.width() - size.width()) / 2), (int)((screen.height() - size.height()) / 2))
        mainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("Centralwidget")

        # Section1: File input and file segmentation

        ##Setting GroupBox_importfile which starts at the point (10, 10) in mainWindow, take the interface 801 long and 461 wide down to the right.
        self.groupBox_section1 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_section1.setGeometry(QtCore.QRect(10, 10, 801, 461))
        self.groupBox_section1.setObjectName("GroupBox_section1")

        ## Place Layout in the groupBox_section1. Starting at the point (20, 20) in GroupBox_section1, take the interface 781 long and 351 wide down to the right.
        self.LayoutWidget = QtWidgets.QWidget(self.groupBox_section1)
        self.LayoutWidget.setGeometry(QtCore.QRect(20, 100, 781, 351))
        self.LayoutWidget.setObjectName("LayoutWidget")
        self.Layout = QtWidgets.QHBoxLayout(self.LayoutWidget)
        ##set the size of the margins
        self.Layout.setContentsMargins(0, 0, 0, 0)
        self.Layout.setObjectName("Layout")

        ## Setting import image of QLabel
        self.IMG_import = QtWidgets.QLabel(self.LayoutWidget)
        self.IMG_import.setStyleSheet("")
        self.IMG_import.setText("")
        self.IMG_import.setObjectName("IMG_import")

        self.IMG_import.setAlignment(QtCore.Qt.AlignCenter)
        ## Automatically adjust the size of pictures to ensure the picture can fit
        self.IMG_import.setScaledContents(True)
        self.IMG_import.setGeometry(QtCore.QRect(1, 1, 376, 349))

        ## Setting export image of QLabel
        self.IMG_export = QtWidgets.QLabel(self.LayoutWidget)
        self.IMG_export.setStyleSheet("")
        self.IMG_export.setText("")
        self.IMG_export.setObjectName("IMG_export")

        self.IMG_export.setAlignment(QtCore.Qt.AlignCenter)
        self.IMG_export.setScaledContents(True)
        self.IMG_export.setGeometry(QtCore.QRect(385, 1, 375, 349))

        # Place Layout_2 in the groupBox_section1. Starting at the point (20, 30) in groupBox_section1, take the interface 761 long and 51 wide down to the right.
        self.LayoutWidget_2 = QtWidgets.QWidget(self.groupBox_section1)
        self.LayoutWidget_2.setGeometry(QtCore.QRect(20, 30, 761, 51))
        self.LayoutWidget_2.setObjectName("LayoutWidget_2")
        self.Layout_2 = QtWidgets.QHBoxLayout(self.LayoutWidget_2)
 
        self.Layout_2.setContentsMargins(0, 0, 0, 0)
        self.Layout_2.setObjectName("Layout_2")

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.Layout_2.addItem(spacerItem)

        ## Place the PushButton for browsing file
        self.PushButton_filebrowsing = QtWidgets.QPushButton(self.LayoutWidget_2)
        self.PushButton_filebrowsing.setObjectName("PushButton_filebrowsing")
        ##automatically resize the location of pushbutton in LayoutWidget_2 for a better look
        self.Layout_2.addWidget(self.PushButton_filebrowsing)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.Layout_2.addItem(spacerItem1)

        ## Place the PushButton for segmentation function

        ##contain pushButton_segmentationprocess in the LayoutWidget_2
        self.pushButton_segmentationprocess = QtWidgets.QPushButton(self.LayoutWidget_2)
        self.pushButton_segmentationprocess.setObjectName("PushButton_segmentationpreprocess")
        
        self.Layout_2.addWidget(self.pushButton_segmentationprocess)

        ## Push button for save function
        self.pushButton_saving = QtWidgets.QPushButton(self.LayoutWidget_2)
        self.pushButton_saving.setObjectName("pushButton_saving")

        self.Layout_2.addWidget(self.pushButton_saving)

        # Section2: Classification function of this models

        ##Setting groupBox_section1 which starts at the point (10, 480) in mainWindow, take the interface 801 long and 200 wide down to the right.
        self.groupBox_section2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_section2.setGeometry(QtCore.QRect(10, 480, 801, 200))
        self.groupBox_section2.setObjectName("groupBox_section2")

        ## Place horinzontalLayout_3 into groupBox_section2
        self.LayoutWidget_3 = QtWidgets.QWidget(self.groupBox_section2)
        self.LayoutWidget_3.setGeometry(QtCore.QRect(20, 50, 761, 51))
        self.LayoutWidget_3.setObjectName("LayoutWidget_3")

        self.Layout_3 = QtWidgets.QHBoxLayout(self.LayoutWidget_3)
        self.Layout_3.setContentsMargins(0, 0, 0, 0)
        self.Layout_3.setObjectName("Layout_3")

        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.Layout_3.addItem(spacerItem2)

        ## Place a Qlabel of model choice
        self.label_model = QtWidgets.QLabel(self.LayoutWidget_3)
        self.label_model.setObjectName("Label_model")

        self.Layout_3.addWidget(self.label_model)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.Layout_3.addItem(spacerItem3)

        ## Place a Learning Model Run Function Button
        self.pushButton_run = QtWidgets.QPushButton(self.LayoutWidget_3)
        self.pushButton_run.setObjectName("pushButton_run")

        self.Layout_3.addWidget(self.pushButton_run)

        ## Categorized Results Text Browser
        self.LayoutWidget_4 = QtWidgets.QWidget(self.groupBox_section2)
        self.LayoutWidget_4.setGeometry(QtCore.QRect(20, 100, 761, 81))
        self.LayoutWidget_4.setObjectName("LayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.LayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("HorizontalLayout_4")

        self.label_4 = QtWidgets.QLabel(self.LayoutWidget_4)
        self.label_4.setObjectName("label_4")
        
        self.horizontalLayout_4.addWidget(self.label_4)

        ## Classification Result Text Browser
        self.textBrowser_result = QtWidgets.QTextBrowser(self.LayoutWidget_4)
        self.textBrowser_result.setObjectName("textBrowser_result")

        self.horizontalLayout_4.addWidget(self.textBrowser_result)
        mainWindow.setCentralWidget(self.centralwidget)
    

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    # Redefine the text in the Qlabel
    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Tuberculosis Detector"))
        self.groupBox_section1.setTitle(_translate("mainWindow", "File Import and Segmentation"))
        self.PushButton_filebrowsing.setText(_translate("mainWindow", "Browse from file"))
        self.pushButton_segmentationprocess.setText(
            _translate("mainWindow", "Segmentation with the U-Net model"))
        self.pushButton_saving.setText(_translate("mainWindow", "Save"))
        self.groupBox_section2.setTitle(_translate("mainWindow", "Classification by models"))
        self.label_model.setText(_translate("mainWindow", "Model Choice: DenseNet 201"))
        self.pushButton_run.setText(_translate("mainWindow", "Run the model"))
        self.label_4.setText(_translate("mainWindow", "Classification Result: "))





