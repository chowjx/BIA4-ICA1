# Classification Model Training and Testing on Kaggle

## Overview
This guide provides instructions for running a classification model using both internal and external datasets. All codes and datasets are publicly available on Kaggle, and you can execute the training and testing process using GPU through [this Kaggle link](https://www.kaggle.com/code/selix075/classification-training-testing-final/edit).

## Steps

### 1. Load Packages and Set Hyperparameters
- **Load Necessary Libraries**: Import all the required packages for the process.
- **Set Hyperparameters**: Define essential hyperparameters for model training.

### 2. Define Important Functions
- **Function Definitions**: Create key functions for both training and testing phases.

  `input_images_preprocess`

  `read_merge_data`

  `create_model`

  `predict_model`

  `train_model`

  `cross_validation`
### 3. Data Loading for Training
- **Load Training Data**: Prepare the dataset to be used for training the model.

### 4. Model Training
- **Conduct 5-Fold Cross Validation Training**: This step includes training with 5-fold cross-validation. To save time, you can stop the process after obtaining the 0-fold model.

### 5. Download Trained Models
- **Retrieve Saved Models**: Download the models saved after the training process.

### 6. Input External Testing Dataset
- **Prepare External Dataset**: Load the external dataset for testing purposes.

### 7. Test the External Dataset
- **Reload the Model**: Bring the trained model back into the environment.
- **Perform Testing**: Test the model using the external dataset.
- **Note**: Remember to modify file paths as needed for your specific testing environment.

### 8. Preparation for GUI
- **Setup for User Interface**: Arrange necessary components for the graphical user interface.

  `input_trained_model`

  `input_and_judge`
## Additional Notes
- Ensure you have access to Kaggle and can utilize GPU for the training and testing process.
- Regularly check and update file paths and environment settings to match your requirements,
  especially when you use your own devices, instead of Kaggle.
