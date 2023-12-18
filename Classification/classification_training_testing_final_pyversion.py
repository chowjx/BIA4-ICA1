# %% [markdown]
# # You can run this train model in kaggle through this link!
# https://www.kaggle.com/code/selix075/classification-training-testing/edit

# %% [markdown]
# # 1. Load packages and set hyperparameters

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:23.345890Z","iopub.execute_input":"2023-12-18T12:30:23.346575Z","iopub.status.idle":"2023-12-18T12:30:36.169920Z","shell.execute_reply.started":"2023-12-18T12:30:23.346547Z","shell.execute_reply":"2023-12-18T12:30:36.169022Z"}}
import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dropout,Dense,Conv2D,BatchNormalization,MaxPooling2D,Flatten
from keras.optimizers import SGD
from keras import initializers
from keras import regularizers

import sys
# if '../input/train-model/' not in sys.path:
#     sys.path.append('../input/train-model/')
#sys.path
#del sys
#sys.path.remove('../input/train-model/')
import sys
if '../input/pre-code' not in sys.path:
    sys.path.append('../input/pre-code')
if '../input/pre-code-nodenoise' not in sys.path:
    sys.path.append('../input/pre-code-nodenoise')

print(sys.path)
#from my_utils import utils_paths
#del utils_paths
#from image_preprocessing import preprocess_image
#from image_preprocessing_noDenoise import preprocess_image

import matplotlib.pyplot as plt
import numpy as np

import random
import pickle
import cv2
import os


# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.171781Z","iopub.execute_input":"2023-12-18T12:30:36.172320Z","iopub.status.idle":"2023-12-18T12:30:36.187057Z","shell.execute_reply.started":"2023-12-18T12:30:36.172292Z","shell.execute_reply":"2023-12-18T12:30:36.186042Z"}}
try:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d","--dataset",#required=True,
                    default="../input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database",
                    help="path to input dataset of images")
    ap.add_argument("-m","--model",#required=True,
                    default="/kaggle/working/",
                    help="path to output trained model ")
    ap.add_argument("-l","-label-bin",#required=True,
                    default=114,
                    help="path to output label binarizer")
    ap.add_argument("-p","--plot",#required=True,
                    default="/kaggle/working/",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
except:
    args={}
    
    #Set up the directories may be used
    Paths=['../input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database/', #0
           '../input/chest-xray-pneumoniacovid19tuberculosis/train/', #1
           '../input/chest-xray-pneumoniacovid19tuberculosis/test/', #2
           '../input/preprocessed/output/output', #3
           '../input/preprocessed/extra/extra', #4
           '../input/preprocessed/extra/extra', #5
           '../input/segmentation/Segmentation/Segmentation',#6
           '../input/segmentation/extra- segmentation/extra', #7
           '../input/segmentation/extra- segmentation/extra',#8
           '../input/old-data-split/training', #9
           '../input/old-data-split/validation', #10
           '../input/old-data-split/testing' #11
          ]
    
    args["training_dataset"] = Paths[0]
    args["finetune_dataset"] = Paths[1]
    args["test_dataset"] = Paths[2]
    
    args["target_classes"] = [["Normal","Tuberculosis"], #0
                              ["NORMAL","TURBERCULOSIS"], #1
                              ["NORMAL","TURBERCULOSIS"], #2
                              ["processed_normal","processed_tb"], #3
                              ["train_normal","train_tb"], #4
                              ["test_normal","test_tb"], #5
                              ["normal_segmentation","tb_segmentation"], #6
                              ["Seg_train_normal","Seg_train_tb"], #7
                              ["Seg_test_normal","Seg_test_tb"], #8
                              ["Normal","Tuberculosis"], #9
                              ["Normal","Tuberculosis"], #10
                              ["Normal","Tuberculosis"] #11
                             ]
    training_classes=args["target_classes"][0]
    finetune_classes=args["target_classes"][1]
    test_classes=args["target_classes"][2]
    #"../input/segmentation-self/Segmentation"
    merged_list=[(Paths[1],args["target_classes"][1]),
                 (Paths[9],args["target_classes"][9]),
                 (Paths[10],args["target_classes"][10])]
    
    #Initialize the hyperparameters
    args["INIT_LR"]=0.01 
    args["EPOCHS"]=200 
    args["MODEL_TYPE"]="densenet201" #（maybe:“CNN”，“MLP”，“densenet201”）
    args["BATCH_SIZE"]=300 
    args["NEW_WIDTH"]=224
    args["NEW_HEIGHT"]=224
    args["SEED"]=114514 #42
    args["SCALE"] = False
    args["PRE"]=False
print(args)



# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.188439Z","iopub.execute_input":"2023-12-18T12:30:36.188973Z","iopub.status.idle":"2023-12-18T12:30:36.219469Z","shell.execute_reply.started":"2023-12-18T12:30:36.188941Z","shell.execute_reply":"2023-12-18T12:30:36.218645Z"}}
random.seed(args["SEED"]) 

# %% [markdown]
# # 2. define important functions for training and testing

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.221919Z","iopub.execute_input":"2023-12-18T12:30:36.222268Z","iopub.status.idle":"2023-12-18T12:30:36.234285Z","shell.execute_reply.started":"2023-12-18T12:30:36.222214Z","shell.execute_reply":"2023-12-18T12:30:36.233531Z"}}
def input_images_preprocess(data_dir,target_classes,scaling=False,select_width=None,preprocess=True): #[256*2+1,256*3+1]
    from sklearn.preprocessing import LabelBinarizer
    from keras.utils import to_categorical
    import cv2
    import os
    from tqdm import tqdm
    from tqdm.notebook import tqdm_notebook
    import numpy as np
    
    print(f"[INFO] starting reading in {data_dir}")
    ## Load and preprocess the test data
    data = []
    labels = []
    image_w=[]
    image_h=[]
    # Iterate over test data
    for class_name in target_classes:
        class_path = os.path.join(data_dir, class_name)
        
        print(f'[INFO] read the {class_name} images')
        
        for imagePath in tqdm_notebook(os.listdir(class_path),dynamic_ncols=True):
            
            image_path = os.path.join(class_path, imagePath)
            image = cv2.imread(image_path)
            
            image_h.append(image.shape[0])
            image_w.append(image.shape[1])
            
            if select_width:
                image=image[:,select_width[0]:select_width[1]]
                #print("select the width from",select_width[0],"to",select_width[1])
            if preprocess:
                image=preprocess_image(image_path, size=(224, 224))
            image = cv2.resize(image, (args["NEW_WIDTH"], args["NEW_HEIGHT"]))
            data.append(image)
            labels.append(class_name)
        print(f'[INFO] {class_name} images reading done')

    print(f"[INFO] {data_dir} both classes reading done")

    if args["MODEL_TYPE"]=="MLP":
        # If you use MLP, you flatten the two-dimensional image into a one-dimensional vector
        data=[image.flatten() for image in data]
        print("Indeed, with the MLP model, the two-dimensional image is flattened into a one-dimensional vector")
    else:
        print("not a MLP model")

    # Convert labels to binary labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    #labels = to_categorical(labels, len(lb.classes_)) 
    labels=np.array(labels)
    data = np.array(data, dtype="float16")
    if scaling:
        data=data / 255.0
    
    print("data.shape:",data.shape)
    print("labels.shape:",labels.shape)
    
    return data,labels

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.235499Z","iopub.execute_input":"2023-12-18T12:30:36.235767Z","iopub.status.idle":"2023-12-18T12:30:36.247132Z","shell.execute_reply.started":"2023-12-18T12:30:36.235744Z","shell.execute_reply":"2023-12-18T12:30:36.246398Z"}}
def read_merge_data(merged_list, n_w, n_h):  
    merged_data = np.zeros((1, n_w*n_h*3))
    merged_labels = np.zeros((1, 1))
    for data_dir, target_classes in merged_list:  
        data, labels = input_images_preprocess(data_dir, target_classes, 
                                               preprocess=args["PRE"],scaling=args["SCALE"])  
        merged_data = np.concatenate((merged_data, data), axis=0) 
        merged_labels = np.concatenate((merged_labels, labels),axis=0)
    merged_data=merged_data[1:,]
    merged_labels=merged_labels[1:,]
    print("[INFO] Merge Done!")  
    print("merged_data.shape:", merged_data.shape)  
    print("merged_labels.shape:", merged_labels.shape)  
    return merged_data, merged_labels

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.248298Z","iopub.execute_input":"2023-12-18T12:30:36.248582Z","iopub.status.idle":"2023-12-18T12:30:36.263210Z","shell.execute_reply.started":"2023-12-18T12:30:36.248559Z","shell.execute_reply":"2023-12-18T12:30:36.262353Z"}}

def create_model(model_type):
    import tensorflow as tf
    model = tf.keras.models.Sequential()
    # kernel regularizer=regularizers,12(0.01)
    # keras.initializers.TruncatedNormal(mean=0.0，stddey=0.05， seed=None)
    # #initializers.random normal
    # model.add(Dronout(0.8))
    
    if model_type=="MLP":
        from keras.layers import Dropout,Dense,Conv2D,BatchNormalization,MaxPooling2D,Flatten
        model.add(Dense(512,input_shape=(args["NEW_WIDTH"]*args["NEW_HEIGHT"]*3,),
                        activation="relu")) 
        #model.add(Dropout(0.1,trainable=True))
        model.add(Dense(256,activation="relu",))
        #model.add(Dropout(0.1,trainable=True))
        model.add(Dense(1,activation="sigmoid"))

    elif model_type=="CNN": # Build a more complex CNN model with Batch Normalization and Dropout

        model.add(Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(args["NEW_WIDTH"], args["NEW_HEIGHT"], 3)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
    
    elif model_type=="densenet201":
        # Pretrained backbone
        #model = keras_cv.models.DenseNetBackbone.from_preset("densenet201_imagenet")
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model

        # load the pretrianed DenseNet201 model
        base_model = DenseNet201(weights='/kaggle/input/densenetbackbone/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                            include_top=False, input_shape=(args["NEW_WIDTH"], args["NEW_HEIGHT"], 3))

        #freeze the backbone layer
        for layer in base_model.layers:
            layer.trainable = False

        # add the classifier to DenseNet
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)  
        predictions = Dense(1, activation='sigmoid')(x)  

        # construct the whole model
        model = Model(inputs=base_model.input, outputs=predictions)
    return model

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.264434Z","iopub.execute_input":"2023-12-18T12:30:36.264749Z","iopub.status.idle":"2023-12-18T12:30:36.275944Z","shell.execute_reply.started":"2023-12-18T12:30:36.264724Z","shell.execute_reply":"2023-12-18T12:30:36.275090Z"}}
def flatten_dict(input):
    result={}
    for k, v in input.items():
        if not isinstance(v, dict):  
            result[k] = v
        else:
            for v_k,v_v in v.items():
                result[f"{k}_{v_k}"] = v_v
    return result

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.277141Z","iopub.execute_input":"2023-12-18T12:30:36.277423Z","iopub.status.idle":"2023-12-18T12:30:36.292627Z","shell.execute_reply.started":"2023-12-18T12:30:36.277399Z","shell.execute_reply":"2023-12-18T12:30:36.291788Z"}}
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn import preprocessing
def plot_confusion_matrix(classes_list, conf_mat, training_output_dir,present=""):
#     display_labels_x = []
#     display_labels_y = []
#     for label in classes_list:
#         display_labels_x += ["{0}\nn={1:.0f}".format(label, sum(conf_mat[:,i]))]
#         display_labels_y += ["{0}\nn={1:.0f}".format(label, sum(conf_mat[i,:]))]
#         print(display_labels_x,display_labels_y)
        #display_labels_x=[1,0]
        #yticks=display_labels_y=[1,0]
    display = ConfusionMatrixDisplay(confusion_matrix=preprocessing.normalize(conf_mat, norm="l1"), 
                                     #xticks=display_labels_x, 
                                     #yticks=display_labels_y
                                     display_labels=classes_list
                                    )
    display.plot(cmap="Blues",values_format=".2g")
    plt.title(present)
    plt.show()
    plt.savefig(f"{present}confusion_matrix.png")

# 计算ROC曲线所需的值  
def ROC_plot(Y_valid, Y_pred,present=""):
    from sklearn.metrics import roc_curve,roc_auc_score
    
    fpr, tpr, thresholds = roc_curve(Y_valid, Y_pred)  

    # Calculate the AUC value 
    auc = roc_auc_score(Y_valid, Y_pred)  
    print('AUC: %.3f' % auc)  

    # draw the ROC plot  
    plt.figure()  
    plt.plot([0, 1], [0, 1], 'k--')    
    plt.plot(fpr, tpr, color='red',label='AUC = {:.3f})'.format(auc))  
    plt.xlabel('False positive rate')    
    plt.ylabel('True positive rate')  
    plt.title('ROC Curve')    
    plt.legend(loc='best')   
    plt.show()
    plt.savefig(f"{present}ROC_plot.png")

def predict_model(model,test_data,binary_labels,present=""):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report 
    
    # Make predictions
    predictions = model.predict(test_data)
    
    #only select first column，Convert all forms of output to one dimension
    binary_labels=binary_labels.flatten()
    
    #convert prediction to 1 or 0
    binary_predictions = np.where(predictions.T > 0.5, 1, 0).flatten()
    
    print(binary_predictions,binary_predictions.shape)
    print(binary_labels,binary_labels.shape)

    # Evaluate the predictions
    report=classification_report(binary_labels,binary_predictions,digits=4)
    print(report)
    dic_report=classification_report(binary_labels,binary_predictions,digits=4,output_dict=True)
    dic_report = flatten_dict(dic_report)  


    # plot confusion matrix
    conf_mat = confusion_matrix(binary_labels, binary_predictions)
    classes_list = ["Normal", "Tuberculosis"]
    plot_confusion_matrix(classes_list, conf_mat, training_output_dir="/.", present=present)
    
    # plot ROC plot
    ROC_plot(binary_labels, predictions, present)
    
    # Find a normal sample with incorrect predictions
    normal_wrong_indices=np.where(binary_predictions[binary_labels==0]!=binary_labels[binary_labels==0])
    tuberculosis_wrong_indices=np.where(binary_predictions[binary_labels==1]!=binary_labels[binary_labels==1])
    print("normal_wrong_indices:",normal_wrong_indices[0].tolist())
    print("tuberculosis_wrong_indices:",tuberculosis_wrong_indices[0].tolist())
    
    return dic_report

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.293633Z","iopub.execute_input":"2023-12-18T12:30:36.293921Z","iopub.status.idle":"2023-12-18T12:30:36.305353Z","shell.execute_reply.started":"2023-12-18T12:30:36.293898Z","shell.execute_reply":"2023-12-18T12:30:36.304536Z"}}
# from sklearn.metrics import classification_report  
# y_true = [0, 1, 1, 0]  
# y_pred = [0, 0, 1, 1]  
# print(classification_report(y_true, y_pred))

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.309690Z","iopub.execute_input":"2023-12-18T12:30:36.309965Z","iopub.status.idle":"2023-12-18T12:30:36.327881Z","shell.execute_reply.started":"2023-12-18T12:30:36.309942Z","shell.execute_reply":"2023-12-18T12:30:36.326909Z"}}
def train_model(model,trainX,testX,trainY,testY,patience, present="", model_type=""):
    import tensorflow as tf
    
    print(f"[INFO] {present} strating training！（happy）")
    
    #create a callback
    from tensorflow.keras.callbacks import Callback,EarlyStopping
    early_stop = EarlyStopping(patience=patience, restore_best_weights=True)
    

    from tensorflow.keras.optimizers import Adam
    opt = Adam(lr=args["INIT_LR"]) 
    
    loss_fuction="binary_crossentropy"
#     if model_type!="MLP":
#         loss_fuction="binary_crossentropy"
#     else: #redefine the loss function to use CBP
#         def CBP_loss(y_true, y_pred):
#             grad_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            
#             return loss
#         loss_fuction=

    #compile the model
    model.compile(loss=loss_fuction,
                  optimizer=opt,
                  metrics=["accuracy"])

    #Stores the accuracy and loss of each epoch    
    training_accuracy = []    
    validation_accuracy = []    
    training_loss = []    
    validation_loss = []    

    # A custom callback function to record accuracy and loss for each epoch   
    class CustomCallback(tf.keras.callbacks.Callback):    
        def on_epoch_end(self, epoch, logs=None):    
            training_accuracy.append(logs['accuracy'])    
            validation_accuracy.append(logs['val_accuracy'])    
            training_loss.append(logs['loss'])    
            validation_loss.append(logs['val_loss'])     
    
    # Create an instance of a custom callback function  
    custom_callback = CustomCallback()    

    # Define a list of callback functions  
    callbacks = [custom_callback, early_stop]   
    
    import time  
    # Start recording the time before training starts  
    start_time = time.time() 

    # Training network model
    H = model.fit(trainX,trainY,validation_data=(testX, testY),callbacks=callbacks,
                  epochs=args["EPOCHS"],batch_size=args["BATCH_SIZE"])

    # End the record and calculate the total time 
    end_time = time.time()  
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)    
    minutes, seconds = divmod(remainder, 60)  
    print(f"{present} Total training time: {hours} hours {minutes} minutes {seconds} seconds")
    
    
    # Plot accuracy and loss over time  
    plt.figure(figsize=(10, 6))      
    plt.plot(range(len(training_loss)), training_loss, label="Training Loss")    
    plt.plot(range(len(validation_loss)), validation_loss, label="Validation Loss")    
    plt.xlabel(f"{present} Epoch")    
    plt.ylabel(f"{present} Loss")    
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))    
    plt.plot(range(len(training_accuracy)), training_accuracy, label="Training Accuracy")    
    plt.plot(range(len(validation_accuracy)), validation_accuracy, label="Validation Accuracy")  
    plt.xlabel(f"{present} Epoch")    
    plt.ylabel(f"{present} Accuracy")    
    plt.legend()
    plt.show()
    print(f"[INFO] {present} training complete！（happy）")

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.329072Z","iopub.execute_input":"2023-12-18T12:30:36.329443Z","iopub.status.idle":"2023-12-18T12:30:36.342689Z","shell.execute_reply.started":"2023-12-18T12:30:36.329409Z","shell.execute_reply":"2023-12-18T12:30:36.341602Z"}}
def cross_validation(data, labels, patience,nfolds=5,random_state=114514, model_type=""):
    from sklearn.model_selection import KFold,StratifiedKFold
    from keras.models import load_model
    import pandas as pd
    import matplotlib.pyplot as plt
    # Define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    # K-fold Cross Validation model evaluation
    fold=0
    
    #establish the reports for the Kfold models
    reports = {}
    
    for train, test in kfold.split(data, labels):    
        import tensorflow as tf 
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"]) 
        # Using dual Gpus or single Gpus, "/gpu:1"
        with strategy.scope(): 
            model = create_model(args["MODEL_TYPE"])
            train_model(model,data[train],data[test],labels[train],labels[test],
                        patience=patience,present=f"{fold} fold training",model_type=model_type)
            #save the model
            print(f"[INFO]{fold} fold {model_type} model storing...")
            model.save(f'{fold}_fold_{model_type}_chest_xray_model.h5')
            print(f"[INFO]{fold} fold {model_type} model storing complete！")
            
            model=load_model(f'{fold}_fold_{model_type}_chest_xray_model.h5')
            
            #fill in the reports for the Kfold models
            report = predict_model(model,data[test],labels[test],f"{fold} fold validation")
            reports[f"{fold}_fold"]=report
        fold=fold+1
    reports=pd.DataFrame.from_dict(reports)
    reports.to_csv('cross_validation_reports.csv', index=True)
    print(reports)

# %% [markdown]
# # 3. Load the data for training

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:30:36.343821Z","iopub.execute_input":"2023-12-18T12:30:36.344106Z","iopub.status.idle":"2023-12-18T12:31:56.264543Z","shell.execute_reply.started":"2023-12-18T12:30:36.344082Z","shell.execute_reply":"2023-12-18T12:31:56.263562Z"}}
try:
    pre_MLP_judge=MLP_judge
except:
    pre_MLP_judge="nothing"

if args["MODEL_TYPE"]=="MLP":
    MLP_judge=True
else:
    MLP_judge=False

try:
    data, labels=data, labels
    if pre_MLP_judge!=MLP_judge:
        raise("reload the data since the MLP requires different data inputs from other models")
except:
    data, labels= input_images_preprocess(data_dir=Paths[0],
                                          target_classes = args["target_classes"][0],
                                          preprocess=args["PRE"],scaling=args["SCALE"])

# data, labels = merged_data, merged_labels

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:31:56.265809Z","iopub.execute_input":"2023-12-18T12:31:56.266142Z","iopub.status.idle":"2023-12-18T12:31:56.271248Z","shell.execute_reply.started":"2023-12-18T12:31:56.266111Z","shell.execute_reply":"2023-12-18T12:31:56.270435Z"}}
print(data.shape,labels.shape)

# %% [markdown]
# # 4. Training (5-fold cross validation)
# # > If you want to save more time, you can intercept codes here once you obtain the 0-fold model.

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T12:31:56.272461Z","iopub.execute_input":"2023-12-18T12:31:56.272750Z","iopub.status.idle":"2023-12-18T13:07:34.023656Z","shell.execute_reply.started":"2023-12-18T12:31:56.272726Z","shell.execute_reply":"2023-12-18T13:07:34.022790Z"}}
from tensorflow import test
if test.is_gpu_available():
    cross_validation(data, labels, nfolds=5, patience=4,random_state = args["SEED"],model_type=args["MODEL_TYPE"])
else:
    print("no GPU, you'd better not run it")

# %% [markdown]
# # 5. Download the saved models from training

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:07:34.024865Z","iopub.execute_input":"2023-12-18T13:07:34.025158Z","iopub.status.idle":"2023-12-18T13:07:34.031918Z","shell.execute_reply.started":"2023-12-18T13:07:34.025132Z","shell.execute_reply":"2023-12-18T13:07:34.031047Z"}}
import os
os.chdir('/kaggle/working')
print(os.getcwd())
print(os.listdir("/kaggle/working"))
from IPython.display import FileLink,FileLinks
for i in os.listdir("/kaggle/working"):
    print(i)
    try:
        FileLink(i)
    except:
        FileLinks(i)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:07:34.033180Z","iopub.execute_input":"2023-12-18T13:07:34.033834Z","iopub.status.idle":"2023-12-18T13:07:34.045696Z","shell.execute_reply.started":"2023-12-18T13:07:34.033798Z","shell.execute_reply":"2023-12-18T13:07:34.044682Z"}}
FileLink(f'0_fold_{args["MODEL_TYPE"]}_chest_xray_model.h5')

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:07:34.046809Z","iopub.execute_input":"2023-12-18T13:07:34.047073Z","iopub.status.idle":"2023-12-18T13:07:34.057190Z","shell.execute_reply.started":"2023-12-18T13:07:34.047049Z","shell.execute_reply":"2023-12-18T13:07:34.056305Z"}}
FileLink(f'1_fold_{args["MODEL_TYPE"]}_chest_xray_model.h5')

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:07:34.058238Z","iopub.execute_input":"2023-12-18T13:07:34.058552Z","iopub.status.idle":"2023-12-18T13:07:34.068878Z","shell.execute_reply.started":"2023-12-18T13:07:34.058523Z","shell.execute_reply":"2023-12-18T13:07:34.068046Z"}}
FileLink(f'2_fold_{args["MODEL_TYPE"]}_chest_xray_model.h5')

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:07:34.069928Z","iopub.execute_input":"2023-12-18T13:07:34.070260Z","iopub.status.idle":"2023-12-18T13:07:34.079471Z","shell.execute_reply.started":"2023-12-18T13:07:34.070232Z","shell.execute_reply":"2023-12-18T13:07:34.078704Z"}}
FileLink(f'3_fold_{args["MODEL_TYPE"]}_chest_xray_model.h5')

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:07:34.080651Z","iopub.execute_input":"2023-12-18T13:07:34.081014Z","iopub.status.idle":"2023-12-18T13:07:34.091260Z","shell.execute_reply.started":"2023-12-18T13:07:34.080986Z","shell.execute_reply":"2023-12-18T13:07:34.090138Z"}}
FileLink(f'4_fold_{args["MODEL_TYPE"]}_chest_xray_model.h5')

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:07:34.092370Z","iopub.execute_input":"2023-12-18T13:07:34.092730Z","iopub.status.idle":"2023-12-18T13:07:34.101269Z","shell.execute_reply.started":"2023-12-18T13:07:34.092699Z","shell.execute_reply":"2023-12-18T13:07:34.100406Z"}}
FileLink(f'cross_validation_reports.csv')

# %% [markdown]
# # 6. input the external testing dataset

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:07:34.102591Z","iopub.execute_input":"2023-12-18T13:07:34.103235Z","iopub.status.idle":"2023-12-18T13:07:41.244953Z","shell.execute_reply.started":"2023-12-18T13:07:34.103183Z","shell.execute_reply":"2023-12-18T13:07:41.243948Z"}}
test_data,binary_labels = input_images_preprocess(Paths[2],args["target_classes"][2],
                                               preprocess=args["PRE"],scaling=args["SCALE"]
                                               )

# %% [markdown]
# # 7. Reload the model and test the external dataset (Remember to change the paths!)
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:07:41.246128Z","iopub.execute_input":"2023-12-18T13:07:41.246408Z","iopub.status.idle":"2023-12-18T13:08:16.728126Z","shell.execute_reply.started":"2023-12-18T13:07:41.246382Z","shell.execute_reply":"2023-12-18T13:08:16.727316Z"}}
from keras.models import load_model
#model=load_model(f'3_fold_{args["MODEL_TYPE"]}_chest_xray_model.h5')
model=load_model(f'/kaggle/input/densenet201-finetuning/4_fold_densenet201_chest_xray_model.keras')
# you can change the directory!

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:08:16.730517Z","iopub.execute_input":"2023-12-18T13:08:16.730810Z","iopub.status.idle":"2023-12-18T13:08:24.861551Z","shell.execute_reply.started":"2023-12-18T13:08:16.730784Z","shell.execute_reply":"2023-12-18T13:08:24.860547Z"}}
other_dataset_report=predict_model(model,test_data,binary_labels,present="test the model")
import pandas as pd
other_dataset_report={k:[v] for k,v in other_dataset_report.items()}
other_dataset_report=pd.DataFrame(other_dataset_report)
print(other_dataset_report)
other_dataset_report.to_csv('other_dataset_report.csv', index=False)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:08:24.862998Z","iopub.execute_input":"2023-12-18T13:08:24.864028Z","iopub.status.idle":"2023-12-18T13:08:24.870923Z","shell.execute_reply.started":"2023-12-18T13:08:24.863991Z","shell.execute_reply":"2023-12-18T13:08:24.869972Z"}}
import os
out_path=os.listdir("../input/chest-xray-pneumoniacovid19tuberculosis/test/NORMAL")
print(out_path[118])
print(out_path[196])
print(out_path[199])

# %% [markdown]
# # 8. Preparation for GUI

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:08:24.872174Z","iopub.execute_input":"2023-12-18T13:08:24.872612Z","iopub.status.idle":"2023-12-18T13:08:31.400101Z","shell.execute_reply.started":"2023-12-18T13:08:24.872576Z","shell.execute_reply":"2023-12-18T13:08:31.399121Z"}}
# test one image(preparation for GUI)
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import keras
print(keras.__version__)


# model = tf.keras.applications.DenseNet121(weights='imagenet') 
# print(model.summary())
model_type="preprocessed_densenet201" # This will change according to your own situation
def input_trained_model(model_type):
    from keras.models import load_model
    path={"raw_densenet201":"You changed it according to the way you put your model, and this path down here is also modified according to the way you put your model",
          "preprocessed_densenet201":"/kaggle/working/0_fold_densenet201_chest_xray_model.h5"
         }
    model_path = path[model_type]
    # This is to choose different paths according to different models (remember to change!) 
    # For example, model_path
    model = load_model(model_path)
    return model
model = input_trained_model(model_type)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-18T13:08:31.402784Z","iopub.execute_input":"2023-12-18T13:08:31.403161Z","iopub.status.idle":"2023-12-18T13:08:36.309135Z","shell.execute_reply.started":"2023-12-18T13:08:31.403125Z","shell.execute_reply":"2023-12-18T13:08:36.308190Z"}}
image_path = "../input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database/Normal/Normal-5.png" 
# This is the path you choose to enter the picture, then you will change the corresponding variable

def input_and_judge(model_type,model,image_path,scaling= False,preprocess=args["PRE"]):
    import cv2
    import numpy as np
    if preprocess:
        image=preprocess_image(image_path, size=(224, 224))
    else:
        image = cv2.imread(image_path)
        #print(image.shape)
        image = cv2.resize(image, (224, 224))
    
    if scaling:
        image = np.array(image) / 255.0 #scaling
    image = np.array([image])
    #because there're only one image, we need to add one dimension to fit the model's shape
    #print(image.shape)
    if model_type == "MLP":
        image=image.flatten()
    #print(image)
    out = model.predict(image)
    out ="turberculosis" if out>0.5 else "normal"
    return out

result = input_and_judge(model_type,model,image_path,scaling=args["SCALE"])

print(f'{image_path}: {result}')

# %% [code]