
model_type="MLP"

from tensorflow.keras.models import load_model
from keras.models import load_model
#image_path = "/Users/user/Desktop/professional courses/BIA 4/ICA1/2022-23-Group-06-main 2/test_chest_xray/NORMAL/test_normal_0001.jpeg"

#model = load_model("/Users/user/Desktop/professional courses/BIA 4/ICA1/2022-23-Group-06-main/Group 5/GUI/3_fold_densenet201_chest_xray_model.h5")
def input_and_judge(model_type,model,image_path):
    import cv2
    import numpy as np
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.array(image)
    image = np.array([image])
    #because there're only one image, we need to add one dimension to fit the model's shape
    if model_type == "MLP":
        image=image.flatten()
    out = model.predict(image)
    out ="turberculosis" if out> 0.5 else "normal"
    return out

#result = input_and_judge(model_type,model,image_path)

#print(f'{image_path}: {result}')
