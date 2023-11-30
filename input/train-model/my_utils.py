import os  
  
class utils_paths:  
    @staticmethod  
    def list_images(directory):  
        image_paths = []  
        
        # 遍历目录中的文件和子目录  
        for foldername in os.listdir(directory):  
            folder_path = os.path.join(directory, foldername)  
            
            # 确保它是一个文件夹  
            if os.path.isdir(folder_path):  
                # 如果是"Normal"或"Tuberculosis"文件夹，则遍历其中的图片文件  
                if foldername.lower() in ["normal", "tuberculosis"]:  
                    for filename in os.listdir(folder_path):  
                        file_path = os.path.join(folder_path, filename)  
                        # 确保它是一个文件，并且是一个图片文件（这里只检查了常见的图片格式）  
                        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.png', '.jpeg')):  
                            image_paths.append(file_path)  
        
        return image_paths
