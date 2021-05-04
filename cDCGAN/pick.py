import os
import shutil
import glob
import random
train_data = os.path.join('dataset','train')
training_data = os.path.join('data','train')
os.makedirs(training_data,exist_ok=True)
class_list =os.listdir(train_data)
for i,class_ in enumerate(class_list):
    image_list = glob.glob(os.path.join(train_data,class_,"*.jpeg"))
    unique_num = random.sample(range(0, len(image_list)), int(len(image_list)))
    os.makedirs(os.path.join(training_data, class_), exist_ok=True)
    for k, num in enumerate(unique_num):
        filename = os.path.basename(image_list[num])
        shutil.copyfile(image_list[num], os.path.join(training_data, class_, filename))
        if k ==32:
            break