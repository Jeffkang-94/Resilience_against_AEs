import os
import shutil
import glob
import random
train_data = os.path.join('dataset','train')
test_data = os.path.join('dataset','test')
os.makedirs(test_data,exist_ok=True)
class_list =os.listdir(train_data)
for i,class_ in enumerate(class_list):
    image_list = glob.glob(os.path.join(train_data,class_,"*.jpeg"))
    unique_num = random.sample(range(0, len(image_list)), int(len(image_list)*0.1))
    os.makedirs(os.path.join(test_data, class_), exist_ok=True)
    for num in unique_num:
        filename = os.path.basename(image_list[num])
        shutil.move(image_list[num], os.path.join(test_data, class_, filename))