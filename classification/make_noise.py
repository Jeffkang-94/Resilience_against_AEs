from PIL import Image
import os, glob
from torchvision.utils import save_image
from torchvision.datasets.folder import pil_loader
from torchvision import transforms
import numpy as np
import torch

test_data = os.path.join('dataset','test')
noise_data = os.path.join('dataset','noise')
class_list =os.listdir(test_data)
for i,class_ in enumerate(class_list):
    image_list = glob.glob(os.path.join(test_data,class_,"*.jpeg"))
    os.makedirs(os.path.join(noise_data, class_), exist_ok=True)
    for img in image_list:
        filename = os.path.basename(img)
        img = pil_loader(img)
        img = transforms.ToTensor()(img)
        img = img.cuda()
        noise = torch.zeros_like(img).cuda()
        noise.data.normal_(0, 0.001)
        img = noise+img

        save_image(img, os.path.join(noise_data, class_, filename), normalize=True)

