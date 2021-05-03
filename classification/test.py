'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
#from utils import ImageFolder
from utils import CustomImageDataset
from torchvision.utils import save_image
import os
import argparse
import numpy as np
from utils import progress_bar
from foolbox import PyTorchModel
import foolbox.attacks as fa
from foolbox.criteria import Misclassification
parser = argparse.ArgumentParser(description='PyTorch Malware Image Classification')
parser.add_argument('--batch_size', default=100, type=int, help='size of batch')
parser.add_argument('--name', type=str, help='name of trial')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
testset = CustomImageDataset(os.path.join('dataset', 'noise'), transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16)


# Model
print('==> Building model..')

net = models.GoogLeNet(init_weights=True)
net = net.to(device)
net.eval()

cudnn.benchmark = True

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/checkpoint/checkpoint.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(testloader):
            imgs, targets = inputs['image'].to(device), inputs['label'].to(device)
            outputs = net(imgs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

test(0)