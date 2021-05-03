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

import os
import argparse

from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch Malware Image Classification')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--batch_size', default=100, type=int, help='size of batch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--name', type=str, help='name of trial')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

trainset = CustomImageDataset(os.path.join('dataset', 'train'), transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)



testset = CustomImageDataset(os.path.join('dataset', 'test'), transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16)



# Model
print('==> Building model..')

net = models.GoogLeNet(init_weights=True)
net = net.to(device)
cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/vgg16.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_lr = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[60,120,180], gamma=0.1)
checkpoint_path = args.name
if not os.path.isdir(args.name):
    os.mkdir(os.path.join('checkpoint',args.name))
f = open(os.path.join('checkpoint',checkpoint_path, "log.txt"),"a")
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print(optimizer)
    for batch_idx,inputs in enumerate(trainloader):
        
        inputs, targets = inputs['image'].to(device), inputs['label'].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs.logits, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        
    optimizer_lr.step()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(testloader):
            inputs, targets = inputs['image'].to(device), inputs['label'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    f.write('Epoch[%d] Test Acc : %.3f%% (%d/%d)' %(epoch, acc, correct, total))
    f.flush()
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        torch.save(state, os.path.join('checkpoint',checkpoint_path,'checkpoint.pth'))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+201):
    train(epoch)
    test(epoch)
f.close()