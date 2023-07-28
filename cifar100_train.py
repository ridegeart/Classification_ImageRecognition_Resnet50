import os
import sys
sys.path.append(os.getcwd())
current = 'C:/Users/CamyTang/FMADefect/ResNet50/'
os.chdir(current)

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cifar100 import CIFAR100

from ResNet import Bottleneck, ResNet, ResNet50
from model.transformer import VisionTransformer
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    print(torch.__version__)
    print('Torch Cuda:',torch.cuda.is_available())
    print('Torch Cuda Device counts:',torch.cuda.device_count())

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train = CIFAR100(root='./dataset', train=True, download=False, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(train, batch_size=42, shuffle=True, num_workers=2)

    test = CIFAR100(root='./dataset', train=False, download=False, transform=transform_test)

    testloader = torch.utils.data.DataLoader(test, batch_size=42,shuffle=False, num_workers=2)

    modelName = './dataset/vit_b_16_lc_swag-4e70ced5.pth'

    net = VisionTransformer(image_size=224, patch_size=16,num_layers=12,num_heads=12,
                            hidden_dim=768, mlp_dim=3072).to(device)
    #-------------Pretrained Weight---------------------------
    model_state = net.state_dict()
    checkpoint = torch.load(modelName)

    # 將權重的輸出改為自定義數據集的num_epoch
    checkpoint['heads.head.weight'] = torch.rand((100, 768))
    checkpoint['heads.head.bias'] = torch.rand(100)
    
    # 將pretrained_dict裡不屬於model_dict的key刪除
    pretrained_dict =  {k: v for k, v in checkpoint.items() if k in model_state}
    # 更新現有的model_dict
    model_state.update(pretrained_dict)
    # load更新後的model_state
    net.load_state_dict(model_state)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

    EPOCHS = 200
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
        
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i%100 == 0 and i > 0:
                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        avg_loss = sum(losses)/len(losses)
        scheduler.step(avg_loss)
                
    print('Training Done')

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on 10,000 test images: ', 100*(correct/total), '%')