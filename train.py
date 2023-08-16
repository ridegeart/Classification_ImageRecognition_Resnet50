import os
import sys
sys.path.append(os.getcwd())
current = '/home/agx/AUO_FMA/Resnet50_SingleLayer/'
os.chdir(current)

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_dataset import LoadDataset
from helper import calculate_accuracy
# 訓練要使用到的model 包含resnet/inception/transformer
from model.ResNet import ResNet50,ResNet101
from model.inceptionV4 import InceptionV4
from model.transformer import VisionTransformer

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from torchsummary import summary
from thop import profile
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
BatchSize = 42
if __name__ == '__main__':
    print(torch.__version__)
    print('Torch Cuda:',torch.cuda.is_available())
    print('Torch Cuda Device counts:',torch.cuda.device_count())

    transform_train = transforms.Compose([
        transforms.RandomAffine(40, scale=(.85, 1.15), shear=0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_csv_path = './dataset/train.csv'
    test_csv_path = './dataset/test.csv'

    # number of workers
    nw = min([os.cpu_count(), BatchSize if BatchSize > 1 else 0, 8])  
    print('Using {} dataloader workers every process'.format(nw))

    train = LoadDataset(csv_path=train_csv_path, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train, batch_size=42, shuffle=True, num_workers=nw)

    test = LoadDataset(csv_path=test_csv_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test, batch_size=42,shuffle=False, num_workers=nw)

    print('train_dataset:'+str(len(train)))
    print('test_dataset:'+str(len(test)))
    
    #------------Normal Trained---------------------------
    modelName = './dataset/resnet50-19c8e357.pth'
    net = ResNet50(14)
    """
    net = VisionTransformer(image_size=224, patch_size=16,num_layers=12,num_heads=12,
                            hidden_dim=768, mlp_dim=3072)
    """
    #-------------Pretrained Weight---------------------------
    model_state = net.state_dict()
    checkpoint = torch.load(modelName)

    # 將預訓練的輸出(fc層)部分刪除
    del_keys = ['fc.weight', 'fc.bias'] 
    for k in del_keys:
        del checkpoint[k]
    # 將pretrained_dict與 model_state 配對
    pretrained_dict =  {k: v for k, v in checkpoint.items() if k in model_state}
    # 更新現有的model_dict
    model_state.update(pretrained_dict)
    # load更新後的model_state
    net.load_state_dict(model_state)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    # pg：只訓練可以訓練的參數。
    pg = [p for p in net.parameters() if p.requires_grad]
    
    #----------optimizer 選擇----------------
    optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)
    '''
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    optimizer = optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.3)
    '''
    #----------scheduler 選擇----------------
    lf = lambda x: ((1 + math.cos(x * math.pi / 100)) / 2) * (1 - 0.01) + 0.01  # cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    '''
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor = 0.1, patience=5)
    main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 - 30, eta_min=0.0)
    warmup_lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.033, total_iters=30)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[30])
    '''
    #-------------Model Size---------------------------
    '''
    summary(net,input_size=(3,224,224))
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(net, inputs=(input,))
    print('flops：',flops,' , params：',params)
    '''
    train_epoch_loss = []
    train_epoch_acc = []
    test_epoch_loss = []
    test_epoch_acc = []
    train_acc = 0
    best_epoch = 0

    EPOCHS = 100
    for epoch in range(EPOCHS):
        losses = []
        epoch_acc = []
        net.train()
        for i, inp in tqdm(enumerate(trainloader)):
            inputs, labels = inp['image'].to(device), inp['label_1'].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            
            epoch_acc.append(calculate_accuracy(predictions=outputs, labels=labels))
        
        avg_loss = sum(losses)/len(losses)
        train_epoch_loss.append(avg_loss)
        avg_acc = sum(epoch_acc)/len(epoch_acc)
        train_epoch_acc.append(avg_acc)
        #scheduler.step(avg_loss)
        scheduler.step() # 更新學習率

        print(f'Training Loss at epoch {epoch} : {avg_loss}')
        print(f'Training accuracy at epoch {epoch} : {avg_acc}')

        losses = []
        epoch_acc = []
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data['image'].to(device), data['label_1'].to(device)
                outputs = net(images)
                
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                
                epoch_acc.append(calculate_accuracy(predictions=outputs, labels=labels))

        avg_testloss = sum(losses)/len(losses)
        test_epoch_loss.append(avg_testloss)
        avg_testacc = sum(epoch_acc)/len(epoch_acc)
        test_epoch_acc.append(avg_testacc)
        
        print(f'Testing Loss at epoch {epoch} : {avg_testloss}')
        print(f'Testing accuracy at epoch {epoch} : {avg_testacc}')

        if train_acc < avg_testacc:
            train_acc = avg_testacc
            best_epoch = epoch
            torch.save(net.state_dict(), './dataset/FMA_best50_pre.pth')
            print("Best Model saved!")

        torch.save(net.state_dict(), './dataset/FMA_fin50_pre.pth')
        print("final Model saved!")

    print('Training Done')

    print('starte picture for acc')
    path='./graph_folder/'
    num_epoch=100, 
    epochs = [x for x in range(num_epoch[0])]
    print(len(epochs),len(train_epoch_acc))

    train_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_epoch_acc, "Mode":['train']*(num_epoch[0])})
    test_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_epoch_acc, "Mode":['test']*(num_epoch[0])})
            
    sns.lineplot(data=data_acc.reset_index(), x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Superclass Accuracy Graph')
    plt.savefig(path+f'accuracy_superclass_epoch.png')
    plt.clf()

    #--LOSS--#
    train_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":train_epoch_loss, "Mode":['train']*(num_epoch[0])})
    test_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":test_epoch_loss, "Mode":['test']*(num_epoch[0])})
    train_test_loss = pd.concat([train_loss_df, test_loss_df])
            
    sns.lineplot(data=train_test_loss.reset_index(), x='Epochs', y='Loss', hue='Mode')
    plt.title('Loss Graph')
    plt.savefig(path+f'loss_epoch.png')
    plt.clf()
    print('picture save done  start next epoch')
    print(f'Best accuracy at {best_epoch}')