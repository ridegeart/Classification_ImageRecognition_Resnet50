# %%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_dataset import LoadDataset
import os
from ResNet import Bottleneck, ResNet, ResNet50

current = 'C:/Users/CamyTang/FMADefect/ResNet50/ResNet/'
os.chdir(current)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# %%
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# %%
train_csv_path = './dataset/train.csv'
test_csv_path = './dataset/test.csv'

train = LoadDataset(csv_path=train_csv_path, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)

test = LoadDataset(csv_path=test_csv_path, transform=transform_test)
testloader = torch.utils.data.DataLoader(test, batch_size=128,shuffle=False, num_workers=2)

# %%
classes = ['CF REPAIR FAIL','PI SPOT-WITH PAR','POLYMER','GLASS BROKEN','PV-HOLE-T','CF DEFECT','CF PS DEFORMATION','FIBER','AS-RESIDUE-E','LIGHT METAL','GLASS CULLET','ITO-RESIDUE-T','M1-ABNORMAL','ESD']

# %%
net = ResNet50(14).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

train_epoch_loss = []
# %%
EPOCHS = 100
for epoch in range(EPOCHS):
    losses = []
    running_loss = 0
    for i, inp in enumerate(trainloader):
        inputs, labels = inp['image'].to(device), inp['label_1'].to(device)
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
    train_epoch_loss.append(avg_loss)
    scheduler.step(avg_loss)
            
print('Training Done')

# %%
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data['image'].to(device), data['label_1'].to(device)
        outputs = net(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy on 10,000 test images: ', 100*(correct/total), '%')

# %%
