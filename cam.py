#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import resnet50
import torch.nn as nn
import matplotlib.pyplot as plt

class_num = 14

modelName = './dataset/FMA_3layer_50.pth'  #更換為訓練好的模型

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = resnet50.ResNet50()
model_ft = model.to(device)
model_ft.load_state_dict(torch.load(modelName, map_location=lambda  storage, loc:storage),False)  
#讀取最後一層的輸出特徵圖
model_features = nn.Sequential(*list(model_ft.children())[:-7])

class_ = {0:'CF REPAIR FAIL',1:'PI SPOT-WITH PAR',2:'POLYMER',3:'GLASS BROKEN',4:'PV-HOLE-T',5:'CF DEFECT',6:'CF PS DEFORMATION',7:'FIBER',8:'AS-RESIDUE-E',9:'LIGHT METAL',10:'GLASS CULLET',11:'ITO-RESIDUE-T',12:'M1-ABNORMAL',13:'ESD'}
model_ft.eval()
model_features.eval()

# Display all model layer weights
"""
for name, para in model_ft.named_parameters():
    print('{}: {}'.format(name, para.shape))
"""
#獲得fc層的權重
fc_weights = model_ft.linear_lvl3.weight   #(14,2048)

data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224), Image.ANTIALIAS),
                                    transforms.RandomAffine(40, scale=(.85, 1.15), shear=0),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomPerspective(distortion_scale=0.2),
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                    transforms.ToTensor()]),
        "val": transforms.Compose([transforms.Resize((224, 224), Image.ANTIALIAS),
                                    transforms.ToTensor()])}

#---------------單張熱力圖------------------

img_path = './dataset/images/CF PS DEFORMATION@NP@CF@CF_20220619_B5564PE-1-2.jpg'     #圖片路徑
CAM_RESULT_PATH = './data/'  #熱力圖的儲存位置

if not os.path.exists(CAM_RESULT_PATH):
    os.mkdir(CAM_RESULT_PATH)

#加載圖像
_, img_name = os.path.split(img_path)
img = Image.open(img_path).convert('RGB')
img_tensor = (data_transform['val'](img)/255.0).unsqueeze(0).to(device) #[1,3,224,224]

# 提取最后一層特徵圖
features = model_features(img_tensor).detach()  #[1,2048,56,56]
# 獲得網路輸出
logit = model_ft(img_tensor)
h_x = torch.nn.functional.softmax(logit[2], dim=1).data.squeeze()  #tensor([0.9981, 0.0019])
probs, idx = h_x.sort(0, True)     #按概率從大到小排列(信心度,索引值)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()
# 獲得所有類別預測結果（含信心值,由大到小）
for i in range(class_num):
    print('{:.3f} -> {}'.format(probs[i], class_[idx[i]]))
# 獲得熱力圖的函數
def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape        #1,2048,56,56
    output_cam = []
    for idx in class_idx: 
        feature_conv = feature_conv.reshape((nc, h*w))  # [2048,56*56]
        cam = torch.mm(weight_softmax[idx].unsqueeze(0),feature_conv)  #(1, 2048) * (2048, 56*56) -> (1, 56*56)
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = cam_img.detach().cpu().numpy()
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)
 
        output_cam.append(cam_img)
    return output_cam
 
# CAMs = returnCAM(features, fc_weights, [idx[0]])  #得到預測信心度最高的類別對應的熱力圖
CAMs = returnCAM(features, fc_weights, idx)  #得到所有預測類別的熱力圖
print(img_name + ' output for the top1 prediction: %s' % class_[idx[0]])
# 讀取圖片
img = cv2.imread(img_path)
height, width, _ = img.shape 

heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)  #CAM上顯示信心度最大值的class
result = heatmap * 0.6 + img * 0.4    #比例可以自己調節

text = '%s %.2f%%' % (class_[idx[0]], probs[0]*100)   #熱力圖上的文字
cv2.putText(result, text, (20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(123, 222, 238), thickness=1, lineType=cv2.LINE_AA)

image_name_ = 'output'
cv2.imwrite(CAM_RESULT_PATH + image_name_ + '_' + 'pred_' + class_[idx[0]] + '.jpg', result)  #写入存储磁盘

"""
#----------------多張熱力圖繪圖-----------------
imgpath = './dataset/detect_imgs/'  #detect圖片所在位置
CAM_RESULT_PATH = './dataset/cam_pic/detect/cam_img/'   #熱力圖儲存資料夾路徑
CAM_FALSE_PATH = './dataset/cam_pic/detect/cam_withorin_false/'  #分類錯誤的熱力圖與原圖儲存路徑
CAM_RIGHT_PATH = './dataset/cam_pic/detect/cam_withorin_right/'  #分類正確的熱力圖與原圖儲存路徑

if not os.path.exists(CAM_RESULT_PATH):
    os.mkdir(CAM_RESULT_PATH)
if not os.path.exists(CAM_FALSE_PATH):
    os.mkdir(CAM_FALSE_PATH)
if not os.path.exists(CAM_RIGHT_PATH):
    os.mkdir(CAM_RIGHT_PATH)

# 獲得熱力圖的函數
def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape        #1,2048,56,56
    output_cam = []
    for idx in class_idx:
        feature_conv = feature_conv.reshape((nc, h*w))  # [2048,56*56]
        cam = torch.mm(weight_softmax[idx].unsqueeze(0),feature_conv)  #(1, 2048) * (2048, 56*56) -> (1, 56*56)
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = cam_img.detach().cpu().numpy()
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)

        output_cam.append(cam_img)
    return output_cam

for j,img in enumerate(os.listdir(imgpath)):
    bg = Image.new('RGB',(448,224), "#000000")
    print(j)
    images = Image.open(imgpath+img).convert('RGB')
    img_tensor = (data_transform['val'](images)/255.0).unsqueeze(0).to(device) #[1,3,224,224]
    # 提取最后一層特徵圖
    features = model_features(img_tensor).detach()  #[1,2048,56,56]
    # 獲得網路輸出
    logit = model_ft(img_tensor) 
    h_x = torch.nn.functional.softmax(logit[2], dim=1).data.squeeze()  #tensor([0.9981, 0.0019])
    probs, idx = h_x.sort(0, True)      #按概率從大到小排列(信心度,索引值)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
        
    CAMs = returnCAM(features, fc_weights, idx)  #得到所有預測類別的熱力圖
    print(img + ' output for the top1 prediction: %s' % class_[idx[0]])
    
    # 讀取圖片    
    imgs = cv2.imread(imgpath+img)
    img_name = img.split('@')
    height, width, _ = imgs.shape
    
    # 獲取正確類別的索引值
    t_key = 0
    for i in range(14):
        if img_name[0] == class_[idx[i]]:
            t_key = i

    #heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)  #CAM上顯示信心度最大值的class
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (height, width)), cv2.COLORMAP_JET)  #CAM上顯示True class
    
    result = heatmap * 0.5 + imgs * 0.5     #比例可以自己調節

    #text = '%s %.2f%%' % (class_[idx[0]], probs[0]*100) 	#信心度最大值的class
    text = '%s %.2f%%' % (class_[idx[0]], probs[0]*100) 	#True Class
    cv2.putText(result, text, (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(123, 222, 238), thickness=1, lineType=cv2.LINE_AA)

    cv2.imwrite(CAM_RESULT_PATH + img[:-4] + '_campred.jpg', result)
    
    #----------------熱力圖與原圖-----------------
    bg = Image.new('RGB',(width*2,height), "#000000") #創建畫布(原圖與熱力圖)
    if img_name[0] != class_[idx[0]]:
        imgs = Image.open(imgpath+img)
        bg.paste(imgs,(0,0))
        im = Image.open(CAM_RESULT_PATH + img[:-4] + '_campred.jpg')
        bg.paste(im,(width,0))
        bg.save(CAM_FALSE_PATH + img[:-4]+'_cam.jpg')
    else:
        imgs = Image.open(imgpath+img)
        bg.paste(imgs,(0,0))
        im = Image.open(CAM_RESULT_PATH + img[:-4] + '_campred.jpg')
        bg.paste(im,(width,0))
        bg.save(CAM_RIGHT_PATH + img[:-4]+'_cam.jpg')
"""