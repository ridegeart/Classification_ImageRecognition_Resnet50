import os
import sys
#curPath=os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import cv2 
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model.ResNet import Bottleneck, ResNet, ResNet50,ResNet101
from load_dataset import LoadDataset
from helper import read_meta
from urllib.request import urlopen
csv_save_name = 'detect0727_ViT_pretrained.csv'
model_save_path = './dataset/'
def makedirs(path):
    try:
        os.makedirs(path)
    except:
        return

if __name__ == "__main__":
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        '''mdoelsave.pth location'''
        os.chdir(model_save_path)
        modelName = 'FMA_finViT_pretrained.pth'

        ''' Mode read'''
        model = ResNet50(14).to(device)
        model.load_state_dict(torch.load(modelName),False) #RuntimeError:Error(s) in loading state_dict for DataParallel 訓練與測試環境不同
        model.eval()

        ''' predict csv'''
        datacsv ='detect.csv' #args.test_csv現在路徑在args.model_save_path

        '''data - loader'''
        batch_size=1
        epoch = 1
        datadic={}
        metafile = 'C:/Users/CamyTang/FMADefect/ResNet50/dataset/pickle_files/meta'
        coarse_labels,fine_labels,third_labels = read_meta(metafile)
        #detect_csv_path = 'detect.csv'

        detect_dataset = LoadDataset(csv_path=datacsv, transform=transforms.ToTensor())
        detect_generator = DataLoader(detect_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        dfsave = pd.DataFrame()

        r=0
        for e in range(epoch):
                for j, sample in tqdm(enumerate(detect_generator)): #detect_generator
                        print("-----------Number-----:"+str(j))
                        #----for test.csv testing----#
                        batch_x ,batch_y1,imgpath= sample['image'].to(device), sample['label_1'].to(device), sample['image_path']
                        print(imgpath,batch_y1)

        #                print(imgpath)
                        ''' Tensor balue'''
                        subtwoclass_pred= model(batch_x) 

                        ''' confidence  & classes'''
                        ''' - subtwoclasses'''
                        probs_subtwo = torch.nn.functional.softmax(subtwoclass_pred, dim=1)
                        subtwo_value,subtwo_index=torch.topk(probs_subtwo,k=5,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                        conftwo_sub,classestwo_sub = torch.max(probs_subtwo,1)
                        imgclasstwo_sub= third_labels[(classestwo_sub.item())]
                        print('subclass',conftwo_sub,classestwo_sub.item(),imgclasstwo_sub)
                        ''' Get into datadic '''
                        output_dic = {
                                'subtwo_conf':[str(index)[:6] for index in subtwo_value[0].tolist()],
                                'subtwo_class':[third_labels[index] for index in subtwo_index[0].tolist()],
                                'ans':imgclasstwo_sub,
                                'conf':str(conftwo_sub[0].tolist())[:6],
                                'True':third_labels[(batch_y1.item())],
                        }
                        ''' dataframe concat'''
                        datadic[imgpath[0]] = output_dic
                        df = pd.DataFrame(datadic)
                        df = df.T

                        if  len(dfsave) == 0 :
                                dfsave = df 
                        else :
                                dfsave = pd.concat([df,dfsave],axis=0)

        '''datasave cleaner'''
        index_duplicates = dfsave.index.duplicated()
        dfsave = dfsave.loc[~index_duplicates]
        #dfsave.reset_index(drop=True,inplace=True)
        
        #makedirs(model_save_path+'result/')
        dfsave.to_csv('./result/'+csv_save_name,index=True,index_label='ImagePath')
        print('data_save:'+'./result/'+csv_save_name)

