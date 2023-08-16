'''Pytorch dataset loading script.
'''

import os
import sys
sys.path.append(os.getcwd())
import pickle
import csv
import cv2
from PIL import Image
from torch.utils.data import Dataset
from helper import read_meta
import torch
from torch.utils.data.dataloader import default_collate
import torchvision

class LoadDataset(Dataset):
    '''Reads the given csv file and loads the data.
    '''


    def __init__(self, csv_path, image_size=32, image_depth=3, return_label=True, transform=None):
        '''Init param.
        '''

        assert os.path.exists(csv_path), 'The given csv path must be valid!'

        self.csv_path = csv_path
        self.image_size = image_size
        self.image_depth = image_depth
        self.return_label = return_label
        self.meta_filename = './dataset/pickle_files/meta'
        self.transform = transform
        self.data_list = self.csv_to_list()
        self.coarse_labels, self.fine_labels ,self.third_labels= read_meta(self.meta_filename)
        self.image_name_list = self.data_to_imagename()


    def csv_to_list(self):
        '''Reads the path of the file and its corresponding label 
        '''

        with open(self.csv_path, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        return data

    def data_to_imagename(self):
        '''Reads csv_to_list , data for imagename'''
        imagelist=[]
        for i in self.data_list:
            imgsname = i[0]
            imagelist.append(imgsname)
        return imagelist

    def __len__(self):
        '''Returns the total amount of data.
        '''
        return len(self.data_list)

    def __getitem__(self, idx):
        '''Returns a single item.
        '''
        image_path, image, superclass = None, None, None #add subtwoclass
        if self.return_label:
            image_path, superclass= self.data_list[idx] #add subtwoclass
        else:
            image_path = self.data_list[idx]

        if self.image_depth == 1:
            image = cv2.imread(image_path, 0)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #if self.image_size != 32:
         #   cv2.resize(image, (self.image_size, self.image_size))


        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.return_label:#add 
            return {
                'image':image/255.0,
                'label_1': self.third_labels.index(superclass.strip(' ')),
                'image_path':image_path
            }
        else:
            return {
                'image':image/255.0,
            }

