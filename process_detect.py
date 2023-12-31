

import os
import sys
#curPath=os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import pickle
import numpy as np
import pandas as pd
import imageio
import cv2
from tqdm import tqdm
from helper import unpickle, read_meta


class Preprocess_detect:
    '''Process the detedt pickle files.
        image_write_dir : dataset/detect_imgs
    '''
    def __init__(self, meta_filename='C:/Users/CamyTang/FMADefect/ResNet50/dataset/pickle_files/meta',detect_file='C:/Users/CamyTang/FMADefect/ResNet50/dataset/pickle_files/detect',
                        image_write_dir='C:/Users/CamyTang/FMADefect/ResNet50/dataset/detect_imgs/', csv_write_dir='C:/Users/CamyTang/FMADefect/ResNet50/dataset/', detect_csv_filename='detect.csv'):
        '''Init params.
        '''
        self.meta_filename = meta_filename
        self.detect_file = detect_file
        self.image_write_dir = image_write_dir
        self.csv_write_dir = csv_write_dir
        self.detect_csv_filename = detect_csv_filename

        if not os.path.exists(self.image_write_dir):
            os.makedirs(self.image_write_dir)

        if not os.path.exists(self.csv_write_dir):
            os.makedirs(self.csv_write_dir)
            
        #--原本的會有問題--#
        #self.coarse_label_names, self.fine_label_names = read_meta(meta_filename=self.meta_filename)
        self.coarse_label_names, self.fine_label_names ,self.third_label_names = read_meta(self.meta_filename)



    def process_data(self):
        '''Read the train/test data and write the image array and its corresponding label into the disk and a csv file respectively.
        '''

 
        pickle_file = unpickle(self.detect_file)

        filenames = pickle_file['filenames']#[t.decode('utf8') for t in pickle_file[b'filenames']]
        coarse_labels = pickle_file['coarse_labels']#pickle_file[b'coarse_labels']
        fine_labels = pickle_file['fine_labels']#pickle_file[b'fine_labels']
        fine_labels = pickle_file['fine_labels']#pickle_file[b'fine_labels']
        third_labels = pickle_file['trhid_labels']#pickle_file[b'fine_labels']
        
        #data = pickle_file[b'data']

        '''
        images = []
        for d in data:
            image = np.zeros((32,32,3), dtype=np.uint8)
            image[:,:,0] = np.reshape(d[:1024], (32,32))
            image[:,:,1] = np.reshape(d[1024:2048], (32,32))
            image[:,:,2] = np.reshape(d[2048:], (32,32))
            images.append(image)
        '''


        csv_filename = self.detect_csv_filename

        c=0
        with open(f'{self.csv_write_dir}/{csv_filename}', 'w+') as f:
            for i, image in enumerate(filenames):
                filename = filenames[i]
                third_label = self.third_label_names[third_labels[i]]

                #imageio.imsave(f'{self.image_write_dir}{filename}', image)
                c=c+1
                #print('count:'+str(c))
                print(f'{self.image_write_dir}{filename}')
                f.write(f'{self.image_write_dir}{filename},{third_label}\n')



p = Preprocess_detect()
##-----detect-----##
p.process_data() #process the testing set
print('detect download ok')


