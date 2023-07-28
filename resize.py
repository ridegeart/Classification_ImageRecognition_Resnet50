import cv2
import os 
import sys
#curPath=os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.getcwd())
image_size = 224

Traintype =True

if Traintype:
    image_root ='C:/Users/CamyTang/FMADefect/ResNet50/dataset/images/'
    root = image_root
else:
    detect_root ='C:/Users/CamyTang/FMADefect/ResNet50/dataset/detect_imgs/'
    root = detect_root


os.chdir(root)
for imgname in os.listdir(root):
    image = cv2.imread(imgname)
    imageresize = cv2.resize(image, (image_size,image_size))
    cv2.imwrite(imgname,imageresize)
    print(imgname)