https://github.com/ridegeart/Classification_ImageRecognition_Resnet50
# Classification
Implementation of ResNet 50, 101, 152 in PyTorch based on paper [Deep Residual Learning for Image Recognition] by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. 
The github form [Github JayPatwardhan]

# FMA_Defect

## setting
- train.py：路徑設定，os.chdir(current) ，更改 current 為自己的路徑。

## Training 
- 使用 train.py 進行訓練。

## Cutsomer Data Training 
1. dataPickle_Transform.py : 建立meta/train/test 資料集資訊。
    - rootPic：訓練照片所在路徑。
    - projectroot：train.py 所在路徑。
    - [資料夾說明]
        1. dataPickle_Transform/preimages/train , dataPickle_Transform/preimage/test 的照片移到 dataset/images。
        2. dataPickle_Transform/picklefiles/meta,train,test 移到dataset/pickle_files。
2. process_dataset.py : 建立只有 FMA Defect 類別的 train.csv /  test.csv。
    - train.csv /  test.csv為 train:test = 9:1。
3. resize.py : 照片 resize 。
    - image_root：更改為照片所在路徑。
    - image_size = 224 #resnet50/Transformer
    - image_size = 160 #resnet101
    - image_size = 299 #inceptionV4
4. ./model/ResNet.py 為有加入 CBAM 之模型檔。
    - 為了接 pretrained weight 更改層命名 
        1. batch_norm1->bn1
        2. i_downsample->downsample
        3. self.relu = nn.ReLU() -> self.relu = nn.ReLU(inplace=True)
        4. max_pool->maxpool
5. load_dataset.py:
    - self.meta_filename 更改路徑
6. train.py：加入自定義訓練集。
    - 不能 import 自定義 model：加入 __init__ 檔，並在 train.py 中加入 sys.path.append(os.getcwd()) 後再 import。
    - 訓練要使用到的model：resnet50/inceptionV4/transformer。
    - DataLoader 的 batch_size：
        - batch_size：62 #resnet50/inceptionV4/resnet101
        - batch_size：42 #Transformer
    - num_workers：BatchSize 參考DataLoader 的 batch_size。
    - Normal Trained：使用 net 新增想用的訓練模型。
        - ResNet50/ResNet101(num_classes)：輸入預計輸出的類別數。
        - ResNet50(num_classes,use_cbam=False)：無CBAM的resnet50。
        - VisionTransformer()：呼叫參數參考'Vit_b_16'。
    - Pretrained Weight：
        - 先建造模型，更改模型dict後，再load model to device。
    - 加入 checkpoint 機制，儲存測試準確率與訓練準確率最高的模型。
    - 加入 net.train()，不加的話訓練準確率不會上升。
    - 加入 net.eval() 固定 batchnorm 的參數，不隨 test 數據的輸入對參數做平均 ( batchnorm )。
    - 加入 plot loss 與 acc 部分，並設定在訓練完成後繪圖；pd.DataFrame 的 輸入型態改為 list。
    - lineplot 中的 reset_index(inplace=False) -> reset_index()。

## Data Inference
1. dataPickle_detect.py : 資料丟入之前先轉換成detect(pickle)，照片名稱預先處理。
    - save detect pickle 轉換在：dataPickle_Transform/pickle_files/detect
    - save detect image 轉換在： dataPickle_Transform/preimages/detect;
2. process_detect.py:需先將必要的資料複製於對應資料夾中，會產生出一個detect.csv。
    - detect pickle 複製:(dataPickle_Transform/pickle_files/) 複製貼到 (data/pickle_files/)。
    - image 複製:(dataPickle_Transform/preimages/) 複製貼到 (data/detect_imgs/)。
3. resize.py : data/detect_imgs 照片記得 resize。
    - Traintype = False
    - image_size = 224 #resnet50/Transformer
    - image_size = 160 #resnet101
    - image_size = 299 #inceptionV4
4. detect.py :  Inference預測，結果./dataset/result/detect_predict.csv。
    - modelName更改為訓練好的權重路徑。
    - 加載模型報錯：RuntimeError:Error(s) in loading state_dict for DataParallel，訓練模型與測試模型環境不一致，加入 load_state_dict(torch.load(modelName)) -> load_state_dict(torch.load(modelName),False)

## .py檔案稍微修改
### Training-Part
1. load_dataset.py : 將輸出改為只有一個 label_1 ，並從 third_labels 取index 而來。
2. helper.py: 使用 read_meta 函式。

## HeatMap熱力圖 (3layer_Resnet50)  
1. cam_1pic.py：繪製單張熱力圖。
    - modelName：更改為已訓練好的權重檔的路徑。
    - img_path：欲輸入模型的圖片路徑。
    - CAM_RESULT_PATH：畫好的熱力圖的儲存路徑。
2. cam_mutiPic.py：繪製多張熱力圖。
    - modelName：更改為已訓練好的權重檔的路徑。
    - imgpath：欲輸入模型的資料夾路徑。
    - CAM_RESULT_PATH：畫好的熱力圖的儲存資料夾路徑。
    - CAM_FALSE_PATH：分類錯誤的熱力圖與原圖儲存路徑
    - CAM_RIGHT_PATH：分類正確的熱力圖與原圖儲存路徑
    - 熱力圖與原圖組合圖
        - 分類正確：生成的圖像 -> (原圖,預測類別)
        - 分類錯誤：生成的圖像 -> (原圖,預測類別,真實類別)
- 更改畫熱力圖的模型
    - model_features:讀取模型最後一層的輸出特徵圖。
    - fc_weights：獲得fc層的權重。

# Cifar 100

## Cutsomer Data Training
1. 解壓縮cifar-100-python.tar.gz至訓練目錄，Cifar 100 資料集下載處：
https://www.cs.toronto.edu/~kriz/cifar.html

## Training
1. cifar100_train.py：直接跑這個。
    - os.chdir(current) ，更改 current 為自己的路徑。
    - from cifar100 import CIFAR100 ： cifar100.py。
2. cifar100.py：cifar 的 data loader。

[Deep Residual Learning for Image Recognition]: https://arxiv.org/pdf/1512.03385.pdf "Deep Residual Learning for Image Recognition"
[Github JayPatwardhan]: https://github.com/JayPatwardhan/ResNet-PyTorch/tree/master "Github - JayPatwardhan"
