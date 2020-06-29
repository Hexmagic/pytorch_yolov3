
最小化实现yolov3,参考 [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

## 安装
### 克隆文件
    $ git clone 
    $ cd pytorch_yolov3
    # pip3 install -r requirements.txt

### 下载数据集

    wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
    wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    tar xf VOCtrainval_11-May-2012.tar
    tar xf VOCtrainval_06-Nov-2007.tar
    tar xf VOCtest_06-Nov-2007.tar

得到VOCdevkit文件夹，重命名为datasets,如下

    D:\PYTORCH_YOLOV3\DATASETS
    ├─VOC2007
    │  ├─Annotations
    │  ├─ImageSets
    │  │  ├─Layout
    │  │  ├─Main
    │  │  └─Segmentation
    │  ├─JPEGImages
    │  ├─labels
    │  ├─SegmentationClass
    │  └─SegmentationObject
    └─VOC2012
        ├─Annotations
        ├─ImageSets
        │  ├─Action
        │  ├─Layout
        │  ├─Main
        │  └─Segmentation
        ├─JPEGImages
        ├─labels
        ├─SegmentationClass
        └─SegmentationObject

## 下载预训练权重
百度云：

    链接：https://pan.baidu.com/s/1s5_gV2YaVvT4u-ZgL2vgjw 
    提取码：1ik1 
    复制这段内容后打开百度网盘手机App，操作更方便哦   


## 推断
默认使用测试集

    $ python3 detect.py 

<p align="center"><img src="assets/2008_006730.png" width="416"></p>
<p align="center"><img src="assets/2008_006733.png" width="480"\></p>
<p align="center"><img src="assets/2008_006807.png" width="480"\></p>


## 训练
下载darknet权重

    wget https://pjreddie.com/media/files/darknet53.conv.74

开始训练

    $ python train.py --iter 120000 --batch_size 18 --pretrained_weights weigths/darknet53.conv.74 --save_folder weights



