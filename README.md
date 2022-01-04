# README

## 文件说明

项目名：Cats_VS_Dogs

文件结构如下：

```
├─data··············数据集，内部均为jpg图片
│  ├─test···········测试集
│  ├─train··········训练集
│  │  ├─cats
│  │  └─dogs
│  └─validation·····验证集
│      ├─cats
│      └─dogs
├─model·············模型存储位置
├─sample············示例图片
├─train.py··········训练并保存模型
└─test.py···········测试模型
```

数据集来自 [kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) ，内含 train.zip 和 test.zip 两个文件，train.zip 解压后可以看到是25000张 jpg 图片，其中猫和狗的图片各占一半，按照 cat/dog.序号.jpg 的格式命名。test.zip 解压后可以看到是乱序混排的12500张猫狗照片，按照 序号.jpg 格式命名。分别选取猫和狗的前2000张作为训练集，再各1000张作为验证集，另取test的500张作为测试集。

由于使用的 [VGG16 模型](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)，所以可将下载好的模型文件置于 C:\Users\your_username\\.keras\models 下。

如果网络状况不佳，也可以从网盘下载：链接: https://pan.baidu.com/s/195FM3YgdrID-bJiSHalDhQ?pwd=s454

（进行了 epoch = 50 的预训练文件也包括在内，置于 model 文件夹下即可直接运行 test.py）

## 环境配置

此处仅列出我完成项目时的环境配置：

```
cuda==10.1
cudnn==7.6.5
python==3.7
tensorflow-gpu==2.2.0
scipy==1.4.1
numpy==1.18.4
matplotlib==3.2.1
opencv_python==4.2.0.34
tqdm==4.46.1
Pillow==8.2.0
h5py==2.10.0
```

## 训练&测试

**训练：**

按照文件说明调整好文件结构，并配置好环境之后，运行 train.py，即可开始训练，训练完成后，模型保存在 ./model 路径下。

**测试：**

在已有模型的情况下，进行测试只需在 test.py 中给出测试图片的路径即可。



