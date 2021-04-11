import os
import torch
from torchvision import  transforms ,utils,datasets
import  numpy as np


data_transform = transforms.Compose([
 transforms.Resize(200), # 缩放图片(Image)，保持长宽比不变，最短边为32像素
 transforms.CenterCrop(200), # 从图片中间切出32*32的图片
 transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
 transforms.Normalize(mean=[0.492, 0.461, 0.417], std=[0.256, 0.248, 0.251]) # 标准化至[-1, 1]，规定均值和标准差
])

cwd=os.getcwd()
train_path=os.path.join(cwd,"../data_set/flower_data/train")
val_path=os.path.join(cwd,"../data_set/flower_data/val")
train_dataset=datasets.ImageFolder(root=train_path,transform=data_transform)
img, label = train_dataset[0] #将启动魔法方法__getitem__(0)
print(label)
print(img.size)
import matplotlib.pyplot as plt
# plt.imshow(np.transpose(img,(1,2,0)))
# plt.show()
import torchvision
import matplotlib.pyplot as plt
import numpy as np

dataset_loader = torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True)


def imshow(img):
 img = img / 2 + 0.5  # unnormalize
 npimg = img.numpy()
 plt.imshow(np.transpose(npimg, (1, 2, 0)))
 plt.show()
# 随机获取部分训练数据
dataiter = iter(dataset_loader)#此处填写加载的数据集
images, labels = dataiter.next()
# 显示图像
imshow(torchvision.utils.make_grid(images))
# 打印标签
print(''.join(str(labels[j].item()) for j in range(4)))