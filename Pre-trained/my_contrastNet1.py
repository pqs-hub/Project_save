import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader, random_split
from Geometric_Similarity import ACC_value, generate_transformations
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
from PIL import Image
import scipy.fftpack  # Importing DCT from scipy
import os
from oas2class_nomarkers import clip_generate
# 定义损失函数，使特征向量距离与ACC值相近
class ACCSimilarityLoss(nn.Module):
    def __init__(self):
        super(ACCSimilarityLoss, self).__init__()
    
    def forward(self, output1, output2, acc_value):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(torch.pow(euclidean_distance - acc_value, 2))
        return loss

# 自定义Dataset类，用于加载Marker图像数据并应用DCT转换
class MarkerDataset(Dataset):
    def __init__(self, markers, transform=None):
        self.markers = markers  # 传入的Marker对象列表
        self.transform = transform or transforms.ToTensor()
    
    def __len__(self):
        return len(self.markers)
    
    def __getitem__(self, idx):
        marker1 = self.markers[idx]
        marker2 = random.choice(self.markers)

        # 将Marker转换为图像
        img1 = marker1.to_grayscale_image((50, 50))  # 灰度图
        img2 = marker2.to_grayscale_image((50, 50))

        # 生成marker1的变换集
        transformations, _ = generate_transformations(marker1)
        
        # 计算ACC相似度作为标签
        best_acc_value = np.inf
        for marker in transformations:
            acc_value = ACC_value(marker, marker2, marker.width)
            if acc_value < best_acc_value:
                best_acc_value = acc_value
        
        # 将DCT图像转换为张量
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, best_acc_value  # 返回ACC值作为标签


# 定义特征提取网络（例如ResNet或自定义卷积神经网络）
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 50 * 50, 512), 
            nn.ReLU(),
            nn.Linear(512, 128) # 提取特征向量
        )
    
    def forward(self, x):
        return self.features(x)

class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoints/', verbose=False):
        """
        Args:
            patience (int): 当验证损失不再改善时等待的epoch数
            delta (float): 改善的最小变化
            path (str): 保存最佳模型的路径
            verbose (bool): 是否显示改善信息
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

        # 检查并创建保存目录
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存验证损失降低的模型，同时记录损失值'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')

        # 构造带有验证损失值的文件名
        filename = f"checkpoint_val_loss_{val_loss:.6f}.pt"

        # 保存模型的state_dict以及当前的验证损失值
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }, os.path.join(self.path, filename))

        self.val_loss_min = val_loss



# 创建数据集和数据加载器
oas_path = r"/home/qspeng/Data/mkLanaiCPU.gds"
layer_indices = [(66, 20)]
clip_cells = clip_generate(oas_path, marker_width = 3000, marker_height = 3000, layer_indices=layer_indices, x_offset=2000, y_offset=2000, marker_extend=False)

# 创建数据集
marker_dataset = MarkerDataset(clip_cells, transform=transforms.ToTensor())
train_size = int(0.8 * len(marker_dataset))  # 80% 作为训练集
val_size = len(marker_dataset) - train_size  # 20% 作为验证集

# 划分数据集
train_dataset, val_dataset = random_split(marker_dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 定义网络和优化器
feature_extractor = FeatureExtractor().cuda()
criterion = ACCSimilarityLoss().cuda()
optimizer = optim.Adam(feature_extractor.parameters(), lr=0.001)

# 初始化早停机制
early_stopping = EarlyStopping(patience=10, verbose=True)

# 训练循环
for epoch in range(200):
    total_loss = 0
    feature_extractor.train()  # 切换到训练模式
    with tqdm(train_loader, unit="batch") as tepoch:
        for img1, img2, acc_value in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            
            img1 = img1.float().cuda()
            img2 = img2.float().cuda()
            acc_value = acc_value.float().cuda()

            # 前向传播
            output1 = feature_extractor(img1)
            output2 = feature_extractor(img2)

            # 计算损失
            loss = criterion(output1, output2, acc_value)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    # 验证阶段
    val_loss = 0.0
    feature_extractor.eval()  # 切换到评估模式
    with torch.no_grad():
        for img1, img2, acc_value in val_loader:
            img1 = img1.float().cuda()
            img2 = img2.float().cuda()
            acc_value = acc_value.float().cuda()

            output1 = feature_extractor(img1)
            output2 = feature_extractor(img2)
            val_loss += criterion(output1, output2, acc_value).item()

    val_loss /= len(val_loader)

    print(f'Epoch {epoch + 1}, Average Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss}')

    # 调用早停机制
    early_stopping(val_loss, feature_extractor)

    # 如果早停机制被触发，退出训练
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
