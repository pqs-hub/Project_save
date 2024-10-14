import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm  # 导入tqdm库用于进度条
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 1. 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.Grayscale(num_output_channels=3),  # 将灰度图转为3通道RGB图像
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                         std=[0.229, 0.224, 0.225]),
])

# 2. 使用预训练的 ResNet18 模型，去除最后的全连接层
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # 去掉最后的分类层
model.to(device)  # 将模型转移到GPU
model.eval()  # 设置为评估模式

# 3. 定义提取特征的函数
def extract_features(image_paths):
    features_list = []
    
    for image_path in tqdm(image_paths, desc="Extracting Features"):  # 添加进度条
        # 读取图像并转换为RGB格式
        img = Image.open(image_path).convert('L')  # 读取灰度图
        img = preprocess(img).unsqueeze(0).to(device)  # 图像预处理并转移到GPU
        
        # 提取特征
        with torch.no_grad():
            features = model(img)
        
        # 将特征展平为一维向量
        features = features.view(features.size(0), -1).cpu().numpy()
        features_list.append(features)
    
    # 将所有图像的特征堆叠成一个矩阵
    return np.vstack(features_list)

# 4. 批量处理图像
def load_images_from_folder(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

# 5. 批量处理并提取特征
folder_path = r"E:\博士生涯\oas_read\ICCAD16-N7M2EUV\marked_testcase3_scaled"  # 假设图像存放在这个文件夹
image_paths = load_images_from_folder(folder_path)
features = extract_features(image_paths)

# 6. 使用 KMeans 进行聚类
n_clusters = 30  # 假设想分成30类，可以根据需要调整
kmeans = KMeans(n_clusters=n_clusters)

# 监控聚类过程
print("Clustering...")
kmeans.fit(features)

# 7. 输出聚类结果
for i, image_path in enumerate(image_paths):
    print(f"Image {image_path} is assigned to cluster {kmeans.labels_[i]}")

# 8. 绘制每个簇的成员图像
def plot_cluster_images(image_paths, labels, n_clusters, images_per_cluster=5, images_per_batch=5):
    # 遍历每个簇
    for cluster in range(n_clusters):
        # 找到属于当前簇的图像路径
        cluster_images = [image_paths[i] for i in range(len(labels)) if labels[i] == cluster]
        
        # 计算当前簇的图像数量
        n_images = min(images_per_cluster, len(cluster_images))  # 确保不会超出图像数量
        n_cols = 5  # 每行展示5个图像
        n_rows = (n_images + n_cols - 1) // n_cols  # 计算行数
        
        # 创建一个画布
        plt.figure(figsize=(15, 15))
        
        # 创建子图
        for i in range(n_images):
            # 显示当前批次信息
            if i % images_per_batch == 0:
                print(f"Displaying Cluster {cluster}, Batch {(i // images_per_batch) + 1}")
            
            plt.subplot(n_rows, n_cols, (i % images_per_batch) + 1)  # 更新为使用 n_rows
            img = Image.open(cluster_images[i]).convert('RGB')  # 读取图像
            plt.imshow(img)
            plt.axis('off')  # 不显示坐标轴
            plt.title(f'Cluster {cluster}')

            # 如果达到每批次显示的图像数量，暂停并显示
            if (i + 1) % images_per_batch == 0 or (i + 1) == n_images:
                plt.tight_layout()
                plt.show()
                input("Press Enter to continue...")  # 暂停，等待用户输入
                plt.figure(figsize=(15, 15))  # 重新创建画布为下一个批次准备

        # 处理完当前簇后，清除当前图形
        plt.clf()  # 清除当前图形以便下一个簇

# 调用绘制函数
plot_cluster_images(image_paths, kmeans.labels_, n_clusters)

# 9. 保存聚类结果
def save_clustered_images(image_paths, labels, output_dir):
    # 创建主目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每个簇
    for cluster in range(n_clusters):
        # 创建簇索引目录
        cluster_dir = os.path.join(output_dir, f'Cluster_{cluster}')
        os.makedirs(cluster_dir, exist_ok=True)
        
        # 找到属于当前簇的图像路径
        cluster_images = [image_paths[i] for i in range(len(labels)) if labels[i] == cluster]
        
        # 保存图像到相应的簇目录
        for img_path in cluster_images:
            img_name = os.path.basename(img_path)  # 获取文件名
            img_save_path = os.path.join(cluster_dir, img_name)  # 生成保存路径
            img = Image.open(img_path)
            img.save(img_save_path)

# 调用保存函数
output_directory = r"E:\博士生涯\聚类结果"  # 设置输出目录
save_clustered_images(image_paths, kmeans.labels_, output_directory)
