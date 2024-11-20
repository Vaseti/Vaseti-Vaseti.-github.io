import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 定义一个简单的二分类 CNN 模型
class BinaryCNN(nn.Module):
    """
    BinaryCNN 模型用于实现二分类任务，分为特征提取和分类两部分：
    - 特征提取：通过卷积层提取图像中的特征信息。
    - 分类：将提取到的特征输入全连接层进行二分类。
    """
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            # 第一层卷积：输入为RGB图片，输出32个特征图，卷积核大小为3x3
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # 激活函数，增加模型的非线性表达能力
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化，减小特征图尺寸
            # 第二层卷积：输入32个特征图，输出64个特征图
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平特征图为一维向量
            nn.Linear(64 * 8 * 8, 128),  # 全连接层，输入维度为 64 * 8 * 8，输出128个神经元
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout 防止过拟合
            nn.Linear(128, 2),  # 输出层，二分类任务，输出为两个类别的概率
            nn.Softmax(dim=1)  # Softmax 激活，转换为概率分布
        )

    def forward(self, x):
        # 前向传播，输入 x，返回分类结果
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# 定义自定义数据集类
class PlantDataset(Dataset):
    """
    PlantDataset 用于加载延时摄影数据集，提供图像及对应的分类标签。
    - images: 图像数据，形状为 (N, C, H, W)。
    - labels: 标签数据，二分类任务中标签为 0 或 1。
    """
    def __init__(self, images, labels):
        self.images = images  # 图像数据
        self.labels = labels  # 对应的分类标签

    def __len__(self):
        return len(self.images)  # 数据集大小

    def __getitem__(self, idx):
        # 根据索引返回图像和标签
        return self.images[idx], self.labels[idx]

# 定义模型的训练函数
def train_model(model, train_loader, epochs=10, lr=0.001):
    """
    训练指定的模型，使用交叉熵损失函数和 Adam 优化器。
    - model: 要训练的 CNN 模型
    - train_loader: 数据加载器，提供训练数据
    - epochs: 训练轮数
    - lr: 学习率
    """
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam 优化器

    for epoch in range(epochs):
        model.train()  # 切换到训练模式
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移到设备（CPU/GPU）
            optimizer.zero_grad()  # 清除梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累计损失
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# 定义模型的推理函数
def infer_model(model, image):
    """
    使用训练好的模型对单张图像进行推理。
    - model: 训练好的 CNN 模型
    - image: 输入图像
    返回值：预测的类别
    """
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        image = image.to(device)
        output = model(image.unsqueeze(0))  # 添加 batch 维度
        predicted = torch.argmax(output, dim=1)  # 获取概率最大的类别
    return predicted.item()

# 主程序入口
if __name__ == "__main__":
    # 检查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据准备：加载 M1, M2 和 M3 的训练数据及标签
    # 注意：用户需要根据实际数据替换以下占位符
    train_images_m1, train_labels_m1 = ..., ...
    train_images_m2, train_labels_m2 = ..., ...
    train_images_m3, train_labels_m3 = ..., ...

    # 数据加载器
    train_loader_m1 = DataLoader(PlantDataset(train_images_m1, train_labels_m1), batch_size=32, shuffle=True)
    train_loader_m2 = DataLoader(PlantDataset(train_images_m2, train_labels_m2), batch_size=32, shuffle=True)
    train_loader_m3 = DataLoader(PlantDataset(train_images_m3, train_labels_m3), batch_size=32, shuffle=True)

    # 初始化三个阶段的模型
    model_m1 = BinaryCNN().to(device)  # 检测 Soil 和 FA
    model_m2 = BinaryCNN().to(device)  # 检测 FA 和 OC
    model_m3 = BinaryCNN().to(device)  # 检测 OC 和 FL

    # 训练每个阶段的模型
    print("Training M1...")
    train_model(model_m1, train_loader_m1, epochs=10)
    print("Training M2...")
    train_model(model_m2, train_loader_m2, epochs=10)
    print("Training M3...")
    train_model(model_m3, train_loader_m3, epochs=10)

    # 模拟延时序列的推理流程
    print("Inferring stages...")
    time_lapse_sequence = [...]  # 延时摄影序列中的图片，需替换为实际数据
    current_model = model_m1  # 初始模型为 M1
    for img in time_lapse_sequence:
        stage = infer_model(current_model, img)  # 当前阶段的推理
        if stage == 1 and current_model == model_m1:
            current_model = model_m2  # 检测到 FA，切换到 M2
        elif stage == 1 and current_model == model_m2:
            current_model = model_m3  # 检测到 OC，切换到 M3
        elif stage == 1 and current_model == model_m3:
            print("First FL detected. Analysis complete.")  # 检测到 FL，结束流程
            break
