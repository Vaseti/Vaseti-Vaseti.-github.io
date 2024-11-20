import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image

# 自定义数据集类
class TimeLapseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        初始化数据集
        :param image_paths: 图像路径列表
        :param labels: 标签列表
        :param transform: 图像预处理
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # 返回数据集大小
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        根据索引返回图像和对应标签
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")  # 打开图像并转换为RGB格式
        label = self.labels[idx]  # 获取标签
        if self.transform:
            image = self.transform(image)  # 应用预处理
        return image, label

# 模型定义
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=4):
        """
        初始化基线卷积神经网络（Baseline CNN）
        :param num_classes: 分类数量（默认为4）
        """
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            # 卷积层1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积层2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积层3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积层4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平特征图
            nn.Linear(256 * 7 * 7, 512),  # 全连接层1
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout防止过拟合
            nn.Linear(512, num_classes),  # 全连接层2，输出4类
            nn.Softmax(dim=1),  # Softmax将输出归一化为概率
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        """
        x = self.features(x)  # 提取特征
        x = self.classifier(x)  # 分类
        return x

# 模型训练函数
def train_model(model, dataloader, criterion, optimizer, device):
    """
    训练模型
    :param model: 待训练的模型
    :param dataloader: 数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 使用的设备（CPU/GPU）
    """
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)  # 将数据移到指定设备
        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        total_loss += loss.item()
    return total_loss / len(dataloader)  # 返回平均损失

# 模型验证函数
def validate_model(model, dataloader, device):
    """
    验证模型性能
    :param model: 待验证的模型
    :param dataloader: 数据加载器
    :param device: 使用的设备（CPU/GPU）
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():  # 禁用梯度计算
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 前向传播
            _, preds = torch.max(outputs, 1)  # 获取预测类别
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)  # 返回准确率

# 主程序
def main():
    """
    主函数：训练和验证基线模型
    """
    # 数据准备
    image_paths = [...]  # 图像路径列表（需要自行填写）
    labels = [...]  # 对应的标签列表（需要自行填写）
    
    # 数据集划分
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    # 数据加载
    train_dataset = TimeLapseDataset(train_paths, train_labels, transform)
    val_dataset = TimeLapseDataset(val_paths, val_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型、损失函数和优化器初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和验证
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_accuracy = validate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "baseline_cnn.pth")
    print("训练完成，模型已保存为 'baseline_cnn.pth'")

if __name__ == "__main__":
    main()
