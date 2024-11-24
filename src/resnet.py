import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断GPU是否可用

# 数据预处理 数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),   # 随机裁剪为224*224
    transforms.RandomRotation(20),       # 随机旋转20度
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.ToTensor()
])

# 读取数据
root = 'Smalldata'  # 数据集目录
train_dataset = datasets.ImageFolder(root + '/train', transform)
test_dataset = datasets.ImageFolder(root + '/test', transform)

# 导入数据，批次大小为8，打乱数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)

# 获取类别名称和索引
classes = train_dataset.classes
classes_index = train_dataset.class_to_idx
print("类别名称", classes)
print("类别编号", classes_index)

# 使用 ResNet-18
model = models.resnet18(pretrained=True)  # 加载预训练的 ResNet-18

# 冻结 ResNet 的卷积层，只训练全连接层
for param in model.parameters():
    param.requires_grad = False

# 修改全连接层，输出类别为2
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 100),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(100, 2)
)

# 将模型转移到设备（GPU 或 CPU）
model = model.to(device)
print("使用GPU:", next(model.parameters()).device)

# 设置学习率
LR = 0.0001

# 定义损失函数和优化器
entropy_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), LR, momentum=0.9)

print("开始训练~")

# 训练函数
def train():
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据发送到GPU上
        out = model(inputs)  # 获得模型预测结果
        loss = entropy_loss(out, labels).to(device)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

# 测试函数
def test():
    model.eval()
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        _, predicted = torch.max(out, 1)  # 获取预测类别
        correct += (predicted == labels).sum()  # 计算预测正确的数量

    test_acc = correct.item() / len(test_dataset)
    print(f"Test acc: {test_acc:.2f}")
    print(f"Test loss: {1 - test_acc:.2f}")  # 损失率+准确率为1

    correct = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum()

    train_acc = correct.item() / len(train_dataset)
    print(f"Train acc: {train_acc:.2f}")
    print(f"Train loss: {1 - train_acc:.2f}")

# 训练20个epoch
for epoch in range(0, 20):
    print(f'epoch: {epoch}')
    train()
    test()

# 保存模型
torch.save(model.state_dict(), '/home/ly/code/opencv/cat_dog/mode/resnet.pth')
print("~结束训练")
