import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理和增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),   # 随机裁剪为224x224
    transforms.RandomRotation(20),       # 随机旋转20度
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
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

# 使用 ConvNeXt-Tiny
model = models.convnext_tiny(pretrained=True)  # 加载预训练的 ConvNeXt-Tiny 模型

# 冻结 ConvNeXt 的卷积层，只训练全连接层
for param in model.parameters():
    param.requires_grad = False

# 修改全连接层，输出类别为2
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 100)  # 修改最后一层
model.classifier.append(nn.ReLU())  # 添加 ReLU 激活函数
model.classifier.append(nn.Dropout(p=0.5))  # 添加 Dropout 层
model.classifier.append(nn.Linear(100, 2))  # 输出层为2个类别

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
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据发送到GPU
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

    return test_acc

# 保存最佳模型
best_acc = 0.0
best_model_wts = model.state_dict()

# 训练20个epoch
for epoch in range(0, 20):
    print(f'epoch: {epoch}')
    train()
    
    # 在每个epoch后进行测试
    test_acc = test()
    
    # 如果当前epoch的测试集准确率更高，则更新最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        best_model_wts = model.state_dict()
        print(f"更新最佳模型：Test acc: {test_acc:.2f}")
        
# 将最佳模型的权重加载到模型中
model.load_state_dict(best_model_wts)

# 保存最佳模型
torch.save(model.state_dict(), '/home/ly/code/opencv/cat_dog/mode/convnext_tiny_best.pth')
print("~结束训练")
