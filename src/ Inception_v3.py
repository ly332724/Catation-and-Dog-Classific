import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断GPU是否可用

# 数据预处理 数据增强
transform = transforms.Compose([
    # 对图像进行随机的裁剪crop以后再resize成固定大小（299*299）
    transforms.RandomResizedCrop(299),
    # 随机旋转20度（顺时针和逆时针）
    transforms.RandomRotation(20),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(p=0.5),
    # 将数据转换为tensor
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 读取数据
root = 'Smalldata'   # root是数据集目录
train_dataset = datasets.ImageFolder(root + '/train', transform)
test_dataset = datasets.ImageFolder(root + '/test', transform)

# 导入数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)

# 类别名称
classes = train_dataset.classes
# 类别编号
classes_index = train_dataset.class_to_idx
print("类别名称", classes)
print("类别编号", classes_index)

# 加载Inception v3模型
model = models.inception_v3(pretrained=True)

# 只训练最后的全连接层
for param in model.parameters():
    param.requires_grad = False

# 修改Inception v3的最后全连接层（Inception v3的fc层名为 'fc'）
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 100),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(100, 2)  # 2表示有2个类别（cat和dog）
)

# 将模型发送到GPU
model = model.to(device)
print("使用GPU:", next(model.parameters()).device)

# 定义学习率和优化器
LR = 0.0001
entropy_loss = nn.CrossEntropyLoss()   # 损失函数z
optimizer = optim.SGD(model.parameters(), LR, momentum=0.9)

# 训练和测试函数
def train():
    model.train()
    for i, data in enumerate(train_loader):
        # 获得数据和对应的标签
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据发送到GPU上

        # 获得模型预测结果
        out = model(inputs)

        # Inception v3 有两个输出：main_output 和 auxiliary_output
        # 这里我们只使用主输出（logits）
        main_output = out[0]  # out[0] 是主输出 logits

        # 交叉熵代价函数
        loss = entropy_loss(main_output, labels).to(device)  # 损失函数也需要发到GPU
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test():
    model.eval()
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 获得模型预测结果
        out = model(inputs)
        _, predicted = torch.max(out, 1)

        correct += (predicted == labels).sum()

    print("Test acc: {:.2f}".format(correct.item() / len(test_dataset)))
    print("Test loss:{:.2f}".format(1 - correct.item() / len(test_dataset)))

    correct = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        out = model(inputs)
        _, predicted = torch.max(out, 1)

        correct += (predicted == labels).sum()

    print("Train acc: {:.2f}".format(correct.item() / len(train_dataset)))
    print("Train loss:{:.2f}".format(1 - correct.item() / len(train_dataset)))

# 训练模型
for epoch in range(0, 20):
    print('epoch:', epoch)
    train()
    test()

# 保存训练好的模型
torch.save(model.state_dict(), '/home/ly/code/opencv/cat_dog/mode/inception.pth')
print("~结束训练")
