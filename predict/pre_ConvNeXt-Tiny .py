import torch
import torch.nn as nn  # 添加这一行
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import random
import time
from PIL import Image
from matplotlib import pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据处理函数
def img_transform(img_rgb, transform=None):
    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")
    img_t = transform(img_rgb)
    return img_t

# 模型加载函数
def get_model(path_state_dict, num_classes, vis_model=False):
    # 使用 weights 代替 pretrained
    model = models.convnext_tiny(weights="IMAGENET1K_V1")  # 使用预训练权重

    # 修改全连接层
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 100)  # 修改最后一层
    model.classifier.append(nn.ReLU())  # 添加 ReLU 激活函数
    model.classifier.append(nn.Dropout(p=0.5))  # 添加 Dropout 层
    model.classifier.append(nn.Linear(100, 2))  # 输出层为2个类别

    # 加载训练好的模型权重
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict, strict=False)  # 使用strict=False来忽略不匹配的层
    model.eval()  # 设置为评估模式
    model.to(device)
    return model

# 图片处理函数
def process_img(path_img):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    img_rgb = Image.open(path_img).convert('RGB')  # 打开图片并转为RGB格式
    img_tensor = img_transform(img_rgb, inference_transform)
    img_tensor.unsqueeze_(0)  # 添加batch维度
    img_tensor = img_tensor.to(device)
    return img_tensor, img_rgb

# 随机选择图片函数
def random_image_from_folder(folder_path):
    all_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    random_image = random.choice(all_images)
    return os.path.join(folder_path, random_image)

# 主函数
if __name__ == "__main__":
    num_classes = 2
    path_state_dict = "/home/ly/code/opencv/cat_dog/mode/convnext_tiny.pth"  # 模型权重路径
    test_folder = "/home/ly/code/opencv/cat_dog/data/test"  # 测试数据路径

    # 1/5 随机选择一张图片
    path_img = random_image_from_folder(test_folder)
    print(f"预测图片: {path_img}")

    # 2/5 图片处理
    img_tensor, img_rgb = process_img(path_img)

    # 3/5 加载模型
    model = get_model(path_state_dict, num_classes)

    # 模型预测
    with torch.no_grad():
        time_tic = time.time()
        outputs = model(img_tensor)  # 获取模型输出
        time_toc = time.time()

    # 4/5 获取预测结果
    _, pred_int = torch.max(outputs.data, 1)
    pred_idx = pred_int.cpu().numpy().item()  # 使用 .item() 提取标量
    pred_str = "cat" if pred_idx == 0 else "dog"

    print(f"img: {os.path.basename(path_img)} is: {pred_str}")
    print("time consuming: {:.2f}s".format(time_toc - time_tic))

    # 5/5 可视化结果
    plt.imshow(img_rgb)
    plt.title(f"predict: {pred_str}")
    plt.text(5, 45, f"Top 1: {pred_str}", bbox=dict(fc='yellow'))
    plt.show()
