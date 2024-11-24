import os
import random
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models

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
    model = models.vgg16(num_classes=num_classes)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 100),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(100, 2)
    )

    # 加载预训练权重
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)
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
    path_state_dict = "/home/ly/code/opencv/cat_dog/mode/vgg.pth"  # 模型权重路径
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
    pred_idx = int(pred_int.cpu().numpy())
    pred_str = "cat" if pred_idx == 0 else "dog"

    print(f"img: {os.path.basename(path_img)} is: {pred_str}")
    print("time consuming: {:.2f}s".format(time_toc - time_tic))

    # 5/5 可视化结果
    plt.imshow(img_rgb)
    plt.title(f"predict: {pred_str}")
    plt.text(5, 45, f"Top 1: {pred_str}", bbox=dict(fc='yellow'))
    plt.show()
