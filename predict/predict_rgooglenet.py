import os
import random
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
import torchsummary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """
    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")
    img_t = transform(img_rgb)
    return img_t

def get_model(path_state_dict, num_classes, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict: 预训练模型的路径
    :param num_classes: 输出类别数
    :param vis_model: 是否打印模型结构
    :return: 加载了权重的模型
    """
    model = models.resnet18(pretrained=True)  # 使用ResNet-18
    # 冻结卷积层
    for param in model.parameters():
        param.requires_grad = False

    # 修改最后的全连接层（输出类别为2）
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 100),  # 输入维度是模型的fc层输入特征数
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(100, num_classes)  # 输出类别数为2
    )

    # 加载训练好的模型参数
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model

def process_img(path_img):
    """
    处理图像
    :param path_img: 图片路径
    :return: 预处理后的图像tensor，原图
    """
    # hard code
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # path --> img
    img_rgb = Image.open(path_img).convert('RGB')

    # img --> tensor
    img_tensor = img_transform(img_rgb, inference_transform)
    img_tensor.unsqueeze_(0)        # chw --> bchw
    img_tensor = img_tensor.to(device)

    return img_tensor, img_rgb

def random_image_from_folder(folder_path):
    """
    从指定文件夹中随机选择一张图片
    :param folder_path: 文件夹路径
    :return: 随机选择的图片路径
    """
    # 获取文件夹中的所有图片
    all_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    # 随机选择一张图片
    random_image = random.choice(all_images)
    return os.path.join(folder_path, random_image)

if __name__ == "__main__":
    num_classes = 2
    # 配置路径
    path_state_dict = os.path.join(BASE_DIR, "/home/ly/code/opencv/cat_dog/mode/resnet.pth")
    test_folder = os.path.join(BASE_DIR, "data/test")

    # 1/5 从文件夹中随机选择一张图片进行预测
    path_img = random_image_from_folder(test_folder)
    print(f"预测图片: {path_img}")

    # 2/5 加载图片
    img_tensor, img_rgb = process_img(path_img)

    # 3/5 加载模型
    model = get_model(path_state_dict, num_classes, True)

    with torch.no_grad():
        time_tic = time.time()
        outputs = model(img_tensor)
        time_toc = time.time()

    # 4/5 获取预测结果
    _, pred_int = torch.max(outputs.data, 1)
    _, top1_idx = torch.topk(outputs.data, 1, dim=1)

    pred_idx = int(pred_int.cpu().numpy())
    if pred_idx == 0:
        pred_str = "cat"
    else:
        pred_str = "dog"

    print(f"img: {os.path.basename(path_img)} is: {pred_str}")
    print("time consuming: {:.2f}s".format(time_toc - time_tic))

    # 5/5 可视化结果
    plt.imshow(img_rgb)
    plt.title(f"predict: {pred_str}")
    plt.text(5, 45, f"top {1}: {pred_str}", bbox=dict(fc='yellow'))
    plt.show()
