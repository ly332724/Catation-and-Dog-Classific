import os, shutil

# 下载的kaggle数据集路径
original_dataset_dir = '/home/ly/code/opencv/cat_dog/data/train'  # 原始数据集路径

# 新的小数据集放置路径
base_dir = '/home/ly/code/opencv/cat_dog/Smalldata'  # 新数据集的路径
os.makedirs(base_dir, exist_ok=True)

# 创建训练集和测试集的目录
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# 创建猫和狗的子目录
train_cats_dir = os.path.join(train_dir, 'cats')
os.makedirs(train_cats_dir, exist_ok=True)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.makedirs(train_dogs_dir, exist_ok=True)

test_cats_dir = os.path.join(test_dir, 'cats')
os.makedirs(test_cats_dir, exist_ok=True)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.makedirs(test_dogs_dir, exist_ok=True)

# 将猫的图片分配到训练集和测试集
train_cat_fnames = ['cat.{}.jpg'.format(i) for i in range(999)]  # 取前200张猫的图片
for fname in train_cat_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

test_cat_fnames = ['cat.{}.jpg'.format(i) for i in range(300, 400)]  # 取300到399的猫的图片
for fname in test_cat_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将狗的图片分配到训练集和测试集
train_dog_fnames = ['dog.{}.jpg'.format(i) for i in range(999)]  # 取前200张狗的图片
for fname in train_dog_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

test_dog_fnames = ['dog.{}.jpg'.format(i) for i in range(300, 400)]  # 取300到399的狗的图片
for fname in test_dog_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 打印文件统计信息
print('Total training cat images:', len(os.listdir(train_cats_dir)))
print('Total training dog images:', len(os.listdir(train_dogs_dir)))
print('Total test cat images:', len(os.listdir(test_cats_dir)))
print('Total test dog images:', len(os.listdir(test_dogs_dir)))
