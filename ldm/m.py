import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor()  # 转换为Tensor
])

# 加载数据集
dataset = ImageFolder("/media/ybxy/c89da59f-580c-440d-bab8-554bd51bb407/archive (6)/flowers", transform=transform)

# 计算均值和标准差
num_samples = len(dataset)
sum_channel = torch.zeros(3)
sum_square_channel = torch.zeros(3)

# for data, _ in dataset:
#     sum_channel += torch.sum(data, dim=(1, 2))
#     sum_square_channel += torch.sum(data ** 2, dim=(1, 2))
#
# mean = sum_channel / (num_samples * 256 * 256)
# std = torch.sqrt((sum_square_channel / (num_samples * 256 * 256)) - mean**2)
#
# # 输出均值和标准差
# print("Mean:", mean)
# print("Std:", std)

# 定义反归一化操作



# 获取一批图像

