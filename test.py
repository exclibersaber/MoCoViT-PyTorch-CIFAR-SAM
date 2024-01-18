import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from mocovit import MoCoViT

# 载入模型
model = MoCoViT()
model.load_state_dict(torch.load('checkpoints/epoch.pt'))  # 模型文件路径
model.eval()  # 将模型设置为评估模式

# 加载 CIFAR-100 训练集来获取类别名称
cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True)
class_names = cifar100_train.classes

# 索引到类别名称的映射
idx_to_class = {i: class_names[i] for i in range(len(class_names))}

# 图像预处理
def preprocess_image(image_path):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 增加一个维度以符合模型的输入要求
    return image

# 分类图像
def classify_image(image_path, model):
    image = preprocess_image(image_path)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    predicted_class_name = idx_to_class[predicted.item()]
    return predicted_class_name  # 返回预测的类别名称

# 使用模型
image_path = 'image.jpg'  # 要分类的图像路径
predicted_class_name = classify_image(image_path, model)
print("Predicted class name:", predicted_class_name)
