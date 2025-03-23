
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import math
import torch.nn.functional as F

# https://arxiv.org/pdf/1606.08415
# gelu激活函数的不同写法
def gelu1(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu2(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def gelu3(x): # slower than gelu2
    return 0.5 * x * (1 + torch.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

 
# ----------数据集----------(推荐用官网的数据集)
 
# 加载FashionMNIST数据集的训练集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
# 加载FasionMNIST数据集的测试集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
 
# batch大小
batch_size = 64
 
# 创建dataloader
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
 
# 遍历dataloader
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)     # 每个batch数据的形状
    print("Shape of y: ", y.shape)                  # 每个batch标签的形状
    break

 
# ----------模型----------
# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):     # 初始化，实例化模型的时候就会调用
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()     # [64, 1, 28, 28] -> [64, 1*28*28]
        self.layer1 = nn.Linear(28*28,512)
        self.layer2 = nn.Linear(512,64)
        self.layer3 = nn.Linear(64,10)
 
    def forward(self, x):   # 前向传播，输入数据进网络的时候才会调用
        x = self.flatten(x)                     # [64, 1*28*28]
        x = self.layer1(x)
        # 通过每一层线性层之后可以选择使用不同的激活函数(relu,elu,gelu)去激活一下。
        # x = F.relu(x)
        m = nn.ELU()
        x = m(x)
        # x = gelu3(x)
        x = self.layer2(x)
        # x = gelu3(x)
        x = m(x)
        x = self.layer3(x)
        return x
 
# 使用gpu或者cpu进行训练
device = "cuda" if torch.cuda.is_available() else "cpu"
# 打印使用的是gpu/cpu
print("Using {} device".format(device))
# 实例化模型
model = NeuralNetwork().to(device)
# 打印模型结构
print(model)

# ----------训练参数设置----------
loss_fn = nn.CrossEntropyLoss()                             # 损失函数设置
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)    # 学习率设置
epochs = 30                                                  # 训练迭代次数设置

 
# 训练函数
def train(train_dataloader, model, loss_fn, optimizer):
    """
        训练网络
        输入:
            train_dataloader:   训练集的dataloader
            model:              网络模型
            loss_fn:            损失函数
            optimizer:          优化器
        """
    # 切换到train模式
    model.train()
    # 遍历dataloader
    for images, labels in train_dataloader:
        # 将数据和标签加载到device上
        images, labels = images.to(device), labels.to(device)
        # 输入数据到模型里得到输出
        pred = model(images)
        # 计算输出和标签的loss
        loss = loss_fn(pred, labels)
        # 反向推导
        optimizer.zero_grad()
        loss.backward()
        # 步进优化器
        optimizer.step()
 
 
# 测试函数
def test(test_dataloader, model, loss_fn):
    """
        测试网络
        输入:
            test_dataloader:    测试集的dataloader
            model:              网络模型
            loss_fn:            损失函数
        """
    # 测试集大小
    size = len(test_dataloader.dataset)
    # 测试集的batch数量
    num_batches = len(test_dataloader)
    # 切换到测试模型
    model.eval()
    # 记录loss和准确率
    test_loss, correct = 0, 0
    # 梯度截断
    with torch.no_grad():
        for images, labels in test_dataloader:  # 遍历batch
            # 加载到device
            images, labels = images.to(device), labels.to(device)
            # 输入数据到模型里得到输出
            pred = model(images)
            # 累加loss
            test_loss += loss_fn(pred, labels).item()
            # 累加正确率
            correct += (pred.argmax(1) == labels).sum().item()
    # 计算平均loss和准确率
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    print("Done!")
