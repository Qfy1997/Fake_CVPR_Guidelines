
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms

#设置全局随机数种子，让每次程序运行时所生成的随机数都是相同的。
torch.manual_seed(66)
# 如果有gpu就用gpu，如果没有就用cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=32
# Compose定义了一系列transform，此操作相当于将多个transform一并执行
# mnist是灰度图，此处只将一个通道标准化
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5), std=(0.5))])    
# 设定数据集，图片领域的数据集推荐在官网直接下载
mnist_data = torchvision.datasets.MNIST("./mnist_data", train=True, download=False, transform=transform)
# 加载数据集，按照上述要求，shuffle指随机排序
dataloader = torch.utils.data.DataLoader(dataset=mnist_data,batch_size=batch_size,shuffle=True)                                       
image_size = 784
hidden_size = 256
# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid() # sigmoid结果为（0，1）
)
# Generator
latent_size = 64 # latent_size，相当于初始噪声的维数
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh() # 转换至（-1，1）
)
# 放到gpu上计算（如果有的话）
D = D.to(device)
G = G.to(device)
# 定义损失函数、优化器、学习率
loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
# 先定义一个梯度清零的函数，方便后续使用
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
# 迭代次数与计时
total_step = len(dataloader)
num_epochs = 200
start = time.perf_counter() # 起始时间
# 进行训练
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader): # 当前step
        batch_size = images.size(0) # 变成一维向量
        images = images.reshape(batch_size, image_size).to(device)
        # 定义真假label，用作评分
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        # 对D进行训练，D的损失函数包含两部分
        # 第一部分，D对真图的判断能力
        outputs = D(images) # 将真图送入D，输出（0，1），应该是越接近1越好
        d_loss_real = loss_fn(outputs, real_labels)
        real_score = outputs # 真图的分数，越大越好
        # 第二部分，D对假图的判断能力
        z = torch.randn(batch_size, latent_size).to(device) # 开始生成一组fake images即32*784的噪声经过G的假图
        fake_images = G(z)
        outputs = D(fake_images.detach()) # 将假图片给D，detach表示不作用于求grad
        d_loss_fake = loss_fn(outputs, fake_labels)
        fake_score = outputs # 假图的分数，越小越好
        # 对discriminator进行优化
        d_loss = d_loss_real + d_loss_fake # 总的损失就是以上两部分相加，越小越好
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        # 对G进行训练，G的损失函数包含一部分
        # 可以用前面的z，也可以新生成，因为模型没有改变，事实上是一样的
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = loss_fn(outputs, real_labels) # 计算output与real_labels的损失值
        # 对generator进行优化
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        # 优化完成，下面进行一些反馈，展示学习进度
        if i % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}".format(epoch, num_epochs, i, total_step, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))
    if (epoch+1)%50==0:
        G_model_name="G_epoch:"+str(epoch+1)
        d_model_name = "D_epoch:"+str(epoch+1)
        torch.save(G,G_model_name)
        torch.save(D,d_model_name)
Gmodel = torch.load("G_epoch:200")
# 训练结束，跳出循环，检验成果
end = time.perf_counter() # 结束时间
total = end - start
minutes = total//60
seconds = total - minutes*60
print("利用GPU总用时：{:.2f}分钟{:.2f}秒".format(minutes, seconds))
# 向G输入一个噪声，根据不同训练循环数所保存的模型生成的图片
z = torch.rand(1, latent_size).to(device)
print(z)
Gmodel = torch.load("G_epoch:50")
fake_images1 = Gmodel(z).view(28, 28).data.cpu().numpy()
Gmodel = torch.load("G_epoch:100")
fake_images2 = Gmodel(z).view(28, 28).data.cpu().numpy()
Gmodel = torch.load("G_epoch:150")
fake_images3 = Gmodel(z).view(28, 28).data.cpu().numpy()
Gmodel = torch.load("G_epoch:200")
fake_images4 = Gmodel(z).view(28, 28).data.cpu().numpy()
fig, ax = plt.subplots(nrows=1,ncols=4,sharex='all',sharey='all')
ax = ax.flatten()
ax[0].imshow(fake_images1,cmap='Greys',interpolation='nearest')
ax[1].imshow(fake_images2,cmap='Greys',interpolation='nearest')
ax[2].imshow(fake_images3,cmap='Greys',interpolation='nearest')
ax[3].imshow(fake_images4,cmap='Greys',interpolation='nearest')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
plt.tight_layout()
plt.show()
# print(fake_images)
# print(next(iter(dataloader))[0][0][0])
# plt.imshow(fake_images1, cmap = plt.cm.gray)
# plt.show()
# plt.imshow(next(iter(dataloader))[0][0][0], cmap = plt.cm.gray)
# plt.show()


