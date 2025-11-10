# cite :https://codeload.github.com/streamer-AP/DomainAdaptionPytorch/zip/refs/heads/master
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import RandomSampler, Dataset, DataLoader
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Function
from torchvision.utils import make_grid
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import shutil
import os

torch.manual_seed(66)



class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def adjust_learning_rate(optimizer,epoch):
    lr = 0.001*0.1**(epoch//10)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    return lr

def matplotlib_imshow(img,one_channel=False):
    if one_channel:
        img=img.mean(dim=0)
    np_img=img.numpy()
    np_img=(np_img-np.min(np_img))/(np.max(np_img)-np.min(np_img))
    if one_channel:
        plt.imshow(np_img,cmap="Greys")
    else:
        plt.imshow(np.transpose(np_img,(1,2,0)))
    plt.show()

def accuracy(output,target,topk=(1,)):
    maxk=max(topk)
    batch_size=target.size(0)
    _,pred=output.topk(maxk,1,True,True)
    pred=pred.t()
    correct=pred.eq(target.view(1,-1).expand_as(pred))
    res=[]
    for k in topk:
        correct_k=correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100/batch_size))
    return res


class mnist_m(Dataset):
    def __init__(self,root,label_file):
        super(mnist_m,self).__init__()
        self.transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        ])
        with open(label_file,"r") as f:
            self.imgs=[]
            self.labels=[]
            for line in f.readlines():
                line=line.strip("\n").split(" ")
                img_name,label=line[0],int(line[1])
                img=Image.open(root+os.sep+img_name)
                self.imgs.append(self.transform(img.convert("RGB")))
                self.labels.append(label)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,index):
        return self.imgs[index],self.labels[index]
    def __add__(self,other):
        pass


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

# class CNN(nn.Module):
#     def __init__(self,num_classes=10):
#         super(CNN,self).__init__()
#         self.features=nn.Sequential(
#             nn.Conv2d(3,32,5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32,48,5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#         )
#         self.avgpool=nn.AdaptiveAvgPool2d((5,5))
#         self.classifier=nn.Sequential(
#             nn.Linear(48*5*5,100),
#             nn.ReLU(inplace=True),
#             nn.Linear(100,100),
#             nn.ReLU(inplace=True),
#             nn.Linear(100,num_classes)
#         )
#     def forward(self,x):
#         x=x.expand(x.data.shape[0],3,image_size,image_size)
#         x=self.features(x)
#         x=self.avgpool(x)
#         x=torch.flatten(x,1)
#         x=self.classifier(x)
#         return x


class DANN(nn.Module):
    def __init__(self,num_classes=10):
        super(DANN,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,48,5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool=nn.AdaptiveAvgPool2d((5,5))
        self.task_classifier=nn.Sequential(
            nn.Linear(48*5*5,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,num_classes)
        )
        self.domain_classifier=nn.Sequential(
            nn.Linear(48*5*5,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,2)
        )
        self.GRL=GRL()
    def forward(self,x,alpha):
        x = x.expand(x.data.shape[0], 3, image_size,image_size)
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        task_predict=self.task_classifier(x)
        x=GRL.apply(x,alpha)
        domain_predict=self.domain_classifier(x)
        return task_predict,domain_predict


if __name__=='__main__':
    root_path=os.path.join("dataset","mnist_m")
    print(root_path)

    image_size=28
    batch_size=128
    start=time.time()
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5])])
    train_ds=datasets.MNIST(root="the_path_you_defined",train=True,transform=transform,download=False)
    test_ds=datasets.MNIST(root="the_path_you_defined",train=False,transform=transform,download=False)
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    test_dl=DataLoader(test_ds,batch_size=batch_size,shuffle=False)
    root_path=os.path.join("the_path_you_defined","mnist_m")
    train_m_ds=mnist_m(os.path.join(root_path,"mnist_m_train"),os.path.join(root_path,"mnist_m_train_labels.txt"))
    test_m_ds=mnist_m(os.path.join(root_path,"mnist_m_test"),os.path.join(root_path,"mnist_m_test_labels.txt"))
    train_m_dl=DataLoader(train_m_ds,batch_size=batch_size,shuffle=True)
    test_m_dl=DataLoader(test_m_ds,batch_size=batch_size,shuffle=False)
    end=time.time()
    print("load finish time:",end-start,"s")

    # show_images=[train_ds[i][0] for i in range(8)]
    # show_labels=[train_ds[i][1] for i in range(8)]
    # print(show_images[0].shape)
    # show_img_grid=make_grid(show_images)
    # matplotlib_imshow(show_img_grid,one_channel=True)

    # show_images=[train_m_ds[i][0] for i in range(8)]
    # show_labels=[train_m_ds[i][1] for i in range(8)]
    # show_img_grid=make_grid(show_images)
    # print(show_labels)
    # matplotlib_imshow(show_img_grid,one_channel=False)
    # cnn_model=CNN()
    # optimizer=Adam(cnn_model.parameters(),lr=0.001)
    # Loss=nn.CrossEntropyLoss()
    # epochs=5
    # train_loss=AverageMeter()
    # test_loss=AverageMeter()
    # test_top1=AverageMeter()
    # train_top1=AverageMeter()
    # train_cnt=AverageMeter()
    # print_freq=200
    # cnn_model.cuda()
    # for epoch in range(epochs):
    #     lr=adjust_learning_rate(optimizer,epoch)
    #     train_loss.reset()
    #     train_top1.reset()
    #     train_cnt.reset()
    #     test_top1.reset()
    #     test_loss.reset()
    #     for images,labels in train_dl:
    #         images=images.cuda()
    #         labels=labels.cuda()
    #         optimizer.zero_grad()
    #         predict=cnn_model(images)
    #         losses=Loss(predict,labels)
    #         train_loss.update(losses.data,images.size(0))
    #         top1=accuracy(predict.data,labels,topk=(1,))[0]
    #         train_top1.update(top1,images.size(0))
    #         train_cnt.update(images.size(0),1)
    #         losses.backward()
    #         optimizer.step()
    #         if train_cnt.count%print_freq==0:
    #             print("Epoch:{}[{}/{}],Loss:[{:.3f},{:.3f}],prec[{:.4f},{:.4f}]".format(epoch,train_cnt.count,len(train_dl),train_loss.val,train_loss.avg,train_top1.val,train_top1.avg))
    #     #这个位置写的挺厉害的，练完了就测，测完了还能涨。
    #     for images,labels in test_dl:
    #         images=images.cuda()
    #         labels=labels.cuda()
    #         predict=cnn_model(images)
    #         losses=Loss(predict,labels)
    #         test_loss.update(losses.data,images.size(0))
    #         top1=accuracy(predict.data,labels,topk=(1,))[0]
    #         test_top1.update(top1,images.size(0))
    #     print("Epoch:{},val,Loss:[{:.3f}],prec[{:.4f}]".format(epoch,test_loss.avg,test_top1.avg))


    train_loss=AverageMeter()
    train_domain_loss=AverageMeter()
    train_task_loss=AverageMeter()
    test_loss=AverageMeter()
    test_top1=AverageMeter()
    test_domain_acc=AverageMeter()
    train_top1=AverageMeter()
    train_cnt=AverageMeter()
    print_freq=200
    #这个模型每次跑完后的结果不同，个人感觉属于迁移学习中的领域适应。
    domain_model=DANN()
    domain_model.cuda()
    domain_loss=nn.CrossEntropyLoss()
    task_loss=nn.CrossEntropyLoss()
    lr=0.001
    optimizer=Adam(domain_model.parameters(),lr=lr)
    epochs=3
    # print("-==============")
    # print("images:",images.size(0))
    # print(images)
    # print('============')
    for epoch in range(epochs):
        lr=adjust_learning_rate(optimizer,epoch)
        train_loss.reset()
        train_domain_loss.reset()
        train_task_loss.reset()
        train_top1.reset()
        train_cnt.reset()
        test_top1.reset()
        test_loss.reset()
        for source,target in zip(train_dl,train_m_dl):
            train_cnt.update(16,1)
            p = float(train_cnt.count + epoch * len(train_dl)) / (epochs *len(train_dl))
            alpha = torch.tensor(2. / (1. + np.exp(-10 * p)) - 1)
            src_imgs=source[0].cuda()
            src_labels=source[1].cuda()
            dst_imgs=target[0].cuda()
            optimizer.zero_grad()
            
            src_predict,src_domains=domain_model(src_imgs,alpha)
            src_label_loss=task_loss(src_predict,src_labels)
            src_domain_loss=domain_loss(src_domains,torch.ones(len(src_domains)).long().cuda())
            
            _,dst_domains=domain_model(dst_imgs,alpha)
            dst_domain_loss=domain_loss(dst_domains,torch.zeros(len(dst_domains)).long().cuda())
            
            losses=src_label_loss+src_domain_loss+dst_domain_loss
            
            train_loss.update(losses.data,16)
            train_domain_loss.update(dst_domain_loss.data,16)
            train_task_loss.update(src_label_loss.data,16)
            top1=accuracy(src_predict.data,src_labels,topk=(1,))[0]
            train_top1.update(top1,16)
            
            losses.backward()
            optimizer.step()
            if train_cnt.count%print_freq==0:
                print("Epoch:{}[{}/{}],Loss:[{:.3f},{:.3f}],domain loss:[{:.3f},{:.3f}],label loss:[{:.3f},{:.3f}],prec[{:.4f},{:.4f}],alpha:{}".format(
                    epoch,train_cnt.count,len(train_dl),train_loss.val,train_loss.avg,
                    train_domain_loss.val,train_domain_loss.avg,
                    train_task_loss.val,train_task_loss.avg,train_top1.val,train_top1.avg,alpha))
        
        for images,labels in test_m_dl:
            images=images.cuda()
            labels=labels.cuda()
            predicts,domains=domain_model(images,0)
            losses=task_loss(predicts,labels)
            test_loss.update(losses.data,images.size(0))
            top1=accuracy(predicts.data,labels,topk=(1,))[0]
            domain_acc=accuracy(domains.data,torch.zeros(len(domains)).long().cuda(),topk=(1,))[0]
            test_top1.update(top1,images.size(0))
            test_domain_acc.update(domain_acc,images.size(0))
        print("Epoch:{},val,Loss:[{:.3f}],prec[{:.4f}],domain_acc[{:.4f}]".format(epoch,test_loss.avg,test_top1.avg,test_domain_acc.avg))
    torch.save(domain_model,"DANN_model.pth")
    
    # torch.save(cnn_model,"cnn_model.pth")
    # test_m_top1=AverageMeter()
    # test_m_loss=AverageMeter()
    # for images,labels in test_m_dl:
    #     images=images.cuda()
    #     labels=labels.cuda()
    #     predict=cnn_model(images)
    #     losses=Loss(predict,labels)
    #     test_m_loss.update(losses.data,images.size(0))
    #     top1=accuracy(predict.data,labels,topk=(1,))[0]
    #     test_m_top1.update(top1,images.size(0))
    # print("Epoch:{},val,Loss:[{:.3f}],prec[{:.4f}]".format(epoch,test_m_loss.avg,test_m_top1.avg))
    # torch.save(cnn_model,"cnn_transfer_model.pth")


    # 我的写法不行，我得保存，保存完了就不想更新参数。
    # state_dict = torch.load('cnn_model.pth')
    # state_dict = torch.load('cnn_transfer_model.pth')
    # total=0
    # total_count=0
    # for images,labels in test_dl:
            # images=images.cuda()
            # labels=labels.cuda()
            # predict=state_dict(images)
            # predict_index=torch.argmax(predict,dim=1)
            # print(predict_index)
            # print(labels)
            # count=torch.eq(predict_index,labels)
            # count=torch.sum(count).item()
            # total+=len(labels)
            # total_count+=count
            # break
    # acc:0.9894 mnist  9894/10000
    # acc: 0.6   mnist_m 5405/9001
    # print("total:",total)
    # print("count:",total_count)
    # acc=total_count/total
    # print("acc:",acc)
    
    
