import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from yaml import load
# from PIL import Image
import cv2
import matplotlib.cm
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

# HYPER PARAM
BATCH_SIZE = 8
MAX_DATA = 10000

class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    #<Network CNN 3 + FC 2> 
        self.conv1 = nn.Conv2d(n_channel, 32,kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512,n_out)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()
        self.liner1 = nn.Linear(64, 8, bias=False)
        self.liner2 = nn.Linear(8, 64, bias=False)
        self.flatten = nn.Flatten()
        self.deconv1 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.deconv2 = nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False)
        self.deconv3 = nn.ConvTranspose2d(1, 1, 3, stride=1, bias=False)
        
        
    #<Weight set>
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        #self.maxpool = nn.MaxPool2d(2,2)
        #self.batch = nn.BatchNorm2d(0.2)
        torch.nn.init.ones_(self.deconv1.weight)
        torch.nn.init.ones_(self.deconv2.weight)
        torch.nn.init.ones_(self.deconv3.weight)
        
    #<CNN layer>   
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            #self.maxpool,
            # self.flatten
        )
    #<SE block>
        self.se_block = nn.Sequential(
            self.liner1,
            self.relu,
            self.liner2,
            self.sig,
        )
        
    #<FC layer (output)>
        self.fc_layer = nn.Sequential(
            self.flatten,
            self.fc4,
            self.relu,
            self.fc5,
        )

    #<forward layer>
    def forward(self,x):
        x1 = self.cnn_layer(x)
        a, b, _, _ = x1.size()
        x2 = self.avg_pool(x1).view(a, b)
        x3 = self.se_block(x2).view(a, b, 1, 1)
        x4 = x1*x3.expand_as(x1)
        x5 = self.fc_layer(x4)
        return x5
    
    def feature2image(self, conv1, conv2, conv3, mul):
        ave1_reshape = torch.reshape(conv1, (1, 1, 11, 15))
        ave2_reshape = torch.reshape(conv2, (1, 1, 5, 7))
        ave3_reshape = torch.reshape(conv3, (1, 1, 3, 5))
        mul_reshape = torch.reshape(mul, (1, 1, 3, 5))
        # image = self.deconv3(mul_reshape)
        # image = self.deconv2(image) 
        # image = self.deconv1(image)
        image = mul_reshape * ave3_reshape
        image = self.deconv3(image) * ave2_reshape
        image = self.deconv2(image) * ave1_reshape
        image = self.deconv1(image)
        image = torch.reshape(image, (48, 64))
        image = image.to('cpu').detach().numpy().copy()
        # min = np.min(image)
        # max = np.max(image)
        # image = (image - min) / (max - min)
        image = (image - image.min())/(image.max() - image.min())
        return image

class deep_learning:
    def __init__(self, n_channel=3, n_action=1):
        #<tensor device choiece>
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)
        self.optimizer = optim.Adam(self.net.parameters(),eps=1e-2,weight_decay=5e-4)
        #self.optimizer.setup(self.net.parameters())
        self.totensor = transforms.ToTensor()
        self.n_action = n_action
        self.count = 0
        self.accuracy = 0
        self.results_train = {}
        self.results_train['loss'], self.results_train['accuracy'] = [], []
        self.loss_list = []
        self.acc_list = []
        self.datas = []
        self.target_angles = []
        self.criterion = nn.MSELoss()
        self.transform=transforms.Compose([transforms.ToTensor()])
        self.first_flag =True
        torch.backends.cudnn.benchmark = True
        self.extractor = create_feature_extractor(self.net, ["relu", "relu_1", "relu_2", "mul"])

    def make_dataset(self,img,target_angle):
        if self.first_flag:
            self.x_cat = torch.tensor(img,dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat=self.x_cat.permute(0,3,1,2)
            self.t_cat = torch.tensor([target_angle],dtype=torch.float32,device=self.device).unsqueeze(0)
            self.first_flag =False
        x = torch.tensor(img,dtype =torch.float32, device=self.device).unsqueeze(0)
        x=x.permute(0,3,1,2)
        t = torch.tensor([target_angle],dtype=torch.float32,device=self.device).unsqueeze(0)
        self.x_cat =torch.cat([self.x_cat,x],dim=0)
        self.t_cat =torch.cat([self.t_cat,t],dim=0)
        
    #<make dataset>
        self.dataset = TensorDataset(self.x_cat,self.t_cat)

    def trains(self):
        self.net.train()
        train_dataset = DataLoader(self.dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'),shuffle=True)
        
    #<only cpu>
        # train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)
        
    #<split dataset and to device>
        for x_train, t_train in train_dataset:
            x_train.to(self.device,non_blocking=True)
            t_train.to(self.device,non_blocking=True)
            break

    #<learning>
        self.optimizer.zero_grad()
        y_train = self.net(x_train)
        loss = self.criterion(y_train, t_train) 
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def act_and_trains(self, img,target_angle):
        self.make_dataset(img,target_angle)
        loss = self.trains()
        #<test>
        self.net.eval()
        x = torch.tensor(img,dtype =torch.float32, device=self.device).unsqueeze(0)
        x=x.permute(0,3,1,2)
        action_value_training = self.net(x)
        return action_value_training[0][0].item(), loss

    def act(self, img):
            self.net.eval()
        #<make img(x_test_ten),cmd(c_test)>
            # x_test_ten = torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
            x_test_ten = torch.tensor(img,dtype=torch.float32, device=self.device).unsqueeze(0)
            x_test_ten = x_test_ten.permute(0,3,1,2)
            #print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
        #<test phase>
            action_value_test = self.net(x_test_ten)
            
            # print("act = " ,action_value_test.item())
            return action_value_test.item()

    def result(self):
            accuracy = self.accuracy
            return accuracy

    def save(self, save_path):
        #<model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(self.net.state_dict(), path + '/model_gpu.pt')


    def load(self, load_path):
        #<model load>
        self.net.load_state_dict(torch.load(load_path))

    def fv(self, img):
        self.net.eval()
        x_ten = torch.tensor(img,dtype=torch.float32, device=self.device).unsqueeze(0)
        x_ten = x_ten.permute(0,3,1,2)
        self.net(x_ten)
        features = self.extractor(x_ten)
        conv1 = features["relu"]
        feature1 = conv1.to('cpu').detach().numpy().copy()
        ave1 = np.average(feature1[0],axis=0)
        ave1_ten = torch.from_numpy(ave1)
        ave1_ten = ave1_ten.cuda()
        conv2 = features["relu_1"]
        feature2 = conv2.to('cpu').detach().numpy().copy()
        ave2 = np.average(feature2[0],axis=0)
        ave2_ten = torch.from_numpy(ave2)
        ave2_ten = ave2_ten.cuda()
        conv3 = features["relu_2"]
        feature3 = conv3.to('cpu').detach().numpy().copy()
        ave3 = np.average(feature3[0],axis=0)
        ave3_ten = torch.from_numpy(ave3)
        ave3_ten = ave3_ten.cuda()
        mul = features["mul"]
        feature = mul.to('cpu').detach().numpy().copy()
        ave = np.average(feature[0],axis=0)
        ave_ten = torch.from_numpy(ave)
        ave_ten = ave_ten.cuda()
        fv_img = self.net.feature2image(ave1_ten, ave2_ten, ave3_ten, ave_ten)
        feature_rgb = np.dstack((fv_img, fv_img, fv_img))
        multi_img = np.multiply(feature_rgb, img)
        return multi_img

if __name__ == '__main__':
        dl = deep_learning()