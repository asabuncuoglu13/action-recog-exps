"""
'act-recog-smth-torch' notebook's executable file to submit to the slurm.ü
The file includes wandb integration, change the credentials to submit to your project.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils.data.data import SomethingSomethingV2
from torch.multiprocessing import cpu_count
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import math

import wandb

class AttentionConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, bias=bias)
        self.query_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, bias=bias)
        self.value_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width, d = x.size()
        #print("x size:", x.size())

        pad = nn.ReplicationPad3d(self.padding)
        padded_x = pad(x)
        #print("padded x size:", padded_x.size())
        
        q_out = self.query_conv(x)
        #print("q_out:", q_out.shape)
        
        k_out = self.key_conv(x)
        #print("k_out:", k_out.shape)
        
        v_out = self.value_conv(x)
        #print("v_out:", v_out.shape)
        
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height - self.padding, width - self.padding, d - self.padding, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height - self.padding, width - self.padding, d - self.padding, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height - self.padding, width - self.padding, d - self.padding, 1)
        
        out = torch.matmul(q_out, (k_out).transpose(-1, -2))
        out = F.softmax(out, dim=-1)
        out = torch.matmul(out, v_out).view(batch, -1, height- self.padding, width - self.padding, d - self.padding)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
        


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel,self).__init__()
        
        self.conv0 = nn.Conv3d(3, 3, 3)
        self.pool0 = nn.MaxPool3d(4)
        self.conv1 = nn.Sequential(
            AttentionConv3d(3, 8, kernel_size=5, padding=2, groups=2),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )        
        #self.conv1 = nn.Conv3d(3,8,3)
        self.conv2 = nn.Conv3d(8,32,3)
        self.conv3 = nn.Conv3d(32,32,3)
        # self.conv4 = nn.Conv3d(64,128,3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool3d(4)
        self.linear1 = nn.Linear(512,256)
        self.linear2 = nn.Linear(256,174)
        self.batchnorm = nn.BatchNorm3d(32)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        #if downsampling:
        x = self.conv0(x)
        x = self.pool0(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        #x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.pool1(x)
        x = self.batchnorm(x)
        x = x.view(x.size()[0],-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    
def main():
    parameters = {
        "dataset" : "smth-smth-small",
        "pretrained_model" : False,
        "cuda" : True,
        "img_size" : 224,
        "stem" : False,
        "batch_size": 32,
        "num_workers" : min(cpu_count(), 2),
        "num_epochs": 40,
        "learning_rate" : 0.0005,
        "momentum" : 0.9,
        "weight_decay" : 1e-4
    }

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    #data_root = "/datasets/20bn_something_something/v2/"
    data_root = "/kuacc/users/asabuncuoglu13/20bn"

    train_dataset = SomethingSomethingV2(root=data_root, mode='train')
    valid_dataset = SomethingSomethingV2(root=data_root, mode='validation')
    train_loader = DataLoader(train_dataset,
                              batch_size=parameters["batch_size"],
                              shuffle=True,
                              num_workers= parameters["num_workers"])

    test_loader = DataLoader(valid_dataset,
                              batch_size=parameters["batch_size"],
                              shuffle=True,
                              num_workers= parameters["num_workers"])

    #downsampling = True

    model = CNNModel()
    model = model.cuda()

    job_id = os.getenv('SLURM_JOB_ID', 'NA')
    wandb.init(project='sasa', entity='asabuncuoglu13', config=parameters)
    wandb.run.name = 'j{}'.format(job_id) # wandb.run.save()
    wandb.watch(model)

    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])

    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    for epoch in range(parameters["num_epochs"]):
        model.train()
        running_acc = 0
        train_loss = 0
        for x, labels in tqdm(train_loader):
            optimizer.zero_grad()
            logits = model(x.cuda())
            #logits = model(x)
            loss = error(logits, labels.cuda())
            #loss = error(logits, labels)
            train_loss+=loss.data
            train_accuracy = accuracy_score(torch.argmax(logits,axis = 1).cpu().numpy(),labels.cpu().numpy())
            running_acc+=train_accuracy*len(x)
            loss.backward()
            optimizer.step()
        running_acc/=len(train_dataset)
        train_loss/=5
        model.eval()
        with torch.no_grad():
            for x, labels in tqdm(test_loader):
                test = x.cuda()
                outputs = model(test).detach()
                test_loss = error(outputs,labels.cuda())
                test_accuracy = accuracy_score(torch.argmax(outputs,axis = 1).cpu().numpy(),labels.cpu().numpy())
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss.data)
        test_accuracy_list.append(test_accuracy)
        train_accuracy_list.append(train_accuracy)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, "model.pt")
        print('Epoch: {}  Train Loss:{:.2f}, Test Loss:{:.2f}, Train Accuracy:{:.2f}, Test Accuracy:{:.2f}'.format(epoch, train_loss,test_loss,train_accuracy,test_accuracy))
        wandb.log({ "trn_loss": train_loss, "tst_loss": test_loss})
        
        
if __name__ == '__main__':
    main()