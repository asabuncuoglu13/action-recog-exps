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
from resnet import *

device = 'cuda'
batch_size = 32
epochs = 20

def train(loader, model, crit, optimizer, epoch):
    model.train()

    loss_sum = 0
    for clips, targets in tqdm(loader):
        clips = clips.to(device)
        targets = targets.to(device)

        logits = model(clips)
        loss = crit(logits, targets)
        loss_sum += loss.data.cpu().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (loss_sum / len(loader))

def bce(probs, labels):
    safelog =  lambda x: np.log(np.maximum(x, np.exp(-50.)))
    return np.mean(-labels * safelog(probs) - (1 - labels) * safelog(1 - probs))

def validate(loader, model, crit):
    model.eval()
    sm = nn.Softmax(dim=1)
    labels = np.zeros((len(loader.dataset)), dtype=np.float32)
    probs = np.zeros((len(loader.dataset), 2), dtype=np.float32)
    with torch.no_grad():
        for i, (clips, targets) in enumerate(tqdm(loader)):
            start = i*batch_size
            end = start + clips.shape[0]
            labels[start:end] = targets
            clips = clips.to(device)

            logits = model(clips)
            probs[start:end] = sm(logits).cpu().numpy()

    probs = probs.reshape(4, -1, 2).mean(axis=0)
    labels = labels.reshape(4, -1).mean(axis=0)

    preds = probs.argmax(axis=1)
    correct = (preds == labels).sum()
    acc = correct*100//preds.shape[0]
    loss = bce(probs[:, 1], labels)
    print('validation accuracy %d%%' % acc)
    return loss

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    data_root = "/datasets/20bn_something_something/v2/"
    
    basedir = "/kuacc/users/asabuncuoglu13/action-recog-exps/notebooks/"
    
    train_dataset = SomethingSomethingV2(root=data_root, mode='train')
    eval_dataset = SomethingSomethingV2(root=data_root, mode='validation')
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers= min(cpu_count(), 2))

    eval_loader = DataLoader(eval_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers= min(cpu_count(), 2))
    
    model = resnet3d18(num_classes=174)
    model = model.to(device)

    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.Adam(model.parameters())

    model_file = basedir + 'logs/resnet_model_batch.pth'
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['state_dict'])
        print('loaded %s' % model_file)

    try:
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = train(train_loader, model, crit, opt, epoch)

            # Evaluate on validation set
            val_loss = validate(eval_loader, model, crit)
            print('epoch %d training loss %.2f validation loss %.2f\n' % (
                  epoch, train_loss, val_loss))
    finally:
        torch.save({'state_dict': model.state_dict()}, model_file)
    print('done')
    
if __name__ == '__main__':
    main()