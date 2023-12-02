import torch
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from torchvision import datasets 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch import optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys
import Nopileup_dataset as dataset
import os
import time


class modified_CNN(nn.Module):
    def __init__(self):
        super(modified_CNN, self).__init__()
        self.conv1 = nn.Conv1d(5, 320, kernel_size = 6)#If input rows = 1000, output of this should be (batchsize, 320,1000-5) assuming no padding and kernel size = num columns 
        self.pool1 = nn.MaxPool1d(5) #If rows = 1000, (Kernel size, stride) = (5,1). So output of this should be (batchsize, 320, 995/5)
        self.conv2 = nn.Conv1d(320,480, kernel_size = 5) #If rows=1000, Output should be (batchsize, 480,195)
        self.pool2 = nn.MaxPool1d(5) #If rows=1000, Output shd be (batchsize, 480, 39)
        
        self.fc1 = nn.Linear(9120, 256) #9120 is based on input row size being 500
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)
    
    def forward(self,x): #x should be (batchsize, channels(columns), length). meta should be (batchsize, 78)
        out = nn.functional.relu(self.conv1(x))
        out = self.pool1(out)
        out = nn.functional.relu(self.conv2(out))
        out = self.pool2(out)
 
        out = torch.flatten(out, start_dim=1) #Should return (batchsize, 480*39)
    
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        
        return(out)
    
    
if __name__ == "__main__":
    

    #Create train, test datasets + dataloaders. Take subset of train/test based on chromosome as indicated in metadata file
    #samplename_locus.npy

    ohe_dir = "/nfs/turbo/dcmb-class/bioinf593/groups/group_05/STRonvoli/data/ohe/"
    meta_file = "/nfs/turbo/dcmb-class/bioinf593/groups/group_05/STRonvoli/data/metadata.tsv"

    dataset = dataset.NOPDataset(ohe_dir, meta_file) 
    meta = pd.read_csv(meta_file, sep = '\t')
    train_indices = meta.index[meta["split"] == 0]
    test_indices = meta.index[meta["split"] == 1]

    trainset = Subset(dataset, train_indices)
    validset = Subset(dataset, test_indices)
    
    num = 6
    trainloader = DataLoader(trainset, batch_size = 64, shuffle=True, num_workers = num)
    validloader = DataLoader(validset, batch_size = 64, shuffle=True, num_workers = num)

    
    #Train
    #Parameters (to change): 
    pos_count = 500
    meta_count = 79
    max_epochs = 10
    learning = 0.001

    net = modified_CNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning)

    losses = [] #Training loss for each epoch (keys are epochs)
    losses_t = [] #Test loss over epochs
    for e in range(max_epochs):
        loss_l = 0
        batch_size = 0 #Track total number of samples.. should just be overall total but just to ensure
        net.train() #Train model
        print(len(trainloader))
        start_time = time.time()
        total_loss = 0
        
        for (batch_idx, batch) in enumerate(trainloader):
            (X, labels) = batch
            X = X.to(device)
            labels = labels.to(device)
            X = torch.transpose(X,1,2)
            X = X.float()
            labels = labels.float()
            labels = torch.reshape(labels, (labels.size(dim=0),1))
            
            optimizer.zero_grad()
            output = net(X) #Will be (batch_size, 10)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += (loss.cpu().detach().numpy()) #Remove grad requirement + convert to np array to be able to plot
        losses.append(total_loss/len(trainloader))
        print(f'Epoch {e} done')
        
        #Save model 
        savedir = "/nfs/turbo/dcmb-class/bioinf593/groups/group_05/STRonvoli/models/NOP_models/"
        path = os.path.join(savedir, f'epoch-{e}-model.pth')
        torch.save({
            'epoch': e,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss}, path)  # saving model
        

        net.eval()
        for (Y,y) in validloader:
            Y = Y.to(device)
            y = y.to(device)
            Y = torch.transpose(Y,1,2)
            Y = Y.float()
            y = y.float()
            y = torch.reshape(y, (y.size(dim=0),1))
            out = net(Y)
            lossl = criterion(out, y)
            loss_l += (lossl.data.item())
        losses_t.append(loss_l/len(validloader))
        print(f'Validation loss after epoch {e} is')
        print(loss_l/len(validloader))


    figure(0)
    plt.plot(losses_t)
    plt.xlabel('Epochs')
    plt.ylabel('Average MSE loss over batches in each epoch')
    plt.title('Validation Loss over epochs for STR prediction using no pileup CNN')
    plt.savefig('Average Validation loss (across all batches) over epochs - no pileup CNN')
    
    
    figure(1)
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Average MSE loss over batches in each epoch')
    plt.title('Training Loss over epochs for STR prediction using no pileup CNN')
    plt.savefig('Average Training loss (across all batches) over epochs- no pileup CNN')
    
    
    
    