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
import STRDataset as dataset
import os
import time


#Design CNN model 

class CNN(nn.Module):
    def __init__(self, size):
        '''size is the length of the metadata'''
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(5, 320, kernel_size = 6) #Input shd be (batchsize, channels, samples/rows)
        self.batchnorm1 = nn.BatchNorm1d(320) 
        self.pool1 = nn.MaxPool1d(5) #(Kernel size, stride) = (5,1)
        self.conv2 = nn.Conv1d(320,480, kernel_size = 5)
        self.pool2 = nn.MaxPool1d(5) 
        
        self.fc1 = nn.Linear(9120 + size, 256) #THE input will be larger based on how big metadata is (we input in forward func)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256,128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,1)
    
    def forward(self,x,meta): #x should be (batchsize, channels(columns), length). meta should be (batchsize, 78)
        out = self.batchnorm1(self.conv1(x)) #Since we take batchnorm before activation, after conv layer
        out = nn.functional.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = self.pool2(out)
        out = torch.flatten(out, start_dim=1) #Should return (batchsize, 480*39)
        #We normally don't add batchnorm to output layers; also here it seems to do worse if I add batchnorm in second conv layer too
        
        #Concatenate metadata here!
        out = torch.cat((out,meta),1)
        out = self.batchnorm2(self.fc1(out)) #Batchnorm before relu
        out = nn.functional.relu(out)
        out = self.batchnorm3(self.fc2(out)) #Batchnorm before relu
        out = nn.functional.relu(out)
        out = nn.functional.relu(self.fc3(out))
        
        return(out)
    


if __name__ == "__main__":
    

    #Create train, test datasets + dataloaders. Take subset of train/test based on chromosome as indicated in metadata file
    #samplename_locus.npy

    ohe_dir = "/nfs/turbo/dcmb-class/bioinf593/groups/group_05/STRonvoli/data/ohe/"
    meta_file = "/nfs/turbo/dcmb-class/bioinf593/groups/group_05/STRonvoli/data/metadata.tsv"

    dataset = dataset.STRDataset(ohe_dir, meta_file) 
    meta = pd.read_csv(meta_file, sep = '\t')
    train_indices = meta.index[meta["split"] == 0]
    test_indices = meta.index[meta["split"] == 1]

    trainset = Subset(dataset, train_indices)
    validset = Subset(dataset, test_indices)
    
    num = 6
    trainloader = DataLoader(trainset, batch_size = 64, shuffle=True, num_workers = num)
    validloader = DataLoader(validset, batch_size = 64, shuffle=True, num_workers = num)
        
    #Test if the loaders work
    '''
    for (batch_idx, batch) in enumerate(trainloader):
        #print(batch)
        (X, labels, meta) = batch
        print(X.shape)
        print(labels.shape)
        print(meta.shape)
    '''
    
    #Sweep through num workers to find ideal
    '''
    for num_workers in range(2, os.cpu_count(), 2):  
        sub = Subset(dataset, train_indices[0:1000])
        train_loader = DataLoader(sub,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
        start = time.time()
        for i, data in enumerate(train_loader):
            pass
        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
   
    '''
    
    
    #Train
    #Parameters (to change): 
    pos_count = 500
    meta_count = 79
    max_epochs = 10
    learning = 0.001

    net = CNN(meta_count).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning)

    losses = [] #training loss for each epoch (keys are epochs)
    losses_t = [] #Test loss over epochs
    for e in range(max_epochs):
        loss_l = 0 #Track validation loss in each epoch
        net.train() #Train model
        print(len(trainloader))
        start_time = time.time()
        total_loss = 0 #Track training loss in each epoch
        
        for (batch_idx, batch) in enumerate(trainloader):
            (X, labels, meta) = batch
            X = X.to(device)
            labels = labels.to(device)
            meta = meta.to(device)
            X = torch.transpose(X,1,2)
            X = X.float()
            meta = meta.float()
            labels = labels.float()
            labels = torch.reshape(labels, (labels.size(dim=0),1))
            
            optimizer.zero_grad()
            output = net(X, meta) #Will be (batch_size,1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            print(f'batch {batch_idx} done, loss is {loss}, time from start of batches is {time.time() - start_time}')
            
            total_loss += (loss.cpu().detach().numpy()) #Remove grad requirement + convert to np array to be able to plot
            #print(total_loss)
        losses.append(total_loss/len(trainloader))
        #print(losses[e])
        print(f'Epoch {e} done')
        
        #Save model 
        savedir = "/nfs/turbo/dcmb-class/bioinf593/groups/group_05/STRonvoli/models/batchnorm/"
        path = os.path.join(savedir, f'epoch-{e}-model.pth')
        torch.save({
            'epoch': e,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss}, path)  # saving model
        

        net.eval()
        for (Y,y,m) in validloader:
            Y = Y.to(device)
            y = y.to(device)
            m = m.to(device)
            Y = torch.transpose(Y,1,2)
            Y = Y.float()
            y = y.float()
            y = torch.reshape(y, (y.size(dim=0),1))
            m = m.float()
            
            out = net(Y,m)
            lossl = criterion(out, y)
            loss_l += (lossl.data.item())
        losses_t.append(loss_l/len(validloader))
        print(f'Validation loss after epoch {e} is')
        print(loss_l/len(validloader))

    
    plt.plot(losses)
    plt.plot(losses_t)
    plt.xlabel('Epochs')
    plt.ylabel('Average MSE loss over batches in each epoch')
    plt.title('Training and validation loss per epoch for baseline CNN with batchnorm')
    plt.legend(['Train','Valid'])
    plt.savefig('Average MSE loss over batches- baseline CNN with batchnorm.png')
    
    
    
    
