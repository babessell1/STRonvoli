import torch
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from torchvision import datasets 
from torchvision.transforms import ToTensor
<<<<<<< HEAD
from torch.utils.data import DataLoader, random_split
=======
from torch.utils.data import DataLoader, Subset
>>>>>>> main
from torch import nn
from torch import optim
import STRDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
<<<<<<< HEAD

metadata_file = '/nfs/turbo/dcmb-class/bioinf593/groups/group_05/output/trgt/repeatregion_10_parsedvcf.txt'
ohe_dir = '/nfs/turbo/dcmb-class/bioinf593/groups/group_05/output/depth'
#Custom Dataset class
data = STRDataset(ohe_dir, metadata_file)
train_ratio = 0.2
train_size = len(data) * train_ratio
test_size = len(data) - train_size


#Custom Dataloader class for train/test
generator = torch.Generator().manual_seed(1)
train_data, test_data = random_split(data, [train_size, test_size], generator=generator)

batch_size = 64
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
=======
import sys
import STRDataset as dataset
import os
import time
>>>>>>> main


#Design CNN model 
# Use dialted model
class CNN(nn.Module):
    def __init__(self, size):
<<<<<<< HEAD
        '''size is the total number of positions (around start + end) we have'''
        super(CNN, self).__init__(size)
        self.conv1 = nn.Conv1d(5, 320, kernel_size = 6,
                               dilation=10) # (batchsize, 320, 945)
        self.pool1 = nn.MaxPool1d(5) # (batchsize, 320, 189)
        self.conv2 = nn.Conv1d(320,480, kernel_size = 5) # (batchsize, 480, 185)
        self.pool2 = nn.MaxPool1d(5) # (batchsize, 480, 37)
        
        self.fc1 = nn.Linear((480*37)+size, 256) #THE input will be larger based on how big metadata is (we input in forward func)
=======
        '''size is the length of the metadata'''
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(5, 320, kernel_size = 6)#If input rows = 1000, output of this should be (batchsize, 320,1000-5) assuming no padding and kernel size = num columns 
        self.pool1 = nn.MaxPool1d(5) #(Kernel size, stride) = (5,1). So output of this should be (batchsize, 320, 995/5)
        self.conv2 = nn.Conv1d(320,480, kernel_size = 5) #Output should be (batchsize, 480,195)
        self.pool2 = nn.MaxPool1d(5) #Output shd be (batchsize, 480, 39)
        
        self.fc1 = nn.Linear(9120 + size, 256) #THE input will be larger based on how big metadata is (we input in forward func)
>>>>>>> main
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)
    
    def forward(self,x,meta): #x should be (batchsize, channels(columns), length). meta should be (batchsize, 78)
        out = nn.functional.relu(self.conv1(x))
        out = self.pool1(out)
        out = nn.functional.relu(self.conv2(out))
        out = self.pool2(out)
 
<<<<<<< HEAD
        out = torch.flatten(out, start_dim=1) #Should return (batchsize, 480*37)
    
        #Concatenate metadata here!
=======
        out = torch.flatten(out, start_dim=1) #Should return (batchsize, 480*39)
        #print(out.shape) #(4,9120)
>>>>>>> main
        
        #Concatenate metadata here!
        out = torch.cat((out,meta),1)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        
        return(out)
    


if __name__ == "__main__":
    

<<<<<<< HEAD
    #Create train, valid, test datasets + dataloaders 

        
        
        
    #Train

    #Parameters (to change): 
    pos_count = 1000
    max_epochs = 50
    learning = 0.0001

    net = CNN(pos_count).to(device)
    criterion = nn.CrossEntropyLoss()  # Same as NLLLoss except NLLLoss takes output of softmax func
    optimizer = optim.Adam(net.parameters(), lr=learning)

    #losses = [] #Gives loss for each epoch (keys are epochs)
    losses_t = [] #Test loss over epochs
    for e in range(max_epochs):
        loss_l = 0
        net.train() #Train mode
        for (batch_idx, batch) in enumerate(trainloader):
            (X, labels) = batch
            X = X.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = net(X) #Will be (batch_size, 10)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            #loss_l.append(loss.cpu().detach().numpy()) #Remove grad requirement + convert to np array to be able to plot
            #batch_size += len(labels)
        #losses.append(sum(loss_l)/batch_size)
        #print(losses[e])
        print(f'Epoch {e} done')
        
        net.eval()
        for (t,y) in validloader:
            t = t.to(device)
            y = y.to(device)
            out = net(t)
            lossl = criterion(out, y)
            loss_l += (lossl.data.item())
        losses_t.append(loss_l/len(validloader))
        print(loss_l/len(validloader))
        

    plt.plot(losses_t)
    plt.xlabel('Epochs')
    plt.ylabel('Average cross entropy loss over batches in each epoch')
    plt.title('Validation Loss over epochs for MNIST classification using CNN')
    plt.savefig('Average Validation loss (across all batches) over epochs')
=======
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
            #print(f'batch {batch_idx} done, loss is {loss}, time from start of batches is {time.time() - start_time}')
            
            total_loss += (loss.cpu().detach().numpy()) #Remove grad requirement + convert to np array to be able to plot
            #print(total_loss)
        losses.append(total_loss/len(trainloader))
        #print(losses[e])
        print(f'Epoch {e} done')
        
        #Save model 
        savedir = "/nfs/turbo/dcmb-class/bioinf593/groups/group_05/STRonvoli/models/"
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
            y = torch.reshape(y, (y, size(dim=0),1))
            m = m.float()
            
            out = net(Y,m)
            lossl = criterion(out, y)
            loss_l += (lossl.data.item())
        losses_t.append(loss_l/len(validloader))
        print(f'Validation loss after epoch {e} is')
        print(loss_l/len(validloader))


    figure(0)
    plt.plot(losses_t)
    plt.xlabel('Epochs')
    plt.ylabel('Average MSE loss over batches in each epoch')
    plt.title('Validation Loss over epochs for STR prediction using CNN')
    plt.savefig('Average Validation loss (across all batches) over epochs')
    
    figure(1)
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Average MSE loss over batches in each epoch')
    plt.title('Training Loss over epochs for STR prediction using CNN')
    plt.savefig('Average Training loss (across all batches) over epochs')
    
    
    
>>>>>>> main
    
