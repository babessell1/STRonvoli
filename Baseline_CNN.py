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



#Design CNN model 

class CNN(nn.Module):
    def __init__(self, size):
        '''size is the length of the metadata'''
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(5, 320, kernel_size = 6)#Output of this should be (batchsize, 320,1000-5) assuming no padding and kernel size = num columns 
        self.pool1 = nn.MaxPool1d(5) #(Kernel size, stride) = (5,1). So output of this should be (batchsize, 320, 995/5)
        self.conv2 = nn.Conv1d(320,480, kernel_size = 5) #Output should be (batchsize, 480,195)
        self.pool2 = nn.MaxPool1d(5) #Output shd be (batchsize, 480, 39)
        
        self.fc1 = nn.Linear((480*39)+size, 256) #THE input will be larger based on how big metadata is (we input in forward func)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)
    
    def forward(self,x,meta): #x should be (batchsize, channels(columns), length). meta should be (batchsize, 78)
        out = nn.functional.relu(self.conv1(x))
        out = self.pool1(out)
        out = nn.functional.relu(self.conv2(out))
        out = self.pool2(out)
 
        out = torch.flatten(out, start_dim=1) #Should return (batchsize, 480*39)
    
        #Concatenate metadata here!
        out = torch.cat((out,meta),0)
        
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        
        return(out)


if __name__ == "__main__":
    

    #Create train, test datasets + dataloaders 
    #samplename_locus.npy

    ohe_dir = "/nfs/turbo/dcmb-class/bioinf593/groups/group_05/STRonvoli/data/ohe/"
    meta_file = "/nfs/turbo/dcmb-class/bioinf593/groups/group_05/STRonvoli/data/metadata.tsv"

    dataset = dataset.STRDataset(ohe_dir, meta_file) 
    meta = pd.read_csv(meta_file, sep = '\t')
    train_indices = meta.index[meta["split"] == 0]
    test_indices = meta.index[meta["split"] == 1]

    trainloader = Subset(dataset, train_indices)
    validloader = Subset(dataset, test_indices)

    #Test if the loaders work
    for (batch_idx, batch) in enumerate(trainloader):
        (X, labels, meta) = batch
        print(X.shape)
        print(labels.shape)
        print(meta.shape)



    '''
    #Train
    #Parameters (to change): 
    pos_count = 1000
    meta_count = 88
    max_epochs = 10
    learning = 0.0001

    net = CNN(meta_count).to(device)
    criterion = nn.CrossEntropyLoss()  # Same as NLLLoss except NLLLoss takes output of softmax func
    optimizer = optim.Adam(net.parameters(), lr=learning)

    #losses = [] #Gives loss for each epoch (keys are epochs)
    losses_t = [] #Test loss over epochs
    for e in range(max_epochs):
        loss_l = 0
        batch_size = 0 #Track total number of samples.. should just be overall total but just to ensure
        net.train() #Train mode
        for (batch_idx, batch) in enumerate(trainloader):
            (X, labels, meta) = batch
            X = X.to(device)
            labels = labels.to(device)
            meta = meta.to(device)
            optimizer.zero_grad()
            output = net(X, meta) #Will be (batch_size, 10)
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
    '''
