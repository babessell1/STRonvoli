import torch
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from torchvision import datasets 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch import optim
import STRDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


#Design CNN model 
# Use dialted model
class CNN(nn.Module):
    def __init__(self, size):
        '''size is the total number of positions (around start + end) we have'''
        super(CNN, self).__init__(size)
        self.conv1 = nn.Conv1d(5, 320, kernel_size = 6,
                               dilation=10) # (batchsize, 320, 945)
        self.pool1 = nn.MaxPool1d(5) # (batchsize, 320, 189)
        self.conv2 = nn.Conv1d(320,480, kernel_size = 5) # (batchsize, 480, 185)
        self.pool2 = nn.MaxPool1d(5) # (batchsize, 480, 37)
        
        self.fc1 = nn.Linear((480*37)+size, 256) #THE input will be larger based on how big metadata is (we input in forward func)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)
    
    def forward(self,x,meta): #x should be (batchsize, channels(columns), length). meta should be (dims,)
        out = nn.functional.relu(self.conv1(x))
        out = self.pool1(out)
        out = nn.functional.relu(self.conv2(out))
        out = self.pool2(out)
 
        out = torch.flatten(out, start_dim=1) #Should return (batchsize, 480*37)
    
        #Concatenate metadata here!
        
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        
        return(out)


    
if __name__ == "__main__":
    

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
    
