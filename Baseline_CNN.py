import torch
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from torchvision import datasets 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Custom Dataset class


#Custom Dataloader class for train/test



#Design CNN model 

#*I changed parameters based on col length 5 - kernel size 5, stride of max pooling 1. 
#Q: I don't think (horizontal/over columns) max pooling makes sense in this context...
#In baseline CNN, we are still adding the metadata before FC? Will need to change below dims 

class CNN(nn.Module):
    def __init__(self, size):
        '''size is the total number of positions (around start + end) we have'''
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size = 5) #Output of this should be (batchsize, 12,size-5+1, 5) assuming no padding and kernel size = num columns 
        self.pool1 = nn.MaxPool2d(2, 1) #(Kernel size, stride). So output of this should be (batchsize, 12, size-5, 4) 
        self.conv2 = nn.Conv2d(12,32, kernel_size = 4) #Output should be (batchsize, 32, (size-5)-5+1, 4)
        self.pool2 = nn.MaxPool2d(2,1) #Output shd be (batchsize, 32, size-8, 3)
        self.fc1 = nn.Linear(32*(size-8)*3, 256) #THE input will be larger based on how big metadata is (we input in forward func)
        self.fc2 = nn.Linear(256,)
        #self.fc3 = nn.Linear(256, 10) #10 classes; output layer 
    
    def forward(self,x,meta): #x should be (batchsize, channels, height, width). meta should be (dims,)
        out = nn.functional.relu(self.conv1(x))
        out = self.pool1(out)
        out = nn.functional.relu(self.conv2(out))
        out = self.pool2(out)
 
        out = torch.flatten(out, start_dim=1) #Should return (batchsize, 64*4*4)
        #Concatenate metadata here!
        
        out = nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        #out = self.fc3(out) 
        
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
    batch_size = 0 #Track total number of samples.. should just be overall total but just to ensure
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
    
