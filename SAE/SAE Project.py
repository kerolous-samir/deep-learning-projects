#Import Libraries
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.autograd import Variable

#Functions
def converter(data):
    new_data = []
    for user in range(1,no_users+1):
        id_movies = data[:,1][data[:,0] == user]
        id_ratings = data[:,2][data[:,0] == user]
        ratings = np.zeros(no_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

#Stacked AutoEncoders Model
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE,self).__init__()
        self.fc1 = nn.Linear(no_movies,20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,no_movies)
        self.activation = nn.Sigmoid()
    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


#Import Data
training_set = pd.read_csv("ml-100k/u2.base",delimiter='\t').values
test_set = pd.read_csv("ml-100k/u2.test",delimiter='\t').values

#Numbers of users & movies
no_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
no_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#Convert Data
training_set = converter(training_set)
test_set = converter(test_set)

#Convert to Float tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


#Train Model         
sae = SAE()
standard = nn.MSELoss()
optimizer = optim.Adam(sae.parameters(),weight_decay=0.5)
epochs = 200
for epoch in range(1,epochs+1):
    train_loss = 0
    s = 0.
    for user in range(no_users):
        input = Variable(training_set[user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = standard(output,target)
            mean_correcter = no_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_correcter)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
            


#Test Model            
test_loss = 0
s = 0.
for user in range(no_users):
    input = Variable(training_set[user]).unsqueeze(0)
    target = Variable(test_set[user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = standard(output,target)
        mean_correcter = no_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_correcter)
        s += 1.
print('loss: '+str(test_loss/s))