#Import Libraries
import pandas as pd
import numpy as np
import torch

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


#Restricted Boltzmann Machine Model
class RBM():
    def __init__(self,nh,nv):
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
    def sample_h(self,x):
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        prob_h_given_v = torch.sigmoid(activation)
        return prob_h_given_v , torch.bernoulli(prob_h_given_v)
    def sample_v(self,y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        prob_v_given_h = torch.sigmoid(activation)
        return prob_v_given_h , torch.bernoulli(prob_v_given_h)
    def train(self,v0,vk,ph0,phk):
        self.W += (torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((ph0-phk),0)
        

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

#Change to binary
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


#Train Model
rbm = RBM(100,len(training_set[0]))
batch_size = 32

for epoch in range(1,11):
    train_loss = 0
    s = 0.
    for user in range(0,no_users-batch_size,batch_size):
        vk = training_set[user:user+batch_size]
        v0 = training_set[user:user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
            

#Test Model
test_loss = 0
s = 0.
for user in range(no_users):
    v = training_set[user:user+1]
    vt = test_set[user:user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0])) # for RMSE np.sqrt(torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))**2)
        s += 1.
print('test loss: '+str(test_loss/s))

