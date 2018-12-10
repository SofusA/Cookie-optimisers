import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np


#Params
batchsize = 100

# Load data
from scripts import movieload_iter
n_users, n_items = movieload_iter.MovieLensDataset().items()
train_loader, val_loader, test_loader = movieload_iter.getLoaders(batchsize = batchsize, shuffle = True, sizes = [0.7, 0.2, 0.1])

# Model
class CFNN(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, n_hidden=10):
        super(CFNN, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.lin1 = nn.Linear(emb_size*2, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.drop1 = nn.Dropout(0.1)
        
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        x = F.relu(torch.cat([U, V], dim=1))
        x = self.drop1(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

## Training loop
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (users, items, ratings) in enumerate(train_loader):
        users = users.long().cuda()
        items = items.long().cuda()
        ratings = ratings.float().cuda()
        ratings = ratings.unsqueeze(1)
        optimizer.zero_grad()
        output = model(users, items)
        loss = criterion(output, ratings)
        loss.backward()
        optimizer.step()
        #if batch_idx % len(train_loader.dataset)/10 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(users), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))

def validate(model, val_loader, criterion, epoch):
    model.eval()
    outputlist = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (users, items, ratings) in val_loader:
            users = users.long().cuda()
            items = items.long().cuda()
            ratings = ratings.float().cuda()
            ratings = ratings.unsqueeze(1)
            output = model(users, items)
            #output = torch.from_numpy(np.array([2.7]*len(ratings))).float().cuda()
            #output = output.unsqueeze(1)
            #print(output)
            
            outputlist += [output]
            test_loss += criterion(output, ratings).item() # sum up batch loss

    test_loss /= len(val_loader)
    print(f'\nEpoch {epoch}: Test average loss: {test_loss:.2f}')
    print(f'{users.data[0]},  {items.data[0]} -> {ratings.data[0].data[0]:.1f}. Prediction: {output.data[0].data[0]:.2f}')

def trainLoop(epochs, lr=0.001, wd = 1e-6):
    # Define model    
    model = CFNN(n_users, n_items).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = wd)
    
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch)
        validate(model, val_loader, criterion, epoch)