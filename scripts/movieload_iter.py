#import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

def condence(nparr):
    uniq = np.unique(nparr)
    name2idx = {o:i for i,o in enumerate(uniq)}
    return np.array([name2idx[o] for o in nparr])

class MovieLensDataset(Dataset):
    def __init__(self, filename='datasets/movielens-small/ratings.csv'):
        self.rawdata = pd.read_csv(filename)
        self.rawdata["userId"] = condence(self.rawdata["userId"].values)
        self.rawdata["movieId"] = condence(self.rawdata["movieId"].values)

    def __len__(self):
        return len(self.rawdata)

    def __getitem__(self, idx):
        idx = idx.item()
        users = self.rawdata.iloc[idx, 0]#.astype(int)
        items = self.rawdata.iloc[idx, 1]#.astype(int)
        ratings = self.rawdata.iloc[idx, 2]#.astype(float)
        
        return (users, items, ratings)
    
    def items (self):
        n_users = self.rawdata["userId"].nunique()
        n_items = self.rawdata["movieId"].nunique()
        
        return [n_users, n_items]

def getLoaders(batchsize = 100, shuffle = True, sizes = [0.7, 0.2, 0.1]):
    dataset = MovieLensDataset()
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    
    return [DataLoader(data, batch_size = batchsize, shuffle = shuffle) for data in [train_data, val_data, test_data]]