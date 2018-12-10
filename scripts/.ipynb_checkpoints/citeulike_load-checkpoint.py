import pandas as pd
import numpy as np

# Stolen from FastAI
def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

def load(filename, trainSize, valSize):
    df = pd.read_csv(filename)
    train, validate, test = np.split(df.sample(frac=1), [int(trainSize*len(df)), int((trainSize+valSize)*len(df))])
    
    #train = encode_data(train)
    #validate = encode_data(validate, train)
    #test = encode_data(test, train)
    
    return train, validate, test


