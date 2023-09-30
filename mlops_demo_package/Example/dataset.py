# from typing import Any
from sklearn import datasets
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np

class IrisData:

    train_X: np.ndarray = None
    train_y: np.ndarray = None

    test_X: np.ndarray = None
    test_y: np.ndarray = None

    def __init__(self, test_size = 0.2, random_state = 8765):

        iris = datasets.load_iris()

        X = iris.data
        y = iris.target

        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)


class IrisDataset(Dataset):

    '''
    X -> tensor [N_elements, 4]
    '''
    def __init__(self, X: torch.Tensor, y: torch.Tensor):

        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size()[0]


    def __getitem__(self, index: int):
        return self.X[index], self.y[index]
