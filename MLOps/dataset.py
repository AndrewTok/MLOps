import numpy as np
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class IrisData:
    train_X: np.ndarray = None
    train_y: np.ndarray = None

    test_X: np.ndarray = None
    test_y: np.ndarray = None

    def __init__(self, train_X: np.ndarray, train_y: np.ndarray, test_X:np.ndarray, test_y:np.ndarray):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def build(test_size = 0.2, random_state = 8765):
        iris = datasets.load_iris()

        X = iris.data
        y = iris.target

        (
            train_X,
            test_X,
            train_y,
            test_y,
        ) = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return IrisData(train_X, train_y, test_X, test_y)

    def save_to_file(self, filename: str):
        # total_dataset = np.stack(self.train_X, self.train_y, self.test_X, self.test_y])
        np.savez(filename, self.train_X, self.train_y, self.test_X, self.test_y)

    def load_from_file(filename: str):
        npz_file = np.load(filename + '.npz')
        files = npz_file.files
        train_X, train_y, test_X, test_y = npz_file[files[0]], npz_file[files[1]], npz_file[files[2]], npz_file[files[3]]
        return IrisData(train_X, train_y, test_X, test_y)


class IrisDataset(Dataset):

    """
    X -> tensor [N_elements, 4]
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]
