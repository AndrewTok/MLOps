import pytorch_lightning as pl
import torch

from .dataset import IrisData, IrisDataset


class IrisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: IrisData,
        batch_size=16,
    ):
        super().__init__()

        self.batch_size = batch_size

        self.train_X = torch.from_numpy(data.train_X).to(torch.float)
        self.train_y = torch.from_numpy(data.train_y).to(torch.float)
        self.test_X = torch.from_numpy(data.test_X).to(torch.float)
        self.test_y = torch.from_numpy(data.test_y).to(torch.float)

        self.train_set = IrisDataset(
            self.train_X,
            self.train_y,
        )
        self.test_set = IrisDataset(
            self.test_X,
            self.test_y,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            drop_last=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=1,
        )
