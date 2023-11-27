import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score

from .data_module import IrisDataModule
from .dataset import IrisData  # , IrisDataset
from .models import SimpleNet

from dvc import api as DVC

import os

# from torch.utils.data import DataLoader


class IrisModule(pl.LightningModule):
    def __init__(self, model: SimpleNet = None):
        super().__init__()

        if model is None:
            self.model = SimpleNet()
        else:
            self.model = model

        self.loss_f = torch.nn.CrossEntropyLoss()
        self.lr = 5e-2
        self.weight_decay = 1e-1

    def training_step(self, batch, batch_idx):
        x, y_gt = batch
        y_pr = self.model(x)

        loss = self.loss_f(y_pr, y_gt.to(torch.long))  # y_gt.float()

        acc = IrisModule.compute_acc(y_gt, y_pr)  # self.metric(y_pr, y_gt)

        metrics = {'loss': loss.detach(), 'accuracy': acc}
        self.log_dict(
            metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        # self.train_acc.append(acc.detach())  # optional

        return loss

    @staticmethod
    def compute_acc(y_true: torch.Tensor, pred_probas: torch.Tensor):
        return accuracy_score(
            y_true.detach().numpy(),
            np.argmax(pred_probas.detach().numpy(), axis=1),
        )

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )  # ,
        return optimizer

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)


# class TrainRunner:
#     data: IrisData

#     def __init__(self, data: IrisData, model: SimpleNet = None):
#         self.train_X = torch.from_numpy(data.train_X).to(torch.float)
#         self.train_y = torch.from_numpy(data.train_y).to(torch.float)
#         self.test_X = torch.from_numpy(data.test_X).to(torch.float)
#         self.test_y = torch.from_numpy(data.test_y).to(torch.float)

#         if model is None:
#             self.model = SimpleNet()
#         else:
#             self.model = model

#     def compute_acc(self, y_true: torch.Tensor, pred_probas: torch.Tensor):
#         return accuracy_score(
#             y_true.detach().numpy(),
#             np.argmax(pred_probas.detach().numpy(), axis=1),
#         )

#     def train(self, batch_size: int, epch_num: int):
#         train_dataset = IrisDataset(self.train_X, self.train_y)
#         train_dataloader = DataLoader(train_dataset, batch_size)
#         loss_func = torch.nn.CrossEntropyLoss()
#         optimizer = torch.optim.SGD(
#             self.model.parameters(), lr=5e-2, weight_decay=1e-1
#         )
#         self.model.train()
#         loss_history = []
#         acc_history = []
#         for epch in range(epch_num):
#             for X, Y in iter(train_dataloader):
#                 optimizer.zero_grad()

#                 Y_pred_probas = self.model(X)
#                 loss = loss_func(Y_pred_probas, Y.to(torch.long))

#                 loss.backward()

#                 optimizer.step()

#                 loss_history.append(loss.detach())

#                 acc_history.append(self.compute_acc(Y, Y_pred_probas))

#         return loss_history, acc_history

#     def save_model(self, filename):
#         torch.save(self.model.state_dict(), filename)

#     def test_current_model(self):
#         pred_probas = self.model(self.test_X)
#         return self.compute_acc(self.test_y, pred_probas), np.argmax(
#             pred_probas.detach().numpy(), axis=1
#         )


def load_data(url: str = './'): #'https://github.com/AndrewTok/ml-ops'
    fs = DVC.DVCFileSystem(url, rev='main')

    # print(fs.find("/", detail=False, dvc_only=True))
    
    # def get_filename(path: str):
    #     idx = path.rfind('/')
    #     return path[idx+1:]

    tracked_lst = fs.find("/", detail=False, dvc_only=True)
    for tracked in tracked_lst:       
        path = tracked[1:]
        if os.path.exists(path):
            continue
        fs.get_file(path, path)

    # fs.get('data', 'data', recursive=True)

    pass

def train():
    # data = IrisData.build(test_size=0.4)
    # data.save_to_file('dataset')

    load_data()

    data = IrisData.load_from_file('data/dataset.npz')

    data_module = IrisDataModule(data, batch_size=64)
    train_module = IrisModule()

    trainer = pl.Trainer(
        accelerator='cpu', devices=1, max_epochs=32, log_every_n_steps=10
    )

    train_loader = data_module.train_dataloader()

    trainer.fit(train_module, train_loader)

    # loss_history, acc_history = trainer.train(batch_size=64, epch_num=32)
    train_module.save_model('trained_model_params.pt')


if __name__ == '__main__':
    train()
