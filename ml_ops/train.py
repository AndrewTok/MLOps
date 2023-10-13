import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from .dataset import IrisData, IrisDataset
from .models import SimpleNet


class TrainRunner:
    data: IrisData

    def __init__(self, data: IrisData, model: SimpleNet = None):
        self.train_X = torch.from_numpy(data.train_X).to(torch.float)
        self.train_y = torch.from_numpy(data.train_y).to(torch.float)
        self.test_X = torch.from_numpy(data.test_X).to(torch.float)
        self.test_y = torch.from_numpy(data.test_y).to(torch.float)

        if model is None:
            self.model = SimpleNet()
        else:
            self.model = model

    def compute_acc(self, y_true: torch.Tensor, pred_probas: torch.Tensor):
        return accuracy_score(
            y_true.detach().numpy(),
            np.argmax(pred_probas.detach().numpy(), axis=1),
        )

    def train(self, batch_size: int, epch_num: int):
        train_dataset = IrisDataset(self.train_X, self.train_y)
        train_dataloader = DataLoader(train_dataset, batch_size)
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=5e-2, weight_decay=1e-1
        )
        self.model.train()
        loss_history = []
        acc_history = []
        for epch in range(epch_num):
            for X, Y in iter(train_dataloader):
                optimizer.zero_grad()

                Y_pred_probas = self.model(X)
                loss = loss_func(Y_pred_probas, Y.to(torch.long))

                loss.backward()

                optimizer.step()

                loss_history.append(loss.detach())

                acc_history.append(self.compute_acc(Y, Y_pred_probas))

        return loss_history, acc_history

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def test_current_model(self):
        pred_probas = self.model(self.test_X)
        return self.compute_acc(self.test_y, pred_probas), np.argmax(
            pred_probas.detach().numpy(), axis=1
        )


def train():
    data = IrisData.build(test_size=0.4)
    trainer = TrainRunner(data)

    data.save_to_file('dataset')

    loss_history, acc_history = trainer.train(batch_size=64, epch_num=32)
    trainer.save_model('trained_model_params.pt')


if __name__ == '__main__':
    train()
