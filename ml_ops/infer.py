import numpy as np
import pandas as pd
import torch

from .data_module import IrisDataModule
from .dataset import IrisData
from .models import SimpleNet
from .train import IrisModule  # TrainRunner


def test_current_model(model, test_X, test_y):
    pred_probas = model(test_X)
    return IrisModule.compute_acc(test_y, pred_probas), np.argmax(
        pred_probas.detach().numpy(), axis=1
    )


def infer(model_file: str = 'trained_model_params.pt', data_file: str = 'data/dataset.npz'):
    data = IrisData.load_from_file(data_file)
    net = SimpleNet()
    net.load_state_dict(torch.load(model_file))

    data_module = IrisDataModule(data)

    accuracy, pred = test_current_model(
        net, data_module.test_X, data_module.test_y
    )
    true = data.test_y

    df = pd.DataFrame({'Predict': pred, 'True': true})
    df.to_csv('predictions.csv')

    print('accuracy: ' + str(accuracy))


if __name__ == '__main__':
    infer('trained_model_params.pt', 'data/dataset.npz')
