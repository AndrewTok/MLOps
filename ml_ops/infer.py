import numpy as np
import pandas as pd
import torch

from .data_module import IrisDataModule
from .dataset import IrisData
from .models import SimpleNet
from .train import IrisModule  # TrainRunner

from .config import Params


def test_current_model(model, test_X, test_y):
    pred_probas = model(test_X)
    return IrisModule.compute_acc(test_y, pred_probas), np.argmax(
        pred_probas.detach().numpy(), axis=1
    )


def infer(cfg: Params): #model_file: str = 'trained_model_params.pt', data_file: str = 'data/dataset.npz'
    data = IrisData.load_from_file(cfg.data.path) #data_file
    net = SimpleNet(hidden_1=cfg.model.hidden_1_size, hidden_2=cfg.model.hidden_2_size)
    net.load_state_dict(torch.load(Params.get_model_save_path(cfg.model))) #model_file

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
