import pandas as pd
import torch

from .dataset import IrisData
from .models import SimpleNet
from .train import TrainRunner


def infer(model_file: str, data_file: str):
    data = IrisData.load_from_file(data_file)
    net = SimpleNet()
    net.load_state_dict(torch.load(model_file))
    train_runner = TrainRunner(data, net)

    accuracy, pred = train_runner.test_current_model()
    true = data.test_y

    df = pd.DataFrame({'Predict': pred, 'True': true})
    df.to_csv('predictions.csv')

    print('accuracy: ' + str(accuracy))


if __name__ == '__main__':
    infer('trained_model_params.pt', 'dataset')
