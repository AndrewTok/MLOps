import dataset
import models
import pandas as pd
import torch
from train import TrainRunner


def infer(model_file: str, data_file: str):
    data = dataset.IrisData.load_from_file(data_file)
    net = models.SimpleNet()
    net.load_state_dict(torch.load(model_file))
    train_runner = TrainRunner(data, net)

    accuracy, pred = train_runner.test_current_model()
    true = data.test_y

    df = pd.DataFrame({'Predict': pred, 'True': true})
    df.to_csv('predictions.csv')

    print('accuracy: ' + str(accuracy))


if __name__ == '__main__':
    infer('trained_model_params', 'dataset')
