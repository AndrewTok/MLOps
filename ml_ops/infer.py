import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from .config import Params
from .data_module import IrisDataModule
from .dataset import IrisData
from .models import SimpleNet
from .train import IrisModule


def compute_acc(
    y_true: torch.Tensor,
    pred_probas: torch.Tensor,
):
    return accuracy_score(
        y_true,
        np.argmax(
            pred_probas,
            axis=1,
        ),
    )


def save_mlflow_model(
    cfg: Params,
    dummy_input,
):
    import mlflow
    import onnx
    from mlflow.models import infer_signature

    onnx_model = onnx.load_model(Params.get_model_onnx_path(cfg.model))

    model = SimpleNet(
        hidden_1=cfg.model.hidden_1_size,
        hidden_2=cfg.model.hidden_2_size,
    )

    model.eval()

    mlflow.set_tracking_uri(cfg.artifacts.log_uri)
    mlflow.set_experiment(experiment_name=cfg.artifacts.experiment_name)

    with mlflow.start_run() as run:
        print(run.info.artifact_uri)
        signature = infer_signature(
            dummy_input.numpy(),
            model(dummy_input).detach().numpy(),
        )
        model_info = mlflow.onnx.log_model(
            onnx_model,
            "model",
            signature=signature,
        )

    return model_info


def get_mlflow_model(
    model_info,
):
    import mlflow

    onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    return onnx_pyfunc
    # predictions = onnx_pyfunc.predict(dummy_input.numpy())
    # print(predictions)


def test_current_model(
    model,
    test_X,
    test_y,
):
    pred_probas = model(test_X)
    return IrisModule.compute_acc(
        test_y,
        pred_probas,
    ), np.argmax(
        pred_probas.detach().numpy(),
        axis=1,
    )


def get_test_info(
    pred_probas,
    test_y,
):
    return compute_acc(
        test_y,
        pred_probas,
    ), np.argmax(
        pred_probas,
        axis=1,
    )


def infer_onnx(
    cfg: Params,
):
    model_info = save_mlflow_model(
        cfg,
        torch.rand(
            1,
            4,
        ),
    )
    onnx_pyfunc = get_mlflow_model(model_info=model_info)
    data = IrisData.load_from_file(cfg.data.path)  # data_file

    # data_module = IrisDataModule(data)

    pred_probas = onnx_pyfunc.predict(data.test_X.astype('float32'))[
        cfg.onnx.pred_name
    ]

    true = data.test_y

    # print(pred_probas)

    (
        accuracy,
        pred,
    ) = get_test_info(
        pred_probas,
        true,
    )

    df = pd.DataFrame(
        {
            'Predict': pred,
            'True': true,
        }
    )
    df.to_csv('predictions.csv')

    print('accuracy: ' + str(accuracy))


def infer(
    cfg: Params,
):
    data = IrisData.load_from_file(cfg.data.path)  # data_file
    net = SimpleNet(
        hidden_1=cfg.model.hidden_1_size,
        hidden_2=cfg.model.hidden_2_size,
    )
    net.load_state_dict(
        torch.load(Params.get_model_save_path(cfg.model))
    )  # model_file

    data_module = IrisDataModule(data)

    (
        accuracy,
        pred,
    ) = test_current_model(
        net,
        data_module.test_X,
        data_module.test_y,
    )
    true = data.test_y

    df = pd.DataFrame(
        {
            'Predict': pred,
            'True': true,
        }
    )
    df.to_csv('predictions.csv')

    print('accuracy: ' + str(accuracy))


if __name__ == '__main__':
    infer(
        'trained_model_params.pt',
        'data/dataset.npz',
    )
