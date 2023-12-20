import os

import mlflow.onnx
import onnx
import onnxruntime as ort
import torch
from dvc import api as DVC
from mlflow.models import infer_signature

from .config import Params, load_cfg
from .models import SimpleNet


def start_mlflow_server():
    import webbrowser

    import mlflow.cli

    cfg: Params = load_cfg()
    webbrowser.open(cfg.artifacts.mlflow_server_address, new=2)

    mlflow.cli.server(
        [
            '--backend-store-uri',
            '.\\logs\\mlflow_runs\\',
        ]
    )

    pass


def load_data(
    url: str = 'https://github.com/AndrewTok/ml-ops',  # './'
):
    # 'https://github.com/AndrewTok/ml-ops'
    fs = DVC.DVCFileSystem(
        url,
        rev='main',
    )

    tracked_lst = fs.find("/", detail=False, dvc_only=True)
    for tracked in tracked_lst:
        path = tracked[1:]
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fs.get_file(path, path)


def export_to_onnx(model: SimpleNet, cfg: Params):
    model.eval()
    dummy_input = SimpleNet.get_dummy_input()
    torch.onnx.export(
        model,
        dummy_input,
        Params.get_save_model_onnx_path(cfg.model),
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=[cfg.onnx.feature_name],
        output_names=[cfg.onnx.pred_name],
        dynamic_axes={
            cfg.onnx.feature_name: {0: "BATCH_SIZE"},
            cfg.onnx.pred_name: {0: "BATCH_SIZE"},
        },
    )


def save_mlflow_model(cfg: Params):
    dummy_input = SimpleNet.get_dummy_input()

    onnx_model = onnx.load_model(cfg.get_serving_onnx_path())

    model = SimpleNet(
        hidden_1=cfg.model.hidden_1_size,
        hidden_2=cfg.model.hidden_2_size,
    )

    model.eval()

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


def make_outputs_for_triton_testing(onnx_path, inputs):
    ort_sess = ort.InferenceSession(onnx_path)
    outputs = ort_sess.run(None, {'IRIS_FEATURES': inputs})
    return outputs
