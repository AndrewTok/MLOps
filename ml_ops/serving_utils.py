import torch




from .config import Params, load_cfg
from .models import SimpleNet
import onnx

import mlflow.onnx
from mlflow.models import infer_signature

def start_mlflow_server():
    import webbrowser

    import mlflow.cli

    cfg: Params = load_cfg()
    webbrowser.open(cfg.artifacts.mlflow_server_address, new=2)

    mlflow.cli.server(
        [
            '--backend-store-uri',
            '.\\logs\\mlflow_runs\\',
        ])

    pass


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
            cfg.onnx.pred_name: {0: "BATCH_SIZE"}})



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