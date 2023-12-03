import os

import git
import numpy as np
import pytorch_lightning as pl
import torch
from dvc import api as DVC

# from mlflow.server import get_app_client
from sklearn.metrics import accuracy_score

from .config import Params, load_cfg
from .data_module import IrisDataModule
from .dataset import IrisData  # , IrisDataset
from .models import SimpleNet

# from torch.utils.data import DataLoader


class IrisModule(pl.LightningModule):
    def __init__(
        self,
        cfg: Params,
        git_commit_id: str,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.model = SimpleNet(
            hidden_1=cfg.model.hidden_1_size,
            hidden_2=cfg.model.hidden_2_size,
        )

        self.loss_f = torch.nn.CrossEntropyLoss()
        self.lr = cfg.training.learning_rate  # 5e-2
        self.weight_decay = cfg.training.weight_decay

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        (
            x,
            y_gt,
        ) = batch
        y_pr = self.model(x)

        loss = self.loss_f(
            y_pr,
            y_gt.to(torch.long),
        )  # y_gt.float()

        acc = IrisModule.compute_acc(
            y_gt,
            y_pr,
        )  # self.metric(y_pr, y_gt)

        metrics = {
            'loss': loss.detach(),
            'accuracy': acc,
            'mistakes': (1 - acc)*y_gt.shape[0]
        }
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        # self.train_acc.append(acc.detach())  # optional

        return loss

    @staticmethod
    def compute_acc(
        y_true: torch.Tensor,
        pred_probas: torch.Tensor,
    ):
        return accuracy_score(
            y_true.detach().numpy(),
            np.argmax(
                pred_probas.detach().numpy(),
                axis=1,
            ),
        )

    def configure_optimizers(
        self,
    ):
        """Define optimizers and LR schedulers."""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )  # ,
        return optimizer

    def save_model(
        self,
        filename,
    ):
        torch.save(
            self.model.state_dict(),
            filename,
        )


def load_data(url: str = './',):
    # 'https://github.com/AndrewTok/ml-ops'
    fs = DVC.DVCFileSystem(
        url,
        rev='main',
    )

    tracked_lst = fs.find(
        "/",
        detail=False,
        dvc_only=True,
    )
    for tracked in tracked_lst:
        path = tracked[1:]
        if os.path.exists(path):
            continue
        fs.get_file(
            path,
            path,
        )


def train():
    load_data()

    cfg = load_cfg()

    repo = git.Repo(search_parent_directories=True)

    data = IrisData.load_from_file('data/dataset.npz')

    data_module = IrisDataModule(
        data,
        batch_size=cfg.training.batch_size,
    )
    train_module = IrisModule(
        cfg,
        git_commit_id=repo.head.object.hexsha,
    )

    _logger = pl.loggers.MLFlowLogger(
        experiment_name=cfg.artifacts.experiment_name,
        tracking_uri=cfg.artifacts.log_uri,  # "file:./logs/mlflow_runs",
        # save_dir = "./logs/mlruns"
    )

    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=cfg.training.epochs,
        logger=_logger,
        log_every_n_steps=cfg.artifacts.checkpoint.every_n_train_steps,
    )

    train_loader = data_module.train_dataloader()

    trainer.fit(
        train_module,
        train_loader,
    )

    # loss_history, acc_history = trainer.train(batch_size=64, epch_num=32)
    train_module.save_model(Params.get_model_save_path(cfg.model))

    export_to_onnx(
        model=train_module.model,
        cfg=cfg,
    )
    


def start_mlflow_server():
    import webbrowser

    import mlflow.cli

    cfg: Params = load_cfg()
    webbrowser.open(
        cfg.artifacts.mlflow_server_address,
        new=2,
    )

    mlflow.cli.server(
        [
            '--backend-store-uri',
            '.\\logs\\mlflow_runs\\',
        ])

    pass


def export_to_onnx(
    model: SimpleNet,
    cfg: Params,
):
    # import onnxruntime as ort

    model.eval()

    dummy_input = torch.randn(1, 4)

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


if __name__ == '__main__':
    train()
