import os
from dataclasses import dataclass

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


@dataclass
class Data:
    name: str
    path: str


@dataclass
class Model:
    name: str
    save_dir: str
    hidden_1_size: int
    hidden_2_size: int


@dataclass
class Training:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float


@dataclass
class Checkpoint:
    use: bool
    dirpath: str
    filename: str
    monitor: str
    save_top_k: int
    every_n_train_steps: int
    every_n_epochs: int


@dataclass
class Artifacts:
    experiment_name: str
    log_uri: str
    checkpoint: Checkpoint
    mlflow_server_address: str
    # mlflow_models_uri: str


@dataclass
class Onnx:
    feature_name: str
    pred_name: str


@dataclass
class Serving:
    onnx_model_path: str
    predictions_save_path: str


@dataclass
class Params:
    data: Data
    model: Model
    training: Training
    artifacts: Artifacts
    onnx: Onnx
    serving: Serving

    @staticmethod
    def get_model_save_path(
        model: Model,
    ):
        return os.path.join(model.save_dir, 'trained_' + model.name + '.pt')

    @staticmethod
    def get_save_model_onnx_path(
        model: Model,
    ):
        return os.path.join(model.save_dir, model.name + '.onnx')

    def get_serving_onnx_path(self):
        if self.serving.onnx_model_path == "auto":
            return Params.get_save_model_onnx_path(self.model)
        return self.serving.onnx_model_path


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


def check_cfg():
    initialize(version_base="1.3", config_path="../configs")

    cfg: Params = compose("config.yaml")

    print(cfg.model.hidden_1_size)
    print(OmegaConf.to_yaml(cfg, resolve=True))


def load_cfg() -> Params:
    initialize(version_base="1.3", config_path="../configs")

    cfg: Params = compose("config.yaml")

    return cfg


def make_params(cfg: DictConfig):
    return Params(
        cfg.data, cfg.model, cfg.training, cfg.artifacts, cfg.onnx, cfg.serving
    )
