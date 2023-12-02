from dataclasses import dataclass


import hydra
from hydra.core.config_store import ConfigStore

from omegaconf import OmegaConf

from hydra import compose, initialize


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


@dataclass
class Params:
    data: Data
    model: Model
    training: Training
    artifacts: Artifacts

    @staticmethod
    def get_model_save_path(model: Model):
        return model.save_dir + 'trained_' + model.name + '.pt'

cs = ConfigStore.instance()
cs.store(name="params", node=Params)

def check_cfg():


    initialize(version_base="1.3", config_path="../configs")

    cfg: Params = compose("config.yaml")

    # data = Data('name', 'path')
    # train = Training(16, 32, 1e2)
    # model = Model('name', 32, 64)

    # cfg = Params(data, model, train)

    print(cfg.model.hidden_1_size)
    print(OmegaConf.to_yaml(cfg, resolve=True))


def load_cfg() -> Params:
    
    initialize(version_base="1.3", config_path="../configs")

    cfg: Params = compose("config.yaml")
    
    return cfg

