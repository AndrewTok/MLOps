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
class Params:
    data: Data
    model: Model
    training: Training

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

