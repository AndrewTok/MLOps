import fire

from ml_ops import infer as m_infer
from ml_ops import train as m_train

from ml_ops import config
def train():
    m_train.train()


def infer():
    cfg = config.load_cfg()
    m_infer.infer(cfg)

def check_cfg():
    config.check_cfg()
    pass

def run_mlflow_tracking_server():
    m_train.start_mlflow_server()

if __name__ == '__main__':
    fire.Fire()
