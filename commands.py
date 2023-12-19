import fire

from ml_ops import config
from ml_ops import infer as m_infer
from ml_ops import serving_utils
from ml_ops import train as m_train
from ml_ops import triton_client


def train():
    m_train.train()


def infer():
    cfg = config.load_cfg()
    m_infer.infer(cfg)


def run_mlflow_tracking_server():
    serving_utils.start_mlflow_server()


def run_server():
    cfg = config.load_cfg()
    m_infer.infer_onnx(cfg)


def run_triton_test():
    triton_client.main()


if __name__ == '__main__':
    fire.Fire()
