import fire

from ml_ops import config
from ml_ops import infer as m_infer
from ml_ops import serving_utils
from ml_ops import train as m_train
from ml_ops import triton_client


def train():
    serving_utils.load_data()
    m_train.train()


def infer():
    serving_utils.load_data()
    cfg = config.load_cfg()
    m_infer.infer(cfg)


def run_mlflow_tracking_server():
    serving_utils.load_data()
    serving_utils.start_mlflow_server()


def run_server():
    serving_utils.load_data()
    cfg = config.load_cfg()
    m_infer.infer_onnx(cfg)


def run_triton_test():
    serving_utils.load_data()
    triton_client.main()


if __name__ == '__main__':
    fire.Fire()
