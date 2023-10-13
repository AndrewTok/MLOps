import fire

from ml_ops import infer as m_infer
from ml_ops import train as m_train


def train():
    m_train.train()


def infer():
    m_infer.infer('trained_model_params.pt', 'dataset')


if __name__ == '__main__':
    fire.Fire()
