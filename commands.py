import fire
from ml_ops import train as m_train, infer as m_infer



def train():
    m_train.train()


def infer():
    m_infer.infer('trained_model_params.pt', 'dataset')



if __name__ == '__main__':
    fire.Fire()