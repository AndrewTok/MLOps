import torch


class LinearBlock(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_out: int,
    ):
        super().__init__()

        self.LinearRelu = torch.nn.Sequential(
            torch.nn.Linear(
                n_input,
                n_out,
            ),
            torch.nn.BatchNorm1d(n_out),
            torch.nn.ReLU(),
        )

    def forward(
        self,
        x,
    ):
        return self.LinearRelu(x)


class SimpleNet(torch.nn.Module):
    def __init__(
        self,
        n_features: int = 4,
        n_classes: int = 3,
        hidden_1=64,
        hidden_2=32,
    ):
        super().__init__()

        # hidden_1 = 64
        # hidden_2 = 32

        self.model = torch.nn.Sequential(
            LinearBlock(
                n_features,
                hidden_1,
            ),
            LinearBlock(
                hidden_1,
                hidden_2,
            ),
            torch.nn.Linear(
                hidden_2,
                n_classes,
            ),
        )

        self.SM = torch.nn.Softmax(dim=1)

    def forward(
        self,
        x,
    ):
        x1 = self.model(x)
        return self.SM(x1)
