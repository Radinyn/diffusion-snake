import torch


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, layers: list[int]):
        super().__init__()

        linear_layers = []
        for i in range(len(layers) - 1):
            linear_layers.append(
                torch.nn.Linear(
                    in_features=layers[i],
                    out_features=layers[i + 1],
                )
            )

            if i != len(layers) - 2:
                linear_layers.append(torch.nn.ReLU())

        self.network = torch.nn.Sequential(*linear_layers, torch.nn.Softmax())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SnakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = MultiLayerPerceptron(layers=[3 * 4 + 4, 32, 32, 5])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
