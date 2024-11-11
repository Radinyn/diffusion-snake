import torch
import matplotlib.pyplot as plt
import random
from torch import Tensor
import torchvision.transforms as transforms
from enum import Enum
from PIL.Image import Image

tensor_to_image = transforms.ToPILImage()


class Tile(Enum):
    BORDER = Tensor([0.0, 0.0, 0.0])
    BACKGROUND = Tensor([0.5, 0.5, 0.5])
    SNAKE = Tensor([0, 1.0, 0.0])
    APPLE = Tensor([1.0, 0.0, 0.0])
    LOST = Tensor([0.0, 0.0, 1.0])

    @classmethod
    def close(cls, tensor: Tensor, eps: float = 0.01):
        for tile in cls:
            if torch.sum(torch.abs(tile.value - tensor)) < eps:
                return tile

    def all(self, n: int) -> Tensor:
        return self.value.expand(n, n, 3).movedim(-1, 0)


class World:
    def __init__(self, n: int = 15):
        self.n = n
        self.board: Tensor = self.__create_empty_board(n)

        self.ALL_APPLES = Tile.APPLE.value.expand(n, n, 3).movedim(-1, 0)
        self.ALL_BACKGROUND = Tile.BACKGROUND.value.expand(n, n, 3).movedim(-1, 0)
        self.DEATH_SCREEN = Tile.LOST.value.expand(n, n, 3).movedim(-1, 0)

    @staticmethod
    def __create_empty_board(n: int) -> Tensor:
        board = Tile.BACKGROUND.value.expand(n, n, 3).movedim(-1, 0).clone()
        board[:, 0, :] = Tile.BORDER.value.unsqueeze(dim=1)
        board[:, -1, :] = Tile.BORDER.value.unsqueeze(dim=1)
        board[:, :, 0] = Tile.BORDER.value.unsqueeze(dim=1)
        board[:, :, -1] = Tile.BORDER.value.unsqueeze(dim=1)
        return board

    def __get_background_indices(self) -> Tensor:
        return torch.all(self.board == self.ALL_BACKGROUND, dim=0).nonzero()

    def __getitem__(self, idx: Tensor) -> Tile:
        if len(idx) != 2:
            return KeyError(f"You have to pass exactly two indices, got {len(idx)}")
        return Tile.close(self.board[:, int(idx[0]), int(idx[1])])

    def __setitem__(self, idx: Tensor, tile: Tile):
        if len(idx) != 2:
            return KeyError(f"You have to pass exactly two indices, got {len(idx)}")
        self.board[:, int(idx[0]), int(idx[1])] = tile.value

    def place_apple(self) -> Tensor:
        possible_indices = self.__get_background_indices()
        ab = random.choice(possible_indices)
        a, b = ab
        self.board[:, a, b] = Tile.APPLE.value
        return ab

    def death_screen(self):
        self.board = self.DEATH_SCREEN

    def to_image(self) -> Image:
        return tensor_to_image(self.board)

    def plot(self):
        plt.imshow(self.to_image())
        plt.yticks([])
        plt.xticks([])
        plt.show()


if __name__ == "__main__":
    board = World()
    board.place_apple()
    board.plot()
