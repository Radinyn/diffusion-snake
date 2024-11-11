import torch
from enum import Enum
from torch import Tensor
from .world import Tile, World
from PIL.Image import Image


class Direction(Enum):
    UP = Tensor([-1, 0])
    DOWN = Tensor([1, 0])
    LEFT = Tensor([0, -1])
    RIGHT = Tensor([0, 1])

    def is_vertical(self):
        return self == Direction.UP or self == Direction.DOWN

    def to_int(self) -> int:
        match self:
            case Direction.UP:
                return 0
            case Direction.DOWN:
                return 1
            case Direction.LEFT:
                return 2
            case Direction.RIGHT:
                return 3


class StepState(Enum):
    APPLE = 0
    ALIVE = 1
    DEAD = 2


class Engine:
    def __init__(self, body: list[Tensor] | None = None):
        self.direction = Direction.UP
        self.world = World()

        if body:
            self.body = body
        else:
            self.body = [
                Tensor([7, 7]),
                Tensor([8, 7]),
                Tensor([9, 7]),
            ]

        for segment in self.body:
            self.world[segment] = Tile.SNAKE

        self.apple = self.world.place_apple()

    def step(self, direction: Direction | None) -> bool:
        if direction and self.direction.is_vertical() != direction.is_vertical():
            self.direction = direction

        head = self.body[0]
        tail = self.body[-1]

        next_head = head + self.direction.value
        next_tile = self.world[next_head]

        match next_tile:
            case Tile.APPLE:
                self.body.insert(0, next_head)
                self.world[next_head] = Tile.SNAKE
                self.apple = self.world.place_apple()
                return StepState.APPLE

            case Tile.BACKGROUND:
                self.body.insert(0, next_head)
                self.world[next_head] = Tile.SNAKE
                self.body.pop()
                self.world[tail] = Tile.BACKGROUND
                return StepState.ALIVE

            case Tile.BORDER | Tile.SNAKE | Tile.LOST:
                self.world.death_screen()
                return StepState.DEAD

    def get_distance_head_to_apple(self):
        head = self.body[0]
        return torch.sum(torch.abs(head - self.apple))

    def raycast_for_target(self, direction: Direction, target: Tile) -> Tensor:
        curr = self.body[0].clone() + direction.value

        i = 0.0
        tile = self.world[curr]
        while tile != target and tile != Tile.BORDER and tile != Tile.LOST:
            curr += direction.value
            tile = self.world[curr]
            i += 1.0

        return i / self.world.n if tile == target else 1.0

    def get_description_vector(self):
        targets = (Tile.APPLE, Tile.SNAKE, Tile.BORDER)
        vector = torch.zeros(len(targets) * len(Direction) + 4)

        i = 0
        for target in targets:
            for direction in Direction:
                vector[i] = self.raycast_for_target(direction, target)
                i += 1

        vector[i] = (self.apple[0] - self.body[0][0]) / self.world.n
        i += 1

        vector[i] = (self.apple[1] - self.body[0][1]) / self.world.n
        i += 1

        vector[i] = self.body[0][0] / self.world.n
        i += 1

        vector[i] = self.body[0][1] / self.world.n
        return vector

    def get_image(self) -> Image:
        return self.world.to_image()

    def get_frame(self) -> Tensor:
        return self.world.board.clone()
