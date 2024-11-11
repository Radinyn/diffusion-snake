import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from torchvision.utils import make_grid
from pathlib import Path
from snake.agent import SnakeNet, direction_from_index
from snake.game import Engine, StepState

tensor_to_image = transforms.ToPILImage()


def load_model(path: str | Path | None = None) -> SnakeNet:
    if path is None:
        path = Path(__file__).parent.parent / "agent/models/best.pt"
    model = SnakeNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Iterate in sliding window for diffuzer training
class SnakeDatasetIterator:
    def __init__(self, dataset: "SnakeDataset", window_size: int = 4):
        self.dataset = dataset
        self.playthrough_index = 0
        self.index = 0
        self.window_size = window_size

    def sliding_window(self) -> tuple[int, list[torch.Tensor]]:
        n = len(self.dataset.actions[self.playthrough_index])
        max_index = n - self.window_size

        if self.index <= max_index:
            # action that caused the last frame
            action = self.dataset.actions[self.playthrough_index][
                self.index + self.window_size - 1
            ]
            frames = self.dataset.frames[self.playthrough_index][
                self.index : self.index + self.window_size
            ]
            self.index += 1
            return (action, frames)

        # next playthrough
        self.index = 0
        self.playthrough_index += 1
        if self.playthrough_index >= len(self.dataset):
            raise StopIteration
        return self.sliding_window()

    def __iter__(self) -> "SnakeDatasetIterator":
        return self

    def __next__(self) -> tuple[int, list[torch.Tensor]]:
        return self.sliding_window()


class SnakeDataset:
    def __init__(self):
        self.actions: list[list[int]] = []
        self.frames: list[list[torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> SnakeDatasetIterator:
        return SnakeDatasetIterator(self)

    @classmethod
    def load(cls, path: str | Path) -> "SnakeDataset":
        with open(path, "rb") as f:
            actions, frames = pickle.load(f)

        dataset = cls()
        dataset.actions = actions
        dataset.frames = frames
        return dataset

    def save(self, path: str | Path):
        with open(path, "wb+") as f:
            pickle.dump((self.actions, self.frames), f)

    def to_device(self, device: torch.device) -> "SnakeDataset":
        self.frames = [
            [frame.to(device=device) for frame in frames_playthrough]
            for frames_playthrough in self.frames
        ]
        return self

    def add_playthrough(self, actions: list[int], frames: list[torch.Tensor]):
        self.actions.append(actions)
        self.frames.append(frames)

    @staticmethod
    def create_playthrough(
        agent_model: SnakeNet,
        death_frames_to_keep: int = 3,
    ) -> tuple[list[int], list[torch.Tensor]]:
        if death_frames_to_keep < 1:
            raise ValueError("`death_frames_to_keep` must be at least 1")

        engine = Engine()

        # action that caused that frame
        actions = []
        frames = []

        while death_frames_to_keep > 0:
            desc_vec = engine.get_description_vector()
            probs = agent_model(desc_vec)
            distribution = torch.distributions.Categorical(probs)
            sample = int(distribution.sample())
            direction = direction_from_index(sample)

            if engine.step(direction) == StepState.DEAD:
                death_frames_to_keep -= 1

            actions.append(sample)
            frames.append(engine.get_frame())

        return actions, frames

    @staticmethod
    def display_frames(
        frames: list[torch.Tensor],
        nrow: int = 25,
    ):
        grid = make_grid(frames, nrow=nrow)
        grid_img = tensor_to_image(grid)

        plt.xticks([])
        plt.yticks([])
        plt.imshow(grid_img)
        plt.show()


if __name__ == "__main__":
    GENERATE = True
    N_PLAYTHROUGHS = 1000
    path = Path(__file__).parent / f"data/dataset{N_PLAYTHROUGHS}.pickle"

    if GENERATE:
        model = load_model()
        dataset = SnakeDataset()

        for _playthrough in tqdm(range(N_PLAYTHROUGHS), desc="Generating SnakeDataset"):
            actions, frames = SnakeDataset.create_playthrough(model)
            dataset.add_playthrough(actions, frames)

        dataset.save(path)
    else:
        dataset = SnakeDataset.load(path)

        try:
            for action, frames in dataset:
                SnakeDataset.display_frames(frames)
        except KeyboardInterrupt:
            pass
